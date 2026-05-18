from typing import Callable

import torch
from torch import nn

from diffusion.refiner.vendor.ltx_core.loader.module_ops import ModuleOps
from diffusion.refiner.vendor.ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from diffusion.refiner.vendor.ltx_core.model.transformer import LTXModel


class FP8Linear(nn.Module):
    """Linear layer with FP8 weight storage for scaled matrix multiplication."""

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        fp8_shape = (in_features, out_features)
        self.weight = nn.Parameter(torch.empty(fp8_shape, dtype=torch.float8_e4m3fn, device=device))
        # Weight scale for FP8 dequantization (shape matches checkpoint format)
        self.weight_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
        # Input scale for static quantization (pre-quantized checkpoints)
        self.input_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape

        # Static quantization: use pre-computed scale
        qinput, cur_input_scale = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, self.input_scale)

        # Flatten to 2D for matmul
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])

        # FP8 scaled matmul
        output = torch.ops.trtllm.cublas_scaled_mm(
            qinput,
            self.weight,
            scale_a=cur_input_scale,
            scale_b=self.weight_scale,
            bias=None,
            out_dtype=x.dtype,
        )

        # Add bias
        if self.bias is not None:
            bias = self.bias
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        # Restore original shape
        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        return output


def quantize_weight_to_fp8_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to FP8 (float8_e4m3fn) using per-tensor scaling.
    Args:
        weight: The weight tensor to quantize (any dtype, will be cast to float32)
    Returns:
        Tuple of (quantized_weight, weight_scale):
        - quantized_weight: FP8 tensor, transposed for cublas_scaled_mm
        - weight_scale: Per-tensor scale factor (reciprocal of quantization scale)
    """
    weight_fp32 = weight.to(torch.float32)

    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    max_abs = torch.amax(torch.abs(weight_fp32))
    scale = fp8_max / max_abs

    @torch.compiler.disable
    def _quantize(
        weight_fp32: torch.Tensor, scale: torch.Tensor, fp8_min: torch.Tensor, fp8_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_weight = torch.clamp(weight_fp32 * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        quantized_weight = quantized_weight.t()
        weight_scale = scale.reciprocal()
        return quantized_weight, weight_scale

    quantized_weight, weight_scale = _quantize(weight_fp32, scale, fp8_min, fp8_max)
    return quantized_weight, weight_scale


def _should_skip_layer(layer_name: str, excluded_layer_substrings: tuple[str, ...]) -> bool:
    return any(substring in layer_name for substring in excluded_layer_substrings)


EXCLUDED_LAYER_SUBSTRINGS = (
    "patchify_proj",
    "adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "caption_projection",
    "proj_out",
    "audio_patchify_proj",
    "audio_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    "audio_caption_projection",
    "audio_proj_out",
    "transformer_blocks.0.",
    *[f"transformer_blocks.{i}." for i in range(43, 48)],
)


def _linear_to_fp8linear(layer: nn.Linear) -> FP8Linear:
    """
    Create an FP8Linear layer from an nn.Linear layer.
    Args:
        layer: The nn.Linear layer to convert (typically on meta device)
    Returns:
        A new FP8Linear with the same configuration
    """
    return FP8Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
    )


def _apply_fp8_prepare_to_model(model: nn.Module, excluded_layer_substrings: tuple[str, ...]) -> nn.Module:
    """Replace nn.Linear layers with FP8Linear in the module tree."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue

        if _should_skip_layer(name, excluded_layer_substrings):
            continue

        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        replacements.append((parent, attr_name, module))

    for parent, attr_name, linear in replacements:
        setattr(parent, attr_name, _linear_to_fp8linear(linear))

    return model


def _create_transpose_kv_operation(
    excluded_layer_substrings: tuple[str, ...],
) -> Callable[[str, torch.Tensor], list[KeyValueOperationResult]]:
    def transpose_if_matches(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        # Only process .weight keys
        if not key.endswith(".weight"):
            return [KeyValueOperationResult(key, value)]

        # Only transpose 2D FP8 tensors (Linear weights)
        if value.dim() != 2 or value.dtype != torch.float8_e4m3fn:
            return [KeyValueOperationResult(key, value)]

        # Check if the layer is excluded
        layer_name = key.rsplit(".weight", 1)[0]
        if _should_skip_layer(layer_name, excluded_layer_substrings):
            return [KeyValueOperationResult(key, value)]

        # Transpose to cuBLAS layout (in, out)
        transposed_weight = value.t()

        return [KeyValueOperationResult(key, transposed_weight)]

    return transpose_if_matches


FP8_TRANSPOSE_SD_OPS = SDOps("fp8_transpose_weights").with_kv_operation(
    _create_transpose_kv_operation(EXCLUDED_LAYER_SUBSTRINGS),
    key_prefix="transformer_blocks.",
    key_suffix=".weight",
)


FP8_PREPARE_MODULE_OPS = ModuleOps(
    name="fp8_prepare_for_loading",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: _apply_fp8_prepare_to_model(model, EXCLUDED_LAYER_SUBSTRINGS),
)
