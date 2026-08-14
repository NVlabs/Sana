import json
import struct
from typing import Callable

import torch
from torch import nn

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.model.transformer import LTXModel
from ltx_core.quantization.policy import QuantizationPolicy


def _read_safetensors_dtypes(path: str) -> dict[str, str]:
    """Return ``{tensor_name: dtype_string}`` from the safetensors header."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
    return {k: v["dtype"] for k, v in header.items() if k != "__metadata__"}


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

        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn, device=device))
        self.weight_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
        self.input_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape

        # Clamp before cast: out-of-range values cast to NaN/saturated FP8, which
        # produces black-screen output on some checkpoints (e.g. ltx-2-19b-dev-fp8).
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        qinput = torch.clamp(x * self.input_scale.reciprocal(), fp8_min, fp8_max).to(torch.float8_e4m3fn)
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])
        output = torch._scaled_mm(
            qinput,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        return output


def quantize_weight_to_fp8_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to ``float8_e4m3fn`` with a per-tensor scale."""
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
        weight_scale = scale.reciprocal()
        return quantized_weight, weight_scale

    quantized_weight, weight_scale = _quantize(weight_fp32, scale, fp8_min, fp8_max)
    return quantized_weight, weight_scale


def _linear_to_fp8linear(layer: nn.Linear) -> FP8Linear:
    """Create an ``FP8Linear`` matching the shape/bias of *layer*."""
    return FP8Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
    )


def _swap_linears_to_fp8(model: nn.Module, should_swap: Callable[[str], bool]) -> nn.Module:
    """Replace nn.Linear layers with FP8Linear where ``should_swap(name)`` returns True."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue
        if not should_swap(name):
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


def get_fp8_swap_module_ops(checkpoint_path: str) -> tuple[ModuleOps, ...]:
    """Return the FP8 swap ``ModuleOps`` for layers whose ``.weight`` is ``F8_E4M3``
    and which have a sibling ``.weight_scale`` tensor in the checkpoint.
    Raises ``ValueError`` if no such layers are found — that combination is ambiguous
    (a BF16 checkpoint with this policy would load as a no-op).
    """
    dtypes = _read_safetensors_dtypes(checkpoint_path)
    fp8_scale_paths = frozenset(
        key.removesuffix(".weight_scale")
        for key in dtypes
        if key.endswith(".weight_scale") and dtypes.get(key.removesuffix(".weight_scale") + ".weight") == "F8_E4M3"
    )
    if not fp8_scale_paths:
        raise ValueError(
            f"fp8_scaled_mm requires a pre-quantized checkpoint with F8_E4M3 .weight + .weight_scale "
            f"tensors, but {checkpoint_path!r} has none. Use QuantizationPolicy.fp8_cast() for BF16 checkpoints."
        )

    def _should_swap(name: str) -> bool:
        suffix = "." + name
        return any(p == name or p.endswith(suffix) for p in fp8_scale_paths)

    return (
        ModuleOps(
            name="fp8_swap_linears",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=lambda model: _swap_linears_to_fp8(model, _should_swap),
        ),
    )


def _fp8_scaled_mm_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize via ``weight.float() * weight_scale``, add the BF16 delta,
    and re-quantize to FP8 with a fresh per-tensor scale.
    Layers that were not swapped to scaled FP8 (e.g. small embedder linears
    excluded from the auto-discovered swap set) stay BF16 and have no
    ``.weight_scale`` companion -- for those, fall back to a plain bf16 fuse.
    """
    scale_key = key.replace(".weight", ".weight_scale")
    if scale_key not in model_sd.sd:
        return bf16_fuse_rule(key, weight, deltas, model_sd)
    weight_scale = model_sd.sd[scale_key]
    original_weight = weight.to(torch.float32) * weight_scale
    new_weight = original_weight + deltas.to(torch.float32)
    new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)
    return {key: new_fp8_weight, scale_key: new_weight_scale}


fp8_scaled_mm_fuse_rule = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_fp8_scaled_mm_fuse)


def build_policy(checkpoint_path: str) -> QuantizationPolicy:
    """FP8 scaled matmul for checkpoints pre-quantized with per-tensor scales.
    The set of layers to swap to ``FP8Linear`` is discovered from the
    checkpoint's ``.weight_scale`` tensors via suffix-matching against the
    model's named modules. Requires a pre-quantized checkpoint; for BF16
    checkpoints, use :func:`ltx_core.quantization.fp8_cast.build_policy`.
    """
    return QuantizationPolicy(
        sd_ops=None,
        module_ops=get_fp8_swap_module_ops(checkpoint_path),
        fuse_rule=fp8_scaled_mm_fuse_rule,
    )
