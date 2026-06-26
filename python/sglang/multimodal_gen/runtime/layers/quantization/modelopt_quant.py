# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/modelopt_quant.py
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight,
    slice_nvfp4_output,
)
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.utils.common import is_flashinfer_available, round_up

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer
else:
    flashinfer = None


def _modelopt_fp4_batched_residual_gate_enabled() -> bool:
    return os.environ.get("SGLANG_LTX2_FP4_FUSED_BATCHED_RESIDUAL_GATE", "0") == "1"


@lru_cache(maxsize=1)
def _get_fp4_quantize_op():
    return current_platform.get_modelopt_fp4_quantize_op()


@lru_cache(maxsize=1)
def _get_fp4_gemm_op():
    return current_platform.get_modelopt_fp4_gemm_op()


@lru_cache(maxsize=1)
def _get_fp4_bias_gelu_gemm_op():
    try:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm_bias_gelu

        return cutlass_scaled_fp4_mm_bias_gelu
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _get_fp4_per_col_residual_gate_gemm_op():
    try:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm_per_col_residual_gate

        return cutlass_scaled_fp4_mm_per_col_residual_gate
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _get_fp4_batched_per_col_residual_gate_gemm_op():
    try:
        from sglang.jit_kernel.nvfp4 import (
            cutlass_scaled_fp4_mm_batched_per_col_residual_gate,
        )

        return cutlass_scaled_fp4_mm_batched_per_col_residual_gate
    except ImportError:
        return None


def _prepare_nvfp4_weight_bytes(
    weight: torch.Tensor, *, swap_weight_nibbles: bool
) -> torch.Tensor:
    """Normalize serialized NVFP4 bytes before padding for the runtime kernel."""
    if not swap_weight_nibbles:
        return weight.contiguous()
    return ((weight >> 4) | (weight << 4)).contiguous()


def _nvfp4_scale_swizzled_to_linear(scales: torch.Tensor) -> torch.Tensor:
    """Convert FlashInfer swizzled block scales back to row-major block scales."""
    scale_ndim = scales.ndim
    if scale_ndim == 2:
        scales = scales.unsqueeze(0)
    if scales.ndim != 3:
        raise ValueError(f"Expected 2D or 3D NVFP4 scales, got {scales.ndim}D")

    B, M, K = scales.shape
    if M % 128 != 0 or K % 4 != 0:
        raise ValueError(
            "FlashInfer swizzled NVFP4 scales must be padded to "
            f"[multiple of 128, multiple of 4], got {tuple(scales.shape)}"
        )

    scales_u8 = scales.contiguous().view(torch.uint8)
    linear = scales_u8.reshape(B, M // 128, K // 4, 32, 4, 4)
    linear = linear.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(B, M, K)
    linear = linear.view(torch.float8_e4m3fn)
    return linear.squeeze(0) if scale_ndim == 2 else linear


def _require_flashinfer():
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is required for the diffusion NVFP4 FlashInfer path."
        )
    return flashinfer


class ModelOptQuantConfig(QuantizationConfig):
    def __init__(
        self,
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
    ):
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.exclude_modules = exclude_modules or []

    def _get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        *,
        Linear: type[LinearMethodBase],
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix) or (
                self.packed_modules_mapping
                and is_layer_skipped(prefix, [], self.packed_modules_mapping)
            ):
                return UnquantizedLinearMethod()
            return Linear(self)
        return None

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant) -> Optional[str]:
        if hf_quant_config is None:
            return None

        quant_algo = (
            hf_quant_config.get("quant_algo")
            or hf_quant_config.get("quantization", {}).get("quant_algo")
            or ""
        ).upper()
        if user_quant in {"modelopt", "modelopt_fp8"} and "FP8" in quant_algo:
            return "modelopt_fp8"
        if user_quant in {"modelopt", "modelopt_fp4"} and (
            "NVFP4" in quant_algo or "FP4" in quant_algo
        ):
            return "modelopt_fp4"
        return None

    def is_layer_excluded(self, prefix: str) -> bool:
        for pattern in self.exclude_modules:
            regex_str = re.escape(pattern).replace(r"\*", r".*")
            if re.fullmatch(regex_str, prefix):
                return True
        return False


class ModelOptFp8Config(ModelOptQuantConfig):
    """Config class for ModelOpt FP8 diffusion checkpoints."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        exclude_modules: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(exclude_modules, packed_modules_mapping)
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt FP8 checkpoint. The format is experimental and subject to change."
            )

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_method = config.get("quant_algo")
        exclude_modules = config.get("ignore")
        if quant_method is None:
            try:
                quantization_section = cls.get_from_keys(config, ["quantization"])
                quant_method = quantization_section.get("quant_algo")
                exclude_modules = quantization_section.get("exclude_modules")
            except ValueError as exc:
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config."
                ) from exc

        if quant_method is None or "FP8" not in quant_method:
            raise ValueError(
                "ModelOptFp8Config only supports static FP8 quantization in SGLang diffusion."
            )

        return cls(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=exclude_modules,
            packed_modules_mapping=config.get("packed_modules_mapping"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(layer, prefix, Linear=ModelOptFp8LinearMethod)


class ModelOptFp4Config(ModelOptQuantConfig):
    """Config class for NVFP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        group_size: int = None,
        exclude_modules: List[str] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        checkpoint_uses_packed_qkv: bool = False,
        swap_weight_nibbles: bool = True,
        weight_scale_layout: str = "swizzled",
    ) -> None:
        super().__init__(exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size
        self.checkpoint_uses_packed_qkv = checkpoint_uses_packed_qkv
        self.swap_weight_nibbles = swap_weight_nibbles
        weight_scale_layout = weight_scale_layout.lower()
        if weight_scale_layout not in {"swizzled", "linear"}:
            raise ValueError(
                "weight_scale_layout must be either 'swizzled' or 'linear', "
                f"got {weight_scale_layout!r}"
            )
        self.weight_scale_layout = weight_scale_layout

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def common_group_size(cfg: dict) -> int:
        """Return the unique group_size across the config; raise if missing/mismatched."""
        sizes = set()

        def _add_group_size_from_dict(config: dict):
            group_size = config.get("group_size")
            if isinstance(group_size, int):
                sizes.add(group_size)

        # Top-level and 'quantization' block
        _add_group_size_from_dict(cfg)
        quantization = cfg.get("quantization")
        if isinstance(quantization, dict):
            _add_group_size_from_dict(quantization)

        # config_groups: accept group-level or nested dicts (e.g., weights/input_activations)
        for config_groups in (cfg.get("config_groups") or {}).values():
            if isinstance(config_groups, dict):
                _add_group_size_from_dict(config_groups)
                for config_group in config_groups.values():
                    if isinstance(config_group, dict):
                        _add_group_size_from_dict(config_group)

        if not sizes:
            raise ValueError("No group_size found in config.")
        if len(sizes) > 1:
            raise ValueError(f"Inconsistent group_size values: {sorted(sizes)}")
        return next(iter(sizes))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4Config:
        group_size = None
        exclude_modules = []
        swap_weight_nibbles = True
        weight_scale_layout = "swizzled"

        # Flat format (config.json quantization_config)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            group_size = config.get("group_size")
            if group_size is None:
                config_groups = config.get("config_groups", {})
                if config_groups:
                    first_group = next(iter(config_groups.values()), {})
                    group_size = first_group.get("weights", {}).get("group_size")
            exclude_modules = config.get("ignore", [])
            swap_weight_nibbles = config.get("swap_weight_nibbles", True)
            weight_scale_layout = config.get("weight_scale_layout", "swizzled")
        else:
            # Nested format (hf_quant_config.json)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
                swap_weight_nibbles = quant_config.get(
                    "swap_weight_nibbles",
                    config.get("swap_weight_nibbles", True),
                )
                weight_scale_layout = quant_config.get(
                    "weight_scale_layout",
                    config.get("weight_scale_layout", "swizzled"),
                )
            except (ValueError, KeyError):
                raise ValueError("Cannot find 'quant_algo' in quantization config.")

        if quant_method not in ["NVFP4"]:
            raise ValueError(
                f"Only NVFP4 quantization is supported for diffusion, got '{quant_method}'."
            )

        if group_size is None or exclude_modules is None:
            raise ValueError(
                "NVFP4 quantization requires group_size and exclude_modules "
                "in the quantization config"
            )
        return cls(
            is_checkpoint_nvfp4_serialized=True,
            group_size=group_size,
            exclude_modules=exclude_modules,
            packed_modules_mapping=config.get("packed_modules_mapping"),
            checkpoint_uses_packed_qkv=config.get("checkpoint_uses_packed_qkv", False),
            swap_weight_nibbles=swap_weight_nibbles,
            weight_scale_layout=weight_scale_layout,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(layer, prefix, Linear=ModelOptFp4LinearMethod)


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt static FP8 checkpoints."""

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=weight_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        if self.quant_config.is_checkpoint_fp8_serialized:
            for scale_name in ["weight_scale", "input_scale"]:
                layer.register_parameter(
                    scale_name,
                    PerTensorScaleParameter(
                        data=torch.full(
                            (len(output_partition_sizes),),
                            torch.finfo(torch.float32).min,
                            dtype=torch.float32,
                        ),
                        weight_loader=weight_loader,
                    ),
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        max_w_scale, quantized_weight = requantize_with_max_scale(
            layer.weight, layer.weight_scale, layer.logical_widths
        )
        # Preserve the parameter subclass metadata while rebinding to the
        # transposed FP8 view expected by the runtime.
        layer.weight.data = quantized_weight.t().detach()
        layer.weight.requires_grad_(False)
        if self.cutlass_fp8_supported:
            max_w_scale = convert_to_channelwise(max_w_scale, layer.logical_widths)
        copy_or_rebind_param(layer, "weight_scale", max_w_scale)
        copy_or_rebind_param(layer, "input_scale", layer.input_scale.max())

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )


def modelopt_fp4_quantize_activation(
    x: torch.Tensor, input_scale_inv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], torch.dtype]:
    input_shape = tuple(x.shape)
    output_dtype = x.dtype
    x = x.view(-1, input_shape[-1])

    fp4_quantize = _get_fp4_quantize_op()
    if fp4_quantize is None:
        raise RuntimeError(
            "No FP4 quantization kernel available. Install flashinfer or sgl_kernel."
        )

    x_fp4, x_scale_interleaved = fp4_quantize(x, input_scale_inv)
    return x_fp4, x_scale_interleaved, input_shape, output_dtype


def modelopt_fp4_apply_quantized_linear(
    layer: torch.nn.Module,
    x_fp4: torch.Tensor,
    x_scale_interleaved: torch.Tensor,
    input_shape: tuple[int, ...],
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output_size = layer.output_size_per_partition
    output_shape = list(input_shape[:-1]) + [output_size]

    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    w = layer.weight
    w_scale_interleaved = layer.weight_scale_interleaved

    if x_scale_interleaved.dtype == torch.uint8:
        x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
    if w_scale_interleaved.dtype == torch.uint8:
        w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)
    fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
    if flashinfer_backend is not None:
        out = fp4_gemm(
            x_fp4,
            w.T,
            x_scale_interleaved,
            w_scale_interleaved.T,
            layer.alpha,
            output_dtype,
            backend=flashinfer_backend,
        )
    elif fp4_gemm is not None:
        out = fp4_gemm(
            x_fp4,
            w,
            x_scale_interleaved,
            w_scale_interleaved,
            layer.alpha,
            output_dtype,
        )
    else:
        raise RuntimeError(
            "No FP4 GEMM kernel available. Install flashinfer or sgl_kernel."
        )

    out = slice_nvfp4_output(out, output_size)

    if bias is not None:
        out.add_(bias)
    return out.view(*output_shape)


def modelopt_fp4_apply_quantized_linear_bias_gelu(
    layer: torch.nn.Module,
    x_fp4: torch.Tensor,
    x_scale_interleaved: torch.Tensor,
    input_shape: tuple[int, ...],
    output_dtype: torch.dtype,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    """Apply CUTLASS NVFP4 GEMM with bias+GELU in the GEMM epilogue."""
    fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
    if fp4_gemm is None or flashinfer_backend is not None:
        return None
    fused_gemm = _get_fp4_bias_gelu_gemm_op()
    if fused_gemm is None:
        return None

    output_size = layer.output_size_per_partition
    output_shape = list(input_shape[:-1]) + [output_size]
    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    w = layer.weight
    w_scale_interleaved = layer.weight_scale_interleaved
    if x_scale_interleaved.dtype == torch.uint8:
        x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
    if w_scale_interleaved.dtype == torch.uint8:
        w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)

    if bias.shape[0] != w.shape[0]:
        if bias.shape[0] > w.shape[0]:
            return None
        padded_bias = torch.zeros((w.shape[0],), dtype=bias.dtype, device=bias.device)
        padded_bias[: bias.shape[0]].copy_(bias)
        bias = padded_bias

    out = fused_gemm(
        x_fp4,
        w,
        x_scale_interleaved,
        w_scale_interleaved,
        layer.alpha,
        bias,
        output_dtype,
    )
    out = slice_nvfp4_output(out, output_size)
    return out.view(*output_shape)


def modelopt_fp4_apply_linear_bias_gelu(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor | None:
    x_fp4, x_scale_interleaved, input_shape, output_dtype = (
        modelopt_fp4_quantize_activation(x, layer.input_scale_inv)
    )
    return modelopt_fp4_apply_quantized_linear_bias_gelu(
        layer, x_fp4, x_scale_interleaved, input_shape, output_dtype, bias
    )


def _modelopt_fp4_get_per_col_gate(
    gate: torch.Tensor,
    output_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    if gate.dtype != output_dtype or gate.shape[-1] != output_size:
        return None
    if gate.ndim == 1:
        return gate.contiguous()
    if all(int(dim) == 1 for dim in gate.shape[:-1]):
        return gate.reshape(output_size).contiguous()
    return None


def _modelopt_fp4_get_batched_per_col_gate(
    gate: torch.Tensor,
    batch_size: int,
    output_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    if gate.dtype != output_dtype or gate.shape[-1] != output_size:
        return None
    if gate.ndim < 2 or int(gate.shape[0]) != int(batch_size):
        return None
    if any(int(dim) != 1 for dim in gate.shape[1:-1]):
        return None
    return gate.reshape(batch_size, output_size).contiguous()


def _modelopt_fp4_accept_batched_activation_scales(
    x_scale_interleaved: torch.Tensor,
    batch_size: int,
    m_per_batch: int,
) -> torch.Tensor | None:
    # NVFP4 scale tensors are swizzled for the GEMM problem shape. A flat
    # [B*M, K] quantization cannot be safely row-copied into [B, M, K]
    # layout when M needs per-batch padding; use per-batch quantization instead.
    rounded_m = round_up(m_per_batch, 128)
    expected_rows = batch_size * rounded_m
    if int(x_scale_interleaved.shape[0]) == expected_rows:
        return x_scale_interleaved
    return None


def _modelopt_fp4_get_batched_weight_scales(
    layer: torch.nn.Module,
    w_scale_interleaved: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    cache_key = (
        int(batch_size),
        int(w_scale_interleaved.data_ptr()),
        tuple(int(dim) for dim in w_scale_interleaved.shape),
        w_scale_interleaved.dtype,
        str(w_scale_interleaved.device),
    )
    cache = getattr(layer, "_sglang_fp4_batched_weight_scale_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1]
    batched = w_scale_interleaved.repeat((batch_size, 1)).contiguous()
    setattr(layer, "_sglang_fp4_batched_weight_scale_cache", (cache_key, batched))
    return batched


def modelopt_fp4_apply_quantized_linear_batched_per_col_residual_gate(
    layer: torch.nn.Module,
    x_fp4: torch.Tensor,
    x_scale_interleaved: torch.Tensor,
    input_shape: tuple[int, ...],
    output_dtype: torch.dtype,
    residual: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    if not _modelopt_fp4_batched_residual_gate_enabled():
        return None
    fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
    if fp4_gemm is None or flashinfer_backend is not None:
        return None
    fused_gemm = _get_fp4_batched_per_col_residual_gate_gemm_op()
    if fused_gemm is None or len(input_shape) < 3:
        return None

    batch_size = int(input_shape[0])
    if batch_size <= 1:
        return None
    m_per_batch = 1
    for dim in input_shape[1:-1]:
        m_per_batch *= int(dim)
    if m_per_batch <= 0 or int(x_fp4.shape[0]) != batch_size * m_per_batch:
        return None

    output_size = layer.output_size_per_partition
    output_shape = list(input_shape[:-1]) + [output_size]
    if list(residual.shape) != output_shape or residual.dtype != output_dtype:
        return None
    if bias.dtype != output_dtype or bias.ndim != 1 or bias.shape[0] != output_size:
        return None

    gate_2d = _modelopt_fp4_get_batched_per_col_gate(
        gate, batch_size, output_size, output_dtype
    )
    if gate_2d is None:
        return None

    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    w = layer.weight
    w_scale_interleaved = layer.weight_scale_interleaved
    if int(w.shape[0]) != int(output_size):
        return None
    if x_scale_interleaved.dtype == torch.uint8:
        x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
    if w_scale_interleaved.dtype == torch.uint8:
        w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)

    x_scale_batched = _modelopt_fp4_accept_batched_activation_scales(
        x_scale_interleaved, batch_size, m_per_batch
    )
    if x_scale_batched is None:
        return None
    w_scale_batched = _modelopt_fp4_get_batched_weight_scales(
        layer, w_scale_interleaved, batch_size
    )

    residual_2d = residual.reshape(batch_size * m_per_batch, output_size)
    if not residual_2d.is_contiguous():
        residual_2d = residual_2d.contiguous()

    alpha = layer.alpha
    if alpha.numel() != 1:
        return None
    gate_alpha = (
        (gate_2d.float() * alpha.float().reshape(())).to(output_dtype).contiguous()
    )
    bias_gate = (
        (bias.float().reshape(1, output_size) * gate_2d.float())
        .to(output_dtype)
        .contiguous()
    )

    out = fused_gemm(
        x_fp4,
        w,
        x_scale_batched,
        w_scale_batched,
        alpha,
        residual_2d,
        gate_alpha,
        bias_gate,
        output_dtype,
        batch_size,
        m_per_batch,
    )
    out = slice_nvfp4_output(out, output_size)
    return out.view(*output_shape)


def modelopt_fp4_apply_linear_batched_per_col_residual_gate(
    layer: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    if not _modelopt_fp4_batched_residual_gate_enabled():
        return None
    input_shape = tuple(x.shape)
    if len(input_shape) < 3:
        return None
    batch_size = int(input_shape[0])
    if batch_size <= 1:
        return None
    m_per_batch = 1
    for dim in input_shape[1:-1]:
        m_per_batch *= int(dim)
    if m_per_batch <= 0:
        return None

    output_dtype = x.dtype
    x_3d = x.reshape(batch_size, m_per_batch, input_shape[-1])
    x_fp4_parts = []
    x_scale_parts = []
    try:
        for batch_idx in range(batch_size):
            x_fp4_part, x_scale_part, _, _ = modelopt_fp4_quantize_activation(
                x_3d[batch_idx], layer.input_scale_inv
            )
            x_fp4_parts.append(x_fp4_part)
            x_scale_parts.append(x_scale_part)
    except RuntimeError:
        return None

    x_fp4 = torch.cat(x_fp4_parts, dim=0).contiguous()
    x_scale_interleaved = torch.cat(x_scale_parts, dim=0).contiguous()
    return modelopt_fp4_apply_quantized_linear_batched_per_col_residual_gate(
        layer,
        x_fp4,
        x_scale_interleaved,
        input_shape,
        output_dtype,
        residual,
        gate,
        bias,
    )


def modelopt_fp4_apply_quantized_linear_per_col_residual_gate(
    layer: torch.nn.Module,
    x_fp4: torch.Tensor,
    x_scale_interleaved: torch.Tensor,
    input_shape: tuple[int, ...],
    output_dtype: torch.dtype,
    residual: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    """Apply CUTLASS NVFP4 GEMM with per-column gate/residual in the epilogue.

    This covers the LTX2 stage-2 case where batch is one and the Ada gate is a
    single channel vector shared by all video tokens. Stage-1 has batch-specific
    gates and should fall back to the standalone epilogue kernel.
    """
    fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
    if fp4_gemm is None or flashinfer_backend is not None:
        return None
    fused_gemm = _get_fp4_per_col_residual_gate_gemm_op()
    if fused_gemm is None:
        return None

    output_size = layer.output_size_per_partition
    output_shape = list(input_shape[:-1]) + [output_size]
    if list(residual.shape) != output_shape or residual.dtype != output_dtype:
        return None
    if bias.dtype != output_dtype or bias.ndim != 1 or bias.shape[0] != output_size:
        return None

    gate_col = _modelopt_fp4_get_per_col_gate(gate, output_size, output_dtype)
    if gate_col is None:
        return modelopt_fp4_apply_quantized_linear_batched_per_col_residual_gate(
            layer,
            x_fp4,
            x_scale_interleaved,
            input_shape,
            output_dtype,
            residual,
            gate,
            bias,
        )

    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    w = layer.weight
    w_scale_interleaved = layer.weight_scale_interleaved
    if int(w.shape[0]) != int(output_size):
        return None
    if x_scale_interleaved.dtype == torch.uint8:
        x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
    if w_scale_interleaved.dtype == torch.uint8:
        w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)

    residual_2d = residual.reshape(-1, output_size)
    if not residual_2d.is_contiguous():
        residual_2d = residual_2d.contiguous()

    alpha = layer.alpha
    if alpha.numel() != 1:
        return None
    gate_alpha = (
        (gate_col.float() * alpha.float().reshape(())).to(output_dtype).contiguous()
    )
    bias_gate = (bias.float() * gate_col.float()).to(output_dtype).contiguous()

    out = fused_gemm(
        x_fp4,
        w,
        x_scale_interleaved,
        w_scale_interleaved,
        alpha,
        residual_2d,
        gate_alpha,
        bias_gate,
        output_dtype,
    )
    out = slice_nvfp4_output(out, output_size)
    return out.view(*output_shape)


def modelopt_fp4_apply_linear_per_col_residual_gate(
    layer: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor | None:
    batched = modelopt_fp4_apply_linear_batched_per_col_residual_gate(
        layer, x, residual, gate, bias
    )
    if batched is not None:
        return batched

    x_fp4, x_scale_interleaved, input_shape, output_dtype = (
        modelopt_fp4_quantize_activation(x, layer.input_scale_inv)
    )
    return modelopt_fp4_apply_quantized_linear_per_col_residual_gate(
        layer,
        x_fp4,
        x_scale_interleaved,
        input_shape,
        output_dtype,
        residual,
        gate,
        bias,
    )


class ModelOptFp4LinearMethod(LinearMethodBase):
    """NVFP4 linear method using CUTLASS FP4 GEMM."""

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                f"Unsupported model when input features size is {input_size_per_partition}, not multiple of 16, for NVFP4 quantization."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_nvfp4_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        set_weight_attrs(input_scale, {"missing_param_init": "ones"})
        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        set_weight_attrs(weight_scale_2, {"missing_param_init": "ones"})
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        set_weight_attrs(weight_scale, {"missing_param_init": "ones"})
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        copy_or_rebind_param(
            layer, "alpha", (input_scale_2 * weight_scale_2).to(torch.float32)
        )
        copy_or_rebind_param(
            layer, "input_scale_inv", (1 / input_scale_2).to(torch.float32)
        )

        layer.output_size_per_partition = layer.weight.shape[0]

        w = layer.weight.data
        w_swapped = _prepare_nvfp4_weight_bytes(
            w,
            swap_weight_nibbles=getattr(self.quant_config, "swap_weight_nibbles", True),
        )

        _, flashinfer_backend = _get_fp4_gemm_op()
        if flashinfer_backend == "trtllm":
            flashinfer_ops = _require_flashinfer()

            weight, _ = pad_nvfp4_weight(w_swapped, n_alignment=128, k_alignment=0)
            scales = layer.weight_scale
            if (
                getattr(self.quant_config, "weight_scale_layout", "swizzled")
                == "swizzled"
            ):
                scales = _nvfp4_scale_swizzled_to_linear(scales)
            if scales.shape[0] != weight.shape[0]:
                pad_n = weight.shape[0] - scales.shape[0]
                scales = torch.nn.functional.pad(scales, (0, 0, 0, pad_n))

            scale_k = scales.shape[1]
            weights_padding_cols = 0
            if scale_k % 4 != 0:
                padded_scale_k = round_up(scale_k, 4)
                pad_scale_k = padded_scale_k - scale_k
                scales = torch.nn.functional.pad(scales, (0, pad_scale_k, 0, 0))
                pad_weight_k = pad_scale_k * 8
                weight = torch.nn.functional.pad(weight, (0, pad_weight_k, 0, 0))
                weights_padding_cols = pad_weight_k

            epilogue_tile_m = 128
            shuffled_scale_shape = scales.shape
            if not weight.is_cuda:
                weight = weight.cuda()
            if scales.device != weight.device:
                scales = scales.to(device=weight.device)
            weight = flashinfer_ops.shuffle_matrix_a(
                weight.view(torch.uint8), epilogue_tile_m
            )
            scales = (
                flashinfer_ops.shuffle_matrix_sf_a(
                    scales.view(torch.uint8), epilogue_tile_m
                )
                .reshape(shuffled_scale_shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weights_padding_cols = weights_padding_cols
            copy_or_rebind_param(layer, "weight", weight)
            copy_or_rebind_param(layer, "weight_scale_interleaved", scales)
            return
        weight, weights_padding_cols = pad_nvfp4_weight(w_swapped)
        layer.weights_padding_cols = weights_padding_cols
        copy_or_rebind_param(layer, "weight", weight)

        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = round_up(M, 128)
        K_padded = round_up(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales

        _, flashinfer_backend = _get_fp4_gemm_op()
        uses_flux1_scale_layout = not getattr(
            self.quant_config, "checkpoint_uses_packed_qkv", False
        ) and getattr(layer, "prefix", "").startswith(
            ("transformer_blocks.", "single_transformer_blocks.")
        )
        if flashinfer_backend is None or uses_flux1_scale_layout:
            # CUTLASS and FLUX.1 CUDNN paths need the TMA scale layout.
            padded_scales = padded_scales.reshape(
                B, M_padded // 128, 4, 32, K_padded // 4, 4
            )
            padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)

        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        copy_or_rebind_param(layer, "weight_scale_interleaved", padded_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_fp4, x_scale_interleaved, input_shape, output_dtype = (
            modelopt_fp4_quantize_activation(x, layer.input_scale_inv)
        )
        return modelopt_fp4_apply_quantized_linear(
            layer, x_fp4, x_scale_interleaved, input_shape, output_dtype, bias
        )
