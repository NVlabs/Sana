"""Implementation of blockwise FP8/FP6 quantization. Depends on ``ltx_kernels``.
This module imports the compiled ``ltx_kernels.blockwise`` kernels at top level
— without them built, simply importing this file raises :class:`ImportError`.
The intended access path is through ``ltx_core.quantization.blockwise.__init__``
which catches that and re-raises as a clean :class:`RuntimeError`. Do not import
this module directly from non-quantization code.
"""

from typing import Callable, ClassVar, List, NamedTuple, Protocol, Type

import torch
from ltx_kernels.blockwise.functional import (
    blockwise_dequantize,
    blockwise_quantize_adanorm_triton,
    blockwise_quantize_rms_fma_triton,
    fp6_blockwise_quantize_weights_torch,
    fp6_pack_tensor,
    fp6_unpack_tensor,
    fp8_blockwise_quantize_weights_torch,
    gated_attention_triton,
    rms_norm_rope,
    rms_norm_split_rope,
)
from ltx_kernels.blockwise.linear import BlockwiseFP6Linear, BlockwiseFP8Linear
from torch import nn

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.transformer import LTXModel
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator, LTXVideoOnlyModelConfigurator
from ltx_core.model.transformer.ops import (
    AdaZeroCallable,
    GatedAttentionCallable,
    PostSACallable,
    PreAttentionCallable,
)
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import TransformerOpsConfig


class FromLinearProtocol(Protocol):
    """Protocol for nn.Module subclasses that can be constructed from an nn.Linear."""

    @classmethod
    def from_linear(cls, linear: nn.Linear, transform_weights: bool = True) -> nn.Module: ...


class BlockwiseQuantizedWeight(NamedTuple):
    """Result of blockwise quantization: a quantized weight tensor and its per-block scale.
    For FP8: ``weight`` is ``float8_e4m3fn``, ``scale`` is ``float32`` shaped
    ``[out // 128, in // 128]``.
    For FP6: ``weight`` is packed ``uint8`` shaped ``[out, (in // 4) * 3]``,
    ``scale`` is ``float32`` shaped ``[out // 128, in // 128]``.
    """

    weight: torch.Tensor
    scale: torch.Tensor


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
    "to_gate_logits",
    "scale_shift_table",
)


_QUANTIZABLE_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _is_quantizable_float(x: torch.Tensor | torch.dtype) -> bool:
    """Whether ``x`` is an unquantized high-precision float (bf16 / fp16 / fp32).
    FP8 / FP6 weights are floats too but they're already in a quantized layout
    and must not be re-quantized.
    """
    dtype = x.dtype if isinstance(x, torch.Tensor) else x
    return dtype in _QUANTIZABLE_FLOAT_DTYPES


def _should_skip_layer(layer_name: str, excluded_layer_substrings: tuple[str, ...]) -> bool:
    return any(substring in layer_name for substring in excluded_layer_substrings)


def _replace_linear_modules(model: torch.nn.Module, linear_cls: Type[FromLinearProtocol]) -> torch.nn.Module:
    skip_list = ["to_gate_logits", "scale_shift_table"]
    for name, module in model.named_modules():
        if "transformer_block" in name and isinstance(module, torch.nn.Linear):
            if _should_skip_layer(name, skip_list):
                continue
            *parent_path, child_name = name.split(".")
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
            setattr(
                parent,
                child_name,
                linear_cls.from_linear(module, False),
            )
            del module.weight
            del module.bias
            torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------
# Weight quantization helpers
# ---------------------------------------------------------------------------


def _blockwise_quantize_weight_helper(
    value: torch.Tensor,
    quant_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    pack_fn: Callable[[torch.Tensor], torch.Tensor],
) -> BlockwiseQuantizedWeight:
    orig_device = value.device
    w_quant, w_scales = quant_fn(value.cuda())
    return BlockwiseQuantizedWeight(
        weight=pack_fn(w_quant).to(device=orig_device),
        scale=w_scales.to(device=orig_device),
    )


def _fp8_blockwise_quantize_weight(value: torch.Tensor) -> BlockwiseQuantizedWeight:
    return _blockwise_quantize_weight_helper(value, fp8_blockwise_quantize_weights_torch, lambda x: x)


def _fp6_blockwise_quantize_weight(value: torch.Tensor) -> BlockwiseQuantizedWeight:
    return _blockwise_quantize_weight_helper(value, fp6_blockwise_quantize_weights_torch, fp6_pack_tensor)


def _create_weight_quantize_op(
    excluded_layer_substrings: tuple[str, ...],
    quantization_func: Callable[[torch.Tensor], BlockwiseQuantizedWeight],
) -> Callable[[str, torch.Tensor], list[KeyValueOperationResult]]:
    """KeyValueOperation that blockwise-quantizes a 2D BF16 ``.weight`` and emits ``.weight_scale``."""

    def quantize_weight(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if _should_skip_layer(key, excluded_layer_substrings):
            return [KeyValueOperationResult(key, value)]
        if value.dim() != 2 or not _is_quantizable_float(value):
            return [KeyValueOperationResult(key, value)]
        quantized = quantization_func(value)
        scale_key = key.replace(".weight", ".weight_scale")
        return [
            KeyValueOperationResult(key, quantized.weight),
            KeyValueOperationResult(scale_key, quantized.scale),
        ]

    return quantize_weight


def _create_bias_to_fp32_op(
    excluded_layer_substrings: tuple[str, ...],
) -> Callable[[str, torch.Tensor], list[KeyValueOperationResult]]:
    """KeyValueOperation that casts a ``.bias`` tensor to FP32.
    ``BlockwiseFP{8,6}Linear`` registers ``.bias`` as float32; the load-time
    cast keeps the checkpoint's BF16 bias compatible with that param dtype.
    """

    def bias_to_fp32(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if _should_skip_layer(key, excluded_layer_substrings):
            return [KeyValueOperationResult(key, value)]
        return [KeyValueOperationResult(key, value.float())]

    return bias_to_fp32


# ---------------------------------------------------------------------------
# Q8 activation callables (formerly in model.transformer.ops)
# ---------------------------------------------------------------------------


class Q8KernelsPreAttention(PreAttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,  # noqa: ARG002
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attn_module.rope_type == LTXRopeType.INTERLEAVED:
            rope_func = rms_norm_rope
        elif attn_module.rope_type == LTXRopeType.SPLIT:
            rope_func = rms_norm_split_rope
        else:
            raise ValueError(f"Invalid rope type: {attn_module.rope_type}")

        if pe is not None:
            k_pe = k_pe if k_pe is not None else pe
            q = rope_func(q, pe[0], pe[1], attn_module.q_norm.weight, False)
            k = rope_func(k, k_pe[0], k_pe[1], attn_module.k_norm.weight, False)
        else:
            q = attn_module.q_norm(q)
            k = attn_module.k_norm(k)
        return q, k


class Q8KernelsAdaZeroFunction(AdaZeroCallable):
    def __call__(
        self,
        x: torch.Tensor,
        eps: float,  # noqa: ARG002
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return blockwise_quantize_adanorm_triton(x, None, scale, shift, torch.float8_e4m3fn, 1.0)


class Q8KernelsPostSAFunction(PostSACallable):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,  # noqa: ARG002
        eps: float,  # noqa: ARG002
        gate: torch.Tensor,
    ) -> List[torch.Tensor]:
        # Dequantize the fused result: the cross-attention AdaLN path applies a BF16
        # scale/shift, which cannot operate on the (fp8, scales) payload.
        normed_fp8 = blockwise_quantize_rms_fma_triton(x, y, gate)
        return x, blockwise_dequantize(normed_fp8)


class Q8KernelsGatedAttention(GatedAttentionCallable):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor:
        # Self-attention path: ``x`` arrives as the ``(fp8, scales)`` tuple
        # produced by Q8KernelsAdaZeroFunction. Cross-attention path
        # (apply_cross_attention_adaln) feeds plain BF16, so dequantize only
        # when needed.
        if isinstance(x, tuple):
            x = blockwise_dequantize(x)
        gate_logits = attn_module.to_gate_logits(x)
        return gated_attention_triton(attn_out, gate_logits)


# ---------------------------------------------------------------------------
# Fuse rules
# ---------------------------------------------------------------------------


_BLOCK = 128


def _blockwise_dequantize_2d(weight_fp8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a 2D blockwise-FP8 weight ``[out, in]`` with per-block scale
    ``[out//128, in//128]`` to BF16.
    ``ltx_kernels.blockwise.blockwise_dequantize`` is built for 3D activations where
    scales are ``[b*s, in//128]`` — one row per token. Weights are block-
    quantized along the row dim too, so we expand the row axis 128x via
    ``repeat_interleave`` and reuse the kernel.
    """
    out_features, in_features = weight_fp8.shape
    scales_per_row = weight_scale.repeat_interleave(_BLOCK, dim=0)
    return blockwise_dequantize((weight_fp8.unsqueeze(0), scales_per_row)).view(out_features, in_features)


def _blockwise_fp8_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize the FP8 weight + per-block scale to BF16, add the BF16 delta,
    and re-quantize blockwise. Both ``.weight`` and the companion
    ``.weight_scale`` are emitted so the loaded layer matches what
    ``BlockwiseFP8Linear`` expects.
    Excluded layers (see ``EXCLUDED_LAYER_SUBSTRINGS``) stay BF16 and have no
    ``.weight_scale`` companion — for those, fall back to a plain bf16 fuse.
    """
    scale_key = key.replace(".weight", ".weight_scale")
    if scale_key not in model_sd.sd:
        return bf16_fuse_rule(key, weight, deltas, model_sd)
    # ``weight`` is already on the fusion device; companion scale may still be CPU-retained.
    weight_scale = model_sd.sd[scale_key].to(device=weight.device)
    bf16_weight = _blockwise_dequantize_2d(weight, weight_scale)
    merged = bf16_weight + deltas.to(dtype=bf16_weight.dtype)
    new_fp8_weight, new_weight_scale = fp8_blockwise_quantize_weights_torch(merged.cuda())
    return {
        key: new_fp8_weight.to(device=weight.device),
        scale_key: new_weight_scale.to(device=weight.device),
    }


def _blockwise_fp6_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Mirror ``BlockwiseFP6Linear.fp8weight`` for the dequant side: unpack the
    packed ``uint8`` weight to ``float8_e4m3fn``, dequantize via the per-block
    scale to BF16, add the BF16 delta, re-quantize to FP6, and pack back to
    uint8. Both ``.weight`` (packed uint8) and ``.weight_scale`` are emitted.
    Note: ``fp6_unpack_tensor`` restores the dropped e_1/e_2 exponent bits as 0,
    so the dequant->add->requant round-trip is lossy on those bits even when no
    LoRA delta is applied. This matches what ``BlockwiseFP6Linear`` already does
    at inference time via its ``fp8weight`` property, so the fused weight is
    numerically consistent with the unfused inference path.
    Excluded layers (see ``EXCLUDED_LAYER_SUBSTRINGS``) stay BF16 and have no
    ``.weight_scale`` companion — for those, fall back to a plain bf16 fuse.
    """
    scale_key = key.replace(".weight", ".weight_scale")
    if scale_key not in model_sd.sd:
        return bf16_fuse_rule(key, weight, deltas, model_sd)
    # ``weight`` is already on the fusion device; companion scale may still be CPU-retained.
    weight_scale = model_sd.sd[scale_key].to(device=weight.device)
    # Packed shape is [out, (in // 4) * 3]; recover in_features.
    original_n = weight.shape[-1] * 4 // 3
    fp8_view = fp6_unpack_tensor(weight, original_n).view(torch.float8_e4m3fn)
    bf16_weight = _blockwise_dequantize_2d(fp8_view, weight_scale)
    merged = bf16_weight + deltas.to(dtype=bf16_weight.dtype)
    new_fp8, new_scale = fp6_blockwise_quantize_weights_torch(merged)
    new_packed = fp6_pack_tensor(new_fp8.view(torch.uint8))
    return {
        key: new_packed.to(device=weight.device),
        scale_key: new_scale.to(device=weight.device),
    }


# ---------------------------------------------------------------------------
# Configurators (TransformerOpsConfig with Q8 activation callables)
# ---------------------------------------------------------------------------


def _build_blockwise_ops_config() -> TransformerOpsConfig:
    return TransformerOpsConfig.from_functions(
        preattention=Q8KernelsPreAttention(),
        gated_attention=Q8KernelsGatedAttention(),
        ada_zero=Q8KernelsAdaZeroFunction(),
        post_sa=Q8KernelsPostSAFunction(),
    )


# FP6 is weight-only; activation ops match FP8.
_BLOCKWISE_OPS = _build_blockwise_ops_config()


class BlockwiseFP8LTXModelConfigurator(ModelConfigurator[LTXModel]):
    BASE: ClassVar[type[ModelConfigurator[LTXModel]]] = LTXModelConfigurator
    OPS: ClassVar[TransformerOpsConfig] = _BLOCKWISE_OPS

    @classmethod
    def from_metadata(cls, metadata: dict) -> LTXModel:
        return cls.BASE.from_metadata(metadata, ops=cls.OPS)


class BlockwiseFP8LTXVideoOnlyModelConfigurator(BlockwiseFP8LTXModelConfigurator):
    BASE = LTXVideoOnlyModelConfigurator


class BlockwiseFP6LTXModelConfigurator(BlockwiseFP8LTXModelConfigurator):
    pass


class BlockwiseFP6LTXVideoOnlyModelConfigurator(BlockwiseFP8LTXVideoOnlyModelConfigurator):
    pass


# ---------------------------------------------------------------------------
# SDOps / ModuleOps / FuseRule assembly
# ---------------------------------------------------------------------------


def build_sd_ops_fp8() -> SDOps:
    return (
        SDOps("blockwise_fp8_weights")
        .with_kv_operation(
            _create_weight_quantize_op(EXCLUDED_LAYER_SUBSTRINGS, _fp8_blockwise_quantize_weight),
            key_prefix="transformer_blocks.",
            key_suffix=".weight",
        )
        .with_kv_operation(
            _create_bias_to_fp32_op(EXCLUDED_LAYER_SUBSTRINGS),
            key_prefix="transformer_blocks.",
            key_suffix=".bias",
        )
    )


def build_sd_ops_fp6() -> SDOps:
    return (
        SDOps("blockwise_fp6_weights")
        .with_kv_operation(
            _create_weight_quantize_op(EXCLUDED_LAYER_SUBSTRINGS, _fp6_blockwise_quantize_weight),
            key_prefix="transformer_blocks.",
            key_suffix=".weight",
        )
        .with_kv_operation(
            _create_bias_to_fp32_op(EXCLUDED_LAYER_SUBSTRINGS),
            key_prefix="transformer_blocks.",
            key_suffix=".bias",
        )
    )


def build_module_ops_fp8() -> ModuleOps:
    return ModuleOps(
        name="blockwise_fp8_prepare_for_loading",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: _replace_linear_modules(model, BlockwiseFP8Linear),
    )


def build_module_ops_fp6() -> ModuleOps:
    return ModuleOps(
        name="blockwise_fp6_prepare_for_loading",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: _replace_linear_modules(model, BlockwiseFP6Linear),
    )


fuse_rule_fp8 = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_blockwise_fp8_fuse)
fuse_rule_fp6 = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_blockwise_fp6_fuse)
