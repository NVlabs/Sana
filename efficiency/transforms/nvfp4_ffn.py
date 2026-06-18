# Copyright 2025 SGLang authors
#
# NVFP4FFN -- LOAD-time transform that quantizes selected linear layers to TE
# NVFP4. The quantization is baked into the weights at load; there is no
# per-step decision, so this is a ModelTransform (not a Technique). set_env()
# emits the existing SGLANG_HQ_ENABLE_TE_NVFP4_FFN recipe consumed by the loader,
# plus metadata env for search axes that a candidate may wire explicitly.

from __future__ import annotations

from efficiency.registry import register_transform
from efficiency.technique import Seam
from efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)


@register_transform("nvfp4_ffn")
class NVFP4FFN(ModelTransform):
    """Quantize selected video linear layers to NVFP4 at load.

    The default values preserve the historical FFN-only recipe. Extra parameters
    expose search axes for recipe variants, fused epilogues, padding policy, and
    candidate metadata. A target loader may ignore metadata env until a candidate
    explicitly wires and validates it.
    """

    name = "nvfp4_ffn"
    phase = TransformPhase.LOAD
    writes = frozenset({Seam.FFN_PRECISION})

    def __init__(
        self,
        module_scope: str = "video_ffn",
        disable_rht: bool = True,
        disable_stochastic_rounding: bool = True,
        disable_2d_quantization: bool = True,
        row_scaled_activation: bool = False,
        fused_proj_in_gelu: bool = False,
        fused_proj_out_bias_gate: bool = False,
        pad_m_to: int = 16,
        fp4_gemm_backend: str = "",
        dense_layers: str = "",
        dense_steps: str = "",
        fallback_policy: str = "bf16",
    ):
        self.module_scope = module_scope
        self.disable_rht = disable_rht
        self.disable_stochastic_rounding = disable_stochastic_rounding
        self.disable_2d_quantization = disable_2d_quantization
        self.row_scaled_activation = row_scaled_activation
        self.fused_proj_in_gelu = fused_proj_in_gelu
        self.fused_proj_out_bias_gate = fused_proj_out_bias_gate
        self.pad_m_to = pad_m_to
        self.fp4_gemm_backend = fp4_gemm_backend
        self.dense_layers = dense_layers
        self.dense_steps = dense_steps
        self.fallback_policy = fallback_policy

    def set_env(self, ctx: TransformContext) -> None:
        e = ctx.env
        e["SGLANG_HQ_ENABLE_TE_NVFP4_FFN"] = "1"
        e["SGLANG_LTX2_TE_NVFP4_VIDEO_FFN"] = "1"
        e["SGLANG_HQ_NVFP4_MODULE_SCOPE"] = self.module_scope
        e["SGLANG_LTX2_TE_NVFP4_DISABLE_RHT"] = "1" if self.disable_rht else "0"
        e["SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] = (
            "1" if self.disable_stochastic_rounding else "0"
        )
        e["SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION"] = (
            "1" if self.disable_2d_quantization else "0"
        )
        e["SGLANG_HQ_NVFP4_ROW_SCALED_ACTIVATION"] = (
            "1" if self.row_scaled_activation else "0"
        )
        e["SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU"] = (
            "1" if self.fused_proj_in_gelu else "0"
        )
        e["SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE"] = (
            "1" if self.fused_proj_out_bias_gate else "0"
        )
        e["SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU"] = (
            "1" if self.fused_proj_in_gelu else "0"
        )
        e["SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE"] = (
            "1" if self.fused_proj_out_bias_gate else "0"
        )
        e["SGLANG_LTX2_TE_NVFP4_PAD_M_TO"] = str(int(self.pad_m_to))
        if self.fp4_gemm_backend:
            e["SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"] = self.fp4_gemm_backend
        if self.dense_layers:
            e["SGLANG_HQ_NVFP4_DENSE_LAYERS"] = self.dense_layers
        if self.dense_steps:
            e["SGLANG_HQ_NVFP4_DENSE_STEPS"] = self.dense_steps
        if self.fallback_policy:
            e["SGLANG_HQ_NVFP4_FALLBACK_POLICY"] = self.fallback_policy
