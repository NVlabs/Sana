"""Blackwell CuTe DSL pathway for DiffVAE (deferred stage-4 + fused stage-5)."""

from ltx_core.model.video_vae.transformer.dsl_kernels.apply_dsl import enable_blackwell_dsl
from ltx_core.model.video_vae.transformer.dsl_kernels.attn import (
    NA_SOFTMAX_BOUND_BUFFER,
    DSLAttention,
    na_attention_dsl,
    na_dsl_available,
    require_softmax_bound,
)
from ltx_core.model.video_vae.transformer.dsl_kernels.block import (
    FUSED_CTX_BIAS_BUFFER,
    FUSED_CTX_WEIGHT_BUFFER,
    DSLDiffusionNABlock,
    fused_na_block_supported,
)
from ltx_core.model.video_vae.transformer.dsl_kernels.chain import (
    DSLChannelLinear,
    DSLDiffusionBlockChain,
    alloc_block_volume,
    linear_into_block_volume,
)

__all__ = [
    "FUSED_CTX_BIAS_BUFFER",
    "FUSED_CTX_WEIGHT_BUFFER",
    "NA_SOFTMAX_BOUND_BUFFER",
    "DSLAttention",
    "DSLChannelLinear",
    "DSLDiffusionBlockChain",
    "DSLDiffusionNABlock",
    "alloc_block_volume",
    "enable_blackwell_dsl",
    "fused_na_block_supported",
    "linear_into_block_volume",
    "na_attention_dsl",
    "na_dsl_available",
    "require_softmax_bound",
]
