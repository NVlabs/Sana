"""Neighborhood-Attention transformer building blocks for the diffusion video VAE."""

from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D, natten_available
from ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock, NABlock
from ltx_core.model.video_vae.transformer.compiling import (
    build_compile_diffusion_decoder_op,
    build_cutlass_fna_diffusion_decoder_op,
    compile_diffusion_decoder,
    configure_cutlass_fna_diffusion_decoder,
    configure_natten_backend,
)
from ltx_core.model.video_vae.transformer.layers import (
    AdaLNZero,
    ChannelLinear,
    LinearPixelShuffleUpsample,
    modulate,
)
from ltx_core.model.video_vae.transformer.qkv import QKVProjections
from ltx_core.model.video_vae.transformer.rope import (
    apply_abs_rope,
    apply_abs_rope_slab,
    configure_abs_rope,
    default_rope_dim_split,
    qkv_rope,
    rope_inv_freqs,
)
from ltx_core.model.video_vae.transformer.swiglu import SwiGLU, SwiGLUMode, configure_swiglu
from ltx_core.model.video_vae.transformer.swiglu_efficient import SwiGLUTileSpec

__all__ = [
    "AdaLNZero",
    "ChannelLinear",
    "DiffusionNABlock",
    "LinearPixelShuffleUpsample",
    "NABlock",
    "NeighborhoodAttention3D",
    "QKVProjections",
    "SwiGLU",
    "SwiGLUMode",
    "SwiGLUTileSpec",
    "apply_abs_rope",
    "apply_abs_rope_slab",
    "build_compile_diffusion_decoder_op",
    "build_cutlass_fna_diffusion_decoder_op",
    "compile_diffusion_decoder",
    "configure_abs_rope",
    "configure_cutlass_fna_diffusion_decoder",
    "configure_natten_backend",
    "configure_swiglu",
    "default_rope_dim_split",
    "modulate",
    "natten_available",
    "qkv_rope",
    "rope_inv_freqs",
]
