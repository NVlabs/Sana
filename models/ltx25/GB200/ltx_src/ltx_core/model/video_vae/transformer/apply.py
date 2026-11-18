"""Install a resolved ``DiffVAEConfig`` onto a loaded ``DiffusionVideoDecoder``.
Owns all module mutation and ModuleOps builders. Preset → recipe lives in
``config.py``; ``torch.compile`` mechanics live in ``compiling.py``.
CHUNKED / BLACKWELL_DSL modes wire ``decoder.upsamples[3]`` onto each diff block
for deferred stage-4 (upsample Linear then ``context_proj`` — fused into the CuTe
kernel for BLACKWELL_DSL).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.transformer.attention import (
    NeighborhoodAttention3D,
    configure_w_chunks,
    natten_available,
)
from ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from ltx_core.model.video_vae.transformer.chunked.block import ChunkedDiffusionNABlock
from ltx_core.model.video_vae.transformer.combined.block import CombinedDiffusionNABlock
from ltx_core.model.video_vae.transformer.compiling import (
    compile_diffusion_decoder,
    configure_natten_backend,
)
from ltx_core.model.video_vae.transformer.config import (
    DiffVAEBlockKind,
    DiffVAEConfig,
    DiffVAEMode,
    NAttentionKind,
)
from ltx_core.model.video_vae.transformer.dsl_kernels.block import DSLDiffusionNABlock
from ltx_core.model.video_vae.transformer.fallback_na import (
    EagerSdpaAttention,
    TritonNaAttention,
    triton_na_available,
    warn_no_natten,
)

if TYPE_CHECKING:
    from ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder


def _block_cls_for(cfg: DiffVAEConfig) -> type[DiffusionNABlock]:
    if cfg.block is DiffVAEBlockKind.COMBINED:
        return CombinedDiffusionNABlock
    if cfg.block is DiffVAEBlockKind.CHUNKED:
        return ChunkedDiffusionNABlock
    if cfg.block is DiffVAEBlockKind.BLACKWELL_DSL:
        return DSLDiffusionNABlock
    raise ValueError(f"Unknown DiffVAEBlockKind: {cfg.block!r}")


def resolve_attention_for_host(cfg: DiffVAEConfig) -> DiffVAEConfig:
    """Remap chunked NATTEN recipes to Triton/eager when natten is missing.
    ``COMBINED`` (full-volume) compile requires natten — Dynamo + fallback NA is
    not supported. Chunked modes remapped to fallback also disable ``torch.compile``
    for now (fallback NA is an opaque custom_op, but the remap still keeps the
    no-natten path on the eager chunked recipe).
    """
    if cfg.attention is not NAttentionKind.NATTEN:
        return cfg
    if natten_available():
        return cfg
    if cfg.block is DiffVAEBlockKind.COMBINED:
        raise RuntimeError(
            "diffvae_optimization=combined_compile requires natten "
            "(full-volume NA + torch.compile). Install with: "
            "uv sync --package ltx-core --extra natten "
            "(or use chunked_eager / chunked_compile, which fall back to Triton/eager)."
        )
    kind = NAttentionKind.TRITON if triton_na_available() else NAttentionKind.EAGER_SDPA
    return replace(
        cfg,
        attention=kind,
        natten_backend=None,
        compile_blocks=False,
        compile_det_stages=False,
    )


def _configure_deferred_stage4(decoder: DiffusionVideoDecoder, enabled: bool) -> None:
    """Toggle deferred stage-4 and wire ``stage4_upsample`` on deferred blocks."""
    decoder.deferred_stage4_upsample = enabled
    upsample = decoder.upsamples[3] if enabled else None
    for block in decoder.diff_blocks:
        if isinstance(block, (ChunkedDiffusionNABlock, DSLDiffusionNABlock)):
            block.stage4_upsample = upsample  # type: ignore[assignment]
        elif hasattr(block, "stage4_upsample"):
            # Drop leftover from a prior deferred → COMBINED class swap.
            delattr(block, "stage4_upsample")


def _set_attention_function(decoder: DiffusionVideoDecoder, attn_fn: object) -> None:
    for module in decoder.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = attn_fn  # type: ignore[assignment]
            module.natten_backend = None


def _install_attention(decoder: DiffusionVideoDecoder, cfg: DiffVAEConfig) -> None:
    if cfg.attention is NAttentionKind.BLACKWELL_DSL:
        from ltx_core.model.video_vae.transformer.dsl_kernels.apply_dsl import (  # noqa: PLC0415
            enable_blackwell_dsl,
        )

        enable_blackwell_dsl(decoder)
        return
    if cfg.attention is NAttentionKind.TRITON:
        warn_no_natten(backend="Triton na3d")
        _set_attention_function(decoder, TritonNaAttention())
        return
    if cfg.attention is NAttentionKind.EAGER_SDPA:
        warn_no_natten(backend="eager tiled SDPA na3d")
        _set_attention_function(decoder, EagerSdpaAttention())
        return
    if cfg.natten_backend is not None:
        configure_natten_backend(decoder, cfg.natten_backend)


def apply_diffvae_config(
    decoder: DiffusionVideoDecoder,
    cfg: DiffVAEConfig,
    *,
    compilation_config: CompilationConfig | None = None,
) -> DiffusionVideoDecoder:
    """Install ``cfg`` onto ``decoder`` (only mutator for DiffVAE presets)."""
    cfg = resolve_attention_for_host(cfg)
    block_cls = _block_cls_for(cfg)
    for block in decoder.diff_blocks:
        if isinstance(block, DiffusionNABlock):
            block.__class__ = block_cls

    deferred = cfg.block in (DiffVAEBlockKind.CHUNKED, DiffVAEBlockKind.BLACKWELL_DSL)
    _configure_deferred_stage4(decoder, deferred)
    # W-chunking + rope_num_tiles=1 coupling is for diffusion residual only.
    for block in decoder.diff_blocks:
        configure_w_chunks(block, cfg.w_chunks)

    _install_attention(decoder, cfg)

    if cfg.compile_blocks:
        compile_diffusion_decoder(
            decoder,
            config=compilation_config,
            compile_det_stages=cfg.compile_det_stages,
        )
    else:
        decoder.mark_dynamic_shapes = False
    return decoder


def apply_diffvae_mode(
    decoder: DiffusionVideoDecoder,
    mode: DiffVAEMode,
    *,
    compilation_config: CompilationConfig | None = None,
) -> DiffusionVideoDecoder:
    """Resolve ``mode`` then :func:`apply_diffvae_config`."""
    return apply_diffvae_config(decoder, mode.resolve(), compilation_config=compilation_config)


def _diffusion_decoder_matcher(model: torch.nn.Module) -> bool:
    from ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder  # noqa: PLC0415

    return isinstance(model, DiffusionVideoDecoder)


def build_diffvae_mode_op(
    mode: DiffVAEMode,
    compilation_config: CompilationConfig | None = None,
) -> ModuleOps:
    """ModuleOps mutator that applies ``mode.resolve()`` on the meta model."""

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_diffvae_mode(model, mode, compilation_config=compilation_config)  # type: ignore[arg-type]

    return ModuleOps(
        name=f"diffvae_mode_{mode.value}",
        matcher=_diffusion_decoder_matcher,
        mutator=mutator,
    )


def build_cutlass_fna_diffusion_decoder_op() -> ModuleOps:
    """Backward-compat ModuleOps name; mutator applies :attr:`DiffVAEMode.CHUNKED_EAGER`."""
    inner = build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER)
    return ModuleOps(
        name="cutlass_fna_diffusion_decoder",
        matcher=inner.matcher,
        mutator=inner.mutator,
    )


def build_compile_diffusion_decoder_op(config: CompilationConfig | None = None) -> ModuleOps:
    """Backward-compat ModuleOps name; mutator applies :attr:`DiffVAEMode.COMBINED_COMPILE`."""
    inner = build_diffvae_mode_op(DiffVAEMode.COMBINED_COMPILE, compilation_config=config)
    return ModuleOps(
        name="compile_diffusion_decoder",
        matcher=inner.matcher,
        mutator=inner.mutator,
    )
