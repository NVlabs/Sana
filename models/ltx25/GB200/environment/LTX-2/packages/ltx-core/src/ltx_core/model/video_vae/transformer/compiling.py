"""``torch.compile`` / NATTEN-backend wiring for DiffusionVideoDecoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.transformer.swiglu import SwiGLUMode, configure_swiglu

if TYPE_CHECKING:
    from ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder

# Default eager DiffVAE pin: lower VRAM than hopper-fna TokPerm (see memory notes).
_CUTLASS_FNA_BACKEND = "cutlass-fna"


def configure_natten_backend(module_root: nn.Module, backend: str | None) -> None:
    """Pin ``natten.na3d`` backend on every ``NeighborhoodAttention3D`` under ``module_root``.
    Stored per-module (no process-global natten monkeypatch). ``backend=None`` restores
    NATTEN auto-selection.
    """
    # Local import avoids a circular import with attention.py.
    from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D  # noqa: PLC0415

    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.natten_backend = backend


def configure_cutlass_fna_diffusion_decoder(decoder: DiffusionVideoDecoder) -> None:
    """Eager DiffVAE path: pin all NA modules to ``cutlass-fna`` for lower peak VRAM."""
    configure_natten_backend(decoder, _CUTLASS_FNA_BACKEND)


def compile_diffusion_decoder(
    decoder: DiffusionVideoDecoder,
    *,
    config: CompilationConfig | None = None,
) -> DiffusionVideoDecoder:
    """``torch.compile`` pre-diffusion and diffusion-step blocks individually.
    Pre-diffusion: each ``NABlock.forward`` (stages differ in shape; whole-graph
    pays full cost for a once-per-decode call).
    Diff-step: each ``DiffusionNABlock.forward_combined``. The outer
    ``forward_diff_step`` loop stays eager: call sites build
    ``[context | conv_in(x)]`` once, and the loop reuses that buffer
    (``copy_`` into the x-half) so each compiled boundary sees one tensor's
    T/H/W symbols (separate ``x``/``latent_context`` args reintroduce
    ``ConstraintViolationError`` under ``mark_dynamic``).
    Also configures hybrid abs-RoPE: opaque ``custom_op`` on pre-diffusion NA
    modules, transparent ``nested_compile_region`` on diffusion-step NA modules.
    Sets ``decoder.mark_dynamic_shapes`` so decode marks T/H/W dynamic.
    Forces every ``SwiGLU`` to ``TILED`` (memory-efficient custom_op; Triton fuse OK).
    Uses the shared :class:`~ltx_core.model.transformer.compiling.CompilationConfig`
    (``seq_dim_dynamic`` / ``recompile_perturbed_block`` are transformer-only and
    ignored here). First-compile cost is paid on the first real ``decode_video``.
    """
    cfg = config or CompilationConfig()
    # Opaque chunked workspace path — not plain F.linear slabs (those blow VRAM).
    configure_swiglu(decoder, SwiGLUMode.TILED)

    # NATTEN's KV-parallelism heuristic (cutlass-fna backend) is backward-pass
    # only -- see natten.context.use_kv_parallelism_in_fused_na's own
    # docstring ("guards for using KV Parallelism in backpropagation").
    # Decoding never calls .backward(), so disabling it costs nothing here,
    # and it must be disabled: its config path (get_default_kv_splits_backward)
    # does `int(x)` on the traced tensor's shape, which forces Dynamo to hard-
    # specialize on that exact size -- under a dynamic-shape tiled decode,
    # every distinct tile size then forces its own full recompile.
    try:
        import natten  # noqa: PLC0415

        natten.use_kv_parallelism_in_fused_na(False)
    except ImportError:
        pass

    compile_kwargs: dict[str, Any] = {
        "mode": cfg.mode,
        "backend": cfg.backend,
        "fullgraph": cfg.fullgraph,
        "dynamic": cfg.dynamic,
    }

    def _compile(fn: Callable[..., Any]) -> Callable[..., Any]:
        with (
            torch._inductor.config.patch(**cfg.inductor_config),
            torch._dynamo.config.patch(**cfg.dynamo_config),  # type: ignore[attr-defined]
        ):
            return torch.compile(fn, **compile_kwargs)

    # Local import avoids a circular import with attention.py.
    from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D  # noqa: PLC0415

    # Stages 1-4 run once per decode, not in a hot loop: compiling
    # forward_pre_diffusion as one graph pays full trace/lower/autotune
    # cost for a function called exactly once, with no amortization and
    # no cross-block fusion win worth that cost (blocks within a stage
    # are structurally identical, just different weights). Compile each
    # NABlock's forward individually instead and leave the stage loop
    # eager -- dynamo's cache hits on every later call to a block at the
    # same shape/dtype, so the cost is paid once per distinct stage
    # shape, not once for the whole unrolled 16-block stack.
    for stage_blocks in decoder.det_stages:
        for block in stage_blocks:
            block.forward = _compile(block.forward)  # type: ignore[method-assign]
    for module in decoder.det_stages.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.rope_use_custom_op = True

    for block in decoder.diff_blocks:
        block.forward_combined = _compile(block.forward_combined)  # type: ignore[method-assign]
    for module in decoder.diff_blocks.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.rope_use_custom_op = False

    decoder.mark_dynamic_shapes = True
    return decoder


def _diffusion_decoder_matcher(model: torch.nn.Module) -> bool:
    from ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder  # noqa: PLC0415

    return isinstance(model, DiffusionVideoDecoder)


def build_cutlass_fna_diffusion_decoder_op() -> ModuleOps:
    """Build a ``ModuleOps`` that pins DiffVAE NA modules to ``cutlass-fna`` (eager)."""

    def cutlass_mutator(model: torch.nn.Module) -> torch.nn.Module:
        configure_cutlass_fna_diffusion_decoder(model)  # type: ignore[arg-type]
        return model

    return ModuleOps(
        name="cutlass_fna_diffusion_decoder",
        matcher=_diffusion_decoder_matcher,
        mutator=cutlass_mutator,
    )


def build_compile_diffusion_decoder_op(config: CompilationConfig | None = None) -> ModuleOps:
    """Build a ``ModuleOps`` that ``torch.compile``s diffusion-decoder block forwards.
    Mirrors :func:`ltx_core.model.transformer.compiling.build_compile_transformer_op`.
    Method-level compile leaves checkpoint keys unchanged (no ``_orig_mod`` rewrite).
    """

    def compile_mutator(model: torch.nn.Module) -> torch.nn.Module:
        return compile_diffusion_decoder(model, config=config)  # type: ignore[arg-type]

    return ModuleOps(
        name="compile_diffusion_decoder",
        matcher=_diffusion_decoder_matcher,
        mutator=compile_mutator,
    )
