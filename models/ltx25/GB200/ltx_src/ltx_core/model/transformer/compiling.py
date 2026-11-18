import contextlib
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.utils._pytree as pytree

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.cudagraph_capture import CudaGraphRunner
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs

# Defaults applied inside the patched forward. Overriding via CompilationConfig
# replaces these wholesale; it does not merge.
_DEFAULT_INDUCTOR_CONFIG: dict[str, Any] = {}
_DEFAULT_DYNAMO_CONFIG: dict[str, Any] = {"inline_inbuilt_nn_modules": True, "cache_size_limit": 256}


@dataclass(frozen=True)
class CompilationConfig:
    """``torch.compile`` configuration for transformer blocks. ``None`` keeps eager."""

    # torch.compile mode (e.g. "reduce-overhead"); None = default.
    # reduce-overhead / max-autotune require GPU-resident weights (keeps_gpu_resident_weights).
    mode: str | None = None
    # torch.compile backend
    backend: str = "inductor"
    # error on graph breaks instead of falling back to eager
    fullgraph: bool = False
    # force dynamic shapes; None lets Dynamo decide
    dynamic: bool | None = None
    # torch._inductor.config overrides
    inductor_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_INDUCTOR_CONFIG))
    # torch._dynamo.config overrides
    dynamo_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_DYNAMO_CONFIG))
    # mark the sequence dimension as dynamic so one artifact serves all sequence lengths
    seq_dim_dynamic: bool = True
    # True: separate graph for guidance pass; False: single runtime-masked graph
    recompile_perturbed_block: bool = True
    # True: per-block compile + one CUDA graph over the block loop. Faster cold start
    # than reduce-overhead's 48-block cudagraph tree. Requires GPU-resident weights
    # (keeps_gpu_resident_weights).
    capture: bool = False


class CompiledBlockPerturbationsProcessor(BlockPerturbationsProcessor):
    """Eager prep of each block's inputs before they enter a compiled region.
    When ``seq_dim_dynamic`` is set, marks sequence-length dims on the tensors about
    to cross into the block so one compile artifact covers all lengths.
    Perturbation state is attached based on ``recompile_perturbed_block``:
    * ``True`` (default) — same as :class:`BlockPerturbationsProcessor`: optional
      masks and Python bool shortcuts. Dynamo specialises on those, so an STG
      pass gets its own graph (extra compile; clean passes skip the mask multiply).
    * ``False`` — always attach both masks and pin the skip flags to ``False``.
      Every pass (cond / uncond / STG) traces the same graph; an all-ones mask is
      a no-op and an all-zeros mask is a skip.
    """

    def __init__(self, seq_dim_dynamic: bool, recompile_perturbed_block: bool) -> None:
        super().__init__()
        self._seq_dim_dynamic = seq_dim_dynamic
        self._recompile_perturbed_block = recompile_perturbed_block

    def _mark_seq_dim_dynamic(self, args: TransformerArgs) -> TransformerArgs:
        # Positional embeddings are second-from-last regardless of rope type:
        # split rope is (B, H, T, D//2) -- dim -2 == 2; interleaved rope is (B, T, D)
        # -- dim -2 == 1. Both work via the negative index.
        torch._dynamo.mark_dynamic(args.x, 1)
        cos, sin = args.positional_embeddings
        torch._dynamo.mark_dynamic(cos, cos.ndim - 2)
        torch._dynamo.mark_dynamic(sin, sin.ndim - 2)
        if args.cross_positional_embeddings is not None:
            cross_cos, cross_sin = args.cross_positional_embeddings
            torch._dynamo.mark_dynamic(cross_cos, cross_cos.ndim - 2)
            torch._dynamo.mark_dynamic(cross_sin, cross_sin.ndim - 2)
        if args.self_attention_mask is not None:
            # Dense form is (B, 1, T, T); key-padding form (from the SP wrapper)
            # is (B, 1, 1, T) -- leave the size-1 query dim static so Dynamo
            # keeps the broadcast.
            if args.self_attention_mask.shape[2] > 1:
                torch._dynamo.mark_dynamic(args.self_attention_mask, 2)
            torch._dynamo.mark_dynamic(args.self_attention_mask, 3)
        if args.context_mask is not None:
            torch._dynamo.mark_dynamic(args.context_mask, 2)
        # `timesteps` / `embedded_timestep` are per-token when conditioning sets a
        # per-position denoise mask, in which case their dim 1 equals the seq length
        # and must vary with it. When they're a single timestep broadcast across the
        # sequence (dim 1 == 1), leaving them static lets Dynamo keep the size-1
        # broadcast.
        if args.timesteps.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.timesteps, 1)
        if args.embedded_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.embedded_timestep, 1)
        # `cross_scale_shift_timestep` is the cross-attn AdaLN scale/shift input
        # derived from the own-modality per-token timesteps (denoise_mask * sigma),
        # so its dim 1 equals the seq length when conditioning is per-token.
        # `cross_gate_timestep` is the cross-modality sigma scalar -- dim 1 is 1
        # and broadcasts, leave it static. Same guard pattern as `timesteps`.
        if args.cross_scale_shift_timestep is not None and args.cross_scale_shift_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.cross_scale_shift_timestep, 1)

        return args

    def __call__(
        self,
        args: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
        cross_attn_type: PerturbationType,
    ) -> TransformerArgs:
        if self._seq_dim_dynamic:
            args = self._mark_seq_dim_dynamic(args)

        if self._recompile_perturbed_block:
            # Conditional None-or-tensor masks + Python bool shortcuts: Dynamo specialises on the
            # STG-perturbed pass and recompiles a separate block graph for it.
            args = super().__call__(args, perturbations, block_idx, self_attn_type, cross_attn_type)
        else:
            # Unconditional runtime masks (skip flags pinned False): the block traces once, no recompile.
            args = replace(
                args,
                self_attn_perturbation_mask=perturbations.mask(self_attn_type, block_idx),
                self_attn_all_perturbed=False,
                cross_attn_perturbation_mask=perturbations.mask(cross_attn_type, block_idx),
                cross_attn_skip_all=False,
            )

        return args

    def graph_signature(self, perturbations: BatchedPerturbationConfig) -> Hashable:
        # recompile_perturbed_block=False attaches unconditional masks with fixed flags -> no
        # perturbation guards, so the block never recompiles per pattern: one constant signature
        # (graph per shape).
        if not self._recompile_perturbed_block:
            return ()
        return super().graph_signature(perturbations)


def _register_transformer_args_pytree() -> None:
    """Register ``TransformerArgs`` as a pytree so :class:`CudaGraphRunner` can flatten it to
    tensor leaves for copy-in/out. Idempotent.
    """
    if TransformerArgs not in pytree.SUPPORTED_NODES:
        pytree.register_dataclass(TransformerArgs)


@contextlib.contextmanager
def _compile_config_patches(config: CompilationConfig) -> Iterator[None]:
    """Apply the config's inductor/dynamo patches for the duration of a compiled call."""
    with (
        torch._inductor.config.patch(**config.inductor_config),
        torch._dynamo.config.patch(**config.dynamo_config),  # type: ignore[attr-defined]
    ):
        yield


def _compile_blocks(model: LTXModel, config: CompilationConfig, *, mode: str | None) -> None:
    """Compile each transformer block and install the perturbation processor.
    ``mode`` is the sole knob that differs between the two compile paths: the captured path passes
    ``None`` (cudagraphs OFF -- it manages its own CUDA graph), the reduce-overhead path passes
    ``config.mode``. Forcing ``dynamic=True`` over-dynamizes and trips a data-dependent ``.item()``
    (DataDependentOutputException); ``config.dynamic`` plus the processor's per-block seq
    ``mark_dynamic`` is what keeps the artifact shape-polymorphic.
    """
    model.transformer_blocks = torch.nn.ModuleList(
        torch.compile(m, mode=mode, backend=config.backend, fullgraph=config.fullgraph, dynamic=config.dynamic)
        for m in model.transformer_blocks
    )
    model.block_input_processor = CompiledBlockPerturbationsProcessor(
        seq_dim_dynamic=config.seq_dim_dynamic, recompile_perturbed_block=config.recompile_perturbed_block
    )


def compile_transformer_captured(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Per-block compile (dynamic, cudagraphs OFF) + self-managed per-shape CUDA-graph capture of
    the block-loop.
    Each block compiles once into a shape-polymorphic Inductor kernel -- AOTAutograd sees ONE
    block (cheap ~38s), not a whole-forward compile whose functionalize metadata pass is O(48
    blocks). ``_process_transformer_blocks`` (the 48-block loop) is then captured as ONE CUDA graph
    per input shape by :class:`CudaGraphRunner` and replayed: single-graph runtime without that
    O(48) AOT cost, staying shape-polymorphic (sweeping down from the max shape re-captures,
    never recompiles). The RoPE/args ``prepare`` and output projection run eager (outside the
    graph) -- they do host->device work that cannot be captured, and widening the boundary to
    include them is only ~3% faster steady-state (not worth the shared RoPE changes it needs).
    """
    _register_transformer_args_pytree()  # so the runner can flatten TransformerArgs to tensor leaves
    _compile_blocks(model, config, mode=None)  # cudagraphs OFF -- the runner captures the graph

    original_block_loop = model._process_transformer_blocks

    def patched_block_loop(
        video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig | None
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        # Config patches matter only during warmup/capture (the first-shape block compile); the
        # steady-state replay path (CudaGraphRunner) never re-enters this.
        with _compile_config_patches(config):
            return original_block_loop(video, audio, perturbations)

    model._process_transformer_blocks = CudaGraphRunner(patched_block_loop, model.block_input_processor)
    return model


def compile_transformer(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Compile each transformer block via ``torch.compile`` with the given settings."""
    if config.capture:
        return compile_transformer_captured(model, config)
    _compile_blocks(model, config, mode=config.mode)

    def patched_dynamo_forward(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # One mark per step so reduce-overhead/max-autotune do not treat each of the
        # 48 compiled block calls as a new iteration. No-op under other modes.
        torch.compiler.cudagraph_mark_step_begin()
        with _compile_config_patches(config):
            return model.forward_without_compilation(*args, **kwargs)

    model.forward_without_compilation = model.forward
    model.forward = patched_dynamo_forward
    return model


def build_compile_transformer_op(config: CompilationConfig) -> ModuleOps:
    """Build a ``ModuleOps`` that compiles transformer blocks with the given settings."""
    return ModuleOps(
        name="compile_transformer",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: compile_transformer(model, config),
    )


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.", f"transformer_blocks.{i}._orig_mod."
        )
    return original_sd_ops
