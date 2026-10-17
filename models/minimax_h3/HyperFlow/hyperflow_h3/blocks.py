# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# The two blocks below subclass and rework `MiniMaxH3SetTimestepsStep` and `MiniMaxH3LoopDenoiser` from diffusers
# (`modular_pipelines/minimax_h3/`), Copyright 2026 The MiniMax and HuggingFace Teams, Apache License 2.0.
# See THIRD_PARTY_NOTICES.md.
"""The two Modular Diffusers blocks HyperFlow swaps into the official MiniMax-H3 workflow.

The official workflow is a ``SequentialPipelineBlocks``; HyperFlow replaces exactly two of its blocks and touches
nothing else:

* ``denoise.set_timesteps`` — [`MiniMaxH3SetTimestepsStep`] → [`HyperFlowSetTimestepsStep`]: the fixed 8-step sigma
  grid instead of ``linspace(1, 0, num_inference_steps)``, and a ``(timestep, endpoint, indices)`` plan per step
  instead of ``(timestep, indices)``.
* ``denoise.denoise.denoiser`` — [`MiniMaxH3LoopDenoiser`] → [`HyperFlowLoopDenoiser`]: hands the endpoints to the
  [`TwoTimeEmbedder`] before the forward and publishes the step index for step-aware attention (Sol-Attn).

Text encoding, VAE encoding, the packed layout, the Euler update and decoding are the official blocks, unchanged. The
``ref2va`` workflow differs from ``fl2va`` only in the component the loop drives (``transformer_ref``), exactly as in
the official code, so one pair of blocks serves all three workflows.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import torch
from diffusers.modular_pipelines import InputParam, OutputParam, PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser

from .embedder import TwoTimeEmbedder
from .schedule import (
    AUDIO_SHIFT,
    DEFAULT_SIGMAS_8STEP,
    VIDEO_SHIFT,
    build_row_time_pairs,
    endpoints_from_sigmas,
    shift_sigmas,
    validate_sigmas,
)

if TYPE_CHECKING:
    from .lora import HyperFlowMetadata

logger = logging.getLogger(__name__)

WORKFLOWS = ("t2va", "fl2va", "ref2va")
_DIFFUSERS_DEFAULT_SIGMA_POINTS = int(InputParam.template("num_inference_steps").default)

_current_step: ContextVar[int | None] = ContextVar("hyperflow_current_step", default=None)


def current_step() -> int | None:
    """The index of the denoising step whose forward is running, or ``None`` outside a HyperFlow loop."""
    return _current_step.get()


def _hyperflow_is_disabled(components: Any) -> bool:
    """Whether every loaded HyperFlow transformer is in base-model passthrough mode."""
    embedders = []
    for name in ("transformer", "transformer_ref"):
        transformer = getattr(components, name, None)
        embedder = getattr(transformer, "time_embedder", None)
        if isinstance(embedder, TwoTimeEmbedder):
            embedders.append(embedder)
    if not embedders:
        return False
    states = {embedder.passthrough for embedder in embedders}
    if len(states) != 1:
        raise RuntimeError("Loaded HyperFlow transformers disagree on whether HyperFlow is enabled.")
    return states.pop()


class HyperFlowSetTimestepsStep(MiniMaxH3SetTimestepsStep):
    """Official ``set_timesteps`` with the HyperFlow sigma grid and per-row ``(t, r)`` pairs.

    The two official schedulers are still the ones doing the work: their configured shifts (``12.0`` video, ``3.0``
    audio) are applied to the raw grid, and ``set_timesteps(sigmas=...)`` installs the shifted grid verbatim, so the
    Euler update block downstream is untouched. ``num_inference_steps`` is accepted only so that existing call sites
    keep working: it must be omitted or equal the grid's step count.

    Args:
        sigmas: Raw sigma grid, ``None`` for [`DEFAULT_SIGMAS_8STEP`]. The loader overrides the default with the grid
            stored in the weights file; an explicit grid is kept.
    """

    model_name = "minimax-h3"

    def __init__(self, sigmas: Sequence[float] | torch.Tensor | None = None):
        self.sigmas = validate_sigmas(DEFAULT_SIGMAS_8STEP if sigmas is None else sigmas)
        self.sigmas_source = "default" if sigmas is None else "user"
        self.expected_shifts: tuple[float, float] = (VIDEO_SHIFT, AUDIO_SHIFT)
        self._warned_shift = False
        super().__init__()

    @property
    def num_steps(self) -> int:
        return int(self.sigmas.numel() - 1)

    @property
    def description(self) -> str:
        return (
            "HyperFlow: installs the adapter's fixed sigma grid on the two official schedulers (video shift 12.0, "
            "audio shift 3.0) and stages one `(timestep, endpoint, timestep_indices)` triple per step — the "
            "endpoint of step i is `1 - sigma[i + 1]`, conditioning rows keep `endpoint == timestep`."
        )

    @property
    def inputs(self) -> list[InputParam]:
        official = [param for param in super().inputs if param.name != "num_inference_steps"]
        return [
            InputParam(
                name="num_inference_steps",
                type_hint=int,
                default=None,
                required=False,
                description=(
                    "Not used by HyperFlow: the step count is the sigma grid's. Accepted only when it equals it."
                ),
            ),
            *official,
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="Timesteps of the video schedule."),
            OutputParam("audio_timesteps", type_hint=torch.Tensor, description="Timesteps of the audio schedule."),
            OutputParam(
                "row_timestep_plan",
                type_hint=list,
                description=(
                    "One `(timestep, endpoint, timestep_indices)` triple per step: the distinct `(t, r)` pairs of "
                    "the sequence and the index of every row into them."
                ),
            ),
        ]

    def _check_shifts(self, video_shift: float, audio_shift: float) -> None:
        if self._warned_shift or (video_shift, audio_shift) == self.expected_shifts:
            return
        self._warned_shift = True
        logger.warning(
            "HyperFlow was trained with scheduler shifts (video %.3g, audio %.3g) but the pipeline uses (%.3g, %.3g). "
            "The adapter's grid is shifted with the pipeline's values; quality is only validated at the trained ones.",
            *self.expected_shifts,
            video_shift,
            audio_shift,
        )

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        if _hyperflow_is_disabled(components):
            # This block made the input optional so an enabled HyperFlow pipeline does not inherit Diffusers' generic
            # 50-point default. Restore that default only for the base-model path, then delegate without changing the
            # official schedule or row-timestep plan.
            if block_state.num_inference_steps is None:
                state.set("num_inference_steps", _DIFFUSERS_DEFAULT_SIGMA_POINTS)
            return super().__call__(components, state)

        device = components._execution_device

        requested = block_state.num_inference_steps
        if requested is not None and int(requested) != self.num_steps:
            raise ValueError(
                f"HyperFlow runs a fixed {self.num_steps}-step grid; `num_inference_steps={requested}` cannot be "
                "honoured. Omit it, or build the blocks with `hyperflow_blocks(..., sigmas=<your grid>)`."
            )

        video_shift = float(components.scheduler.shift)
        audio_shift = float(components.audio_scheduler.shift)
        self._check_shifts(video_shift, audio_shift)
        video_sigmas = shift_sigmas(self.sigmas, video_shift)
        audio_sigmas = shift_sigmas(self.sigmas, audio_shift)

        components.scheduler.set_timesteps(sigmas=video_sigmas, device=device)
        components.audio_scheduler.set_timesteps(sigmas=audio_sigmas, device=device)
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps

        # The loop's `t` comes from `scheduler.timesteps`; read the same values back so the plan matches bit for bit.
        video_timesteps = components.scheduler.timesteps.cpu()
        audio_timesteps = components.audio_scheduler.timesteps.cpu()
        video_endpoints = endpoints_from_sigmas(video_sigmas)
        audio_endpoints = endpoints_from_sigmas(audio_sigmas)

        plan = []
        for step in range(self.num_steps):
            video_timestep = float(video_timesteps[step])
            triple = build_row_time_pairs(
                video_indices=block_state.video_indices,
                audio_indices=block_state.audio_indices,
                num_condition_video_rows=block_state.num_condition_video_rows,
                num_condition_audio_rows=block_state.num_condition_audio_rows,
                num_text_tokens=block_state.text_indices.numel(),
                video_timestep=video_timestep,
                video_endpoint=float(video_endpoints[step]),
                audio_timestep=float(audio_timesteps[step]),
                audio_endpoint=float(audio_endpoints[step]),
                condition_video_timestep=max(video_timestep, components.keyframe_noise_aug),
                condition_audio_timestep=1.0,
            )
            plan.append(tuple(tensor.to(device) for tensor in triple))
        block_state.row_timestep_plan = plan

        self.set_block_state(state, block_state)
        return components, state


class HyperFlowLoopDenoiser(MiniMaxH3LoopDenoiser):
    """Official loop denoiser plus the endpoint hand-off. ``transformer_name`` selects the checkpoint partition."""

    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            f"HyperFlow: one MiniMax-H3 forward against `{self.transformer_name}` per step, with the step's endpoints "
            "handed to the two-time embedder and the step index published for step-aware attention."
        )

    @property
    def inputs(self) -> list[InputParam]:
        params = []
        for param in super().inputs:
            if param.name == "row_timestep_plan":
                param = InputParam(
                    name="row_timestep_plan",
                    type_hint=list,
                    required=True,
                    description="One `(timestep, endpoint, timestep_indices)` triple per step.",
                )
            params.append(param)
        return params

    @torch.no_grad()
    def __call__(self, components: Any, block_state: Any, i: int, t: torch.Tensor):
        transformer = getattr(components, self.transformer_name)
        embedder = transformer.time_embedder
        if not isinstance(embedder, TwoTimeEmbedder):
            raise RuntimeError(
                f"`{self.transformer_name}` has no HyperFlow LoRA loaded. Call `load_hyperflow_lora(pipe, ...)` after "
                "`pipe.load_components(...)`."
            )
        if embedder.passthrough:
            return super().__call__(components, block_state, i, t)

        entry = block_state.row_timestep_plan[i]
        if len(entry) != 3:
            raise RuntimeError(
                "`row_timestep_plan` holds `(timestep, timestep_indices)` pairs, i.e. it was built by the official "
                "`set_timesteps` block. Build the blocks with `hyperflow_blocks(...)` so both blocks are swapped."
            )
        timestep, endpoint, timestep_indices = entry

        layout_kwargs = {
            name: value
            for name, value in block_state.denoiser_input_fields.items()
            if name in inspect.signature(transformer.forward).parameters
        }
        token = _current_step.set(i)
        try:
            with embedder.endpoint_context(endpoint):
                block_state.noise_pred, block_state.audio_noise_pred = transformer(
                    hidden_states=block_state.latents[None],
                    audio_hidden_states=block_state.audio_latents[None],
                    encoder_hidden_states=block_state.prompt_embeds,
                    timestep=timestep,
                    timestep_indices=timestep_indices,
                    attention_kwargs=block_state.attention_kwargs,
                    return_dict=False,
                    **layout_kwargs,
                )
        finally:
            _current_step.reset(token)
        return components, block_state


def _visit(container: Any, fn) -> None:
    sub_blocks = getattr(container, "sub_blocks", None)
    if not sub_blocks:
        return
    for name, block in list(sub_blocks.items()):
        replacement = fn(block)
        if replacement is not None:
            sub_blocks[name] = replacement
        else:
            _visit(block, fn)


def swap_hyperflow_blocks(blocks: Any, *, sigmas: Sequence[float] | torch.Tensor | None = None) -> Any:
    """Replace the official ``set_timesteps`` and loop ``denoiser`` blocks in place; returns ``blocks``.

    Works on a pruned workflow (``MiniMaxH3Blocks().get_workflow(...)``, ``pipe.blocks``) and on the unpruned
    ``MiniMaxH3Blocks`` (every workflow branch is swapped). Idempotent.
    """
    counts = {"set_timesteps": 0, "denoiser": 0}

    def replace(block: Any):
        if isinstance(block, HyperFlowSetTimestepsStep):
            counts["set_timesteps"] += 1
            return block
        if isinstance(block, HyperFlowLoopDenoiser):
            counts["denoiser"] += 1
            return block
        if isinstance(block, MiniMaxH3SetTimestepsStep):
            counts["set_timesteps"] += 1
            return HyperFlowSetTimestepsStep(sigmas)
        if isinstance(block, MiniMaxH3LoopDenoiser):
            counts["denoiser"] += 1
            return HyperFlowLoopDenoiser(transformer_name=block.transformer_name)
        return None  # a container (sequential / conditional / loop): recurse

    _visit(blocks, replace)
    if counts["set_timesteps"] == 0 or counts["denoiser"] == 0:
        raise ValueError(
            "No MiniMax-H3 `set_timesteps` / loop `denoiser` blocks found to swap. Pass `MiniMaxH3Blocks()`, one of "
            "its workflows, or a MiniMax-H3 pipeline's `.blocks`."
        )
    return blocks


def hyperflow_blocks(
    workflow: str = "fl2va", *, sigmas: Sequence[float] | torch.Tensor | None = None
) -> SequentialPipelineBlocks:
    """The official MiniMax-H3 workflow with the two HyperFlow blocks swapped in.

    ``blocks.init_pipeline("MiniMaxAI/MiniMax-H3", components_manager=...)`` then builds the pipeline exactly as the
    official documentation does.
    """
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown workflow {workflow!r}; expected one of {WORKFLOWS}.")
    return swap_hyperflow_blocks(MiniMaxH3Blocks().get_workflow(workflow), sigmas=sigmas)


def configure_hyperflow_blocks(blocks: Any, metadata: HyperFlowMetadata) -> int:
    """Point every [`HyperFlowSetTimestepsStep`] under ``blocks`` at the grid and shifts stored in a weights file.

    Explicit user grids are kept. Raises when there is no HyperFlow ``set_timesteps`` block at all, which means the
    pipeline would run the official single-time loop against a two-time adapter.
    """
    found = 0

    def visit(block: Any):
        nonlocal found
        if isinstance(block, HyperFlowSetTimestepsStep):
            found += 1
            if metadata.sigmas is not None and block.sigmas_source == "default":
                block.sigmas = validate_sigmas(metadata.sigmas)
                block.sigmas_source = "metadata"
            if metadata.video_shift is not None and metadata.audio_shift is not None:
                block.expected_shifts = (metadata.video_shift, metadata.audio_shift)
            return block
        return None

    _visit(blocks, visit)
    if found == 0:
        raise ValueError(
            "The pipeline runs the official MiniMax-H3 blocks. HyperFlow needs its own `set_timesteps` and "
            "`denoiser`: build the pipeline from `hyperflow_blocks(workflow)`, or call "
            "`swap_hyperflow_blocks(pipe.blocks)`. "
            "Pass `configure_blocks=False` to `load_hyperflow_lora` if you really only want the weights."
        )
    return found


__all__ = [
    "WORKFLOWS",
    "HyperFlowLoopDenoiser",
    "HyperFlowSetTimestepsStep",
    "configure_hyperflow_blocks",
    "current_step",
    "hyperflow_blocks",
    "swap_hyperflow_blocks",
]
