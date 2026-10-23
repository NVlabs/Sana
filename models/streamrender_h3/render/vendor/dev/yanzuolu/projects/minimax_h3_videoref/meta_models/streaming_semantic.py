# SPDX-License-Identifier: Apache-2.0
"""Semantic-video injections shared by streaming teacher forcing and resampling forcing.

Both injections keep the paired semantic video out of the DiT sequence. The
windows carry no reference rows, and each sample's complete semantic latent
travels with its payload as ``source_latents``. Target and semantic latents
share one VAE and temporal grid, so a window row reads the semantic latent at
its own video index.

``SemanticSourceMixin`` starts the current chunk from the semantic latent, a
trained SDEdit bridge. The schedule is ``x_t = (1 - t) x0 + t eps``. With
``meta_model.source_sigma`` sigma in (0, 1], a chunk starts at
``z = (1 - sigma) src + sigma eps`` and its noisy video rows are
``x_t = (1 - t / sigma) x0 + (t / sigma) z`` at the video time ``t = sigma u``,
where u is the host's paired video draw. The noise coefficient is exactly t,
so the DiT reads the pretrained schedule at t and only the signal blends
target and source. Equivalently x_t is the schedule's forward with the
effective noise ``x0 + (z - x0) / sigma``, which is what the flow_v conversion
reads. The target stays x0. Audio has no source and keeps its unscaled time
from the same paired draw. History rows, the picture, the captionless branch
and the fitting are the host's. Every eps comes from the host's draw, so the
RNG streams are the host's. Sampling starts each chunk at z, built from the
request's semantic latent and noise from that request's stream, on video time
sigma, and walks sigma times the configured video grid. Audio walks its own
grid from 1. DDIM on x0 predictions moves ``x_t`` to
``x0 + (t' / t) (x_t - x0)``, the same bridge, so the sampler is unchanged.

``SemanticConcatMixin`` denoises every chunk as plain text-to-audio-video from
pure noise at the unscaled paired time. The clean semantic latent is
concatenated channel-wise to each target video row at the patch embedding.
History and current-chunk rows read it at their own video indices. Text,
keyframe or picture, audio and padding rows read zeros. The backbone's
``video_condition_channels`` must equal the video latent channels.

Both set ``data.args.reference_video_rows`` false, so the dataset budgets
windows without reference rows and still loads the semantic latents. Qwen
must not read the semantic video, and ``keep_negative_reference`` must stay
true because sampling keeps the picture rows in the captionless branch. The
validation video still shows the semantic video left of the sample.

In ``guidance_branches`` the reference is the semantic video alone. Qwen
never reads it, so no branch needs a reference-free text pass. A concat
branch that drops the reference zeroes its semantic channels and otherwise
matches the same branch without that drop, keeping the picture rows and
Qwen's picture prefix. The source bridge starts every chunk from the
semantic latent, so no branch may drop the reference.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch, StreamingLatentChunk, StreamingState
from dev.yanzuolu.projects.minimax_h3.meta_models.streaming_guidance import GUIDANCE_CONDITIONS, StreamingGuidanceBranch
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import VideoRefStreamingBatch
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_concat import StreamingVideoConditionMixin


@dataclass(frozen=True)
class SourceStreamingBatch(VideoRefStreamingBatch):
    """Streaming inputs whose semantic latents travel beside the windows, never as reference rows."""

    source_latents: list[torch.Tensor] | None = None


class SemanticLatentMixin:
    """Carry each sample's semantic latent as ``source_latents`` instead of reference rows.

    Hosts are VideoRef streaming metas. Training reads the sources from the
    window payload, and sampling from ``_stream_sources``. The picture is the
    text-to-audio-video keyframe these windows always keep, so no guidance
    branch may drop ``picture``.
    """

    guidance_conditions: tuple[str, ...] = GUIDANCE_CONDITIONS

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if self.qwen_visual_context and self.qwen_reference_video:
            raise ValueError("the semantic video is injected into the target rows, so Qwen must not read it: set qwen_reference_video false")
        if not self.keep_negative_reference:
            raise ValueError("sampling keeps the picture rows in the captionless branch, so keep_negative_reference must stay true")
        configured = config.data.args.get("reference_video_rows")
        if configured is not None and bool(configured):
            raise ValueError("data.args.reference_video_rows must be false because windows carry no reference rows")
        config.data.args.reference_video_rows = False
        self.reference_video_rows = False
        self._stream_sources: list[torch.Tensor] | None = None

    @property
    def needs_reference_free_text_conditioning(self) -> bool:
        """Qwen never reads the semantic video, so dropping it keeps each window's Qwen context."""
        return False

    @torch.no_grad()
    def _validation_inputs_for_requests(self, config: Any, models: dict[str, Any], requests: Sequence[dict[str, Any]],
                                        *, prompts: Sequence[str] | None = None) -> SourceStreamingBatch:
        """Carry each request's semantic latents as sources and keep its pixels for the side-by-side video."""
        inputs = super()._validation_inputs_for_requests(config, models, requests, prompts=prompts)
        assert all(tuple(value.shape) == tuple(shape) for value, shape in zip(inputs.reference_latents, inputs.video_shapes, strict=True)), (
            "the semantic source must share the target's latent grid"
        )
        values = {field.name: getattr(inputs, field.name) for field in fields(inputs)}
        return SourceStreamingBatch(**dict(values, reference_latents=None), source_latents=inputs.reference_latents)

    def _validation_latent_metadata(
        self, config: Any, inputs: SourceStreamingBatch, request: Any, *, index: int,
    ) -> dict[str, Any]:
        return super()._validation_latent_metadata(
            config, replace(inputs, reference_latents=inputs.source_latents), request, index=index,
        )

    def _iter_stream_latents(self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]],
                             rngs: Sequence[Any], *, models: dict[str, Any] | None = None) -> Iterator[StreamingLatentChunk]:
        """Condition every window of a stream on its sample's sources, which all branches share."""
        self._stream_sources = branch_inputs[0][1].source_latents
        try:
            yield from super()._iter_stream_latents(backbone, branch_inputs, rngs, models=models)
        finally:
            self._stream_sources = None


class SemanticSourceMixin(SemanticLatentMixin):
    """Bridge each chunk from its semantic latent at ``source_sigma``."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        sigma = config.meta_model.get("source_sigma")
        if sigma is None:
            raise ValueError("source-bridged streaming requires meta_model.source_sigma")
        self.source_sigma = float(sigma)
        if not 0 < self.source_sigma <= 1:
            raise ValueError("source_sigma must be in (0, 1]")
        if any("reference" in branch.drop for branch in self.guidance_branches or ()):
            raise ValueError("the semantic video starts every chunk and scales its video time, so no guidance branch can drop the reference")

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Scale the paired video draw by sigma. Audio keeps its time from the same draw."""
        ctx = super().sample_timesteps(ctx)
        video, audio = ctx["train_timesteps"]
        ctx["train_timesteps"] = (video * self.source_sigma, audio)
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Move each chunk's noisy video rows onto the bridge from its source start point.

        The host's chunk noise is the eps of z, and the stored target noise
        becomes the effective noise, so the flow_v conversion reads this x_t.
        """
        ctx = super().add_noise(ctx)
        sigma = self.source_sigma
        sources = ctx["encoded_batch"]["source_latents"]
        noisy, targets, noises = ctx["noisy_latents"][0], ctx["window_targets"][0], ctx["target_noises"][0]
        for index, plan in enumerate(ctx["plans"]):
            mask = plan["video_noisy_mask"]
            target = targets[index]
            start = (1 - sigma) * self._select(sources[index], plan["video_indices"][mask]).float() + sigma * noises[index]
            timestep = ctx["train_timesteps"][0][index]
            noisy[index] = noisy[index].index_copy(1, mask.nonzero().flatten().to(target.device),
                                                   (1 - timestep / sigma) * target + timestep / sigma * start)
            noises[index] = target + (start - target) / sigma
        return ctx

    def _state_plan(self, state: StreamingState, stop: int, audio_stop: int, text_len: int) -> dict[str, Any]:
        """Attach the semantic latent of the window's noisy rows as ``source_latents``."""
        plan = super()._state_plan(state, stop, audio_stop, text_len)
        rows = plan["video_indices"][plan["video_noisy_mask"]]
        return dict(plan, source_latents=self._select(self._stream_sources[state.sample_index], rows))

    def _denoise_window(
        self, backbone: Any, branches: Sequence[tuple[float, StreamingInputs]], plans: list[dict[str, Any]],
        histories: tuple[list[torch.Tensor], list[torch.Tensor]], video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor], *, rngs: list[Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Start each chunk at z, using the stream's noise draw as its eps."""
        sigma = self.source_sigma
        video_xts = [(1 - sigma) * plan["source_latents"].to(noise) + sigma * noise
                     for plan, noise in zip(plans, video_xts, strict=True)]
        return super()._denoise_window(backbone, branches, plans, histories, video_xts, audio_xts, rngs=rngs)

    def _sampling_grid(self, node: Any, seqlens: Sequence[int], device: torch.device) -> torch.Tensor:
        """Scale the video grid by sigma, so its first time is the chunk's start time."""
        grid = super()._sampling_grid(node, seqlens, device)
        return grid * self.source_sigma if node is self.sampling_timesteps else grid


class SemanticConcatMixin(StreamingVideoConditionMixin, SemanticLatentMixin):
    """Concatenate each target video row's semantic latent to its input channels."""

    def __init__(self, config: Any) -> None:
        configured = config.meta_model.get("source_sigma")
        if configured is not None and float(configured) != 1.0:
            raise ValueError("channel-concatenated conditioning denoises from pure noise, so source_sigma must be 1")
        super().__init__(config)
        channels = int(config.models.backbone.args.get("video_condition_channels", 0))
        if channels != int(config.data.args.video_latent_channels):
            raise ValueError("models.backbone.args.video_condition_channels must equal data.args.video_latent_channels")

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Attach each window's semantic condition, then noise the chunk as plain text-to-video."""
        sources = ctx["encoded_batch"]["source_latents"]
        ctx["plans"] = [self._with_video_condition(plan, source)
                        for plan, source in zip(ctx["plans"], sources, strict=True)]
        return super().add_noise(ctx)

    def _guidance_available_conditions(self, ctx: dict[str, Any]) -> tuple[frozenset[str], ...]:
        """Every sample carries its semantic latent as the reference."""
        return tuple(available | {"reference"} for available in super()._guidance_available_conditions(ctx))

    def _guidance_window_inputs(
        self, ctx: dict[str, Any], branch: StreamingGuidanceBranch,
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Build the branch without the reference drop, then attach each surviving row's semantic channels.

        A reference drop zeroes them. Picture rows and Qwen context follow the
        remaining drops.
        """
        drop_reference = "reference" in branch.drop
        inputs, values = super()._guidance_window_inputs(ctx, replace(branch, drop=branch.drop - {"reference"}))
        plans = []
        for plan, source in zip(inputs.plans, ctx["encoded_batch"]["source_latents"], strict=True):
            condition = self._with_video_condition(plan, source)["video_condition_latents"]
            plans.append(dict(plan, video_condition_latents=condition.zero_() if drop_reference else condition))
        return replace(inputs, plans=plans), values

    def _state_plan(self, state: StreamingState, stop: int, audio_stop: int, text_len: int) -> dict[str, Any]:
        plan = super()._state_plan(state, stop, audio_stop, text_len)
        return self._with_video_condition(plan, self._stream_sources[state.sample_index])


__all__ = ["SemanticConcatMixin", "SemanticLatentMixin", "SemanticSourceMixin", "SourceStreamingBatch"]
