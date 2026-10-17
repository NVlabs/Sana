# SPDX-License-Identifier: Apache-2.0
"""A pixel condition video read through the DiT's input condition by streaming text-to-audio-video windows.

``PixelConditionMixin`` keeps the text-to-audio-video layout of
``SemanticConcatMixin`` in ``streaming_semantic``: the windows carry no
reference rows, every chunk is plain text-to-audio-video denoising from pure
noise at the unscaled paired time, Qwen never reads the condition video, the
captionless branch keeps the condition, and the validation video shows the
request's ``reference_video_path`` left of the sample. The condition is not a
VAE latent but any RGB video aligned frame by frame with the target, which the
backbone's input condition encoder reads from pixels, as
``StreamingPixelConditionMixin`` in ``modeling/streaming_pixel_condition``
routes it: every target video row, history and current chunk alike, adds the
encoder's delta for its own latent to its patch embedding. Keyframe or
picture, text, audio and padding rows keep their embeddings.

The encoder's map grid is ``s = 2 ** (len(input_condition_widths) - 1)``
times the token grid, so each frame is pixel-unshuffled by
``u = spatial_vae_stride * 2 / s``, 8 for the 16x VAE and three widths, and
``input_condition_channels`` must be ``4 (3 u^2 + 1)``. The backbone must not
widen its patch embedding with ``video_condition_channels``.

Training reads each clip's frames from the dataset, which decodes
``data.args.pixel_condition``, and attaches them to every window plan before
noising, so the resample pass and the captionless branch keep them. The SP
input broadcast carries no uint8 tensors, so the payload holds the frames'
bytes viewed as int8, ``payload_condition_frames`` writes and
``condition_frames_of`` reads them, without copies.
Validation requests name the condition with ``condition_video_path`` and an
optional ``condition_start_frame``, its frame beside the request's first
frame, 0 by default. In ``guidance_branches`` the reference is the condition
video. A branch that drops it lists no condition rows, so those rows keep
exactly their plain embedding, and otherwise matches the same branch without
that drop. A distillation host's guided fitting drops the condition video the
same way on the student's window. In sampling each weighted branch reads its
own batch's ``condition_frames``, and a branch whose batch holds none lists no
condition rows, like a training branch that drops the reference.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch, StreamingLatentChunk, StreamingState
from dev.yanzuolu.projects.minimax_h3.meta_models.streaming_guidance import StreamingGuidanceBranch
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.config import MiniMaxH3DiTArchConfig
from dev.yanzuolu.projects.minimax_h3_videoref.data.pixel_condition import decode_condition_frames, pixel_condition_channels
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_semantic import SemanticLatentMixin, SourceStreamingBatch
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_pixel_condition import StreamingPixelConditionMixin


def payload_condition_frames(frames: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """View uint8 frames as int8 for the payload."""
    return [value.view(torch.int8) for value in frames]


def condition_frames_of(payload: dict[str, Any]) -> list[torch.Tensor]:
    """The payload's condition frames as uint8."""
    return [value.view(torch.uint8) for value in payload["condition_frames"]]


@dataclass(frozen=True)
class PixelStreamingBatch(SourceStreamingBatch):
    """Streaming inputs with each sample's complete uint8 ``[F, H, W, 3]`` condition frames."""

    condition_frames: list[torch.Tensor] | None = None


class PixelConditionMixin(StreamingPixelConditionMixin, SemanticLatentMixin):
    """Attach each sample's condition frames to its training and sampling window plans.

    Training reads them from the payload's ``condition_frames``, and sampling
    from each branch's batch through ``_stream_branch_condition_frames``.
    """

    def __init__(self, config: Any) -> None:
        configured = config.meta_model.get("source_sigma")
        if configured is not None and float(configured) != 1.0:
            raise ValueError("the pixel condition denoises from pure noise, so source_sigma must be 1")
        super().__init__(config)
        args = config.models.backbone.args
        if int(args.get("video_condition_channels", 0)):
            raise ValueError("the pixel condition enters through input_condition_channels, so "
                             "models.backbone.args.video_condition_channels must stay 0")
        widths = args.get("input_condition_widths", MiniMaxH3DiTArchConfig.input_condition_widths)
        self.pixel_condition_scale = 2 ** (len(widths) - 1)
        token = int(config.data.args.get("spatial_vae_stride", 16)) * 2
        if token % self.pixel_condition_scale:
            raise ValueError(f"{len(widths)} input condition stages cannot reach {token}-pixel tokens by pixel unshuffle")
        self.pixel_condition_unshuffle = token // self.pixel_condition_scale
        channels = pixel_condition_channels(self.pixel_condition_unshuffle)
        if int(args.get("input_condition_channels", 0)) != channels:
            raise ValueError(f"the pixel condition unshuffled by {self.pixel_condition_unshuffle} needs "
                             f"models.backbone.args.input_condition_channels: {channels}")
        if config.data.args.get("pixel_condition") is None:
            raise ValueError("the pixel condition requires data.args.pixel_condition")
        self._stream_condition_frames: list[torch.Tensor] | None = None
        self._stream_branch_condition_frames: list[list[torch.Tensor] | None] | None = None

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Attach every sample's condition frames to its window plan, then noise the chunk as plain text-to-video."""
        frames = condition_frames_of(ctx["encoded_batch"])
        ctx["plans"] = [dict(plan, condition_frames=value) for plan, value in zip(ctx["plans"], frames, strict=True)]
        return super().add_noise(ctx)

    def _guidance_available_conditions(self, ctx: dict[str, Any]) -> tuple[frozenset[str], ...]:
        """Every sample carries its condition video as the reference."""
        return tuple(available | {"reference"} for available in super()._guidance_available_conditions(ctx))

    def _guidance_window_inputs(
        self, ctx: dict[str, Any], branch: StreamingGuidanceBranch,
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Build the branch without the reference drop, then attach the condition unless the branch drops it."""
        drop_reference = "reference" in branch.drop
        inputs, values = super()._guidance_window_inputs(ctx, replace(branch, drop=branch.drop - {"reference"}))
        plans = [dict(plan, condition_frames=None if drop_reference else frames)
                 for plan, frames in zip(inputs.plans, condition_frames_of(ctx["encoded_batch"]), strict=True)]
        return replace(inputs, plans=plans), values

    def _student_branch_inputs(self, view: dict[str, Any], branch: StreamingGuidanceBranch) -> StreamingInputs:
        """A distillation branch drops the condition video for ``reference`` and keeps the keyframe."""
        return self._guidance_window_inputs(view, branch)[0]

    @torch.no_grad()
    def _validation_inputs_for_requests(self, config: Any, models: dict[str, Any], requests: Sequence[dict[str, Any]],
                                        *, prompts: Sequence[str] | None = None) -> PixelStreamingBatch:
        """Decode each request's ``condition_video_path`` for its complete target video."""
        inputs = super()._validation_inputs_for_requests(config, models, requests, prompts=prompts)
        stride = int(config.data.args.get("spatial_vae_stride", 16))
        frames = [decode_condition_frames(
            Path(request["condition_video_path"]).expanduser(), start_frame=int(request.get("condition_start_frame", 0)),
            num_frames=sum(self.video_temporal_mapping.decode_timeline.spans(shape[1])),
            height=shape[2] * stride, width=shape[3] * stride, fps=int(request.get("fps", config.validation.get("fps", 24))),
        ).to(get_device()) for request, shape in zip(requests, inputs.video_shapes, strict=True)]
        values = {field.name: getattr(inputs, field.name) for field in fields(inputs)}
        return PixelStreamingBatch(**values, condition_frames=frames)

    def _iter_stream_latents(self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]],
                             rngs: Sequence[Any], *, models: dict[str, Any] | None = None) -> Iterator[StreamingLatentChunk]:
        """Condition every window of a stream on its sample's frames, each branch on its own batch's."""
        self._stream_condition_frames = branch_inputs[0][1].condition_frames
        self._stream_branch_condition_frames = [branch.condition_frames for weight, branch in branch_inputs if weight != 0]
        try:
            yield from super()._iter_stream_latents(backbone, branch_inputs, rngs, models=models)
        finally:
            self._stream_condition_frames = self._stream_branch_condition_frames = None

    def _state_plan(self, state: StreamingState, stop: int, audio_stop: int, text_len: int) -> dict[str, Any]:
        plan = super()._state_plan(state, stop, audio_stop, text_len)
        return dict(plan, condition_frames=self._stream_condition_frames[state.sample_index])

    def _branch_state_plan(self, state: StreamingState, plan: dict[str, Any], text_len: int,
                           branch_index: int) -> dict[str, Any]:
        """A branch reads its own batch's frames, and none when its batch holds none."""
        frames = self._stream_branch_condition_frames[branch_index]
        return dict(super()._branch_state_plan(state, plan, text_len, branch_index),
                    condition_frames=None if frames is None else frames[state.sample_index])


__all__ = ["PixelConditionMixin", "PixelStreamingBatch", "condition_frames_of", "payload_condition_frames"]
