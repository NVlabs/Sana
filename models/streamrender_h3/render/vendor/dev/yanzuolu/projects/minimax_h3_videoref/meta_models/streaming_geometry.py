# SPDX-License-Identifier: Apache-2.0
"""Backward optical flow as per-token AdaLN conditioning of streaming VideoRef windows.

Every target video row, history and current chunk alike, receives the flow
maps of its own latent through the DiT's per-token AdaLN condition, as
``StreamingAdalnConditionMixin`` routes them. Reference, picture or keyframe,
text and audio rows receive none. The maps are the game-agnostic backward flow,
occlusion and validity of ``flow_geometry``, 20 channels per latent.

Training reads each clip's maps from the dataset, which loads them with
``data.args.geometry_root``, and attaches them to every window plan before
noising. Every forward built from those plans carries them: the resample pass,
the conditional and captionless branches and every guidance branch, including
one that drops the target history. Validation requests name their clip's
``geometry_path``, with an optional ``geometry_start_frame``, the geometry
frame of the request's first RGB frame, 0 by default, and every sampled window
and branch reads the maps of its own latents.

The backbone needs ``adaln_condition_dim`` and ``adaln_condition_channels: 20``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch, StreamingLatentChunk, StreamingState
from dev.yanzuolu.projects.minimax_h3_videoref.data.flow_geometry import FLOW_CONDITION_CHANNELS, load_flow_condition_maps
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import VideoRefStreamingBatch
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_adaln_condition import StreamingAdalnConditionMixin


@dataclass(frozen=True)
class GeometryStreamingBatch(VideoRefStreamingBatch):
    """Streaming inputs with each sample's complete ``[T, 20, H, W]`` flow condition maps."""

    adaln_condition_maps: list[torch.Tensor] | None = None


class StreamingGeometryMixin(StreamingAdalnConditionMixin):
    """Attach each sample's flow condition maps to its training and sampling window plans.

    Training reads them from the payload's ``adaln_condition_maps``, and
    sampling from ``_stream_condition_maps``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        args = config.models.backbone.args
        if int(args.get("adaln_condition_dim", 0)) <= 0 or int(args.get("adaln_condition_channels", 0)) != FLOW_CONDITION_CHANNELS:
            raise ValueError("flow conditioning requires models.backbone.args.adaln_condition_dim > 0 and "
                             f"adaln_condition_channels: {FLOW_CONDITION_CHANNELS}")
        if config.data.args.get("geometry_root") is None:
            raise ValueError("flow conditioning requires data.args.geometry_root")
        self._stream_condition_maps: list[torch.Tensor] | None = None

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Attach every sample's complete maps to its window plan before noising."""
        maps = ctx["encoded_batch"]["adaln_condition_maps"]
        ctx["plans"] = [dict(plan, adaln_condition_maps=value) for plan, value in zip(ctx["plans"], maps, strict=True)]
        return super().add_noise(ctx)

    @torch.no_grad()
    def _validation_inputs_for_requests(self, config: Any, models: dict[str, Any], requests: Sequence[dict[str, Any]],
                                        *, prompts: Sequence[str] | None = None) -> GeometryStreamingBatch:
        """Load each request's maps from ``geometry_path`` for its complete target video."""
        inputs = super()._validation_inputs_for_requests(config, models, requests, prompts=prompts)
        maps = [load_flow_condition_maps(
            Path(request["geometry_path"]).expanduser(), start_frame=int(request.get("geometry_start_frame", 0)),
            latent_count=shape[1], video_temporal_mapping=self.video_temporal_mapping,
        ).to(get_device()) for request, shape in zip(requests, inputs.video_shapes, strict=True)]
        values = {field.name: getattr(inputs, field.name) for field in fields(inputs)}
        return GeometryStreamingBatch(**values, adaln_condition_maps=maps)

    def _iter_stream_latents(self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]],
                             rngs: Sequence[Any], *, models: dict[str, Any] | None = None) -> Iterator[StreamingLatentChunk]:
        """Condition every window of a stream on its sample's maps, which all branches share."""
        self._stream_condition_maps = branch_inputs[0][1].adaln_condition_maps
        try:
            yield from super()._iter_stream_latents(backbone, branch_inputs, rngs, models=models)
        finally:
            self._stream_condition_maps = None

    def _state_plan(self, state: StreamingState, stop: int, audio_stop: int, text_len: int) -> dict[str, Any]:
        plan = super()._state_plan(state, stop, audio_stop, text_len)
        return dict(plan, adaln_condition_maps=self._stream_condition_maps[state.sample_index])


__all__ = ["GeometryStreamingBatch", "StreamingGeometryMixin"]
