# SPDX-License-Identifier: Apache-2.0
"""Resampling forcing that reads a pixel condition video through the DiT's input condition.

The pixel-condition counterpart of ``MiniMaxH3VideoRefStreamingConcatRF``.
This meta keeps the RF chains, pictures, Qwen context, resample pass and
validation requests, while every chunk is plain text-to-audio-video denoising
from pure noise and every target video row reads the condition video of its
own latent through the backbone's input condition encoder, as
``PixelConditionMixin`` in ``streaming_pixel`` describes. Each chain keeps
every clip's complete condition frames on the device with its conditioning,
decoded once at the chain's head, and each window's payload carries them, so
with a training UP size above 1 the SP input broadcast gives every group rank
the source rank's frames along with its working copies.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_rf
      class_name: MiniMaxH3VideoRefStreamingPixelRF
      qwen_reference_video: false
      picture_mode: keyframe
      resample_timesteps: {...}
    data:
      args:
        pixel_condition: {root: /path/to/clips, filename: semantics.mp4}
    models:
      backbone:
        args:
          input_condition_channels: 772
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import (
    MiniMaxH3VideoRefStreamingRF,
    _ChainState,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_rf import (
    SemanticChainMixin,
    _SourceChainConditioning,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_pixel import PixelConditionMixin, payload_condition_frames


@dataclass
class _PixelChainConditioning(_SourceChainConditioning):
    """The parent's chain conditioning plus every clip's complete condition frames."""

    condition_frames: list[torch.Tensor] | None = None


class PixelChainMixin:
    """Keep each clip's condition frames with the chain and carry them in every window's payload."""

    def _chain_head_conditioning(self, ctx: dict[str, Any], entry: _ChainState) -> _PixelChainConditioning:
        conditioning = super()._chain_head_conditioning(ctx, entry)
        frames = [value.to(get_device()) for value in ctx["batch"]["condition_frames"]]
        values = {field.name: getattr(conditioning, field.name) for field in fields(conditioning)}
        return _PixelChainConditioning(**values, condition_frames=frames)

    def _chain_window_conditioning(
        self, ctx: dict[str, Any], entry: _ChainState, window: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        plans, conditioning = super()._chain_window_conditioning(ctx, entry, window)
        return plans, dict(conditioning, condition_frames=payload_condition_frames(entry.conditioning.condition_frames))


class MiniMaxH3VideoRefStreamingPixelRF(PixelConditionMixin, PixelChainMixin, SemanticChainMixin, MiniMaxH3VideoRefStreamingRF):
    """Resampling forcing from pure noise with a pixel condition video on every target video row."""


EntryClass = MiniMaxH3VideoRefStreamingPixelRF

__all__ = ["MiniMaxH3VideoRefStreamingPixelRF", "PixelChainMixin", "EntryClass"]
