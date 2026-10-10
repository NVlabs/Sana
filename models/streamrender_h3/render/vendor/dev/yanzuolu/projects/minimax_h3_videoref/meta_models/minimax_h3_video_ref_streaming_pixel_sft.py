# SPDX-License-Identifier: Apache-2.0
"""Streaming teacher forcing that reads a pixel condition video through the DiT's input condition.

The pixel-condition counterpart of ``MiniMaxH3VideoRefStreamingConcatSFT``.
Each sample trains one randomly chosen window of its clip with corpus history
anchored at the clean time. Every chunk is plain text-to-audio-video denoising
from pure noise, and every target video row reads the condition video of its
own latent through the backbone's input condition encoder, as
``PixelConditionMixin`` in ``streaming_pixel`` describes. The payload keeps
every clip's complete condition frames on the device, so the captionless
branch, which reuses the window plans, and a training UP group both see them.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_sft
      class_name: MiniMaxH3VideoRefStreamingPixelSFT
      qwen_reference_video: false
      picture_mode: keyframe
    data:
      args:
        pixel_condition: {root: /path/to/clips, filename: semantics.mp4}
    validation:
      requests:
        - reference_video_path: /path/to/clips/<window>/semantics.mp4
          condition_video_path: /path/to/clips/<window>/semantics.mp4
    models:
      backbone:
        args:
          input_condition_channels: 772
"""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import (
    MiniMaxH3VideoRefStreamingSFT,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_pixel import PixelConditionMixin, payload_condition_frames


class MiniMaxH3VideoRefStreamingPixelSFT(PixelConditionMixin, MiniMaxH3VideoRefStreamingSFT):
    """Teacher forcing from pure noise with a pixel condition video on every target video row."""

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Carry every clip's condition frames in the payload."""
        ctx = super().prepare_inputs(ctx)
        ctx["encoded_batch"]["condition_frames"] = self._to_device(payload_condition_frames(ctx["batch"]["condition_frames"]))
        return ctx


EntryClass = MiniMaxH3VideoRefStreamingPixelSFT

__all__ = ["MiniMaxH3VideoRefStreamingPixelSFT", "EntryClass"]
