# SPDX-License-Identifier: Apache-2.0
"""Streaming teacher forcing that reads the semantic video as extra video input channels.

The teacher-forcing counterpart of ``MiniMaxH3VideoRefStreamingConcatRF``.
Each sample trains one randomly chosen window of its clip with corpus history
anchored at the clean time. Every chunk is plain text-to-audio-video
denoising from pure noise, and the clean semantic latent is concatenated at
the video patch embedding, as ``SemanticConcatMixin`` in
``streaming_semantic`` describes. The payload keeps every clip's semantic
latents as ``source_latents``, so the captionless branch, which reuses the
window plans, and a training UP group both see them.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_concat_sft
      class_name: MiniMaxH3VideoRefStreamingConcatSFT
      qwen_reference_video: false
      picture_mode: keyframe
    models:
      backbone:
        args:
          video_condition_channels: 24
"""

from __future__ import annotations

from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import (
    MiniMaxH3VideoRefStreamingSFT,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_semantic import SemanticConcatMixin


class MiniMaxH3VideoRefStreamingConcatSFT(SemanticConcatMixin, MiniMaxH3VideoRefStreamingSFT):
    """Teacher forcing from pure noise with the semantic latent concatenated to every target video row."""


EntryClass = MiniMaxH3VideoRefStreamingConcatSFT

__all__ = ["MiniMaxH3VideoRefStreamingConcatSFT", "EntryClass"]
