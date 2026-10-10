# SPDX-License-Identifier: Apache-2.0
"""Resampling forcing that reads the semantic video as extra video input channels.

This meta keeps the RF chains, pictures, Qwen context, resample pass and
validation requests, while every chunk is plain text-to-audio-video denoising
from pure noise and the clean semantic latent is concatenated at the video
patch embedding, as ``SemanticConcatMixin`` in ``streaming_semantic``
describes. Training attaches the condition from the window's
``source_latents`` payload before noising, so the resample pass and the
captionless branch, which reuse those plans, keep it, and the SP input
broadcast already carries the sources.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_concat_rf
      class_name: MiniMaxH3VideoRefStreamingConcatRF
      qwen_reference_video: false
      picture_mode: keyframe
      resample_timesteps: {...}
    models:
      backbone:
        args:
          video_condition_channels: 24
"""

from __future__ import annotations

from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import MiniMaxH3VideoRefStreamingRF
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_rf import SemanticChainMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_semantic import SemanticConcatMixin


class MiniMaxH3VideoRefStreamingConcatRF(SemanticConcatMixin, SemanticChainMixin, MiniMaxH3VideoRefStreamingRF):
    """Resampling forcing from pure noise with the semantic latent concatenated to every target video row."""


EntryClass = MiniMaxH3VideoRefStreamingConcatRF

__all__ = ["MiniMaxH3VideoRefStreamingConcatRF", "EntryClass"]
