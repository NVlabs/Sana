# SPDX-License-Identifier: Apache-2.0
"""Streaming teacher forcing whose current chunk starts from the semantic video.

The teacher-forcing counterpart of ``MiniMaxH3VideoRefStreamingSourceRF``.
Each sample trains one randomly chosen window of its clip with corpus history
anchored at the clean time, and the current chunk follows the bridge of
``SemanticSourceMixin`` in ``streaming_semantic``. The windows carry no
reference rows, and the payload keeps every clip's semantic latents as
``source_latents``, so a training UP group broadcasts them with the rest of
the inputs. Pictures and validation are the host's.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_sft
      class_name: MiniMaxH3VideoRefStreamingSourceSFT
      source_sigma: 0.95
      qwen_reference_video: false
      picture_mode: keyframe
"""

from __future__ import annotations

from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import (
    MiniMaxH3VideoRefStreamingSFT,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_semantic import SemanticSourceMixin


class MiniMaxH3VideoRefStreamingSourceSFT(SemanticSourceMixin, MiniMaxH3VideoRefStreamingSFT):
    """Teacher forcing that bridges each chunk from its semantic latent instead of reading reference rows."""


EntryClass = MiniMaxH3VideoRefStreamingSourceSFT

__all__ = ["MiniMaxH3VideoRefStreamingSourceSFT", "EntryClass"]
