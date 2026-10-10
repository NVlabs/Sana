# SPDX-License-Identifier: Apache-2.0
"""Online, cycle-aligned RF crops from complete normalized GT mother clips, one emission per window."""

from __future__ import annotations

from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_latent_chain import (
    VideoRefStreamingLatentChainDataset,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_long_latent import LongLatentCatalogMixin


class VideoRefStreamingLongLatentChainDataset(LongLatentCatalogMixin, VideoRefStreamingLatentChainDataset):
    """Long-GT crops delivered as chains. The chain iterator adds phase staggering and mid-chain replay."""

    _STATE_SCHEMA = "minimax_h3_videoref_streaming_long_latent_chain_worker"
    _STATE_VERSION = 1


EntryClass = VideoRefStreamingLongLatentChainDataset

__all__ = ["VideoRefStreamingLongLatentChainDataset", "EntryClass"]
