# SPDX-License-Identifier: Apache-2.0
"""Deliver each streaming VideoRef latent pack as one emission per training window.

The chain protocol, phase stagger and mid-chain worker replay live in the
shared ``StreamingLatentChainMixin``. This entry binds them to the paired
VideoRef latent corpus and keeps its own worker-state schema.
"""

from __future__ import annotations

from dev.yanzuolu.projects.minimax_h3.data.streaming_chain import StreamingLatentChainMixin
from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_latent import (
    VideoRefStreamingLatentT2AVDataset,
)


class VideoRefStreamingLatentChainDataset(StreamingLatentChainMixin, VideoRefStreamingLatentT2AVDataset):
    """Emit a head batch with the paired pack, then light batches for its remaining windows."""

    _STATE_SCHEMA = "minimax_h3_videoref_streaming_latent_chain_worker"
    _STATE_VERSION = 1


EntryClass = VideoRefStreamingLatentChainDataset

__all__ = ["VideoRefStreamingLatentChainDataset", "EntryClass"]
