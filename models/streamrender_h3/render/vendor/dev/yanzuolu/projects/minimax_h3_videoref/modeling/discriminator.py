# SPDX-License-Identifier: Apache-2.0
"""Chunk-wise DMD2 discriminator that scores one explicit row set per sample.

A streaming window's target is several disjoint packed ranges, the noisy
suffix of each audio channel and the noisy video frames. The V2 head keeps its
parameters and checkpoint layout, and every live sample's explicit ranges
together form that sample's single chunk.
"""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.discriminator import MiniMaxH3DMD2DiscriminatorV2


class MiniMaxH3SampleChunkDiscriminatorV2(MiniMaxH3DMD2DiscriminatorV2):
    """V2 head whose explicit ranges of one sample form one chunk and one logit."""

    @staticmethod
    def _explicit_chunk_layout(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the ranges as V2 does, then pool each sample's ranges into one chunk."""
        rows, lengths, samples = MiniMaxH3DMD2DiscriminatorV2._explicit_chunk_layout(**kwargs)
        count = int(samples[-1].item()) + 1
        pooled = torch.zeros(count, dtype=torch.long, device=lengths.device).index_add_(0, samples, lengths.long())
        return rows, pooled.to(torch.int32), torch.arange(count, device=samples.device)


__all__ = ["MiniMaxH3SampleChunkDiscriminatorV2"]
