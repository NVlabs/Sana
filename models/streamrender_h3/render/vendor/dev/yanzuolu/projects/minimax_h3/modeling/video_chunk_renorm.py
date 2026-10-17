# SPDX-License-Identifier: Apache-2.0
"""Match generated video chunks to one fixed chunk's token statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)


@dataclass(frozen=True)
class VideoChunkAnchor:
    """Detached FP32 statistics for each feature of one generated chunk."""

    mean: torch.Tensor
    std: torch.Tensor


def renorm_video_chunk_rows(
    rows: torch.Tensor,
    anchor: VideoChunkAnchor | None = None,
) -> tuple[torch.Tensor, VideoChunkAnchor]:
    """Keep the first chunk unchanged and match later rows to its statistics.

    Rows are patchified video tokens. Each feature is reduced over all tokens
    in this chunk. Both the fixed anchor and the current mean and population
    standard deviation are detached. Gradients flow through the current row values.
    """
    current = rows.float()
    statistics = current.detach()
    mean = statistics.mean(dim=0, keepdim=True)
    std = statistics.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    if anchor is None:
        return rows, VideoChunkAnchor(mean.detach(), std.detach())
    normalized = (current - mean).div(std).mul(anchor.std).add(anchor.mean)
    return normalized.to(dtype=rows.dtype), anchor


def renorm_video_chunk(
    latent: torch.Tensor,
    anchor: VideoChunkAnchor | None = None,
) -> tuple[torch.Tensor, VideoChunkAnchor]:
    """Apply token-column statistics to one video latent shaped [C, T, H, W]."""
    rows = minimax_h3_patchify_video_latent(
        latent.unsqueeze(0), patch_size=(1, 2, 2)
    )
    normalized, updated_anchor = renorm_video_chunk_rows(rows, anchor)
    if anchor is None:
        return latent, updated_anchor
    channels, frames, height, width = latent.shape
    normalized_latent = minimax_h3_unpatchify_video_tokens(
        normalized,
        latent_shape=(frames, height // 2, width // 2, channels),
        patch_size=(1, 2, 2),
    )[0]
    return normalized_latent, updated_anchor


__all__ = ["VideoChunkAnchor", "renorm_video_chunk", "renorm_video_chunk_rows"]
