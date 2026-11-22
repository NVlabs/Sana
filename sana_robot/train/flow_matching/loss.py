# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.distributed as dist


def chunk_video_losses(prediction, target, weights, valid, chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the video loss of every chunk [B, L] and the latent values it averages: the weighted squared error of the chunk's target latents [B, C, L * f, tokens], averaged over the valid ones. ``weights`` and ``valid`` hold one value per target latent frame [B, L * f]."""
    error = (target.float() - prediction.float()).square() * weights.float()[:, None, :, None]
    mask = valid.bool()[:, None, :, None].expand(error.shape)
    per_chunk = (int(chunks), error.shape[2] // int(chunks))
    sums = torch.where(mask, error, torch.zeros_like(error)).unflatten(2, per_chunk).sum(dim=(1, 3, 4))
    counts = mask.unflatten(2, per_chunk).sum(dim=(1, 3, 4))
    return sums / counts.clamp_min(1), counts


def chunk_action_losses(prediction, target, weights, mask, chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the action loss of every chunk [B, L] and the values it averages: the weighted squared error of the chunk's action rows [B, L * A, 80], averaged over the cells of ``mask``. ``weights`` holds one value per action row [B, L * A]."""
    error = (prediction.float() - target.float()).square() * weights.float()[:, :, None]
    mask = mask.bool()
    per_chunk = (int(chunks), error.shape[1] // int(chunks))
    sums = torch.where(mask, error, torch.zeros_like(error)).unflatten(1, per_chunk).sum(dim=(2, 3))
    counts = mask.unflatten(1, per_chunk).sum(dim=(2, 3))
    return sums / counts.clamp_min(1), counts


def mean_over_trained_chunks(losses: torch.Tensor, trained: torch.Tensor) -> torch.Tensor:
    """Return the mean of chunk losses [B, L] over the trained chunks; 0, still in the graph, when there is none."""
    trained = trained.to(device=losses.device, dtype=torch.bool)
    return torch.where(trained, losses, torch.zeros_like(losses)).sum() / trained.sum().clamp_min(1)


def global_chunk_scale(trained: torch.Tensor, world_size: int) -> torch.Tensor:
    """Return the factor that turns a rank's mean over its own trained chunks into its share of the mean over the trained chunks of all ranks, once the ranks' gradients are averaged: this rank's count times the number of ranks over the count of all ranks."""
    count = trained.sum().to(torch.float32).reshape(1)
    total = count.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
    return count[0] * world_size / total[0].clamp_min(1.0)
