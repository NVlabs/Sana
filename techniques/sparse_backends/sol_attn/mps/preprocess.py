"""Routing thresholds for the Apple Silicon Metal backend."""

from __future__ import annotations

import math

import torch


def _routing_thresholds(
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    scale: float,
    tau: float,
    thresh_type: str,
) -> torch.Tensor:
    log2_scale = scale * math.log2(math.e)
    k_mean = k_centroids.mean(dim=2)
    raw_mean = (q_centroids * k_mean.unsqueeze(2)).sum(dim=-1)

    if thresh_type == "exact":
        blocks = k_centroids.shape[2]
        second_moment = (
            torch.matmul(
                k_centroids.transpose(-1, -2),
                k_centroids,
            )
            / blocks
        )
        projected = torch.matmul(q_centroids, second_moment)
        raw_second_moment = (projected * q_centroids).sum(dim=-1)
        raw_variance = raw_second_moment - raw_mean.square()
    else:
        k_variance = (k_centroids - k_mean.unsqueeze(2)).square().mean(dim=2)
        raw_variance = (q_centroids.square() * k_variance.unsqueeze(2)).sum(dim=-1)

    mean = raw_mean * log2_scale
    variance = torch.clamp_min(raw_variance, 0.0) * (log2_scale * log2_scale)
    return mean + tau * torch.sqrt(variance + 1.0e-6)


__all__ = ["_routing_thresholds"]
