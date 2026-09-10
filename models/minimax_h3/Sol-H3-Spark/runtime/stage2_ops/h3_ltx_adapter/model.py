"""Inference-only Conv3D architecture for the promoted adapter."""

from __future__ import annotations

import torch
from torch import nn


class FactorizedResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int, groups: int):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.spatial = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels)
        self.in_proj = nn.Conv3d(channels, hidden * 2, kernel_size=1)
        self.out_proj = nn.Conv3d(hidden, channels, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.norm(inputs))
        hidden = self.temporal(self.spatial(hidden))
        value, gate = self.in_proj(self.activation(hidden)).chunk(2, dim=1)
        return inputs + self.out_proj(value * self.activation(gate))


class H3ToLTXConvAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, num_blocks: int, expansion: int, groups: int):
        super().__init__()
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.stem = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *(FactorizedResidualBlock(width, expansion=expansion, groups=groups) for _ in range(num_blocks))
        )
        self.final_norm = nn.GroupNorm(groups, width, eps=1e-6)
        self.final_activation = nn.SiLU()
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, aligned_h3: torch.Tensor) -> torch.Tensor:
        residual = self.blocks(self.stem(aligned_h3))
        residual = self.head(self.final_activation(self.final_norm(residual)))
        return self.skip(aligned_h3) + residual


def build_model(config: dict) -> H3ToLTXConvAdapter:
    if config.get("model_type") != "tiny":
        raise ValueError(f"handoff supports only the promoted tiny/Conv3D model, got {config.get('model_type')!r}")
    unsupported = {
        "dense_temporal_blocks": int(config.get("dense_temporal_blocks", 0)),
        "coordinate_channels": int(config.get("coordinate_channels", 0)),
        "output_refiner_blocks": int(config.get("output_refiner_blocks", 0)),
        "source_detail_blocks": int(config.get("source_detail_blocks", 0)),
    }
    if any(unsupported.values()):
        raise ValueError(f"checkpoint enables unsupported training variants: {unsupported}")
    return H3ToLTXConvAdapter(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        width=int(config["width"]),
        num_blocks=int(config["num_blocks"]),
        expansion=int(config["expansion"]),
        groups=int(config["groups"]),
    )
