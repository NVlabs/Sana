# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer: ``down_proj(silu(gate_proj(x)) * up_proj(x))``."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=True)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=True)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
