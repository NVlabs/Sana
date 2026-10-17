# SPDX-License-Identifier: Apache-2.0
"""Per-token AdaLN conditioning of the MiniMax H3 DiT.

Not vendored code -- this file is ours.

A DiT built with ``adaln_condition_dim`` d > 0 takes two more forward inputs.
``adaln_condition_maps`` [N, C, 4h, 4w] holds N per-latent maps with
``C = adaln_condition_channels`` at four times the h x w video token grid, or
is a list of such tensors whose grids may differ. ``adaln_condition_rows``
lists the packed row of every token they describe, tensor after tensor and
in (latent, h, w) order, which is the order the video patchify writes rows.
The encoder maps them once per forward to one d-wide feature per listed row.
Every block owns a head that turns a row's feature into deltas of its
attention and MLP shift and scale::

    h = norm(x) * (1 + scale[idx] + d_scale) + shift[idx] + d_shift

Rows not listed keep the plain AdaLN, so the caller decides which rows are
conditioned and no head output reaches any other row. The heads start at zero,
so a fresh condition leaves the DiT computing exactly its checkpoint's function.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_BF16_DTYPE = torch.bfloat16


class MiniMaxH3AdalnConditionEncoder(nn.Module):
    """Per-latent maps at four times the token grid -> one normalized feature per video token.

    A stride-1 convolution, two stride-2 convolutions down to the token grid
    and a final convolution there, SiLU between them, then RMSNorm over the
    channels. Returns ``[N*h*w, dim]`` rows in (latent, h, w) order.
    """

    def __init__(self, in_channels: int, dim: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, dtype=dtype),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, dtype=dtype),
            nn.SiLU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1, dtype=dtype),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, dtype=dtype),
        )
        self.norm = nn.RMSNorm(dim, eps=eps, dtype=dtype)

    def reset_parameters(self) -> None:
        """PyTorch's default initialization of every layer."""
        for module in (*self.layers, self.norm):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        features = self.layers(maps.to(self.layers[0].weight.dtype))
        return self.norm(features.permute(0, 2, 3, 1)).flatten(0, 2)


class MiniMaxH3AdalnConditionHead(nn.Linear):
    """Zero-initialized ``dim -> 4H``: attention shift and scale deltas, then the MLP's."""

    def __init__(self, dim: int, hidden_size: int, *, dtype: torch.dtype = _BF16_DTYPE) -> None:
        super().__init__(dim, 4 * hidden_size, bias=True, dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


__all__ = ["MiniMaxH3AdalnConditionEncoder", "MiniMaxH3AdalnConditionHead"]
