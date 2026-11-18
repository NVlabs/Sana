# SPDX-License-Identifier: Apache-2.0
"""Per-token input conditioning of the MiniMax H3 DiT.

Not vendored code -- this file is ours.

A DiT built with ``input_condition_channels`` C > 0 takes two more forward
inputs. ``input_condition_maps`` [N, C, s*h, s*w] holds N per-latent maps at s
times the h x w video token grid, ``s = 2 ** (len(input_condition_widths) - 1)``,
or is a list of such tensors whose grids may differ. ``input_condition_rows``
lists the packed row of every token they describe, tensor after tensor and in
(latent, h, w) order, which is the order the video patchify writes rows. The
encoder turns every listed row's map cell into a hidden-wide delta that is
added to that row's input embedding, on top of what the video patch embedding
wrote there. Rows not listed keep their embedding, so the caller decides which
rows are conditioned. The encoder's output projection starts at zero, weight
and bias, so a fresh condition leaves the DiT computing exactly its
checkpoint's function.

The encoder is a residual CNN. A 1x1 stem maps the C channels to
``widths[0]``. Stage k holds ``input_condition_blocks`` residual blocks of
``widths[k]`` channels, each ``GroupNorm(32) -> SiLU -> conv3x3`` twice plus
the identity, and every stage after the first opens with a stride-2 3x3
convolution from the previous width, so the last stage runs on the token grid.
RMSNorm over the channels and the zero-initialized linear to the hidden size
follow. With ``checkpoint`` every stem, block and downsampling layer is
recomputed in backward, so only the layer boundaries stay alive.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_layer

_BF16_DTYPE = torch.bfloat16
_GROUPS = 32


class MiniMaxH3InputConditionBlock(nn.Module):
    """``x + conv(SiLU(GroupNorm(conv(SiLU(GroupNorm(x))))))`` at a fixed width."""

    def __init__(self, channels: int, *, dtype: torch.dtype = _BF16_DTYPE) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_GROUPS, channels, dtype=dtype)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dtype=dtype)
        self.norm2 = nn.GroupNorm(_GROUPS, channels, dtype=dtype)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(hidden)))


class MiniMaxH3InputConditionEncoder(nn.Module):
    """Per-latent maps at ``2 ** (len(widths) - 1)`` times the token grid -> one delta per kept video token."""

    def __init__(
        self, in_channels: int, widths: Sequence[int], blocks: int, hidden_size: int, *,
        eps: float, dtype: torch.dtype = _BF16_DTYPE,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, widths[0], 1, dtype=dtype)]
        for stage, width in enumerate(widths):
            if stage:
                layers.append(nn.Conv2d(widths[stage - 1], width, 3, stride=2, padding=1, dtype=dtype))
            layers.extend(MiniMaxH3InputConditionBlock(width, dtype=dtype) for _ in range(blocks))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.RMSNorm(widths[-1], eps=eps, dtype=dtype)
        self.out = nn.Linear(widths[-1], hidden_size, dtype=dtype)
        self._zero_out()

    def _zero_out(self) -> None:
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def reset_parameters(self) -> None:
        """PyTorch's default initialization of every layer, then the zero output projection."""
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self._zero_out()

    def forward(self, maps: Sequence[torch.Tensor], keep: torch.Tensor, *, checkpoint: bool = False) -> torch.Tensor:
        """Encode every map, keep the tokens ``keep`` selects in (map, latent, h, w) order -> ``[kept, hidden]``."""
        dtype = self.out.weight.dtype
        features = []
        for value in maps:
            hidden = value.to(dtype)
            for layer in self.layers:
                hidden = (checkpoint_layer(layer, hidden, use_reentrant=False)
                          if checkpoint and torch.is_grad_enabled() else layer(hidden))
            features.append(hidden.permute(0, 2, 3, 1).flatten(0, 2))
        features = torch.cat(features)
        if features.shape[0] != keep.shape[0]:
            raise ValueError(
                f"input_condition_maps encode {features.shape[0]} tokens but "
                f"input_condition_rows lists {keep.shape[0]} rows"
            )
        return self.out(self.norm(features[keep]))


__all__ = ["MiniMaxH3InputConditionBlock", "MiniMaxH3InputConditionEncoder"]
