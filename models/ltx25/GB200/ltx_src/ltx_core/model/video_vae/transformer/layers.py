"""Shared small layers for the diffusion-VAE NA transformer stack."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class ChannelLinear(nn.Linear):
    """``nn.Linear`` exposing ``in_channels``/``out_channels`` for config introspection."""

    @property
    def in_channels(self) -> int:
        return self.in_features

    @property
    def out_channels(self) -> int:
        return self.out_features


class LinearPixelShuffleUpsample(nn.Module):
    """Decoder-side resampler: Linear channel-expand, then channels-last PixelShuffle."""

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, self.proj_out_channels, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        """Upsample; when ``stride[0] == 2`` the pixel-shuffle produces a duplicate
        leading frame that must be dropped to preserve the causal 1:2 (then
        composed 1:8) frame mapping. ``drop_leading_frame`` gates that drop: it
        must be ``True`` only for the chunk that contains the tensor's true
        temporal origin (t=0). Tiled callers processing a later chunk in
        isolation must pass ``False`` -- that chunk has no duplicate leading
        frame of its own to drop, since the one duplicate frame in the full
        (untiled) tensor belongs solely to the origin chunk.
        """
        x = self.proj(x)
        x = rearrange(
            x,
            "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2 and drop_leading_frame:
            x = x[:, 1:, :, :, :]
        return x


class AdaLNZero(nn.Module):
    """Per-block AdaLN-Zero modulation: ``t_emb`` -> 7 (scale/shift/gate) chunks.
    Zero-init output projection so the block is an identity at every timestep
    until the modulation pathway opens up during training.
    """

    NUM_CHUNKS: int = 7  # scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.proj(F.silu(t_emb))
        chunks = h.chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN-style scale + shift modulation to a channels-last tensor."""
    return x * (1.0 + scale) + shift
