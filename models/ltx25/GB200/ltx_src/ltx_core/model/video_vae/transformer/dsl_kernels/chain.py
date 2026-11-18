"""Stage-5 block chain ping-pong buffers for fused DSL launches."""

from __future__ import annotations

import torch
from torch import nn

from ltx_core.model.video_vae.transformer.layers import ChannelLinear


def _out_slack_rows() -> int:
    from ltx_kernels.vae import OUT_SLACK_ROWS  # noqa: PLC0415

    return int(OUT_SLACK_ROWS)


def alloc_block_volume(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A ``(1,T,H,W,C)`` tensor whose storage carries the fused kernel's lane slack."""
    _, t, h, w, c = shape
    rows = torch.empty((t * h * w + _out_slack_rows(), c), device=device, dtype=dtype)
    return rows[: t * h * w].view(1, t, h, w, c)


def linear_into_block_volume(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """``linear(x)`` written straight into an :func:`alloc_block_volume` buffer."""
    out = alloc_block_volume((*x.shape[:-1], linear.out_features), device=x.device, dtype=x.dtype)
    n_rows = out.numel() // linear.out_features
    rows = out.view(n_rows, linear.out_features)
    torch.mm(x.reshape(n_rows, linear.in_features), linear.weight.t(), out=rows)
    if linear.bias is not None:
        rows.add_(linear.bias)
    return out


def _recyclable_rows(volume: torch.Tensor) -> torch.Tensor | None:
    """``volume``'s backing ``(rows, C)`` buffer, if it was allocated with lane slack."""
    if volume.ndim != 5 or volume.dtype != torch.bfloat16 or volume.storage_offset() != 0:
        return None
    channels = volume.shape[-1]
    n_rows = volume.numel() // channels
    want = n_rows + _out_slack_rows()
    have = volume.untyped_storage().nbytes() // (volume.element_size() * channels)
    if not volume.is_contiguous() or have < want:
        return None
    return torch.as_strided(volume, (want, channels), (channels, 1))


class DSLChannelLinear(ChannelLinear):
    """``conv_in_x_t`` producing a volume the fused block chain can recycle."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_into_block_volume(self, x)


class DSLDiffusionBlockChain(nn.ModuleList):
    """Stage-5 blocks as one callable chain, alternating two output buffers.
    Signature matches deferred decode: ``(x, stage4_feat, modulation)``.
    """

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        spare: torch.Tensor | None = None
        for block in self:
            y = block(
                x,
                stage4_feat,
                modulation,
                drop_leading_frame=drop_leading_frame,
                out=spare,
            )
            spare = _recyclable_rows(x)
            x = y
        return x
