"""Deferred stage-4 context inject: sequential upsample then context_proj (W-chunked).
Eager in-place Python W-loop — stays outside ChunkedCompile's ``forward_attn_mlp``
graph so Inductor never sees inject. Runs ``upsamples[3].proj`` (pixel-shuffle)
then ``context_proj``; no fused Linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _upsample_then_ctx(
    feat: torch.Tensor,
    up_w: torch.Tensor,
    up_b: torch.Tensor,
    ctx_w: torch.Tensor,
    ctx_b: torch.Tensor | None,
    stride: tuple[int, int, int],
    *,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """``context_proj(pixel_shuffle(upsample_proj(feat)))``."""
    p1, p2, p3 = stride
    up = F.linear(feat, up_w, up_b)
    up = rearrange(
        up,
        "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
        p1=p1,
        p2=p2,
        p3=p3,
    )
    if p1 == 2 and drop_leading_frame:
        up = up[:, 1:, :, :, :]
    return F.linear(up, ctx_w, ctx_b)


def deferred(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample_proj: nn.Linear,
    context_proj: nn.Linear,
    stride: tuple[int, int, int],
    *,
    w_chunks: int,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """Chunk stage-4 feat along W, upsample+``context_proj`` each slab, ``add_`` into ``x``.
    Returns ``x`` (mutated).
    """
    assert upsample_proj.bias is not None
    p3 = stride[2]
    up_w, up_b = upsample_proj.weight, upsample_proj.bias
    ctx_w, ctx_b = context_proj.weight, context_proj.bias

    if w_chunks <= 1:
        x.add_(
            _upsample_then_ctx(
                stage4_feat,
                up_w,
                up_b,
                ctx_w,
                ctx_b,
                stride,
                drop_leading_frame=drop_leading_frame,
            )
        )
        return x

    feat_chunks = torch.chunk(stage4_feat, w_chunks, dim=3)
    w_hi = x.shape[3]
    lo = 0
    for feat_chunk in feat_chunks:
        hi = min(w_hi, lo + feat_chunk.shape[3] * p3)
        ctx = _upsample_then_ctx(
            feat_chunk,
            up_w,
            up_b,
            ctx_w,
            ctx_b,
            stride,
            drop_leading_frame=drop_leading_frame,
        )
        x[:, :, :, lo:hi, :].add_(ctx[:, :, :, : hi - lo, :])
        lo = hi
    return x
