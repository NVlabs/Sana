# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import triton
import triton.language as tl

_CA = 2048


@triton.jit
def sana_cross_qk_rmsnorm(
    Q,
    KV,
    WQ,
    WK,
    n_q,
    kv_stride,
    C: tl.constexpr,
    CA: tl.constexpr,
    CB: tl.constexpr,
    EPS: tl.constexpr,
):
    """One row per program, in place: query row ``row`` of Q for the first n_q programs, then key row ``row - n_q`` at KV + (row - n_q) * kv_stride; weight * (x * rsqrt(mean(x^2) + EPS)) in float32, rounded once to bf16."""
    row = tl.program_id(0)
    ca = tl.arange(0, CA)
    cb = CA + tl.arange(0, CB)
    if row < n_q:
        base = Q + row * C
        wa = tl.load(WQ + ca).to(tl.float32)
        wb = tl.load(WQ + cb).to(tl.float32)
    else:
        base = KV + (row - n_q) * kv_stride
        wa = tl.load(WK + ca).to(tl.float32)
        wb = tl.load(WK + cb).to(tl.float32)

    xa = tl.load(base + ca).to(tl.float32)
    xb = tl.load(base + cb).to(tl.float32)
    ss = tl.sum(xa * xa) + tl.sum(xb * xb)
    r = tl.math.rsqrt(ss * (1.0 / C) + EPS)
    tl.store(base + ca, (wa * (xa * r)).to(tl.bfloat16))
    tl.store(base + cb, (wb * (xb * r)).to(tl.bfloat16))


def supports(channels: int) -> bool:
    """Return whether the kernel serves rows of ``channels`` channels: 2048 plus a power of two."""
    tail = channels - _CA
    return tail > 0 and tail & (tail - 1) == 0


def cross_qk_norm(q_rows, k_rows, q_weight, k_weight, eps: float) -> None:
    """Apply the query and key RMS norms of a text cross-attention in place with one launch: to the contiguous bf16 query rows [N, C] and to the bf16 key rows [L, C] at any row stride, such as the key half of the packed key / value projection [L, 2, C]. The sum of squares is reduced in another order than ``Tensor.mean``, so the result matches the norm modules to the last bits, not bitwise."""
    (tokens, channels) = q_rows.shape
    sana_cross_qk_rmsnorm[(tokens + k_rows.shape[0],)](
        q_rows,
        k_rows,
        q_weight,
        k_weight,
        tokens,
        k_rows.stride(0),
        C=channels,
        CA=_CA,
        CB=channels - _CA,
        EPS=float(eps),
        num_warps=2,
        enable_fp_fusion=False,
    )
