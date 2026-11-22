# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.cross_qk_norm import portable

_TILE = 128
_LANES = 32
_CHANNELS = 2560
_MIN_ROWS = 16
_VERIFIED_TORCH_PREFIX = "2.9."
_VERIFIED_SHAPES = set()
_FACTORS = {}
TILE = tl.constexpr(_TILE)
LANES = tl.constexpr(_LANES)


@triton.jit
def _finish_inv_rows(acc, factor, eps, ROWS: tl.constexpr):
    """Return rsqrt(mean(x^2) + eps) per row from the [ROWS, 128] sums of squares per channel of a tile, combined in the order of torch 2.9's CUDA reduction: channel 4 * lane + i of the tile is value i of lane ``lane``, the four values of a lane add in order, then the 32 lanes add as a pairwise tree."""
    even, odd = tl.split(tl.reshape(acc, [ROWS, LANES, 2, 2]))
    c0, c2 = tl.split(even)
    c1, c3 = tl.split(odd)
    u = ((c0 + c1) + c2) + c3
    lo, hi = tl.split(tl.reshape(u, [ROWS, 16, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [ROWS, 8, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [ROWS, 4, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [ROWS, 2, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [ROWS, 1, 2]))
    total = tl.reshape(lo + hi, [ROWS])
    return tl.math.rsqrt(total * factor + eps)


@triton.jit
def sana_cross_qk_rmsnorm_sm120_unrolled_c2560(
    Q,
    KV,
    WQ,
    WK,
    INV_OUT,
    n_q,
    n_k,
    kv_stride,
    factor_q,
    factor_k,
    ROWS: tl.constexpr,
    WRITE_INV: tl.constexpr,
    EPS: tl.constexpr,
):
    """ROWS rows of 2560 channels per program, in place: query rows for the first ceil(n_q / ROWS) programs, then key rows at stride kv_stride. Each row is read once as 20 tiles of 128 channels that stay in registers between the reduction and the normalization; with WRITE_INV the inverse RMS of every row also goes to INV_OUT."""
    pid = tl.program_id(0)
    n_q_programs = tl.cdiv(n_q, ROWS)
    rr = tl.arange(0, ROWS)
    span = tl.arange(0, TILE)[None, :]
    if pid < n_q_programs:
        rows = pid * ROWS + rr
        valid = rows < n_q
        ptr = Q + rows.to(tl.int64)[:, None] * 2560 + span
        w_ptr = WQ
        factor = factor_q
        inv_base = 0
    else:
        rows = (pid - n_q_programs) * ROWS + rr
        valid = rows < n_k
        ptr = KV + rows.to(tl.int64)[:, None] * kv_stride + span
        w_ptr = WK
        factor = factor_k
        inv_base = n_q
    mask = valid[:, None] & (span >= 0)
    x0 = tl.load(ptr + 0, mask=mask, other=0.0)
    x1 = tl.load(ptr + 128, mask=mask, other=0.0)
    x2 = tl.load(ptr + 256, mask=mask, other=0.0)
    x3 = tl.load(ptr + 384, mask=mask, other=0.0)
    x4 = tl.load(ptr + 512, mask=mask, other=0.0)
    x5 = tl.load(ptr + 640, mask=mask, other=0.0)
    x6 = tl.load(ptr + 768, mask=mask, other=0.0)
    x7 = tl.load(ptr + 896, mask=mask, other=0.0)
    x8 = tl.load(ptr + 1024, mask=mask, other=0.0)
    x9 = tl.load(ptr + 1152, mask=mask, other=0.0)
    x10 = tl.load(ptr + 1280, mask=mask, other=0.0)
    x11 = tl.load(ptr + 1408, mask=mask, other=0.0)
    x12 = tl.load(ptr + 1536, mask=mask, other=0.0)
    x13 = tl.load(ptr + 1664, mask=mask, other=0.0)
    x14 = tl.load(ptr + 1792, mask=mask, other=0.0)
    x15 = tl.load(ptr + 1920, mask=mask, other=0.0)
    x16 = tl.load(ptr + 2048, mask=mask, other=0.0)
    x17 = tl.load(ptr + 2176, mask=mask, other=0.0)
    x18 = tl.load(ptr + 2304, mask=mask, other=0.0)
    x19 = tl.load(ptr + 2432, mask=mask, other=0.0)
    w0 = tl.load(w_ptr + 0 + span)
    w1 = tl.load(w_ptr + 128 + span)
    w2 = tl.load(w_ptr + 256 + span)
    w3 = tl.load(w_ptr + 384 + span)
    w4 = tl.load(w_ptr + 512 + span)
    w5 = tl.load(w_ptr + 640 + span)
    w6 = tl.load(w_ptr + 768 + span)
    w7 = tl.load(w_ptr + 896 + span)
    w8 = tl.load(w_ptr + 1024 + span)
    w9 = tl.load(w_ptr + 1152 + span)
    w10 = tl.load(w_ptr + 1280 + span)
    w11 = tl.load(w_ptr + 1408 + span)
    w12 = tl.load(w_ptr + 1536 + span)
    w13 = tl.load(w_ptr + 1664 + span)
    w14 = tl.load(w_ptr + 1792 + span)
    w15 = tl.load(w_ptr + 1920 + span)
    w16 = tl.load(w_ptr + 2048 + span)
    w17 = tl.load(w_ptr + 2176 + span)
    w18 = tl.load(w_ptr + 2304 + span)
    w19 = tl.load(w_ptr + 2432 + span)
    t = x0.to(tl.float32)
    acc = t * t
    t = x1.to(tl.float32)
    acc = acc + t * t
    t = x2.to(tl.float32)
    acc = acc + t * t
    t = x3.to(tl.float32)
    acc = acc + t * t
    t = x4.to(tl.float32)
    acc = acc + t * t
    t = x5.to(tl.float32)
    acc = acc + t * t
    t = x6.to(tl.float32)
    acc = acc + t * t
    t = x7.to(tl.float32)
    acc = acc + t * t
    t = x8.to(tl.float32)
    acc = acc + t * t
    t = x9.to(tl.float32)
    acc = acc + t * t
    t = x10.to(tl.float32)
    acc = acc + t * t
    t = x11.to(tl.float32)
    acc = acc + t * t
    t = x12.to(tl.float32)
    acc = acc + t * t
    t = x13.to(tl.float32)
    acc = acc + t * t
    t = x14.to(tl.float32)
    acc = acc + t * t
    t = x15.to(tl.float32)
    acc = acc + t * t
    t = x16.to(tl.float32)
    acc = acc + t * t
    t = x17.to(tl.float32)
    acc = acc + t * t
    t = x18.to(tl.float32)
    acc = acc + t * t
    t = x19.to(tl.float32)
    acc = acc + t * t
    inv = _finish_inv_rows(acc, factor, EPS, ROWS)
    if WRITE_INV:
        tl.store(INV_OUT + inv_base + rows, inv, mask=valid)
    inv2 = inv[:, None]
    tl.store(ptr + 0, (w0.to(tl.float32) * (x0.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 128, (w1.to(tl.float32) * (x1.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 256, (w2.to(tl.float32) * (x2.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 384, (w3.to(tl.float32) * (x3.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 512, (w4.to(tl.float32) * (x4.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 640, (w5.to(tl.float32) * (x5.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 768, (w6.to(tl.float32) * (x6.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 896, (w7.to(tl.float32) * (x7.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1024, (w8.to(tl.float32) * (x8.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1152, (w9.to(tl.float32) * (x9.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1280, (w10.to(tl.float32) * (x10.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1408, (w11.to(tl.float32) * (x11.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1536, (w12.to(tl.float32) * (x12.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1664, (w13.to(tl.float32) * (x13.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1792, (w14.to(tl.float32) * (x14.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 1920, (w15.to(tl.float32) * (x15.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 2048, (w16.to(tl.float32) * (x16.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 2176, (w17.to(tl.float32) * (x17.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 2304, (w18.to(tl.float32) * (x18.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)
    tl.store(ptr + 2432, (w19.to(tl.float32) * (x19.to(tl.float32) * inv2)).to(tl.bfloat16), mask=mask)


def supports(channels: int) -> bool:
    """Return whether rows of ``channels`` channels are served: by this kernel at 2560 channels, by the portable kernel at its own widths."""
    return portable.supports(channels)


def reproduces_torch(n_q: int, n_k: int, channels: int, kv_stride: int) -> bool:
    """Return whether the kernel reproduces torch's RMS reduction for these rows: torch 2.9, 2560 channels, at least 16 query rows and 16 key rows, and key rows 16-byte aligned."""
    return (
        torch.__version__.startswith(_VERIFIED_TORCH_PREFIX)
        and channels == _CHANNELS
        and n_q >= _MIN_ROWS
        and n_k >= _MIN_ROWS
        and kv_stride >= channels
        and kv_stride % 8 == 0
    )


def mean_factor(rows: int, channels: int) -> float:
    """Return the float32 factor torch's mean over the last dimension of [rows, channels] multiplies a row sum by: the output count over the element count."""
    key = (rows, channels)
    value = _FACTORS.get(key)
    if value is None:
        num_outputs = np.float32(rows)
        numel = np.float32(rows * channels)
        value = float(np.float32(num_outputs / numel))
        _FACTORS[key] = value
    return value


def _eager_inverse_rms(rows: torch.Tensor, eps: float) -> torch.Tensor:
    """Return rsqrt(mean(x^2) + eps) of every row as the norm modules compute it."""
    return torch.rsqrt(rows.float().pow(2).mean(-1) + eps)


def cross_qk_norm(q_rows, k_rows, q_weight, k_weight, eps: float) -> None:
    """The sm120 form of ``portable.cross_qk_norm``: bitwise equal to the norm modules for the rows ``reproduces_torch`` accepts; other rows go to the portable kernel.

    The first call of a shape outside a CUDA graph capture compares the kernel's inverse RMS rows bitwise with torch's.
    """
    (n_q, channels) = q_rows.shape
    n_k = k_rows.shape[0]
    if not reproduces_torch(n_q, n_k, channels, k_rows.stride(0)):
        portable.cross_qk_norm(q_rows, k_rows, q_weight, k_weight, eps)
        return
    shape_key = (n_q, n_k, channels)
    verify = shape_key not in _VERIFIED_SHAPES and not torch.cuda.is_current_stream_capturing()
    if verify:
        expected = torch.cat((_eager_inverse_rms(q_rows, eps), _eager_inverse_rms(k_rows, eps)))
        inv_out = torch.empty(n_q + n_k, dtype=torch.float32, device=q_rows.device)
    sana_cross_qk_rmsnorm_sm120_unrolled_c2560[(n_q + n_k,)](
        q_rows,
        k_rows,
        q_weight,
        k_weight,
        inv_out if verify else q_rows,
        n_q,
        n_k,
        k_rows.stride(0),
        mean_factor(n_q, channels),
        mean_factor(n_k, channels),
        ROWS=1,
        WRITE_INV=verify,
        EPS=float(eps),
        num_warps=1,
        enable_fp_fusion=False,
    )
    if verify:
        if not torch.equal(inv_out.view(torch.int32), expected.view(torch.int32)):
            raise RuntimeError(
                f"the cross-attention q / k norm kernel does not reproduce torch's RMS reduction for shape {shape_key}"
            )
        _VERIFIED_SHAPES.add(shape_key)
