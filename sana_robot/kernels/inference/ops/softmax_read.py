# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.dispatch import SM110, TunedTable, device_architecture
from sana_robot.kernels.inference.ops.qk_norm_rope.portable import _eager_inverse_rms
from sana_robot.kernels.inference.ops.qk_norm_rope.sm120 import (
    LANES,
    TILE,
    VEC,
    _mean_factor,
    _row_inv_rms_torch_order,
)

NORM_CHANNELS = 2560
MAX_HEADS = 64
_ROPE_ARITH = "comp"
_ROPE_SPLIT = "int"
_ROPE_BLOCK_L = 1
_ROPE_WARPS = 2
_ARITH = {"fp64": 0, "fp32": 1, "comp": 2}
_SPLITS = {"fp64": 0, "int": 1}
_NORM_WARPS = 2
_NORM_SPLIT = False
_NORM_LAUNCH: dict[int, tuple[int, bool]] = {193: (1, True), 160: (1, True), 240: (1, True), 273: (1, False)}
_NORM_LAUNCH_SM110 = TunedTable(_NORM_LAUNCH, tuned=_NORM_LAUNCH)
_NORM_VERIFIED: set[tuple[int, float, float]] = set()


@triton.jit
def _two_sum(x, y):
    """Return the rounded sum of x and y and its exact rounding error."""
    s = x + y
    yv = s - x
    xv = s - yv
    return s, (x - xv) + (y - yv)


@triton.jit
def _split3(ph):
    """Split float64 phases into three float32 parts by float64 arithmetic: the top 16 significant bits, the rest of the float32 value and the float32 of what float32 drops."""
    hi = ph.to(tl.float32)
    lo = (ph - hi.to(tl.float64)).to(tl.float32)
    p0 = (hi.to(tl.int32, bitcast=True) & -256).to(tl.float32, bitcast=True)
    return p0, hi - p0, lo


@triton.jit
def _split3_int(ph):
    """Split float64 phases into the three float32 parts of ``_split3`` by integer operations on their bits."""
    bits = ph.to(tl.int64, bitcast=True)
    e = ((bits >> 52) & 0x7FF).to(tl.int32)
    hi = tl.where(e > 896, ((e - 896) << 23 | ((bits >> 29) & 0x7FFFFF).to(tl.int32)).to(tl.float32, bitcast=True), 0.0)
    scale = tl.where(e > 948, ((e - 948) << 23).to(tl.float32, bitcast=True), 0.0)
    lo = (bits & 0x1FFFFFFF).to(tl.int32).to(tl.float32) * scale
    neg = bits < 0
    hi = tl.where(neg, -hi, hi)
    lo = tl.where(neg, -lo, lo)
    p0 = (hi.to(tl.int32, bitcast=True) & -256).to(tl.float32, bitcast=True)
    return p0, hi - p0, lo


@triton.jit
def _comp_dot2(x, u0, u1, ul, y, v0, v1, vl):
    """Return the float32 rounding of x * (u0 + u1 + ul) + y * (v0 + v1 + vl) for bf16 values x and y, whose products with u0, u1, v0 and v1 are exact."""
    s, e = _two_sum(x * u0, y * v0)
    s1, e1 = _two_sum(x * u1, y * v1)
    s2, e2 = _two_sum(s, s1)
    return s2 + (((e + e1) + e2) + (x * ul + y * vl))


@triton.jit
def _rotate_rows(
    x_ptr,
    out_ptr,
    rows,
    mask,
    cols,
    c,
    s,
    c0,
    s0,
    c1,
    s1,
    cl,
    sl,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    ARITH: tl.constexpr,
    ORDER: tl.constexpr,
):
    """Rotate BLOCK_L rows of x [*, H, D] for all H heads by the rows' phase parts and store them in bf16."""
    for h in tl.static_range(H):
        offs = rows[:, None] * (H * D) + h * D + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if ARITH == 0:
            x = x.to(tl.float64)
        else:
            x = x.to(tl.float32)
        a, b = tl.split(tl.reshape(x, (BLOCK_L, D // 2, 2)))
        if ARITH == 2:
            re = _comp_dot2(a, c0, c1, cl, -b, s0, s1, sl)
            im = _comp_dot2(a, s0, s1, sl, b, c0, c1, cl)
        elif ORDER == 0:
            re = tl.fma(a, c, -(b * s))
            im = tl.fma(a, s, b * c)
        else:
            re = tl.fma(-b, s, a * c)
            im = tl.fma(b, c, a * s)
        if ARITH == 0:
            re = re.to(tl.float32)
            im = im.to(tl.float32)
        re = re.to(tl.bfloat16)
        im = im.to(tl.bfloat16)
        y = tl.reshape(tl.join(re, im), (BLOCK_L, D))
        tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def sana_softmax_rope_read_sm110(
    x_ptr,
    ph_ptr,
    out_ptr,
    L,
    ph_row0,
    q_ptr,
    q_out_ptr,
    q_row0,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    ARITH: tl.constexpr,
    ORDER: tl.constexpr,
    SPLIT: tl.constexpr,
    ONE_LAUNCH: tl.constexpr,
):
    """BLOCK_L rows of the keys [L, H, D] rotated by their phases [L, D / 2] (cos, sin) in float64, loaded once for all heads; with ONE_LAUNCH the programs at the last rows also rotate the queries that share those phases."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_L + tl.arange(0, BLOCK_L)
    rmask = rows < L
    cols = tl.arange(0, D)
    mask = rmask[:, None]
    ph = tl.load(ph_ptr + (ph_row0 + rows)[:, None] * D + cols[None, :], mask=mask, other=0.0)
    if ARITH == 2:
        if SPLIT == 0:
            p0, p1, pl = _split3(ph)
        else:
            p0, p1, pl = _split3_int(ph)
        c0, s0 = tl.split(tl.reshape(p0, (BLOCK_L, D // 2, 2)))
        c1, s1 = tl.split(tl.reshape(p1, (BLOCK_L, D // 2, 2)))
        cl, sl = tl.split(tl.reshape(pl, (BLOCK_L, D // 2, 2)))
        c = c0
        s = s0
    else:
        c, s = tl.split(tl.reshape(ph, (BLOCK_L, D // 2, 2)))
        if ARITH == 1:
            c = c.to(tl.float32)
            s = s.to(tl.float32)
        c0 = c
        s0 = s
        c1 = c
        s1 = s
        cl = c
        sl = s
    _rotate_rows(x_ptr, out_ptr, rows, mask, cols, c, s, c0, s0, c1, s1, cl, sl, H, D, BLOCK_L, ARITH, ORDER)
    if ONE_LAUNCH:
        if pid * BLOCK_L + BLOCK_L > q_row0:
            q_rows = rows - q_row0
            q_mask = (rmask & (q_rows >= 0))[:, None]
            _rotate_rows(
                q_ptr, q_out_ptr, q_rows, q_mask, cols, c, s, c0, s0, c1, s1, cl, sl, H, D, BLOCK_L, ARITH, ORDER
            )


def _launch(x, phases_f64, ph_row0, out, arith, order, block_l, warps, split="fp64", q=None, q_out=None, q_row0=0):
    """Launch the rotation over the rows of ``x``, and over the queries ``q`` at its last rows when they are given."""
    _, length, heads, dim = x.shape
    grid = (triton.cdiv(length, block_l),)
    one = q is not None
    return sana_softmax_rope_read_sm110[grid](
        x,
        phases_f64,
        out,
        length,
        ph_row0,
        q if one else x,
        q_out if one else out,
        q_row0,
        H=heads,
        D=dim,
        BLOCK_L=block_l,
        ARITH=_ARITH[arith],
        ORDER=order,
        SPLIT=_SPLITS[split],
        ONE_LAUNCH=one,
        num_warps=warps,
        enable_fp_fusion=False,
    )


def rope_read_serves(q: torch.Tensor, keys: torch.Tensor, phases: torch.Tensor) -> bool:
    """Return whether ``softmax_rope_read`` serves a read: complex128 phases [1, 1, T, head_dim / 2] shared by the heads, one row per key."""
    return phases.dtype == torch.complex128 and tuple(phases.shape) == (1, 1, keys.shape[1], q.shape[3] // 2)


def softmax_rope_read(q: torch.Tensor, keys: torch.Tensor, phases: torch.Tensor):
    """Return the queries [1, N, heads, head_dim] rotated by the last N rows of ``phases`` and the keys [1, T, heads, head_dim] of a read rotated by all T rows, in one launch.

    The kernel computes in compensated float32 and equals the policy's float64 rotation bit for bit; inputs and outputs are bf16.
    """
    (q, keys, phases) = (q.contiguous(), keys.contiguous(), phases.contiguous())
    n, t = q.shape[1], keys.shape[1]
    ph = torch.view_as_real(phases[0, 0])
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(keys)
    _launch(keys, ph, 0, k_rot, _ROPE_ARITH, 0, _ROPE_BLOCK_L, _ROPE_WARPS, _ROPE_SPLIT, q=q, q_out=q_rot, q_row0=t - n)
    return (q_rot, k_rot)


@triton.jit
def _norm_store_row(row_ptr, w_ptr, out_ptr, inv, tile_off, K_TILES: tl.constexpr):
    """Store one row normalized in float32 and rounded to bf16."""
    for k in tl.static_range(K_TILES):
        x = tl.load(row_ptr + k * TILE + tile_off).to(tl.float32)
        w = tl.load(w_ptr + k * TILE + tile_off).to(tl.float32)
        tl.store(out_ptr + k * TILE + tile_off, (w * (x * inv)).to(tl.bfloat16))


@triton.jit
def sana_softmax_qk_norm_sm110(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_out_ptr,
    k_out_ptr,
    inv_out_ptr,
    q_stride_n,
    k_stride_n,
    factor,
    q_eps,
    k_eps,
    C: tl.constexpr,
    WRITE_INV: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """The RMSNorm of the q row and the k row of one token (SPLIT: of one row per program), with the inverse RMS in torch's reduction order."""
    K_TILES: tl.constexpr = C // TILE
    tile_off = tl.arange(0, LANES)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    if SPLIT:
        pid = tl.program_id(0)
        n = pid // 2
        if pid % 2 == 0:
            q_row = q_ptr + n * q_stride_n
            q_inv = _row_inv_rms_torch_order(q_row, tile_off, factor, q_eps, K_TILES)
            if WRITE_INV:
                tl.store(inv_out_ptr + 2 * n, q_inv)
            _norm_store_row(q_row, q_weight_ptr, q_out_ptr + n * C, q_inv, tile_off, K_TILES)
        else:
            k_row = k_ptr + n * k_stride_n
            k_inv = _row_inv_rms_torch_order(k_row, tile_off, factor, k_eps, K_TILES)
            if WRITE_INV:
                tl.store(inv_out_ptr + 2 * n + 1, k_inv)
            _norm_store_row(k_row, k_weight_ptr, k_out_ptr + n * C, k_inv, tile_off, K_TILES)
    else:
        n = tl.program_id(0)
        q_row = q_ptr + n * q_stride_n
        k_row = k_ptr + n * k_stride_n
        q_inv = _row_inv_rms_torch_order(q_row, tile_off, factor, q_eps, K_TILES)
        k_inv = _row_inv_rms_torch_order(k_row, tile_off, factor, k_eps, K_TILES)
        if WRITE_INV:
            tl.store(inv_out_ptr + 2 * n, q_inv)
            tl.store(inv_out_ptr + 2 * n + 1, k_inv)
        _norm_store_row(q_row, q_weight_ptr, q_out_ptr + n * C, q_inv, tile_off, K_TILES)
        _norm_store_row(k_row, k_weight_ptr, k_out_ptr + n * C, k_inv, tile_off, K_TILES)


def qk_norm_serves(tokens: int, q_eps: float, k_eps: float) -> bool:
    """Return whether ``softmax_qk_norm`` serves a call: at least 16 token rows, the row count from which torch reduces one row per warp, and outside a CUDA graph capture unless the token count already passed its first-call check."""
    if tokens < 16:
        return False
    verified = (int(tokens), float(q_eps), float(k_eps)) in _NORM_VERIFIED
    return verified or not torch.cuda.is_current_stream_capturing()


def softmax_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    heads: int,
):
    """Return the RMS-normalized queries and keys [1, N, heads, head_dim] of the bf16 projections q and k [1, N, 2560] in one launch, bitwise the policy's two RMSNorms.

    The first call of a token count compares the kernel's inverse RMS rows bitwise with torch's.
    """
    _, n, c = q.shape
    table = _NORM_LAUNCH_SM110 if device_architecture(q.device) == SM110 else _NORM_LAUNCH
    warps, split = table.get(int(n), (_NORM_WARPS, _NORM_SPLIT))
    key = (int(n), float(q_eps), float(k_eps))
    verify = key not in _NORM_VERIFIED
    q_out = torch.empty(1, n, c, dtype=q.dtype, device=q.device)
    k_out = torch.empty(1, n, c, dtype=k.dtype, device=k.device)
    inv = torch.empty(n, 2, dtype=torch.float32, device=q.device) if verify else q_out
    sana_softmax_qk_norm_sm110[(2 * n if split else n,)](
        q,
        k,
        q_weight,
        k_weight,
        q_out,
        k_out,
        inv,
        q.stride(1),
        k.stride(1),
        _mean_factor(1, n, c),
        q_eps,
        k_eps,
        C=c,
        WRITE_INV=verify,
        SPLIT=split,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    if verify:
        torch_rows = torch.stack(
            (_eager_inverse_rms(q, q_eps).reshape(-1), _eager_inverse_rms(k, k_eps).reshape(-1)), dim=1
        )
        if not torch.equal(inv, torch_rows):
            raise RuntimeError(f"the softmax q / k norm kernel does not reproduce torch's RMS reduction at {n} tokens")
        _NORM_VERIFIED.add(key)
    return (q_out.view(1, n, heads, c // heads), k_out.view(1, n, heads, c // heads))
