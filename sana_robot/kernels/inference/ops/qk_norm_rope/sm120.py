# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.qk_norm_rope.portable import (
    GDN_LAYOUT,
    QKNormRopeResult,
    _allocate_bnhd,
    _eager_inverse_rms,
)

_VERIFIED_TORCH_PREFIX = "2.9."
_VERIFIED_SHAPES: set[tuple[int, int, int]] = set()
_NUM_WARPS = {128: 2, 256: 2}
_LAUNCH_PDL = False
_TILE = 128
_LANES = 32
_VEC = 4
TILE = tl.constexpr(_TILE)
LANES = tl.constexpr(_LANES)
VEC = tl.constexpr(_VEC)


@triton.jit
def _row_inv_rms_torch_order(row_ptr, tile_off, factor, eps, K_TILES: tl.constexpr):
    """Return rsqrt(mean(x^2) + eps) of one row, summed in the order of torch 2.9's CUDA reduction: four values per lane of one warp, then the pairwise tree over the 32 lanes."""
    acc = tl.zeros([LANES, VEC], tl.float32)
    for k in tl.static_range(K_TILES):
        x = tl.load(row_ptr + k * TILE + tile_off).to(tl.float32)
        acc = acc + x * x
    even, odd = tl.split(tl.reshape(acc, [LANES, 2, 2]))
    c0, c2 = tl.split(even)
    c1, c3 = tl.split(odd)
    u = ((c0 + c1) + c2) + c3
    lo, hi = tl.split(tl.reshape(u, [16, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [8, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [4, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [2, 2]))
    u = lo + hi
    lo, hi = tl.split(tl.reshape(u, [1, 2]))
    total = tl.sum(lo + hi, axis=0)
    return tl.math.rsqrt(total * factor + eps)


@triton.jit
def _norm_rotate_store_row(
    row_ptr,
    w_ptr,
    rot_ptr,
    norm_ptr,
    inv,
    cos_lo,
    sin_lo,
    cos_hi,
    sin_hi,
    tile_off,
    HEAD_DIM: tl.constexpr,
    K_TILES: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    OUTPUT_NORMALIZED: tl.constexpr,
):
    """Normalize one row tile by tile, round it to the input dtype, and store it rotated in float64 with the fused multiply-adds of torch's complex128 product."""
    for k in tl.static_range(K_TILES):
        x = tl.load(row_ptr + k * TILE + tile_off).to(tl.float32)
        w = tl.load(w_ptr + k * TILE + tile_off).to(tl.float32)
        n_ = w * (x * inv)
        if ROUND_BF16:
            n_ = n_.to(tl.bfloat16)
        if OUTPUT_NORMALIZED:
            tl.store(norm_ptr + k * TILE + tile_off, n_)
        n0, n1 = tl.split(tl.reshape(n_, [LANES, 2, 2]))
        r0 = n0.to(tl.float64)
        r1 = n1.to(tl.float64)
        if HEAD_DIM == TILE or (k % 2) == 0:
            cos = cos_lo
            sin = sin_lo
        else:
            cos = cos_hi
            sin = sin_hi
        m = r1 * sin
        neg_m = (m.to(tl.int64, bitcast=True) ^ (-(1 << 63))).to(tl.float64, bitcast=True)
        y0 = tl.fma(r0, cos, neg_m).to(tl.float32)
        y1 = tl.fma(r1, cos, r0 * sin).to(tl.float32)
        tl.store(rot_ptr + k * TILE + tile_off, tl.reshape(tl.join(y0, y1), [LANES, VEC]))


@triton.jit
def sana_qk_norm_rope_sm120(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freq_ptr,
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    q_rot_ptr,
    k_rot_ptr,
    q_norm_ptr,
    k_norm_ptr,
    inv_out_ptr,
    num_tokens,
    eps,
    factor,
    q_stride_b,
    q_stride_n,
    k_stride_b,
    k_stride_n,
    freq_stride_b,
    freq_stride_n,
    freq_stride_pair,
    freq_stride_comp,
    out_stride_b,
    out_stride_n,
    inv_stride_b,
    inv_stride_n,
    C_TOTAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FREQ_BATCHED: tl.constexpr,
    IN_KERNEL_RMS: tl.constexpr,
    OUTPUT_NORMALIZED: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    WRITE_INV: tl.constexpr,
):
    """One token row of q and of k, walked in tiles of 128 channels: the inverse RMS in torch's reduction order, the RMSNorm, and the rotation in float64."""
    K_TILES: tl.constexpr = C_TOTAL // TILE
    pid = tl.program_id(0)
    b = pid // num_tokens
    n = pid % num_tokens
    lane = tl.arange(0, LANES)
    comp = tl.arange(0, VEC)
    tile_off = lane[:, None] * VEC + comp[None, :]
    pair_in_tile = tl.reshape(tl.arange(0, TILE // 2), [LANES, 2])

    q_row = q_ptr + b * q_stride_b + n * q_stride_n
    k_row = k_ptr + b * k_stride_b + n * k_stride_n
    out_off = b * out_stride_b + n * out_stride_n
    if FREQ_BATCHED:
        freq_row = freq_ptr + b * freq_stride_b + n * freq_stride_n
    else:
        freq_row = freq_ptr + n * freq_stride_n
    cos_lo = tl.load(freq_row + pair_in_tile * freq_stride_pair)
    sin_lo = tl.load(freq_row + pair_in_tile * freq_stride_pair + freq_stride_comp)
    if HEAD_DIM == TILE:
        cos_hi = cos_lo
        sin_hi = sin_lo
    else:
        cos_hi = tl.load(freq_row + (pair_in_tile + TILE // 2) * freq_stride_pair)
        sin_hi = tl.load(freq_row + (pair_in_tile + TILE // 2) * freq_stride_pair + freq_stride_comp)

    if IN_KERNEL_RMS:
        q_inv = _row_inv_rms_torch_order(q_row, tile_off, factor, eps, K_TILES)
        k_inv = _row_inv_rms_torch_order(k_row, tile_off, factor, eps, K_TILES)
    else:
        inv_off = b * inv_stride_b + n * inv_stride_n
        q_inv = tl.load(q_inv_rms_ptr + inv_off)
        k_inv = tl.load(k_inv_rms_ptr + inv_off)
    if WRITE_INV:
        tl.store(inv_out_ptr + 2 * pid, q_inv)
        tl.store(inv_out_ptr + 2 * pid + 1, k_inv)

    _norm_rotate_store_row(
        q_row,
        q_weight_ptr,
        q_rot_ptr + out_off,
        q_norm_ptr + out_off,
        q_inv,
        cos_lo,
        sin_lo,
        cos_hi,
        sin_hi,
        tile_off,
        HEAD_DIM,
        K_TILES,
        ROUND_BF16,
        OUTPUT_NORMALIZED,
    )
    _norm_rotate_store_row(
        k_row,
        k_weight_ptr,
        k_rot_ptr + out_off,
        k_norm_ptr + out_off,
        k_inv,
        cos_lo,
        sin_lo,
        cos_hi,
        sin_hi,
        tile_off,
        HEAD_DIM,
        K_TILES,
        ROUND_BF16,
        OUTPUT_NORMALIZED,
    )


def serves_shape(channels: int, head_dim: int, rows: int) -> bool:
    """Return whether the kernel reproduces torch's RMS reduction for ``rows`` token rows of ``channels`` channels: torch 2.9, a head dimension of 128 or 256, a multiple of 128 channels in [2048, 8192) and at least 16 rows."""
    return (
        torch.__version__.startswith(_VERIFIED_TORCH_PREFIX)
        and head_dim in (_TILE, 2 * _TILE)
        and channels % _TILE == 0
        and 2048 <= channels < 8192
        and rows >= 16
    )


def _mean_factor(batch: int, tokens: int, channels: int) -> float:
    """Return the float32 factor torch's mean multiplies a row sum by: the output count over the element count."""
    num_outputs = np.float32(batch * tokens)
    numel = np.float32(batch * tokens * channels)
    return float(np.float32(num_outputs / numel))


def fused_qk_norm_rope(q, k, q_weight, k_weight, freqs, layout, head_dim, eps) -> QKNormRopeResult:
    """The sm120 form of ``portable.fused_qk_norm_rope`` for the shapes ``serves_shape`` accepts, with the inverse RMS computed inside the kernel.

    The first call of a shape, which must come before any CUDA graph capture, compares the kernel's inverse RMS rows bitwise with torch's.
    """
    (batch, tokens, channels) = q.shape
    heads = channels // head_dim
    is_gdn = layout == GDN_LAYOUT
    frequency_components = torch.view_as_real(freqs)
    q_rotated = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
    k_rotated = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
    if is_gdn:
        q_normalized = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
        k_normalized = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
    else:
        q_normalized = q_rotated
        k_normalized = k_rotated

    shape_key = (batch, tokens, channels)
    verify = shape_key not in _VERIFIED_SHAPES
    inv_out = torch.empty(batch * tokens * 2, dtype=torch.float32, device=q.device) if verify else None

    sana_qk_norm_rope_sm120[(batch * tokens,)](
        q,
        k,
        q_weight,
        k_weight,
        frequency_components,
        q_rotated,
        k_rotated,
        q_rotated,
        k_rotated,
        q_normalized,
        k_normalized,
        inv_out if verify else q_rotated,
        tokens,
        float(eps),
        _mean_factor(batch, tokens, channels),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        frequency_components.stride(0),
        frequency_components.stride(2),
        frequency_components.stride(3),
        frequency_components.stride(4),
        q_rotated.stride(0),
        q_rotated.stride(1),
        0,
        0,
        C_TOTAL=channels,
        HEAD_DIM=head_dim,
        FREQ_BATCHED=freqs.shape[0] > 1,
        IN_KERNEL_RMS=True,
        OUTPUT_NORMALIZED=is_gdn,
        ROUND_BF16=q.dtype == torch.bfloat16,
        WRITE_INV=verify,
        num_warps=_NUM_WARPS.get(head_dim, 2),
        launch_pdl=_LAUNCH_PDL,
        enable_fp_fusion=False,
    )
    if verify:
        expected = torch.stack(
            (_eager_inverse_rms(q, eps).reshape(-1), _eager_inverse_rms(k, eps).reshape(-1)), dim=1
        ).reshape(-1)
        if not torch.equal(inv_out.view(torch.int32), expected.view(torch.int32)):
            raise RuntimeError(
                f"the fused q / k norm kernel does not reproduce torch's RMS reduction for shape {shape_key}"
            )
        _VERIFIED_SHAPES.add(shape_key)

    if is_gdn:
        return QKNormRopeResult(
            q_rotated.permute(0, 2, 3, 1),
            k_rotated.permute(0, 2, 3, 1),
            q_normalized.permute(0, 2, 3, 1),
            k_normalized.permute(0, 2, 3, 1),
        )
    return QKNormRopeResult(q_rotated, k_rotated, None, None)
