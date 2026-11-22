# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

_RMS_EPS = 1e-6
_LN_EPS = 1e-6
CHANNELS = 2560
BOUNDARY_DEFAULT = (4, 2, 0, 0, 1, 0)
BOUNDARY_CONFIGS: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int]] = {
    (4533, 4, 1, 1): (4, 2, 0, 0, 1, 1),
    (8165, 4, 2, 1): (8, 2, 0, 0, 1, 0),
    (2717, 1, 2, 1): (4, 2, 0, 1, 1, 0),
}
LAYER_NORM_DEFAULT = (4, 2, 1)
LAYER_NORM_CONFIGS: dict[int, tuple[int, int, int]] = {2700: (2, 2, 1), 4500: (2, 2, 1), 8100: (2, 2, 1)}
GATE_RESIDUAL_DEFAULT = (1, 2)
GATE_RESIDUAL_CONFIGS: dict[int, tuple[int, int]] = {}
_SCRATCH_HIDDEN: dict[tuple[int, int, int], torch.Tensor] = {}
_LN_CONSTS: dict[int, torch.Tensor] = {}


def serves_width(channels: int) -> bool:
    """Return whether the kernels serve token rows of ``channels`` channels: they hold a row as five tiles of 512."""
    return channels == CHANNELS


@triton.jit
def _add_rn(a, b):
    """Float32 add that the assembler cannot fuse with a multiply before it."""
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _welford_combine(m_own, s_own, m_other, s_other, consts_ptr, level: tl.constexpr, LEVEL_BASE: tl.constexpr):
    """Combine two (mean, sum of squared deviations) pairs with the constants (nA, nB, cA) of one level, in the order of torch's CUDA LayerNorm."""
    n_a = tl.load(consts_ptr + LEVEL_BASE + 3 * level)
    n_b = tl.load(consts_ptr + LEVEL_BASE + 3 * level + 1)
    c_a = tl.load(consts_ptr + LEVEL_BASE + 3 * level + 2)
    mean = tl.fma(n_a, m_other, n_b * m_own)
    d = m_own - m_other
    t = (d * d) * c_a
    sigma2 = tl.fma(t, n_b, s_other + s_own)
    return mean, sigma2


@triton.jit
def _tree_level_warp(m, s, consts_ptr, level: tl.constexpr, HALF: tl.constexpr, LEVEL_BASE: tl.constexpr):
    """One level of the reduction tree inside each of the four warps: lanes [0, HALF) combine with lanes [HALF, 2 * HALF)."""
    m_own, m_other = tl.split(tl.permute(tl.reshape(m, [4, 2, HALF]), [0, 2, 1]))
    s_own, s_other = tl.split(tl.permute(tl.reshape(s, [4, 2, HALF]), [0, 2, 1]))
    return _welford_combine(m_own, s_own, m_other, s_other, consts_ptr, level, LEVEL_BASE)


@triton.jit
def _row_layernorm_stats_aten(src_row, consts_ptr, eps, C_TOTAL: tl.constexpr):
    """Return (mean, rstd, var) of one stored row in the order of torch's CUDA LayerNorm: 128 lanes of four channels per step, online Welford per lane, a tree over the lanes, var = sigma2 / N, rstd = rsqrt(var + eps)."""
    LEVEL_BASE: tl.constexpr = C_TOTAL // 128
    thrx = tl.arange(0, 128)
    ii = tl.arange(0, 4)
    m = tl.zeros([128], tl.float32)
    s = tl.zeros([128], tl.float32)
    for k in tl.static_range(C_TOTAL // 512):
        x = tl.load(src_row + 512 * k + thrx[:, None] * 4 + ii[None, :]).to(tl.float32)
        even, odd = tl.split(tl.reshape(x, [128, 2, 2]))
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        for jj in tl.static_range(4):
            if jj == 0:
                xv = x0
            elif jj == 1:
                xv = x1
            elif jj == 2:
                xv = x2
            else:
                xv = x3
            rcp = tl.load(consts_ptr + 4 * k + jj)
            d = xv - m
            m = tl.fma(d, rcp, m)
            s = tl.fma(d, xv - m, s)
    m, s = _tree_level_warp(m, s, consts_ptr, 0, 16, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [64]), tl.reshape(s, [64]), consts_ptr, 1, 8, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [32]), tl.reshape(s, [32]), consts_ptr, 2, 4, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [16]), tl.reshape(s, [16]), consts_ptr, 3, 2, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [8]), tl.reshape(s, [8]), consts_ptr, 4, 1, LEVEL_BASE)
    m4 = tl.reshape(m, [4])
    s4 = tl.reshape(s, [4])
    m_own, m_other = tl.split(tl.permute(tl.reshape(m4, [2, 2]), [1, 0]))
    s_own, s_other = tl.split(tl.permute(tl.reshape(s4, [2, 2]), [1, 0]))
    m2, s2 = _welford_combine(m_own, s_own, m_other, s_other, consts_ptr, 5, LEVEL_BASE)
    m_own, m_other = tl.split(tl.reshape(m2, [1, 2]))
    s_own, s_other = tl.split(tl.reshape(s2, [1, 2]))
    m1, s1 = _welford_combine(m_own, s_own, m_other, s_other, consts_ptr, 6, LEVEL_BASE)
    mean = tl.sum(m1, axis=0)
    sigma2 = tl.sum(s1, axis=0)
    var = tl.math.div_rn(sigma2, C_TOTAL * 1.0)
    rstd = tl.math.rsqrt(var + eps)
    return mean, rstd, var


@triton.jit
def _new_partial_tile(p, cr, ex, gt0, gtb, PREFIX: tl.constexpr, HAS_P: tl.constexpr):
    """One [64, 8] tile of the new source row, rounded to bf16 after every operation."""
    if PREFIX == 1:
        g = (gtb.to(tl.float32) + gt0.to(tl.float32)).to(tl.bfloat16)
        d = (g.to(tl.float32) * cr.to(tl.float32)).to(tl.bfloat16)
    elif PREFIX == 2:
        d = (ex.to(tl.float32) + cr.to(tl.float32)).to(tl.bfloat16)
    if HAS_P:
        if PREFIX > 0:
            out = (p.to(tl.float32) + d.to(tl.float32)).to(tl.bfloat16)
        else:
            out = p
    else:
        out = d
    return out


@triton.jit
def _upd5(v0, v1, v2, v3, v4, a0, a1, a2, a3, a4, m, den, q0, q1, q2, q3, q4, C: tl.constexpr, EPS: tl.constexpr):
    """One source of the online softmax over the five row tiles: its logit from the value row and the query, then the rescaled accumulation."""
    f0 = v0.to(tl.float32)
    f1 = v1.to(tl.float32)
    f2 = v2.to(tl.float32)
    f3 = v3.to(tl.float32)
    f4 = v4.to(tl.float32)
    dot = tl.sum(f0 * q0 + f1 * q1 + f2 * q2 + f3 * q3 + f4 * q4)
    ssq = tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4)
    lg = dot * libdevice.rsqrt(ssq * (1.0 / C) + EPS)
    m2 = tl.maximum(m, lg)
    c = libdevice.exp(m - m2)
    e = libdevice.exp(lg - m2)
    return a0 * c + e * f0, a1 * c + e * f1, a2 * c + e * f2, a3 * c + e * f3, a4 * c + e * f4, m2, den * c + e


@triton.jit
def _ln_stats_tiles(h0, h1, h2, h3, h4, consts_ptr, eps):
    """Return (mean, rstd) of one row held as five float32 [64, 8] tiles, in the order of torch's CUDA LayerNorm."""
    LEVEL_BASE: tl.constexpr = 20
    m = tl.zeros([128], tl.float32)
    s = tl.zeros([128], tl.float32)
    for k in tl.static_range(5):
        if k == 0:
            x = tl.reshape(h0, [128, 4])
        elif k == 1:
            x = tl.reshape(h1, [128, 4])
        elif k == 2:
            x = tl.reshape(h2, [128, 4])
        elif k == 3:
            x = tl.reshape(h3, [128, 4])
        else:
            x = tl.reshape(h4, [128, 4])
        even, odd = tl.split(tl.reshape(x, [128, 2, 2]))
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        for jj in tl.static_range(4):
            if jj == 0:
                xv = x0
            elif jj == 1:
                xv = x1
            elif jj == 2:
                xv = x2
            else:
                xv = x3
            rcp = tl.load(consts_ptr + 4 * k + jj)
            d = xv - m
            m = tl.fma(d, rcp, m)
            s = tl.fma(d, xv - m, s)
    m, s = _tree_level_warp(m, s, consts_ptr, 0, 16, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [64]), tl.reshape(s, [64]), consts_ptr, 1, 8, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [32]), tl.reshape(s, [32]), consts_ptr, 2, 4, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [16]), tl.reshape(s, [16]), consts_ptr, 3, 2, LEVEL_BASE)
    m, s = _tree_level_warp(tl.reshape(m, [8]), tl.reshape(s, [8]), consts_ptr, 4, 1, LEVEL_BASE)
    m4 = tl.reshape(m, [4])
    s4 = tl.reshape(s, [4])
    m_own, m_other = tl.split(tl.permute(tl.reshape(m4, [2, 2]), [1, 0]))
    s_own, s_other = tl.split(tl.permute(tl.reshape(s4, [2, 2]), [1, 0]))
    m2, s2 = _welford_combine(m_own, s_own, m_other, s_other, consts_ptr, 5, LEVEL_BASE)
    m_own, m_other = tl.split(tl.reshape(m2, [1, 2]))
    s_own, s_other = tl.split(tl.reshape(s2, [1, 2]))
    m1, s1 = _welford_combine(m_own, s_own, m_other, s_other, consts_ptr, 6, LEVEL_BASE)
    mean = tl.sum(m1, axis=0)
    sigma2 = tl.sum(s1, axis=0)
    var = tl.math.div_rn(sigma2, 2560.0)
    rstd = tl.math.rsqrt(var + eps)
    return mean, rstd


@triton.jit
def _modulate_tile(y, sc0, sh0, sctbl, shtbl, TABLE_BF16: tl.constexpr):
    """Modulate one float32 LayerNorm tile as the plain modules do: scale and shift rounded to bf16 when the table is bf16, a float32 product and sum, one rounding to bf16."""
    sc = sctbl.to(tl.float32) + sc0.to(tl.float32)
    sh = shtbl.to(tl.float32) + sh0.to(tl.float32)
    if TABLE_BF16:
        sc = sc.to(tl.bfloat16).to(tl.float32)
        sh = sh.to(tl.bfloat16).to(tl.float32)
        op = (1.0 + sc).to(tl.bfloat16).to(tl.float32)
    else:
        op = 1.0 + sc
    return _add_rn(y * op, sh).to(tl.bfloat16)


@triton.jit
def sana_prefix_add_attend_adaln_sm120(
    VB,
    KB,
    Q,
    P,
    CORE,
    EXTRA,
    GT0,
    GTBL,
    HID,
    OUTM,
    SH0,
    SC0,
    SHTBL,
    SCTBL,
    CONSTS,
    SRC_STRIDE,
    GT0_S,
    SH0_S,
    SC0_S,
    ln_eps,
    NA: tl.constexpr,
    PREFIX: tl.constexpr,
    HAS_P: tl.constexpr,
    SUFFIX: tl.constexpr,
    TABLE_BF16: tl.constexpr,
    EPS: tl.constexpr,
    SEV: tl.constexpr,
    LEV: tl.constexpr,
    PLEV: tl.constexpr,
    SPRE: tl.constexpr,
    WHID: tl.constexpr,
    WKB: tl.constexpr,
    LN_PATH: tl.constexpr,
    C: tl.constexpr,
):
    """One token row per program: the new source row, ``P`` plus ``(GTBL + GT0) * CORE`` (``PREFIX`` 1) or ``EXTRA + CORE`` (``PREFIX`` 2) in bf16 steps, is stored as row ``NA`` of ``VB`` and mixed in float32 with the ``NA`` rows before it by the softmax of their logits against ``Q`` (into ``HID`` with ``WHID``); with ``SUFFIX`` the LayerNorm of the mix, modulated as the plain modules do, goes to ``OUTM``. Launch knobs: ``SEV`` 2 stores with the streaming hint, ``LEV`` / ``PLEV`` 0 load the source rows / the single-use rows through L2 only, ``SPRE`` 1 loads the modulation rows before the mix, ``LN_PATH`` 1 takes the LayerNorm statistics from a re-read of the stored mix instead of the register tiles."""
    t = tl.program_id(0)
    lane = tl.arange(0, 64)
    vec = tl.arange(0, 8)
    tile = lane[:, None] * 8 + vec[None, :]
    r0 = t * C + tile
    r1 = r0 + 512
    r2 = r0 + 1024
    r3 = r0 + 1536
    r4 = r0 + 2048
    st_cm: tl.constexpr = ".cs" if SEV == 2 else ""
    ld_cm: tl.constexpr = ".cg" if LEV == 0 else ""
    pf_cm: tl.constexpr = ".cg" if PLEV == 0 else ""

    if PREFIX == 1:
        g0 = t * GT0_S + tile
        gt00 = tl.load(GT0 + g0, cache_modifier=pf_cm)
        gt01 = tl.load(GT0 + g0 + 512, cache_modifier=pf_cm)
        gt02 = tl.load(GT0 + g0 + 1024, cache_modifier=pf_cm)
        gt03 = tl.load(GT0 + g0 + 1536, cache_modifier=pf_cm)
        gt04 = tl.load(GT0 + g0 + 2048, cache_modifier=pf_cm)
        gtb0 = tl.load(GTBL + tile)
        gtb1 = tl.load(GTBL + tile + 512)
        gtb2 = tl.load(GTBL + tile + 1024)
        gtb3 = tl.load(GTBL + tile + 1536)
        gtb4 = tl.load(GTBL + tile + 2048)
    else:
        gt00 = 0
        gt01 = 0
        gt02 = 0
        gt03 = 0
        gt04 = 0
        gtb0 = 0
        gtb1 = 0
        gtb2 = 0
        gtb3 = 0
        gtb4 = 0
    if PREFIX > 0:
        cr0 = tl.load(CORE + r0, cache_modifier=pf_cm)
        cr1 = tl.load(CORE + r1, cache_modifier=pf_cm)
        cr2 = tl.load(CORE + r2, cache_modifier=pf_cm)
        cr3 = tl.load(CORE + r3, cache_modifier=pf_cm)
        cr4 = tl.load(CORE + r4, cache_modifier=pf_cm)
    else:
        cr0 = 0
        cr1 = 0
        cr2 = 0
        cr3 = 0
        cr4 = 0
    if PREFIX == 2:
        ex0 = tl.load(EXTRA + r0, cache_modifier=pf_cm)
        ex1 = tl.load(EXTRA + r1, cache_modifier=pf_cm)
        ex2 = tl.load(EXTRA + r2, cache_modifier=pf_cm)
        ex3 = tl.load(EXTRA + r3, cache_modifier=pf_cm)
        ex4 = tl.load(EXTRA + r4, cache_modifier=pf_cm)
    else:
        ex0 = 0
        ex1 = 0
        ex2 = 0
        ex3 = 0
        ex4 = 0
    if HAS_P:
        p0 = tl.load(P + r0, cache_modifier=pf_cm)
        p1 = tl.load(P + r1, cache_modifier=pf_cm)
        p2 = tl.load(P + r2, cache_modifier=pf_cm)
        p3 = tl.load(P + r3, cache_modifier=pf_cm)
        p4 = tl.load(P + r4, cache_modifier=pf_cm)
    else:
        p0 = 0
        p1 = 0
        p2 = 0
        p3 = 0
        p4 = 0
    q0 = tl.load(Q + tile).to(tl.float32)
    q1 = tl.load(Q + tile + 512).to(tl.float32)
    q2 = tl.load(Q + tile + 1024).to(tl.float32)
    q3 = tl.load(Q + tile + 1536).to(tl.float32)
    q4 = tl.load(Q + tile + 2048).to(tl.float32)
    v00 = tl.load(VB + r0, cache_modifier=ld_cm)
    v01 = tl.load(VB + r1, cache_modifier=ld_cm)
    v02 = tl.load(VB + r2, cache_modifier=ld_cm)
    v03 = tl.load(VB + r3, cache_modifier=ld_cm)
    v04 = tl.load(VB + r4, cache_modifier=ld_cm)
    if NA >= 2:
        v10 = tl.load(VB + SRC_STRIDE + r0, cache_modifier=ld_cm)
        v11 = tl.load(VB + SRC_STRIDE + r1, cache_modifier=ld_cm)
        v12 = tl.load(VB + SRC_STRIDE + r2, cache_modifier=ld_cm)
        v13 = tl.load(VB + SRC_STRIDE + r3, cache_modifier=ld_cm)
        v14 = tl.load(VB + SRC_STRIDE + r4, cache_modifier=ld_cm)
    if NA >= 3:
        v20 = tl.load(VB + 2 * SRC_STRIDE + r0, cache_modifier=ld_cm)
        v21 = tl.load(VB + 2 * SRC_STRIDE + r1, cache_modifier=ld_cm)
        v22 = tl.load(VB + 2 * SRC_STRIDE + r2, cache_modifier=ld_cm)
        v23 = tl.load(VB + 2 * SRC_STRIDE + r3, cache_modifier=ld_cm)
        v24 = tl.load(VB + 2 * SRC_STRIDE + r4, cache_modifier=ld_cm)
    if NA >= 4:
        v30 = tl.load(VB + 3 * SRC_STRIDE + r0, cache_modifier=ld_cm)
        v31 = tl.load(VB + 3 * SRC_STRIDE + r1, cache_modifier=ld_cm)
        v32 = tl.load(VB + 3 * SRC_STRIDE + r2, cache_modifier=ld_cm)
        v33 = tl.load(VB + 3 * SRC_STRIDE + r3, cache_modifier=ld_cm)
        v34 = tl.load(VB + 3 * SRC_STRIDE + r4, cache_modifier=ld_cm)
    if SUFFIX == 1 and SPRE == 1:
        s0 = t * SC0_S + tile
        h0 = t * SH0_S + tile
        sc00 = tl.load(SC0 + s0, cache_modifier=pf_cm)
        sc01 = tl.load(SC0 + s0 + 512, cache_modifier=pf_cm)
        sc02 = tl.load(SC0 + s0 + 1024, cache_modifier=pf_cm)
        sc03 = tl.load(SC0 + s0 + 1536, cache_modifier=pf_cm)
        sc04 = tl.load(SC0 + s0 + 2048, cache_modifier=pf_cm)
        sh00 = tl.load(SH0 + h0, cache_modifier=pf_cm)
        sh01 = tl.load(SH0 + h0 + 512, cache_modifier=pf_cm)
        sh02 = tl.load(SH0 + h0 + 1024, cache_modifier=pf_cm)
        sh03 = tl.load(SH0 + h0 + 1536, cache_modifier=pf_cm)
        sh04 = tl.load(SH0 + h0 + 2048, cache_modifier=pf_cm)

    np0 = _new_partial_tile(p0, cr0, ex0, gt00, gtb0, PREFIX, HAS_P)
    np1 = _new_partial_tile(p1, cr1, ex1, gt01, gtb1, PREFIX, HAS_P)
    np2 = _new_partial_tile(p2, cr2, ex2, gt02, gtb2, PREFIX, HAS_P)
    np3 = _new_partial_tile(p3, cr3, ex3, gt03, gtb3, PREFIX, HAS_P)
    np4 = _new_partial_tile(p4, cr4, ex4, gt04, gtb4, PREFIX, HAS_P)
    nf0 = np0.to(tl.float32)
    nf1 = np1.to(tl.float32)
    nf2 = np2.to(tl.float32)
    nf3 = np3.to(tl.float32)
    nf4 = np4.to(tl.float32)

    ss = tl.sum(nf0 * nf0 + nf1 * nf1 + nf2 * nf2 + nf3 * nf3 + nf4 * nf4)
    pq = tl.sum(nf0 * q0 + nf1 * q1 + nf2 * q2 + nf3 * q3 + nf4 * q4)
    scale = libdevice.rsqrt(ss * (1.0 / C) + EPS)
    lg_p = pq * scale
    tl.store(VB + NA * SRC_STRIDE + r0, np0, cache_modifier=st_cm)
    tl.store(VB + NA * SRC_STRIDE + r1, np1, cache_modifier=st_cm)
    tl.store(VB + NA * SRC_STRIDE + r2, np2, cache_modifier=st_cm)
    tl.store(VB + NA * SRC_STRIDE + r3, np3, cache_modifier=st_cm)
    tl.store(VB + NA * SRC_STRIDE + r4, np4, cache_modifier=st_cm)
    if WKB:
        tl.store(KB + NA * SRC_STRIDE + r0, (nf0 * scale).to(tl.bfloat16), cache_modifier=st_cm)
        tl.store(KB + NA * SRC_STRIDE + r1, (nf1 * scale).to(tl.bfloat16), cache_modifier=st_cm)
        tl.store(KB + NA * SRC_STRIDE + r2, (nf2 * scale).to(tl.bfloat16), cache_modifier=st_cm)
        tl.store(KB + NA * SRC_STRIDE + r3, (nf3 * scale).to(tl.bfloat16), cache_modifier=st_cm)
        tl.store(KB + NA * SRC_STRIDE + r4, (nf4 * scale).to(tl.bfloat16), cache_modifier=st_cm)

    a0 = nf0
    a1 = nf1
    a2 = nf2
    a3 = nf3
    a4 = nf4
    m = lg_p
    den = 1.0
    a0, a1, a2, a3, a4, m, den = _upd5(v00, v01, v02, v03, v04, a0, a1, a2, a3, a4, m, den, q0, q1, q2, q3, q4, C, EPS)
    if NA >= 2:
        a0, a1, a2, a3, a4, m, den = _upd5(
            v10, v11, v12, v13, v14, a0, a1, a2, a3, a4, m, den, q0, q1, q2, q3, q4, C, EPS
        )
    if NA >= 3:
        a0, a1, a2, a3, a4, m, den = _upd5(
            v20, v21, v22, v23, v24, a0, a1, a2, a3, a4, m, den, q0, q1, q2, q3, q4, C, EPS
        )
    if NA >= 4:
        a0, a1, a2, a3, a4, m, den = _upd5(
            v30, v31, v32, v33, v34, a0, a1, a2, a3, a4, m, den, q0, q1, q2, q3, q4, C, EPS
        )
    r = tl.math.div_rn(1.0, den)
    h0b = (a0 * r).to(tl.bfloat16)
    h1b = (a1 * r).to(tl.bfloat16)
    h2b = (a2 * r).to(tl.bfloat16)
    h3b = (a3 * r).to(tl.bfloat16)
    h4b = (a4 * r).to(tl.bfloat16)
    if WHID == 1:
        if LN_PATH == 1 and SUFFIX == 1:
            tl.store(HID + r0, h0b)
            tl.store(HID + r1, h1b)
            tl.store(HID + r2, h2b)
            tl.store(HID + r3, h3b)
            tl.store(HID + r4, h4b)
        else:
            tl.store(HID + r0, h0b, cache_modifier=st_cm)
            tl.store(HID + r1, h1b, cache_modifier=st_cm)
            tl.store(HID + r2, h2b, cache_modifier=st_cm)
            tl.store(HID + r3, h3b, cache_modifier=st_cm)
            tl.store(HID + r4, h4b, cache_modifier=st_cm)

    if SUFFIX == 1:
        hf0 = h0b.to(tl.float32)
        hf1 = h1b.to(tl.float32)
        hf2 = h2b.to(tl.float32)
        hf3 = h3b.to(tl.float32)
        hf4 = h4b.to(tl.float32)
        if LN_PATH == 1:
            tl.debug_barrier()
            mean, rstd, _var = _row_layernorm_stats_aten(HID + t * C, CONSTS, ln_eps, C)
        else:
            mean, rstd = _ln_stats_tiles(hf0, hf1, hf2, hf3, hf4, CONSTS, ln_eps)
        if SPRE == 0:
            s0 = t * SC0_S + tile
            h0 = t * SH0_S + tile
            sc00 = tl.load(SC0 + s0, cache_modifier=pf_cm)
            sc01 = tl.load(SC0 + s0 + 512, cache_modifier=pf_cm)
            sc02 = tl.load(SC0 + s0 + 1024, cache_modifier=pf_cm)
            sc03 = tl.load(SC0 + s0 + 1536, cache_modifier=pf_cm)
            sc04 = tl.load(SC0 + s0 + 2048, cache_modifier=pf_cm)
            sh00 = tl.load(SH0 + h0, cache_modifier=pf_cm)
            sh01 = tl.load(SH0 + h0 + 512, cache_modifier=pf_cm)
            sh02 = tl.load(SH0 + h0 + 1024, cache_modifier=pf_cm)
            sh03 = tl.load(SH0 + h0 + 1536, cache_modifier=pf_cm)
            sh04 = tl.load(SH0 + h0 + 2048, cache_modifier=pf_cm)
        tl.store(
            OUTM + r0,
            _modulate_tile(rstd * (hf0 - mean), sc00, sh00, tl.load(SCTBL + tile), tl.load(SHTBL + tile), TABLE_BF16),
            cache_modifier=st_cm,
        )
        tl.store(
            OUTM + r1,
            _modulate_tile(
                rstd * (hf1 - mean), sc01, sh01, tl.load(SCTBL + tile + 512), tl.load(SHTBL + tile + 512), TABLE_BF16
            ),
            cache_modifier=st_cm,
        )
        tl.store(
            OUTM + r2,
            _modulate_tile(
                rstd * (hf2 - mean), sc02, sh02, tl.load(SCTBL + tile + 1024), tl.load(SHTBL + tile + 1024), TABLE_BF16
            ),
            cache_modifier=st_cm,
        )
        tl.store(
            OUTM + r3,
            _modulate_tile(
                rstd * (hf3 - mean), sc03, sh03, tl.load(SCTBL + tile + 1536), tl.load(SHTBL + tile + 1536), TABLE_BF16
            ),
            cache_modifier=st_cm,
        )
        tl.store(
            OUTM + r4,
            _modulate_tile(
                rstd * (hf4 - mean), sc04, sh04, tl.load(SCTBL + tile + 2048), tl.load(SHTBL + tile + 2048), TABLE_BF16
            ),
            cache_modifier=st_cm,
        )


@triton.jit
def sana_ln_modulate_sm120(
    SRC,
    OUTM,
    SH0,
    SC0,
    SHTBL,
    SCTBL,
    CONSTS,
    SH0_S,
    SC0_S,
    ln_eps,
    TABLE_BF16: tl.constexpr,
    SEV: tl.constexpr,
    PLEV: tl.constexpr,
    C: tl.constexpr,
):
    """One token row per program: the LayerNorm of the bf16 row ``SRC`` in the order of torch's CUDA LayerNorm, modulated as the plain modules do. Launch knobs: ``SEV`` 2 stores with the streaming hint, ``PLEV`` 0 loads through L2 only."""
    t = tl.program_id(0)
    lane = tl.arange(0, 64)
    vec = tl.arange(0, 8)
    tile = lane[:, None] * 8 + vec[None, :]
    r0 = t * C + tile
    st_cm: tl.constexpr = ".cs" if SEV == 2 else ""
    pf_cm: tl.constexpr = ".cg" if PLEV == 0 else ""
    x0 = tl.load(SRC + r0, cache_modifier=pf_cm).to(tl.float32)
    x1 = tl.load(SRC + r0 + 512, cache_modifier=pf_cm).to(tl.float32)
    x2 = tl.load(SRC + r0 + 1024, cache_modifier=pf_cm).to(tl.float32)
    x3 = tl.load(SRC + r0 + 1536, cache_modifier=pf_cm).to(tl.float32)
    x4 = tl.load(SRC + r0 + 2048, cache_modifier=pf_cm).to(tl.float32)
    s0 = t * SC0_S + tile
    h0 = t * SH0_S + tile
    sc00 = tl.load(SC0 + s0, cache_modifier=pf_cm)
    sc01 = tl.load(SC0 + s0 + 512, cache_modifier=pf_cm)
    sc02 = tl.load(SC0 + s0 + 1024, cache_modifier=pf_cm)
    sc03 = tl.load(SC0 + s0 + 1536, cache_modifier=pf_cm)
    sc04 = tl.load(SC0 + s0 + 2048, cache_modifier=pf_cm)
    sh00 = tl.load(SH0 + h0, cache_modifier=pf_cm)
    sh01 = tl.load(SH0 + h0 + 512, cache_modifier=pf_cm)
    sh02 = tl.load(SH0 + h0 + 1024, cache_modifier=pf_cm)
    sh03 = tl.load(SH0 + h0 + 1536, cache_modifier=pf_cm)
    sh04 = tl.load(SH0 + h0 + 2048, cache_modifier=pf_cm)
    mean, rstd = _ln_stats_tiles(x0, x1, x2, x3, x4, CONSTS, ln_eps)
    tl.store(
        OUTM + r0,
        _modulate_tile(rstd * (x0 - mean), sc00, sh00, tl.load(SCTBL + tile), tl.load(SHTBL + tile), TABLE_BF16),
        cache_modifier=st_cm,
    )
    tl.store(
        OUTM + r0 + 512,
        _modulate_tile(
            rstd * (x1 - mean), sc01, sh01, tl.load(SCTBL + tile + 512), tl.load(SHTBL + tile + 512), TABLE_BF16
        ),
        cache_modifier=st_cm,
    )
    tl.store(
        OUTM + r0 + 1024,
        _modulate_tile(
            rstd * (x2 - mean), sc02, sh02, tl.load(SCTBL + tile + 1024), tl.load(SHTBL + tile + 1024), TABLE_BF16
        ),
        cache_modifier=st_cm,
    )
    tl.store(
        OUTM + r0 + 1536,
        _modulate_tile(
            rstd * (x3 - mean), sc03, sh03, tl.load(SCTBL + tile + 1536), tl.load(SHTBL + tile + 1536), TABLE_BF16
        ),
        cache_modifier=st_cm,
    )
    tl.store(
        OUTM + r0 + 2048,
        _modulate_tile(
            rstd * (x4 - mean), sc04, sh04, tl.load(SCTBL + tile + 2048), tl.load(SHTBL + tile + 2048), TABLE_BF16
        ),
        cache_modifier=st_cm,
    )


@triton.jit
def sana_attn_gate_residual_sm120(
    XSA,
    H,
    GT0,
    GTAB,
    DELTA,
    HAFTER,
    g_stride,
    C: tl.constexpr,
    MODE: tl.constexpr,
):
    """One token row per program: ``DELTA = (GTAB + GT0) * XSA`` and ``HAFTER = H + DELTA``, rounded to bf16 after every operation; ``MODE`` 0 holds the row as five [64, 8] tiles, 1 as 2048 + 512 channels."""
    t = tl.program_id(0)
    if MODE == 1:
        ca = tl.arange(0, 2048)
        cb = 2048 + tl.arange(0, C - 2048)
        oa = t * C + ca
        ob = t * C + cb
        gta = tl.load(GTAB + ca)
        gtb = tl.load(GTAB + cb)
        g0a = tl.load(GT0 + t * g_stride + ca)
        g0b = tl.load(GT0 + t * g_stride + cb)
        xa = tl.load(XSA + oa)
        xb = tl.load(XSA + ob)
        ha = tl.load(H + oa)
        hb = tl.load(H + ob)
        ga = (gta.to(tl.float32) + g0a.to(tl.float32)).to(tl.bfloat16)
        gb = (gtb.to(tl.float32) + g0b.to(tl.float32)).to(tl.bfloat16)
        da = (ga.to(tl.float32) * xa.to(tl.float32)).to(tl.bfloat16)
        db = (gb.to(tl.float32) * xb.to(tl.float32)).to(tl.bfloat16)
        tl.store(DELTA + oa, da)
        tl.store(DELTA + ob, db)
        tl.store(HAFTER + oa, (ha.to(tl.float32) + da.to(tl.float32)).to(tl.bfloat16))
        tl.store(HAFTER + ob, (hb.to(tl.float32) + db.to(tl.float32)).to(tl.bfloat16))
    else:
        lane = tl.arange(0, 64)
        vec = tl.arange(0, 8)
        tile = lane[:, None] * 8 + vec[None, :]
        r0 = t * C + tile
        g0 = t * g_stride + tile
        gt0 = tl.load(GTAB + tile)
        gt1 = tl.load(GTAB + tile + 512)
        gt2 = tl.load(GTAB + tile + 1024)
        gt3 = tl.load(GTAB + tile + 1536)
        gt4 = tl.load(GTAB + tile + 2048)
        t0 = tl.load(GT0 + g0)
        t1 = tl.load(GT0 + g0 + 512)
        t2 = tl.load(GT0 + g0 + 1024)
        t3 = tl.load(GT0 + g0 + 1536)
        t4 = tl.load(GT0 + g0 + 2048)
        x0 = tl.load(XSA + r0)
        x1 = tl.load(XSA + r0 + 512)
        x2 = tl.load(XSA + r0 + 1024)
        x3 = tl.load(XSA + r0 + 1536)
        x4 = tl.load(XSA + r0 + 2048)
        h0 = tl.load(H + r0)
        h1 = tl.load(H + r0 + 512)
        h2 = tl.load(H + r0 + 1024)
        h3 = tl.load(H + r0 + 1536)
        h4 = tl.load(H + r0 + 2048)
        d0 = ((gt0.to(tl.float32) + t0.to(tl.float32)).to(tl.bfloat16).to(tl.float32) * x0.to(tl.float32)).to(
            tl.bfloat16
        )
        d1 = ((gt1.to(tl.float32) + t1.to(tl.float32)).to(tl.bfloat16).to(tl.float32) * x1.to(tl.float32)).to(
            tl.bfloat16
        )
        d2 = ((gt2.to(tl.float32) + t2.to(tl.float32)).to(tl.bfloat16).to(tl.float32) * x2.to(tl.float32)).to(
            tl.bfloat16
        )
        d3 = ((gt3.to(tl.float32) + t3.to(tl.float32)).to(tl.bfloat16).to(tl.float32) * x3.to(tl.float32)).to(
            tl.bfloat16
        )
        d4 = ((gt4.to(tl.float32) + t4.to(tl.float32)).to(tl.bfloat16).to(tl.float32) * x4.to(tl.float32)).to(
            tl.bfloat16
        )
        tl.store(DELTA + r0, d0)
        tl.store(DELTA + r0 + 512, d1)
        tl.store(DELTA + r0 + 1024, d2)
        tl.store(DELTA + r0 + 1536, d3)
        tl.store(DELTA + r0 + 2048, d4)
        tl.store(HAFTER + r0, (h0.to(tl.float32) + d0.to(tl.float32)).to(tl.bfloat16))
        tl.store(HAFTER + r0 + 512, (h1.to(tl.float32) + d1.to(tl.float32)).to(tl.bfloat16))
        tl.store(HAFTER + r0 + 1024, (h2.to(tl.float32) + d2.to(tl.float32)).to(tl.bfloat16))
        tl.store(HAFTER + r0 + 1536, (h3.to(tl.float32) + d3.to(tl.float32)).to(tl.bfloat16))
        tl.store(HAFTER + r0 + 2048, (h4.to(tl.float32) + d4.to(tl.float32)).to(tl.bfloat16))


def ln_consts(device) -> torch.Tensor:
    """Return the float32 constants of the in-kernel LayerNorm on ``device``: the reciprocals 1 / (t + 1) of the 20 running counts of a lane, then (nA, nB, cA) of each of the seven combine levels."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    consts = _LN_CONSTS.get(key)
    if consts is None:
        per_lane = CHANNELS // 128
        rcp = [np.float32(1) / np.float32(t + 1) for t in range(per_lane)]
        levels = []
        c = np.float32(per_lane)
        for _level in range(7):
            coef = np.float32(1) / (c + c)
            levels += [c * coef, c * coef, c]
            c = c + c
        consts = torch.from_numpy(np.array(rcp + levels, dtype=np.float32)).to(device)
        _LN_CONSTS[key] = consts
    return consts


def gate_residual(x_sa, hidden, gate_t0, gate_table, configs=GATE_RESIDUAL_CONFIGS):
    """Return ``delta = (gate_table + gate_t0) * x_sa`` and ``hidden + delta``, both bf16 [1, N, C], from one launch; ``configs`` holds the launch's (MODE, warps) per token count."""
    tokens, channels = x_sa.shape[1], x_sa.shape[2]
    delta = torch.empty(x_sa.shape, dtype=torch.bfloat16, device=x_sa.device)
    h_after = torch.empty_like(delta)
    mode, warps = configs.get(tokens, GATE_RESIDUAL_DEFAULT)
    sana_attn_gate_residual_sm120[(tokens,)](
        x_sa,
        hidden,
        gate_t0,
        gate_table,
        delta,
        h_after,
        gate_t0.stride(0),
        C=channels,
        MODE=mode,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return delta, h_after


def sublayer_boundary(
    value_buffer,
    query,
    n_active,
    partial,
    core_out,
    extra_delta=None,
    gate_t0=None,
    gate_table=None,
    shift_t0=None,
    scale_t0=None,
    shift_table=None,
    scale_table=None,
    write_hidden=True,
    configs=BOUNDARY_CONFIGS,
):
    """Add the closed sublayer's update, ``(gate_table + gate_t0) * core_out`` or ``extra_delta + core_out``, to the running block ``partial`` (None at a block's first update), store the sum as ``value_buffer[n_active]`` and mix it with the ``n_active`` rows before it (at most four) per token by the softmax of their logits against ``query``. Returns the mix (None without ``write_hidden``) and, when a shift and scale are given, the LayerNorm of the mix modulated by them; ``configs`` holds the launch knobs per case."""
    N, C = value_buffer.shape[2], value_buffer.shape[3]
    prefix_mode = 1 if extra_delta is None else 2
    suffix_mode = 0 if shift_t0 is None else 1
    dev = value_buffer.device
    warps, sev, lev, plev, spre, ln_path = configs.get((N, n_active, prefix_mode, suffix_mode), BOUNDARY_DEFAULT)
    store_hidden = write_hidden or (ln_path == 1 and suffix_mode == 1)
    if write_hidden:
        hidden = torch.empty((1, N, C), dtype=torch.bfloat16, device=dev)
    elif store_hidden:
        key = (dev.index if dev.index is not None else torch.cuda.current_device(), N, C)
        hidden = _SCRATCH_HIDDEN.get(key)
        if hidden is None:
            hidden = torch.empty((1, N, C), dtype=torch.bfloat16, device=dev)
            _SCRATCH_HIDDEN[key] = hidden
    else:
        hidden = query
    out_mod = torch.empty((1, N, C), dtype=torch.bfloat16, device=dev) if suffix_mode == 1 else None
    dummy = query
    sana_prefix_add_attend_adaln_sm120[(N,)](
        value_buffer,
        value_buffer,
        query,
        partial if partial is not None else dummy,
        core_out,
        extra_delta if prefix_mode == 2 else dummy,
        gate_t0 if prefix_mode == 1 else dummy,
        gate_table if prefix_mode == 1 else dummy,
        hidden,
        out_mod if suffix_mode == 1 else dummy,
        shift_t0 if suffix_mode == 1 else dummy,
        scale_t0 if suffix_mode == 1 else dummy,
        shift_table if suffix_mode == 1 else dummy,
        scale_table if suffix_mode == 1 else dummy,
        ln_consts(dev) if suffix_mode == 1 else dummy,
        value_buffer.stride(0),
        gate_t0.stride(0) if prefix_mode == 1 else 0,
        shift_t0.stride(0) if suffix_mode == 1 else 0,
        scale_t0.stride(0) if suffix_mode == 1 else 0,
        float(_LN_EPS),
        NA=n_active,
        PREFIX=prefix_mode,
        HAS_P=partial is not None,
        SUFFIX=suffix_mode,
        TABLE_BF16=bool(suffix_mode == 1 and shift_table.dtype == torch.bfloat16),
        EPS=_RMS_EPS,
        SEV=sev,
        LEV=lev,
        PLEV=plev,
        SPRE=spre if suffix_mode == 1 else 0,
        WHID=1 if store_hidden else 0,
        WKB=0,
        LN_PATH=ln_path if suffix_mode == 1 else 0,
        C=C,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return (hidden if write_hidden else None), out_mod


def layer_norm_modulate(src, shift_t0, scale_t0, shift_table, scale_table, configs=LAYER_NORM_CONFIGS):
    """Return the LayerNorm of the bf16 rows ``src`` [1, N, C], modulated by ``scale_table + scale_t0`` and ``shift_table + shift_t0`` as the plain modules do, bf16 [1, N, C], from one launch; ``configs`` holds the launch's (warps, SEV, PLEV) per token count."""
    N, C = src.shape[1], src.shape[2]
    out_mod = torch.empty((1, N, C), dtype=torch.bfloat16, device=src.device)
    warps, sev, plev = configs.get(N, LAYER_NORM_DEFAULT)
    sana_ln_modulate_sm120[(N,)](
        src,
        out_mod,
        shift_t0,
        scale_t0,
        shift_table,
        scale_table,
        ln_consts(src.device),
        shift_t0.stride(0),
        scale_t0.stride(0),
        float(_LN_EPS),
        TABLE_BF16=bool(shift_table.dtype == torch.bfloat16),
        SEV=sev,
        PLEV=plev,
        C=C,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out_mod
