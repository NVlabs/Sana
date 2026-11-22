# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

_RMS_EPS = 1e-6
_LN_EPS = 1e-6
BOUNDARY_DEFAULT = (2, 2, 0, 0, 0, 1)
BOUNDARY_CONFIGS: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int]] = {
    (2717, 1, 2, 1): (2, 2, 0, 0, 0, 1),
    (4533, 1, 2, 1): (4, 2, 0, 0, 0, 1),
    (8165, 1, 2, 1): (4, 2, 1, 0, 1, 1),
    (2717, 2, 1, 1): (4, 2, 1, 0, 1, 1),
    (4533, 2, 1, 1): (2, 2, 1, 0, 1, 1),
    (8165, 2, 1, 1): (4, 2, 1, 0, 1, 1),
    (2717, 4, 2, 1): (2, 2, 1, 0, 0, 1),
    (4533, 4, 2, 1): (2, 2, 1, 0, 0, 1),
    (8165, 4, 2, 1): (2, 2, 1, 0, 0, 1),
    (2717, 4, 1, 0): (2, 2, 1, 0, 1, 1),
    (4533, 4, 1, 0): (2, 2, 0, 0, 1, 1),
    (8165, 4, 1, 0): (2, 2, 1, 0, 1, 1),
}
LAYER_NORM_DEFAULT = (2, 2, 0, 1)
LAYER_NORM_CONFIGS: dict[int, tuple[int, int, int, int]] = {
    2717: (8, 0, 1, 1),
    4533: (8, 2, 1, 1),
    8165: (4, 2, 1, 1),
}


def serves_width(channels: int) -> bool:
    """Return whether the kernels serve token rows of ``channels`` channels: 2048 plus a power of two."""
    tail = channels - 2048
    return tail > 0 and not tail & (tail - 1)


@triton.jit
def sana_attn_gate_residual(
    XSA,
    H,
    GT0,
    GTAB,
    DELTA,
    HAFTER,
    g_stride,
    C: tl.constexpr,
    CA: tl.constexpr,
    CB: tl.constexpr,
):
    """One token row per program: ``DELTA = (GTAB + GT0) * XSA`` and ``HAFTER = H + DELTA``, rounded to bf16 after every operation."""
    t = tl.program_id(0)
    ca = tl.arange(0, CA)
    cb = CA + tl.arange(0, CB)
    oa = t * C + ca
    ob = t * C + cb

    ga = (tl.load(GTAB + ca).to(tl.float32) + tl.load(GT0 + t * g_stride + ca).to(tl.float32)).to(tl.bfloat16)
    gb = (tl.load(GTAB + cb).to(tl.float32) + tl.load(GT0 + t * g_stride + cb).to(tl.float32)).to(tl.bfloat16)

    xa = tl.load(XSA + oa)
    xb = tl.load(XSA + ob)
    da = (ga.to(tl.float32) * xa.to(tl.float32)).to(tl.bfloat16)
    db = (gb.to(tl.float32) * xb.to(tl.float32)).to(tl.bfloat16)
    tl.store(DELTA + oa, da)
    tl.store(DELTA + ob, db)

    ha = tl.load(H + oa)
    hb = tl.load(H + ob)
    tl.store(HAFTER + oa, (ha.to(tl.float32) + da.to(tl.float32)).to(tl.bfloat16))
    tl.store(HAFTER + ob, (hb.to(tl.float32) + db.to(tl.float32)).to(tl.bfloat16))


@triton.jit
def _upd(va, vbb, acc_a, acc_b, m, den, qa, qb, C: tl.constexpr, EPS: tl.constexpr):
    """One source of the online softmax: its logit from the value row and the query, then the rescaled accumulation."""
    vaf = va.to(tl.float32)
    vbf = vbb.to(tl.float32)
    dot = tl.sum(vaf * qa) + tl.sum(vbf * qb)
    ssq = tl.sum(vaf * vaf) + tl.sum(vbf * vbf)
    lg = dot * tl.math.rsqrt(ssq * (1.0 / C) + EPS)
    m2 = tl.maximum(m, lg)
    c = tl.exp(m - m2)
    e = tl.exp(lg - m2)
    return acc_a * c + e * vaf, acc_b * c + e * vbf, m2, den * c + e


@triton.jit
def sana_prefix_add_attend_adaln_fused(
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
    SRC_STRIDE,
    GT0_S,
    SH0_S,
    SC0_S,
    NA: tl.constexpr,
    PREFIX: tl.constexpr,
    HAS_P: tl.constexpr,
    SUFFIX: tl.constexpr,
    C: tl.constexpr,
    CA: tl.constexpr,
    CB: tl.constexpr,
    EPS: tl.constexpr,
    LNEPS: tl.constexpr,
    SEV: tl.constexpr,
    LEV: tl.constexpr,
    PLEV: tl.constexpr,
    SPRE: tl.constexpr,
    HOIST: tl.constexpr,
    WHID: tl.constexpr,
    WKB: tl.constexpr,
):
    """One token row per program: the new source row, ``P`` plus ``(GTBL + GT0) * CORE`` (``PREFIX`` 1) or ``EXTRA + CORE`` (``PREFIX`` 2) in bf16 steps, is stored as row ``NA`` of ``VB`` and mixed in float32 with the ``NA`` rows before it by the softmax of their logits against ``Q`` (into ``HID`` with ``WHID``); with ``SUFFIX`` the LayerNorm of the mix times ``1 + SCTBL + SC0`` plus ``SHTBL + SH0``, in bf16 steps, goes to ``OUTM``. Launch knobs: ``SEV`` 2 stores with the streaming hint, ``LEV`` / ``PLEV`` 0 load the source rows / the single-use rows through L2 only, ``SPRE`` 1 loads the modulation rows before the mix, ``HOIST`` 1 loads every row before the first reduction."""
    t = tl.program_id(0)
    ca = tl.arange(0, CA)
    cb = CA + tl.arange(0, CB)
    oa = t * C + ca
    ob = t * C + cb

    st_cm: tl.constexpr = ".cs" if SEV == 2 else ""
    ld_cm: tl.constexpr = ".cg" if LEV == 0 else ""
    pf_cm: tl.constexpr = ".cg" if PLEV == 0 else ""

    if PREFIX == 1:
        gt0a = tl.load(GT0 + t * GT0_S + ca, cache_modifier=pf_cm)
        gt0b = tl.load(GT0 + t * GT0_S + cb, cache_modifier=pf_cm)
        gtba = tl.load(GTBL + ca)
        gtbb = tl.load(GTBL + cb)
        cra = tl.load(CORE + oa, cache_modifier=pf_cm)
        crb = tl.load(CORE + ob, cache_modifier=pf_cm)
    elif PREFIX == 2:
        exa = tl.load(EXTRA + oa, cache_modifier=pf_cm)
        exb = tl.load(EXTRA + ob, cache_modifier=pf_cm)
        cra = tl.load(CORE + oa, cache_modifier=pf_cm)
        crb = tl.load(CORE + ob, cache_modifier=pf_cm)
    if HAS_P:
        pa = tl.load(P + oa, cache_modifier=pf_cm)
        pb = tl.load(P + ob, cache_modifier=pf_cm)
    q_a = tl.load(Q + ca)
    q_b = tl.load(Q + cb)
    if HOIST == 1:
        v0a = tl.load(VB + 0 * SRC_STRIDE + oa, cache_modifier=ld_cm)
        v0b = tl.load(VB + 0 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        if NA >= 2:
            v1a = tl.load(VB + 1 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v1b = tl.load(VB + 1 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        if NA >= 3:
            v2a = tl.load(VB + 2 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v2b = tl.load(VB + 2 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        if NA >= 4:
            v3a = tl.load(VB + 3 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v3b = tl.load(VB + 3 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        if NA >= 5:
            v4a = tl.load(VB + 4 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v4b = tl.load(VB + 4 * SRC_STRIDE + ob, cache_modifier=ld_cm)
    if SUFFIX == 1 and SPRE == 1:
        sc0a = tl.load(SC0 + t * SC0_S + ca, cache_modifier=pf_cm)
        sc0b = tl.load(SC0 + t * SC0_S + cb, cache_modifier=pf_cm)
        sh0a = tl.load(SH0 + t * SH0_S + ca, cache_modifier=pf_cm)
        sh0b = tl.load(SH0 + t * SH0_S + cb, cache_modifier=pf_cm)

    qa = q_a.to(tl.float32)
    qb = q_b.to(tl.float32)

    if PREFIX == 1:
        ga = (gtba.to(tl.float32) + gt0a.to(tl.float32)).to(tl.bfloat16)
        gb = (gtbb.to(tl.float32) + gt0b.to(tl.float32)).to(tl.bfloat16)
        da = (ga.to(tl.float32) * cra.to(tl.float32)).to(tl.bfloat16)
        db = (gb.to(tl.float32) * crb.to(tl.float32)).to(tl.bfloat16)
    elif PREFIX == 2:
        da = (exa.to(tl.float32) + cra.to(tl.float32)).to(tl.bfloat16)
        db = (exb.to(tl.float32) + crb.to(tl.float32)).to(tl.bfloat16)

    if HAS_P:
        if PREFIX > 0:
            npa = (pa.to(tl.float32) + da.to(tl.float32)).to(tl.bfloat16)
            npb = (pb.to(tl.float32) + db.to(tl.float32)).to(tl.bfloat16)
        else:
            npa = pa
            npb = pb
    else:
        npa = da
        npb = db

    nfa = npa.to(tl.float32)
    nfb = npb.to(tl.float32)
    ss = tl.sum(nfa * nfa) + tl.sum(nfb * nfb)
    pq = tl.sum(nfa * qa) + tl.sum(nfb * qb)
    scale = tl.math.rsqrt(ss * (1.0 / C) + EPS)
    lg_p = pq * scale

    tl.store(VB + NA * SRC_STRIDE + oa, npa, cache_modifier=st_cm)
    tl.store(VB + NA * SRC_STRIDE + ob, npb, cache_modifier=st_cm)
    if WKB:
        tl.store(KB + NA * SRC_STRIDE + oa, (nfa * scale).to(tl.bfloat16), cache_modifier=st_cm)
        tl.store(KB + NA * SRC_STRIDE + ob, (nfb * scale).to(tl.bfloat16), cache_modifier=st_cm)

    acc_a = nfa
    acc_b = nfb
    m = lg_p
    den = 1.0
    if HOIST == 0:
        v0a = tl.load(VB + 0 * SRC_STRIDE + oa, cache_modifier=ld_cm)
        v0b = tl.load(VB + 0 * SRC_STRIDE + ob, cache_modifier=ld_cm)
    acc_a, acc_b, m, den = _upd(v0a, v0b, acc_a, acc_b, m, den, qa, qb, C, EPS)
    if NA >= 2:
        if HOIST == 0:
            v1a = tl.load(VB + 1 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v1b = tl.load(VB + 1 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        acc_a, acc_b, m, den = _upd(v1a, v1b, acc_a, acc_b, m, den, qa, qb, C, EPS)
    if NA >= 3:
        if HOIST == 0:
            v2a = tl.load(VB + 2 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v2b = tl.load(VB + 2 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        acc_a, acc_b, m, den = _upd(v2a, v2b, acc_a, acc_b, m, den, qa, qb, C, EPS)
    if NA >= 4:
        if HOIST == 0:
            v3a = tl.load(VB + 3 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v3b = tl.load(VB + 3 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        acc_a, acc_b, m, den = _upd(v3a, v3b, acc_a, acc_b, m, den, qa, qb, C, EPS)
    if NA >= 5:
        if HOIST == 0:
            v4a = tl.load(VB + 4 * SRC_STRIDE + oa, cache_modifier=ld_cm)
            v4b = tl.load(VB + 4 * SRC_STRIDE + ob, cache_modifier=ld_cm)
        acc_a, acc_b, m, den = _upd(v4a, v4b, acc_a, acc_b, m, den, qa, qb, C, EPS)

    r = 1.0 / den
    ha = (acc_a * r).to(tl.bfloat16)
    hb = (acc_b * r).to(tl.bfloat16)
    if WHID == 1:
        tl.store(HID + oa, ha, cache_modifier=st_cm)
        tl.store(HID + ob, hb, cache_modifier=st_cm)

    if SUFFIX == 1:
        hfa = ha.to(tl.float32)
        hfb = hb.to(tl.float32)
        s1m = tl.sum(hfa) + tl.sum(hfb)
        s2m = tl.sum(hfa * hfa) + tl.sum(hfb * hfb)
        mean = s1m * (1.0 / C)
        var = s2m * (1.0 / C) - mean * mean
        rstd = tl.math.rsqrt(var + LNEPS)
        lna = ((hfa - mean) * rstd).to(tl.bfloat16)
        lnb = ((hfb - mean) * rstd).to(tl.bfloat16)
        if SPRE == 0:
            sc0a = tl.load(SC0 + t * SC0_S + ca, cache_modifier=pf_cm)
            sc0b = tl.load(SC0 + t * SC0_S + cb, cache_modifier=pf_cm)
            sh0a = tl.load(SH0 + t * SH0_S + ca, cache_modifier=pf_cm)
            sh0b = tl.load(SH0 + t * SH0_S + cb, cache_modifier=pf_cm)
        sca = (tl.load(SCTBL + ca).to(tl.float32) + sc0a.to(tl.float32)).to(tl.bfloat16)
        scb = (tl.load(SCTBL + cb).to(tl.float32) + sc0b.to(tl.float32)).to(tl.bfloat16)
        opa = (1.0 + sca.to(tl.float32)).to(tl.bfloat16)
        opb = (1.0 + scb.to(tl.float32)).to(tl.bfloat16)
        oma = (lna.to(tl.float32) * opa.to(tl.float32)).to(tl.bfloat16)
        omb = (lnb.to(tl.float32) * opb.to(tl.float32)).to(tl.bfloat16)
        sha = (tl.load(SHTBL + ca).to(tl.float32) + sh0a.to(tl.float32)).to(tl.bfloat16)
        shb = (tl.load(SHTBL + cb).to(tl.float32) + sh0b.to(tl.float32)).to(tl.bfloat16)
        oma = (oma.to(tl.float32) + sha.to(tl.float32)).to(tl.bfloat16)
        omb = (omb.to(tl.float32) + shb.to(tl.float32)).to(tl.bfloat16)
        tl.store(OUTM + oa, oma, cache_modifier=st_cm)
        tl.store(OUTM + ob, omb, cache_modifier=st_cm)


@triton.jit
def sana_ln_modulate(
    SRC,
    OUTM,
    SH0,
    SC0,
    SHTBL,
    SCTBL,
    SH0_S,
    SC0_S,
    C: tl.constexpr,
    CA: tl.constexpr,
    CB: tl.constexpr,
    LNEPS: tl.constexpr,
    SEV: tl.constexpr,
    PLEV: tl.constexpr,
    HOIST: tl.constexpr,
):
    """One token row per program: the LayerNorm of ``SRC`` times ``1 + SCTBL + SC0`` plus ``SHTBL + SH0``, rounded to bf16 after every operation. Launch knobs: ``SEV`` 2 stores with the streaming hint, ``PLEV`` 0 loads through L2 only, ``HOIST`` 1 loads the modulation rows before the statistics."""
    t = tl.program_id(0)
    ca = tl.arange(0, CA)
    cb = CA + tl.arange(0, CB)
    oa = t * C + ca
    ob = t * C + cb
    st_cm: tl.constexpr = ".cs" if SEV == 2 else ""
    pf_cm: tl.constexpr = ".cg" if PLEV == 0 else ""

    ha = tl.load(SRC + oa, cache_modifier=pf_cm)
    hb = tl.load(SRC + ob, cache_modifier=pf_cm)
    if HOIST == 1:
        sc0a = tl.load(SC0 + t * SC0_S + ca, cache_modifier=pf_cm)
        sc0b = tl.load(SC0 + t * SC0_S + cb, cache_modifier=pf_cm)
        sh0a = tl.load(SH0 + t * SH0_S + ca, cache_modifier=pf_cm)
        sh0b = tl.load(SH0 + t * SH0_S + cb, cache_modifier=pf_cm)
    hfa = ha.to(tl.float32)
    hfb = hb.to(tl.float32)
    s1m = tl.sum(hfa) + tl.sum(hfb)
    s2m = tl.sum(hfa * hfa) + tl.sum(hfb * hfb)
    mean = s1m * (1.0 / C)
    var = s2m * (1.0 / C) - mean * mean
    rstd = tl.math.rsqrt(var + LNEPS)
    lna = ((hfa - mean) * rstd).to(tl.bfloat16)
    lnb = ((hfb - mean) * rstd).to(tl.bfloat16)
    if HOIST == 0:
        sc0a = tl.load(SC0 + t * SC0_S + ca, cache_modifier=pf_cm)
        sc0b = tl.load(SC0 + t * SC0_S + cb, cache_modifier=pf_cm)
        sh0a = tl.load(SH0 + t * SH0_S + ca, cache_modifier=pf_cm)
        sh0b = tl.load(SH0 + t * SH0_S + cb, cache_modifier=pf_cm)
    sca = (tl.load(SCTBL + ca).to(tl.float32) + sc0a.to(tl.float32)).to(tl.bfloat16)
    scb = (tl.load(SCTBL + cb).to(tl.float32) + sc0b.to(tl.float32)).to(tl.bfloat16)
    opa = (1.0 + sca.to(tl.float32)).to(tl.bfloat16)
    opb = (1.0 + scb.to(tl.float32)).to(tl.bfloat16)
    oma = (lna.to(tl.float32) * opa.to(tl.float32)).to(tl.bfloat16)
    omb = (lnb.to(tl.float32) * opb.to(tl.float32)).to(tl.bfloat16)
    sha = (tl.load(SHTBL + ca).to(tl.float32) + sh0a.to(tl.float32)).to(tl.bfloat16)
    shb = (tl.load(SHTBL + cb).to(tl.float32) + sh0b.to(tl.float32)).to(tl.bfloat16)
    oma = (oma.to(tl.float32) + sha.to(tl.float32)).to(tl.bfloat16)
    omb = (omb.to(tl.float32) + shb.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUTM + oa, oma, cache_modifier=st_cm)
    tl.store(OUTM + ob, omb, cache_modifier=st_cm)


def gate_residual(x_sa, hidden, gate_t0, gate_table):
    """Return ``delta = (gate_table + gate_t0) * x_sa`` and ``hidden + delta``, both bf16 [1, N, C], from one launch."""
    tokens, channels = x_sa.shape[1], x_sa.shape[2]
    delta = torch.empty(x_sa.shape, dtype=torch.bfloat16, device=x_sa.device)
    h_after = torch.empty_like(delta)
    sana_attn_gate_residual[(tokens,)](
        x_sa,
        hidden,
        gate_t0,
        gate_table,
        delta,
        h_after,
        gate_t0.stride(0),
        C=channels,
        CA=2048,
        CB=channels - 2048,
        num_warps=2,
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
):
    """Add the closed sublayer's update, ``(gate_table + gate_t0) * core_out`` or ``extra_delta + core_out``, to the running block ``partial`` (None at a block's first update), store the sum as ``value_buffer[n_active]`` and mix it with the ``n_active`` rows before it per token by the softmax of their logits against ``query``. Returns the mix (None without ``write_hidden``) and, when a shift and scale are given, the LayerNorm of the mix modulated by them."""
    N, C = value_buffer.shape[2], value_buffer.shape[3]
    prefix_mode = 1 if extra_delta is None else 2
    suffix_mode = 0 if shift_t0 is None else 1
    dev = value_buffer.device
    hidden = torch.empty((1, N, C), dtype=torch.bfloat16, device=dev) if write_hidden else query
    out_mod = torch.empty((1, N, C), dtype=torch.bfloat16, device=dev) if suffix_mode == 1 else None
    warps, sev, lev, plev, spre, hoist = BOUNDARY_CONFIGS.get((N, n_active, prefix_mode, suffix_mode), BOUNDARY_DEFAULT)
    dummy = query
    sana_prefix_add_attend_adaln_fused[(N,)](
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
        value_buffer.stride(0),
        gate_t0.stride(0) if prefix_mode == 1 else 0,
        shift_t0.stride(0) if suffix_mode == 1 else 0,
        scale_t0.stride(0) if suffix_mode == 1 else 0,
        NA=n_active,
        PREFIX=prefix_mode,
        HAS_P=partial is not None,
        SUFFIX=suffix_mode,
        C=C,
        CA=2048,
        CB=C - 2048,
        EPS=_RMS_EPS,
        LNEPS=_LN_EPS,
        SEV=sev,
        LEV=lev,
        PLEV=plev,
        SPRE=spre if suffix_mode == 1 else 0,
        HOIST=hoist,
        WHID=1 if write_hidden else 0,
        WKB=0,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return (hidden if write_hidden else None), out_mod


def layer_norm_modulate(src, shift_t0, scale_t0, shift_table, scale_table):
    """Return the LayerNorm of the bf16 rows ``src`` [1, N, C] times ``1 + scale_table + scale_t0`` plus ``shift_table + shift_t0``, bf16 [1, N, C], from one launch."""
    N, C = src.shape[1], src.shape[2]
    out_mod = torch.empty((1, N, C), dtype=torch.bfloat16, device=src.device)
    warps, sev, plev, lhoist = LAYER_NORM_CONFIGS.get(N, LAYER_NORM_DEFAULT)
    sana_ln_modulate[(N,)](
        src,
        out_mod,
        shift_t0,
        scale_t0,
        shift_table,
        scale_table,
        shift_t0.stride(0),
        scale_t0.stride(0),
        C=C,
        CA=2048,
        CB=C - 2048,
        LNEPS=_LN_EPS,
        SEV=sev,
        PLEV=plev,
        HOIST=lhoist,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out_mod
