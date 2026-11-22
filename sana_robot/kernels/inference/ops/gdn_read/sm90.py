# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda as tec
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.ampere import async_copy

H100_SM_COUNT = 132
CHANNELS = 2560
HEAD_DIM = 128
_BLOCK_A = 8
_BLOCK_AC = 512
_BLOCK_B = 64
_CHUNKS_B = 6
_BLOCK_C = 64


@gluon.jit
def causal_gdn_prep(
    qkv_ptr,
    beta_ptr,
    k_w_ptr,
    parts_ptr,
    cos64_ptr,
    sin64_ptr,
    rstd_k_ptr,
    rb_ptr,
    cs_ptr,
    qmom_ptr,
    state_ptr,
    ksum_ptr,
    cnt_ptr,
    carry_s_ptr,
    carry_z_ptr,
    decay_ptr,
    stride_decay,
    N,
    NP,
    stride_qkv_n,
    stride_beta_n,
    norm_eps,
    key_scale,
    H: ttgl.constexpr,
    HALF_D: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_C: ttgl.constexpr,
    NC: ttgl.constexpr,
):
    """First kernel of the chain, 8 tokens per program: the RMS statistics of the key and query rows, the write strength ``rstd_k * sigmoid(beta) * key_scale`` of every head and the half-precision (cos, sin) table. It also seeds the state scratch with the carry scaled by the decay, transposed, and lets the second kernel launch at once."""
    tec.gdc_launch_dependents()
    pid = ttgl.program_id(0)
    C: ttgl.constexpr = NC * BLOCK_C
    HB: ttgl.constexpr = BLOCK_N // 2
    D2C: ttgl.constexpr = 2 * HALF_D

    M4: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [1, 32], [4, 2], [1, 0])
    C4: ttgl.constexpr = ttgl.BlockedLayout([HB, 2], [1, 32], [1, 8], [1, 0])
    sm_k: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    rows4_m = ttgl.arange(0, HB, layout=ttgl.SliceLayout(1, M4))
    cols_m = ttgl.arange(0, BLOCK_C, layout=ttgl.SliceLayout(0, M4))
    rows4_c = ttgl.arange(0, HB, layout=ttgl.SliceLayout(1, C4))
    cols_c = ttgl.arange(0, BLOCK_C, layout=ttgl.SliceLayout(0, C4))

    ring0a = ttgl.allocate_shared_memory(ttgl.bfloat16, [4, HB, BLOCK_C], sm_k)
    ring0b = ttgl.allocate_shared_memory(ttgl.bfloat16, [HB, BLOCK_C], sm_k)
    ring1a = ttgl.allocate_shared_memory(ttgl.bfloat16, [4, HB, BLOCK_C], sm_k)
    ring1b = ttgl.allocate_shared_memory(ttgl.bfloat16, [HB, BLOCK_C], sm_k)

    r0 = pid * BLOCK_N + rows4_m
    r1 = r0 + HB
    m0 = r0 < N
    m1 = r1 < N

    r0c = pid * BLOCK_N + rows4_c
    r1c = r0c + HB
    m0c = r0c < N
    m1c = r1c < N

    mom0 = ttgl.zeros([HB, BLOCK_C], ttgl.float32, layout=M4)
    for c in tl.static_range(NC):
        kv = ttgl.load(
            qkv_ptr + C + r0[:, None] * stride_qkv_n + (c * BLOCK_C + cols_m)[None, :],
            mask=m0[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        if c < 4:
            ring0a.index(c).store(kv)
        else:
            ring0b.store(kv)
        kvf = kv.to(ttgl.float32)
        mom0 += kvf * kvf
    rstd0 = ttgl.rsqrt(ttgl.sum(mom0, axis=1) / C + norm_eps)
    ttgl.store(rstd_k_ptr + r0, rstd0, mask=m0)
    rstd0c = ttgl.convert_layout(rstd0, ttgl.SliceLayout(1, C4))
    ttgl.thread_barrier()

    mom1 = ttgl.zeros([HB, BLOCK_C], ttgl.float32, layout=M4)
    ca0 = ttgl.zeros([BLOCK_C], ttgl.float32, layout=ttgl.SliceLayout(0, C4))
    ca1 = ttgl.zeros([BLOCK_C], ttgl.float32, layout=ttgl.SliceLayout(0, C4))
    ca2 = ttgl.zeros([BLOCK_C], ttgl.float32, layout=ttgl.SliceLayout(0, C4))
    ca3 = ttgl.zeros([BLOCK_C], ttgl.float32, layout=ttgl.SliceLayout(0, C4))
    ca4 = ttgl.zeros([BLOCK_C], ttgl.float32, layout=ttgl.SliceLayout(0, C4))
    for c in tl.static_range(NC):
        kv = ttgl.load(
            qkv_ptr + C + r1[:, None] * stride_qkv_n + (c * BLOCK_C + cols_m)[None, :],
            mask=m1[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        if c < 4:
            ring1a.index(c).store(kv)
        else:
            ring1b.store(kv)
        kvf = kv.to(ttgl.float32)
        mom1 += kvf * kvf

        kw = ttgl.load(k_w_ptr + c * BLOCK_C + cols_c).to(ttgl.float32)
        if c < 4:
            kb = ring0a.index(c).load(C4)
        else:
            kb = ring0b.load(C4)
        hcol = c * (BLOCK_C // D2C) + cols_c // D2C
        bl0 = ttgl.load(
            beta_ptr + r0c[:, None] * stride_beta_n + hcol[None, :],
            mask=m0c[:, None],
            other=0.0,
        ).to(ttgl.float32)
        sig0 = 1.0 / (1.0 + ttgl.exp(-bl0))
        contrib = ttgl.maximum(kb.to(ttgl.float32) * kw[None, :], 0.0) * sig0
        colp = ttgl.sum(contrib * rstd0c[:, None], axis=0)
        if c == 0:
            ca0 += colp
        elif c == 1:
            ca1 += colp
        elif c == 2:
            ca2 += colp
        elif c == 3:
            ca3 += colp
        else:
            ca4 += colp
    rstd1 = ttgl.rsqrt(ttgl.sum(mom1, axis=1) / C + norm_eps)
    ttgl.store(rstd_k_ptr + r1, rstd1, mask=m1)
    rstd1c = ttgl.convert_layout(rstd1, ttgl.SliceLayout(1, C4))
    ttgl.thread_barrier()

    for c in tl.static_range(NC):
        kw = ttgl.load(k_w_ptr + c * BLOCK_C + cols_c).to(ttgl.float32)
        if c < 4:
            kb = ring1a.index(c).load(C4)
        else:
            kb = ring1b.load(C4)
        hcol = c * (BLOCK_C // D2C) + cols_c // D2C
        bl1 = ttgl.load(
            beta_ptr + r1c[:, None] * stride_beta_n + hcol[None, :],
            mask=m1c[:, None],
            other=0.0,
        ).to(ttgl.float32)
        sig1 = 1.0 / (1.0 + ttgl.exp(-bl1))
        contrib = ttgl.maximum(kb.to(ttgl.float32) * kw[None, :], 0.0) * sig1
        colp = ttgl.sum(contrib * rstd1c[:, None], axis=0)
        if c == 0:
            ca0 += colp
        elif c == 1:
            ca1 += colp
        elif c == 2:
            ca2 += colp
        elif c == 3:
            ca3 += colp
        else:
            ca4 += colp
    ttgl.store(parts_ptr + pid * C + 0 * BLOCK_C + cols_c, (ca0 * key_scale).to(ttgl.bfloat16))
    ttgl.store(parts_ptr + pid * C + 1 * BLOCK_C + cols_c, (ca1 * key_scale).to(ttgl.bfloat16))
    ttgl.store(parts_ptr + pid * C + 2 * BLOCK_C + cols_c, (ca2 * key_scale).to(ttgl.bfloat16))
    ttgl.store(parts_ptr + pid * C + 3 * BLOCK_C + cols_c, (ca3 * key_scale).to(ttgl.bfloat16))
    ttgl.store(parts_ptr + pid * C + 4 * BLOCK_C + cols_c, (ca4 * key_scale).to(ttgl.bfloat16))

    R: ttgl.constexpr = ttgl.BlockedLayout([1, 1], [1, 32], [8, 1], [1, 0])
    rows_r = pid * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, R))
    hs = ttgl.arange(0, 32, layout=ttgl.SliceLayout(0, R))
    rmask_r = rows_r < N
    bmask = rmask_r[:, None] & (hs[None, :] < H)
    bl = ttgl.load(
        beta_ptr + rows_r[:, None] * stride_beta_n + hs[None, :],
        mask=bmask,
        other=0.0,
    ).to(ttgl.float32)
    rstd8 = ttgl.load(rstd_k_ptr + rows_r, mask=rmask_r, other=0.0)
    rb = rstd8[:, None] * key_scale / (1.0 + ttgl.exp(-bl))
    ttgl.store(rb_ptr + hs[None, :] * NP + rows_r[:, None], rb, mask=bmask)

    T: ttgl.constexpr = ttgl.BlockedLayout([1, 2], [1, 32], [8, 1], [1, 0])
    rows_t = pid * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, T))
    dh = ttgl.arange(0, HALF_D, layout=ttgl.SliceLayout(0, T))
    tmask = (rows_t < N)[:, None] & (dh[None, :] < HALF_D)
    tptr = rows_t[:, None] * HALF_D + dh[None, :]
    cosv = ttgl.load(cos64_ptr + tptr, mask=tmask, other=0.0, eviction_policy="evict_first").to(ttgl.float32)
    sinv = ttgl.load(sin64_ptr + tptr, mask=tmask, other=0.0, eviction_policy="evict_first").to(ttgl.float32)
    D2: tl.constexpr = 2 * HALF_D
    eptr = rows_t[:, None] * D2 + (2 * dh)[None, :]
    ttgl.store(cs_ptr + eptr, cosv.to(ttgl.float16), mask=tmask)
    ttgl.store(cs_ptr + eptr + 1, sinv.to(ttgl.float16), mask=tmask)

    momq0 = ttgl.zeros([HB, BLOCK_C], ttgl.float32, layout=M4)
    momq1 = ttgl.zeros([HB, BLOCK_C], ttgl.float32, layout=M4)
    for c in tl.static_range(NC):
        qv0 = ttgl.load(
            qkv_ptr + r0[:, None] * stride_qkv_n + (c * BLOCK_C + cols_m)[None, :],
            mask=m0[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(ttgl.float32)
        momq0 += qv0 * qv0
        qv1 = ttgl.load(
            qkv_ptr + r1[:, None] * stride_qkv_n + (c * BLOCK_C + cols_m)[None, :],
            mask=m1[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(ttgl.float32)
        momq1 += qv1 * qv1
    ttgl.store(qmom_ptr + r0, ttgl.sum(momq0, axis=1), mask=m0)
    ttgl.store(qmom_ptr + r1, ttgl.sum(momq1, axis=1), mask=m1)

    Z: ttgl.constexpr = ttgl.BlockedLayout([8], [32], [8], [0])
    DD: ttgl.constexpr = D2C * D2C
    nprog = ttgl.num_programs(0)
    ZB: ttgl.constexpr = 2048
    total_state = H * DD
    per = (total_state + nprog - 1) // nprog
    z0 = pid * per
    for off in range(0, per, ZB):
        idx = z0 + off + ttgl.arange(0, ZB, layout=Z)
        m = (idx < total_state) & (idx < z0 + per)
        hh = idx // DD
        rem = idx - hh * DD
        dv = rem // D2C
        dk = rem - dv * D2C
        sv = ttgl.load(carry_s_ptr + hh * DD + dk * D2C + dv, mask=m, other=0.0)
        gv = ttgl.load(decay_ptr + hh * stride_decay, mask=m, other=0.0)
        ttgl.store(state_ptr + idx, sv * gv, mask=m)
    total_ksum = H * D2C + 32
    perk = (total_ksum + nprog - 1) // nprog
    k0 = pid * perk
    for off in range(0, perk, ZB):
        idx = k0 + off + ttgl.arange(0, ZB, layout=Z)
        m = (idx < total_ksum) & (idx < k0 + perk)
        mk = m & (idx < H * D2C)
        hh = idx // D2C
        zv = ttgl.load(carry_z_ptr + idx, mask=mk, other=0.0)
        gv = ttgl.load(decay_ptr + hh * stride_decay, mask=mk, other=0.0)
        ttgl.store(ksum_ptr + idx, zv * gv, mask=m)


ROPE_ASM = ttgl.constexpr(
    "{ .reg .f32 c, s, t; .reg .b16 hc, hs;"
    " mov.b32 {hc, hs}, $4;"
    " cvt.f32.f16 c, hc; cvt.f32.f16 s, hs;"
    " neg.f32 t, s;"
    " mul.f32 $0, $2, c; fma.rn.f32 $0, $3, t, $0;"
    " mul.f32 $1, $2, s; fma.rn.f32 $1, $3, c, $1; }"
)


@triton.jit
def causal_gdn_kv_state(
    qkv_ptr,
    parts_ptr,
    rstd_k_ptr,
    rb_ptr,
    cs_ptr,
    k_w_ptr,
    state_ptr,
    spart_ptr,
    st16_ptr,
    ksum_ptr,
    cnt_ptr,
    gpart_ptr,
    zpart_ptr,
    s_new_ptr,
    z_new_ptr,
    carry_z_ptr,
    decay_ptr,
    stride_decay,
    N,
    NP,
    stride_qkv_n,
    rows_per_chunk,
    chunks_live,
    n_parts,
    key_scale,
    MAIN: tl.constexpr,
    NLIVE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EMIT_STATE: tl.constexpr,
):
    """Second kernel, a dependent launch: every (token block, head) program stores the product of its normalized, scaled and rotated keys with the values as a partial, and the last program of a head adds the partials to the seeded state in a fixed order and stores the bf16 state the queries read. ``EMIT_STATE`` also writes the float32 state after the chunk's delta-rule write."""
    tec.gdc_launch_dependents()
    pid = tl.program_id(0)

    HALF: tl.constexpr = D // 2
    d = tl.arange(0, D)
    dh = tl.arange(0, HALF)

    if pid >= MAIN:
        tec.gdc_wait()
        r = pid - MAIN
        rows_g = tl.arange(0, 64)
        for hh in range(r, H, 12):
            racc = tl.zeros((64, D), tl.float32)
            for g0 in range(0, n_parts, 64):
                m = (g0 + rows_g) < n_parts
                t = tl.load(
                    parts_ptr + (g0 + rows_g)[:, None] * C + hh * D + d[None, :],
                    mask=m[:, None],
                    other=0.0,
                )
                racc += t.to(tl.float32)
            seed = tl.load(ksum_ptr + hh * D + d)
            tl.store(ksum_ptr + hh * D + d, seed + tl.sum(racc, axis=0))
        return

    h = pid % H
    p = pid // H
    n_start = p * rows_per_chunk
    if n_start >= N:
        return
    tec.gdc_wait()
    n_end = tl.minimum(n_start + rows_per_chunk, N)

    kw = tl.load(k_w_ptr + h * D + d).to(tl.float32)
    acc = tl.zeros((D, D), tl.float32)
    if EMIT_STATE:
        g_acc = tl.zeros((D, D), tl.float32)
        zacc = tl.zeros((D,), tl.float32)
        gz = tl.load(carry_z_ptr + h * D + d) * tl.load(decay_ptr + h * stride_decay)

    rows = n_start + tl.arange(0, BLOCK_N)
    kptrs = qkv_ptr + C + h * D + rows[:, None] * stride_qkv_n + d[None, :]
    vptrs = qkv_ptr + 2 * C + h * D + rows[None, :] * stride_qkv_n + d[:, None]
    cptrs = rows[:, None] * D + d[None, :]
    for _ in range(n_start, n_end, BLOCK_N):
        rmask = rows < n_end
        rb = tl.load(rb_ptr + h * NP + rows, mask=rmask, other=0.0)
        csv = tl.load(cs_ptr + cptrs, mask=rmask[:, None], other=0.0)
        cosv, sinv = tl.split(tl.reshape(csv, (BLOCK_N, HALF, 2)))
        cosv = cosv.to(tl.float32)
        sinv = sinv.to(tl.float32)
        kv = tl.load(kptrs, mask=rmask[:, None], other=0.0, eviction_policy="evict_first").to(tl.float32)

        base = kv * kw[None, :]
        kb = base * rb[:, None]
        re, im = tl.split(tl.reshape(kb, (BLOCK_N, HALF, 2)))
        kg = tl.reshape(tl.join(re * cosv - im * sinv, re * sinv + im * cosv), (BLOCK_N, D)).to(tl.bfloat16)

        vv_t = tl.load(vptrs, mask=rmask[None, :], other=0.0, eviction_policy="evict_first")
        acc = tl.dot(vv_t, kg, acc)

        if EMIT_STATE:
            rsr = tl.load(rstd_k_ptr + rows, mask=rmask, other=0.0)
            kpb = base * (rsr * key_scale)[:, None]
            pre, pim = tl.split(tl.reshape(kpb, (BLOCK_N, HALF, 2)))
            kp = tl.reshape(
                tl.join(pre * cosv - pim * sinv, pre * sinv + pim * cosv),
                (BLOCK_N, D),
            ).to(tl.bfloat16)
            g_acc = tl.dot(tl.trans(kp), kg, g_acc)
            kdn = tl.maximum(base, 0.0)
            zp = tl.sum(kdn * gz[None, :], axis=1) * (rsr * key_scale)
            zacc += tl.sum(kdn * (rb * (1.0 - zp))[:, None], axis=0)

        rows += BLOCK_N
        kptrs += BLOCK_N * stride_qkv_n
        vptrs += BLOCK_N * stride_qkv_n
        cptrs += BLOCK_N * D

    tl.store(spart_ptr + (p * H + h) * D * D + d[:, None] * D + d[None, :], acc)
    if EMIT_STATE:
        tl.store(gpart_ptr + (p * H + h) * D * D + d[:, None] * D + d[None, :], g_acc)
        tl.store(zpart_ptr + (p * H + h) * D + d, zacc)
    done = tl.atomic_add(cnt_ptr + h, 1.0)
    if done == chunks_live - 1.0:
        sptr = h * D * D + d[:, None] * D + d[None, :]
        seed = tl.load(state_ptr + sptr)
        full = seed
        for pp in tl.static_range(NLIVE):
            full += tl.load(
                spart_ptr + (pp * H + h) * D * D + d[:, None] * D + d[None, :],
                cache_modifier=".cg",
            )
        tptr = h * D * D + d[None, :] * D + d[:, None]
        tl.store(st16_ptr + tptr, full.to(tl.bfloat16))
        if EMIT_STATE:
            gsum = tl.zeros((D, D), tl.float32)
            for pp in tl.static_range(NLIVE):
                gsum += tl.load(
                    gpart_ptr + (pp * H + h) * D * D + d[:, None] * D + d[None, :],
                    cache_modifier=".cg",
                )
            erase = tl.dot(seed, gsum, input_precision="ieee")
            tl.store(s_new_ptr + tptr, full - erase)
            zfull = tl.load(carry_z_ptr + h * D + d) * tl.load(decay_ptr + h * stride_decay)
            for pp in tl.static_range(NLIVE):
                zfull += tl.load(zpart_ptr + (pp * H + h) * D + d, cache_modifier=".cg")
            tl.store(z_new_ptr + h * D + d, zfull)


@gluon.jit
def causal_gdn_q_out(
    qkv_ptr,
    gate_ptr,
    qmom_ptr,
    cs_ptr,
    q_w_ptr,
    o_w_ptr,
    st16_ptr,
    ksum_ptr,
    out_ptr,
    N,
    stride_qkv_n,
    stride_gate_n,
    stride_out_n,
    norm_eps,
    z_eps,
    H: ttgl.constexpr,
    D: ttgl.constexpr,
    C: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    NORMALIZER: ttgl.constexpr,
):
    """Third kernel, a dependent launch, 64 tokens of one head per program: the rotated queries read the state, followed by the output RMSNorm and the sigmoid output gate. It fetches its queries while the second kernel runs and waits for the chain before reading anything the chain writes (the cos/sin table, the query moments, the state)."""
    mma: ttgl.constexpr = ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    dopA: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    blk: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    sm_b: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([D, D], ttgl.bfloat16)

    pid = ttgl.program_id(0)
    h = pid % H
    t = pid // H

    sm_t: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_N, D], ttgl.float16)
    dr = ttgl.arange(0, D, layout=ttgl.SliceLayout(1, blk))
    dc = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, blk))
    rows_b = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blk))
    rows_bc = ttgl.minimum(rows_b + t * BLOCK_N, N - 1)
    st_base = st16_ptr + h * D * D
    sm_q: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for([BLOCK_N, D], ttgl.bfloat16)
    q_sm = ttgl.allocate_shared_memory(ttgl.bfloat16, [BLOCK_N, D], sm_q)
    st1_sm = ttgl.allocate_shared_memory(ttgl.bfloat16, [D, D], sm_b)
    ce_sm = ttgl.allocate_shared_memory(ttgl.float16, [BLOCK_N, D], sm_t)
    qb = ttgl.load(
        qkv_ptr + h * D + rows_bc[:, None] * stride_qkv_n + dc[None, :],
        eviction_policy="evict_first",
    )
    q_sm.store(qb)
    rows_a = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, dopA))
    cols_a = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, dopA))
    ttgl.thread_barrier()
    qa = q_sm.load(dopA).to(ttgl.float32)
    rows_amc = ttgl.minimum(rows_a + t * BLOCK_N, N - 1)

    tec.gdc_wait()
    async_copy.async_copy_global_to_shared(ce_sm, cs_ptr + rows_bc[:, None] * D + dc[None, :])
    async_copy.commit_group()
    async_copy.async_copy_global_to_shared(st1_sm, st_base + dr[:, None] * D + dc[None, :])
    async_copy.commit_group()
    ksum = ttgl.load(ksum_ptr + h * D + cols_a)
    qm = ttgl.load(qmom_ptr + rows_amc, cache_modifier=".cg")
    rstd = ttgl.rsqrt(qm / C + norm_eps)

    qw = ttgl.load(q_w_ptr + h * D + cols_a).to(ttgl.float32)

    rows_m = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, mma))
    cols_m = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, mma))
    rows_mc = ttgl.minimum(rows_m + t * BLOCK_N, N - 1)
    gl = ttgl.load(
        gate_ptr + rows_mc[:, None] * stride_gate_n + h * D + cols_m[None, :],
        eviction_policy="evict_first",
    )
    ow = ttgl.load(o_w_ptr + cols_m).to(ttgl.float32)

    u = qa * qw[None, :]
    s_z = ttgl.sum(ttgl.maximum(u, 0.0) * ksum[None, :], axis=1)

    async_copy.wait_group(1)
    ttgl.thread_barrier()
    cs = ce_sm.load(dopA)

    a1 = ttgl.inline_asm_elementwise(
        ROPE_ASM,
        "=r,=r,r,r,r",
        [u, cs],
        dtype=ttgl.float32,
        is_pure=True,
        pack=2,
    ).to(ttgl.bfloat16)

    async_copy.wait_group(0)
    ttgl.thread_barrier()
    acc = ttgl.zeros([BLOCK_N, D], ttgl.float32, layout=mma)
    acc = hopper.warpgroup_mma(a1, st1_sm, acc, is_async=True)

    acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    rstd_m = ttgl.convert_layout(rstd, ttgl.SliceLayout(1, mma), assert_trivial=True)
    if NORMALIZER:
        s_m = ttgl.convert_layout(s_z, ttgl.SliceLayout(1, mma), assert_trivial=True)
        z = rstd_m / (rstd_m * s_m + z_eps)
    else:
        z = rstd_m
    val = acc * z[:, None]

    rstd_o = ttgl.rsqrt(ttgl.sum(val * val, axis=1) / D + norm_eps)
    sig = 1.0 / (1.0 + ttgl.exp(-gl.to(ttgl.float32)))
    res = val * rstd_o[:, None] * ow[None, :] * sig

    out_sm = ce_sm._reinterpret(ttgl.bfloat16, [BLOCK_N, D], sm_t)
    ttgl.thread_barrier()
    out_sm.store(res.to(ttgl.bfloat16))
    ttgl.thread_barrier()
    res_b = out_sm.load(blk)
    rows_s = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(1, blk))
    cols_s = ttgl.arange(0, D, layout=ttgl.SliceLayout(0, blk))
    rmask = (rows_s + t * BLOCK_N) < N
    optr = out_ptr + (rows_s + t * BLOCK_N)[:, None] * stride_out_n + h * D + cols_s[None, :]
    ttgl.store(
        optr,
        res_b,
        mask=rmask[:, None],
        eviction_policy="evict_first",
    )


_scratch_cache = {}


def supports(device, channels: int, heads: int) -> bool:
    """Return whether the kernels serve Gated DeltaNet layers of ``channels`` in ``heads`` on ``device``: an H100, whose 132 SMs the grid of the second kernel fills once, and 2560 channels in heads of 128."""
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return sm_count == H100_SM_COUNT and channels == CHANNELS and channels == HEAD_DIM * heads


def _get_scratch(N, C, H, D, dev):
    """Return the scratch buffers and the launch grids of a chunk of N tokens, built once per (N, C, H, device) and never dropped: a captured CUDA graph holds their addresses. The arrival counters ``cnt`` follow ``ksum`` in memory, since one sweep of the first kernel seeds the one and clears the other."""
    key = (N, C, H, dev)
    ent = _scratch_cache.get(key)
    if ent is not None:
        return ent
    Np = (N + 15) & ~15
    n_state = H * D * D
    n_ksum = H * D
    n_cnt = 32
    n_spart = _CHUNKS_B * H * D * D
    n_zpart = _CHUNKS_B * H * D
    f32 = torch.empty(
        (1 + H) * Np + n_state + n_ksum + n_cnt + Np + 2 * n_spart + n_zpart + n_state // 2,
        dtype=torch.float32,
        device=dev,
    )
    o_s = (1 + H) * Np
    o_q = o_s + n_state + n_ksum + n_cnt
    f16 = torch.empty(Np * D, dtype=torch.float16, device=dev)
    parts = torch.empty(triton.cdiv(N, _BLOCK_A) * C, dtype=torch.bfloat16, device=dev)
    ent = {
        "Np": Np,
        "rstd_k": f32[:Np],
        "rb": f32[Np : (1 + H) * Np],
        "state": f32[o_s : o_s + n_state],
        "ksum": f32[o_s + n_state : o_s + n_state + n_ksum],
        "cnt": f32[o_s + n_state + n_ksum : o_s + n_state + n_ksum + n_cnt],
        "qmom": f32[o_q : o_q + Np],
        "spart": f32[o_q + Np : o_q + Np + n_spart],
        "gpart": f32[o_q + Np + n_spart : o_q + Np + 2 * n_spart],
        "zpart": f32[o_q + Np + 2 * n_spart : o_q + Np + 2 * n_spart + n_zpart],
        "st16": f32[o_q + Np + 2 * n_spart + n_zpart :].view(torch.bfloat16),
        "cs16": f16,
        "parts": parts,
        "carry_z": torch.zeros(n_ksum, dtype=torch.float32, device=dev),
        "z_new": torch.empty(n_ksum, dtype=torch.float32, device=dev),
        "grid_a": (triton.cdiv(N, _BLOCK_A),),
        "grid_b": (_CHUNKS_B * H + 12,),
        "grid_c": (triton.cdiv(N, _BLOCK_C) * H,),
        "rpc": triton.cdiv(N, _CHUNKS_B * _BLOCK_B) * _BLOCK_B,
    }
    ent["chunks_live"] = float(triton.cdiv(N, ent["rpc"]))
    ent["nlive"] = int(triton.cdiv(N, ent["rpc"]))
    _scratch_cache[key] = ent
    return ent


def _launch(
    s,
    out,
    qkv,
    beta_logits,
    gate_logits,
    rotary_emb,
    q_norm_weight,
    k_norm_weight,
    o_norm_weight,
    carry_s,
    decay,
    key_scale,
    N,
    C,
    H,
    D,
    norm_eps,
    s_new=None,
):
    """Launch the three chained kernels of one chunk on device tensors only; with ``s_new`` the second kernel also writes the state after the chunk."""
    flat = rotary_emb.reshape(N, -1)
    rope_cos = flat.real.contiguous()
    rope_sin = flat.imag.contiguous()
    causal_gdn_prep[s["grid_a"]](
        qkv,
        beta_logits,
        k_norm_weight,
        s["parts"],
        rope_cos,
        rope_sin,
        s["rstd_k"],
        s["rb"],
        s["cs16"],
        s["qmom"],
        s["state"],
        s["ksum"],
        s["cnt"],
        carry_s,
        s["carry_z"],
        decay,
        decay.stride(0),
        N,
        s["Np"],
        qkv.stride(0),
        beta_logits.stride(0),
        norm_eps,
        key_scale,
        H=H,
        HALF_D=D // 2,
        BLOCK_N=_BLOCK_A,
        BLOCK_C=_BLOCK_AC,
        NC=C // _BLOCK_AC,
        num_warps=8,
    )
    emit = s_new is not None
    causal_gdn_kv_state[s["grid_b"]](
        qkv,
        s["parts"],
        s["rstd_k"],
        s["rb"],
        s["cs16"],
        k_norm_weight,
        s["state"],
        s["spart"],
        s["st16"],
        s["ksum"],
        s["cnt"],
        s["gpart"],
        s["zpart"],
        s_new if emit else s["gpart"],
        s["z_new"],
        s["carry_z"],
        decay,
        decay.stride(0),
        N,
        s["Np"],
        qkv.stride(0),
        s["rpc"],
        s["chunks_live"],
        s["grid_a"][0],
        key_scale,
        MAIN=_CHUNKS_B * H,
        NLIVE=s["nlive"],
        H=H,
        D=D,
        C=C,
        BLOCK_N=_BLOCK_B,
        num_warps=16,
        num_stages=3,
        EMIT_STATE=emit,
        launch_pdl=True,
    )
    causal_gdn_q_out[s["grid_c"]](
        qkv,
        gate_logits,
        s["qmom"],
        s["cs16"],
        q_norm_weight,
        o_norm_weight,
        s["st16"],
        s["ksum"],
        out,
        N,
        qkv.stride(0),
        gate_logits.stride(0),
        out.stride(0),
        norm_eps,
        0.0,
        H=H,
        D=D,
        C=C,
        BLOCK_N=_BLOCK_C,
        NORMALIZER=False,
        num_warps=4,
        maxnreg=168,
        launch_pdl=True,
    )


def seeded_gdn_read(
    qkv,
    beta_logits,
    gate_logits,
    rotary_emb,
    q_norm_weight,
    k_norm_weight,
    o_norm_weight,
    carry_s,
    decay,
    key_scale,
    num_heads,
    norm_eps,
):
    """Return the output [N, C] (bf16, before the output projection) of one chunk that reads the state ``carry_s`` [H, D, D] (float32) scaled by ``decay`` [H] together with its own tokens. ``qkv`` [N, 3C], ``beta_logits`` [N, H] and ``gate_logits`` [N, C] are bf16 projections of the chunk's tokens and ``rotary_emb`` their complex phases."""
    tokens, packed = qkv.shape
    channels = packed // 3
    head_dim = channels // num_heads
    s = _get_scratch(tokens, channels, num_heads, head_dim, qkv.device)
    out = torch.empty((tokens, channels), dtype=torch.bfloat16, device=qkv.device)
    _launch(
        s,
        out,
        qkv,
        beta_logits,
        gate_logits,
        rotary_emb,
        q_norm_weight,
        k_norm_weight,
        o_norm_weight,
        carry_s.contiguous(),
        decay,
        float(key_scale),
        tokens,
        channels,
        num_heads,
        head_dim,
        norm_eps,
    )
    return out


def seeded_gdn_read_commit(
    qkv,
    beta_logits,
    gate_logits,
    rotary_emb,
    q_norm_weight,
    k_norm_weight,
    o_norm_weight,
    carry_s,
    decay,
    key_scale,
    num_heads,
    norm_eps,
):
    """Return the output of ``seeded_gdn_read`` and the state [H, D, D] (float32, a new tensor) after the chunk has written its tokens into the decayed ``carry_s`` by the delta rule."""
    tokens, packed = qkv.shape
    channels = packed // 3
    head_dim = channels // num_heads
    s = _get_scratch(tokens, channels, num_heads, head_dim, qkv.device)
    out = torch.empty((tokens, channels), dtype=torch.bfloat16, device=qkv.device)
    s_new = torch.empty((num_heads, head_dim, head_dim), dtype=torch.float32, device=qkv.device)
    _launch(
        s,
        out,
        qkv,
        beta_logits,
        gate_logits,
        rotary_emb,
        q_norm_weight,
        k_norm_weight,
        o_norm_weight,
        carry_s.contiguous(),
        decay,
        float(key_scale),
        tokens,
        channels,
        num_heads,
        head_dim,
        norm_eps,
        s_new=s_new,
    )
    return out, s_new
