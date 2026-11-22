# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.dispatch import TunedTable

HEAD_DIM = 128
MAX_HEADS = 32
_BLOCK_AC = 512
_FOLD_KB = 16


class GdnLaunchConfig(NamedTuple):
    """Launch constants of the three kernels: tokens per program and warps of the first (``block_a``, ``warps_a``); token chunks per head, tokens per dot, warps of the read and of the commit form and pipeline stages of the second (``chunks_b``, ``block_b``, ``warps_b``, ``warps_b_emit``, ``stages_b``); tokens per program and warps of the third (``block_c``, ``warps_c``)."""

    block_a: int
    warps_a: int
    chunks_b: int
    block_b: int
    warps_b: int
    warps_b_emit: int
    stages_b: int
    block_c: int
    warps_c: int


DEFAULT_CONFIG = GdnLaunchConfig(
    block_a=16, warps_a=8, chunks_b=8, block_b=16, warps_b=8, warps_b_emit=16, stages_b=1, block_c=16, warps_c=8
)
READ_CONFIGS = TunedTable(
    {
        193: GdnLaunchConfig(
            block_a=16, warps_a=8, chunks_b=1, block_b=16, warps_b=4, warps_b_emit=16, stages_b=1, block_c=16, warps_c=8
        ),
        273: GdnLaunchConfig(
            block_a=16, warps_a=8, chunks_b=1, block_b=16, warps_b=4, warps_b_emit=16, stages_b=1, block_c=16, warps_c=8
        ),
    },
    tuned=(193, 273),
)
COMMIT_CONFIGS = TunedTable(
    {
        160: GdnLaunchConfig(
            block_a=16, warps_a=8, chunks_b=1, block_b=16, warps_b=8, warps_b_emit=8, stages_b=1, block_c=16, warps_c=4
        ),
        240: GdnLaunchConfig(
            block_a=16, warps_a=8, chunks_b=1, block_b=16, warps_b=8, warps_b_emit=8, stages_b=1, block_c=16, warps_c=8
        ),
    },
    tuned=(160, 240),
)


@triton.jit
def sana_gdn_causal_sm110_prep(
    qkv_ptr,
    beta_ptr,
    rope_ptr,
    rstd_q_ptr,
    rstd_k_ptr,
    beta_s_ptr,
    cs32_ptr,
    cnt_ptr,
    N,
    NP,
    stride_qkv_n,
    stride_beta_n,
    stride_rope_n,
    norm_eps,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """First kernel, ``BLOCK_N`` tokens per program: the RMS statistics of the query and key rows, sigmoid(beta) of every head rounded to bf16 and the float32 copy of the (cos, sin) table. Program 0 resets the arrival counters of the second kernel."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    cols = tl.arange(0, BLOCK_C)

    q_acc = tl.zeros((BLOCK_N,), tl.float32)
    k_acc = tl.zeros((BLOCK_N,), tl.float32)
    for c0 in tl.static_range(0, C, BLOCK_C):
        qv = tl.load(
            qkv_ptr + rows[:, None] * stride_qkv_n + (c0 + cols)[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        kv = tl.load(
            qkv_ptr + C + rows[:, None] * stride_qkv_n + (c0 + cols)[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        q_acc += tl.sum(qv * qv, axis=1)
        k_acc += tl.sum(kv * kv, axis=1)
    tl.store(rstd_q_ptr + rows, tl.rsqrt(q_acc / C + norm_eps), mask=rmask)
    tl.store(rstd_k_ptr + rows, tl.rsqrt(k_acc / C + norm_eps), mask=rmask)

    hs = tl.arange(0, 32)
    bmask = rmask[:, None] & (hs[None, :] < H)
    bl = tl.load(
        beta_ptr + rows[:, None] * stride_beta_n + hs[None, :],
        mask=bmask,
        other=0.0,
    ).to(tl.float32)
    sig = (1.0 / (1.0 + tl.exp(-bl))).to(tl.bfloat16).to(tl.float32)
    tl.store(beta_s_ptr + hs[None, :] * NP + rows[:, None], sig, mask=bmask)

    d = tl.arange(0, D)
    tab = tl.load(
        rope_ptr + rows[:, None] * stride_rope_n + d[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(cs32_ptr + rows[:, None] * D + d[None, :], tab, mask=rmask[:, None])

    if pid == 0:
        tl.store(cnt_ptr + hs, tl.zeros((32,), tl.int32))


@triton.jit
def sana_gdn_causal_sm110_kv_state(
    qkv_ptr,
    rstd_k_ptr,
    beta_s_ptr,
    cs32_ptr,
    k_w_ptr,
    carry_s_ptr,
    carry_z_ptr,
    decay_ptr,
    spart_ptr,
    zpart_ptr,
    cnt_ptr,
    st_ptr,
    zeff_ptr,
    gpart_ptr,
    zwpart_ptr,
    s_new_ptr,
    z_new_ptr,
    N,
    NP,
    stride_qkv_n,
    rows_per_chunk,
    key_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EMIT_STATE: tl.constexpr,
    FOLD_KB: tl.constexpr,
):
    """Second kernel, one program per (token chunk, head): the product of the normalized, rotated and scaled keys with the values, stored as a partial; the last program of a head adds the partials in a fixed order to the decayed carry and stores the bf16 state the queries read. ``EMIT_STATE`` also writes the float32 state after the chunk's delta-rule write, its erase term as the Gram matrix of the keys applied to the decayed carry in blocks of ``FOLD_KB`` columns."""
    pid = tl.program_id(0)
    h = pid % H
    p = pid // H
    d = tl.arange(0, D)
    d_sw = d ^ 1
    d_pair = 2 * (d // 2)
    sgn = tl.where((d % 2) == 0, -1.0, 1.0).to(tl.float32)

    n_start = p * rows_per_chunk
    n_end = tl.minimum(n_start + rows_per_chunk, N)

    kw = tl.load(k_w_ptr + h * D + d).to(tl.float32)
    kw_sw = tl.load(k_w_ptr + h * D + d_sw).to(tl.float32)
    acc = tl.zeros((D, D), tl.float32)
    zpart = tl.zeros((D,), tl.float32)
    if EMIT_STATE:
        g_emit = tl.load(decay_ptr + h)
        gram = tl.zeros((D, D), tl.float32)
        zw = tl.zeros((D,), tl.float32)
        gz = tl.load(carry_z_ptr + h * D + d) * g_emit

    for n0 in range(n_start, n_end, BLOCK_N):
        rows = n0 + tl.arange(0, BLOCK_N)
        rmask = rows < n_end
        rstd = tl.load(rstd_k_ptr + rows, mask=rmask, other=0.0)
        bs = tl.load(beta_s_ptr + h * NP + rows, mask=rmask, other=0.0)
        krow = qkv_ptr + C + h * D + rows[:, None] * stride_qkv_n
        kf = tl.load(krow + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        kf_sw = tl.load(krow + d_sw[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        u = (kw[None, :] * (kf * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
        u_sw = (kw_sw[None, :] * (kf_sw * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
        kdc = (tl.maximum(u, 0.0) * key_scale).to(tl.bfloat16).to(tl.float32)
        zpart += tl.sum(kdc * bs[:, None], axis=0)
        trow = cs32_ptr + rows[:, None] * D
        cosf = tl.load(trow + d_pair[None, :], mask=rmask[:, None], other=0.0)
        sinf = tl.load(trow + (d_pair + 1)[None, :], mask=rmask[:, None], other=0.0)
        krot = (u * cosf + u_sw * (sgn[None, :] * sinf)).to(tl.bfloat16).to(tl.float32)
        kwr = (krot * key_scale).to(tl.bfloat16).to(tl.float32)
        kg = (kwr * bs[:, None]).to(tl.bfloat16)
        vv = tl.load(
            qkv_ptr + 2 * C + h * D + rows[:, None] * stride_qkv_n + d[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        acc = tl.dot(tl.trans(kg), vv, acc)
        if EMIT_STATE:
            gram = tl.dot(tl.trans(kg), kwr.to(tl.bfloat16), gram)
            zp = tl.sum(kdc * gz[None, :], axis=1)
            zw += tl.sum(kdc * (bs * (1.0 - zp))[:, None], axis=0)

    tl.store(spart_ptr + (p * H + h) * D * D + d[:, None] * D + d[None, :], acc)
    tl.store(zpart_ptr + (p * H + h) * D + d, zpart)
    if EMIT_STATE:
        tl.store(gpart_ptr + (p * H + h) * D * D + d[:, None] * D + d[None, :], gram)
        tl.store(zwpart_ptr + (p * H + h) * D + d, zw)
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + h, 1, sem="acq_rel")
    if done == P - 1:
        full = tl.zeros((D, D), tl.float32)
        zs = tl.zeros((D,), tl.float32)
        for pp in tl.static_range(P):
            full += tl.load(
                spart_ptr + (pp * H + h) * D * D + d[:, None] * D + d[None, :],
                cache_modifier=".cg",
            )
            zs += tl.load(zpart_ptr + (pp * H + h) * D + d, cache_modifier=".cg")
        g = tl.load(decay_ptr + h)
        intra = full.to(tl.bfloat16).to(tl.float32)
        cs = tl.load(carry_s_ptr + h * D * D + d[:, None] * D + d[None, :])
        s_eff = (cs * g + intra).to(tl.bfloat16)
        tl.store(st_ptr + h * D * D + d[:, None] * D + d[None, :], s_eff)
        iz = zs.to(tl.bfloat16).to(tl.float32)
        cz = tl.load(carry_z_ptr + h * D + d)
        z_eff = (cz * g + iz).to(tl.bfloat16).to(tl.float32)
        tl.store(zeff_ptr + h * D + d, z_eff)
        if EMIT_STATE:
            gsum = tl.zeros((D, D), tl.float32)
            for pp in tl.static_range(P):
                gsum += tl.load(
                    gpart_ptr + (pp * H + h) * D * D + d[:, None] * D + d[None, :],
                    cache_modifier=".cg",
                )
            tl.store(gpart_ptr + h * D * D + d[:, None] * D + d[None, :], gsum)
            tl.debug_barrier()
            esum = tl.zeros((D, D), tl.float32)
            kb = tl.arange(0, FOLD_KB)
            for k0 in range(0, D, FOLD_KB):
                g_blk = tl.load(gpart_ptr + h * D * D + d[:, None] * D + (k0 + kb)[None, :], cache_modifier=".cg")
                s_blk = tl.load(carry_s_ptr + h * D * D + (k0 + kb)[:, None] * D + d[None, :]) * g
                esum = tl.dot(g_blk, s_blk, esum, input_precision="ieee")
            tl.store(s_new_ptr + h * D * D + d[:, None] * D + d[None, :], cs * g + full - esum)
            zw_sum = tl.zeros((D,), tl.float32)
            for pp in tl.static_range(P):
                zw_sum += tl.load(zwpart_ptr + (pp * H + h) * D + d, cache_modifier=".cg")
            tl.store(z_new_ptr + h * D + d, cz * g + zw_sum)


@triton.jit
def sana_gdn_causal_sm110_q_out(
    qkv_ptr,
    gate_ptr,
    rstd_q_ptr,
    cs32_ptr,
    q_w_ptr,
    o_w_ptr,
    st_ptr,
    zeff_ptr,
    out_ptr,
    N,
    stride_qkv_n,
    stride_gate_n,
    stride_out_n,
    norm_eps,
    z_eps,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NORMALIZER: tl.constexpr,
):
    """Third kernel, ``BLOCK_N`` tokens of one head per program: the queries normalized and rotated, their read of the state, the output RMSNorm and the sigmoid output gate. Values round to bf16 where the plain module rounds them."""
    pid = tl.program_id(0)
    h = pid % H
    t = pid // H
    d = tl.arange(0, D)
    d_sw = d ^ 1
    d_pair = 2 * (d // 2)
    sgn = tl.where((d % 2) == 0, -1.0, 1.0).to(tl.float32)
    rows = t * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N

    qw = tl.load(q_w_ptr + h * D + d).to(tl.float32)
    qw_sw = tl.load(q_w_ptr + h * D + d_sw).to(tl.float32)
    ow = tl.load(o_w_ptr + d).to(tl.float32)
    zeff = tl.load(zeff_ptr + h * D + d)
    rstd = tl.load(rstd_q_ptr + rows, mask=rmask, other=0.0)

    qrow = qkv_ptr + h * D + rows[:, None] * stride_qkv_n
    qf = tl.load(qrow + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    qf_sw = tl.load(qrow + d_sw[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    u = (qw[None, :] * (qf * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
    u_sw = (qw_sw[None, :] * (qf_sw * rstd[:, None])).to(tl.bfloat16).to(tl.float32)
    trow = cs32_ptr + rows[:, None] * D
    cosf = tl.load(trow + d_pair[None, :], mask=rmask[:, None], other=0.0)
    sinf = tl.load(trow + (d_pair + 1)[None, :], mask=rmask[:, None], other=0.0)
    qrot = (u * cosf + u_sw * (sgn[None, :] * sinf)).to(tl.bfloat16)
    st = tl.load(st_ptr + h * D * D + d[:, None] * D + d[None, :])
    num = tl.dot(qrot, st).to(tl.bfloat16).to(tl.float32)
    if NORMALIZER:
        qdc = tl.maximum(u, 0.0)
        den = tl.sum(qdc * zeff[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
        den = (den + z_eps).to(tl.bfloat16).to(tl.float32)
        o = tl.fdiv(num, den[:, None], ieee_rounding=True).to(tl.bfloat16).to(tl.float32)
    else:
        o = num
    r_o = tl.rsqrt(tl.sum(o * o, axis=1) / D + norm_eps)
    on = (ow[None, :] * (o * r_o[:, None])).to(tl.bfloat16).to(tl.float32)
    gl = tl.load(
        gate_ptr + rows[:, None] * stride_gate_n + h * D + d[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)
    g = (1.0 / (1.0 + tl.exp(-gl))).to(tl.bfloat16).to(tl.float32)
    y = (on * g).to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * stride_out_n + h * D + d[None, :],
        y,
        mask=rmask[:, None],
    )


_scratch_cache: dict = {}


def supports(device, channels: int, heads: int) -> bool:
    """Return whether the kernels serve Gated DeltaNet layers of ``channels`` in ``heads``: at most 32 heads of 128 and a multiple of 512 channels."""
    return channels == HEAD_DIM * heads and heads <= MAX_HEADS and channels % _BLOCK_AC == 0


def _get_scratch(N: int, C: int, H: int, D: int, dev: torch.device, cfg: GdnLaunchConfig) -> dict:
    """Return the scratch buffers and the launch grids of a chunk of N tokens under one launch configuration, built once per (N, C, H, D, device, configuration) and never dropped: a captured CUDA graph holds their addresses. Every call rewrites the buffers."""
    key = (N, C, H, D, str(dev), cfg)
    ent = _scratch_cache.get(key)
    if ent is not None:
        return ent
    Np = (N + 15) & ~15
    P = cfg.chunks_b
    rpc = triton.cdiv(triton.cdiv(N, P), cfg.block_b) * cfg.block_b
    ent = {
        "Np": Np,
        "rpc": rpc,
        "rstd_q": torch.empty(Np, dtype=torch.float32, device=dev),
        "rstd_k": torch.empty(Np, dtype=torch.float32, device=dev),
        "beta_s": torch.empty(H * Np, dtype=torch.float32, device=dev),
        "cs32": torch.empty(Np * D, dtype=torch.float32, device=dev),
        "spart": torch.empty(P * H * D * D, dtype=torch.float32, device=dev),
        "zpart": torch.empty(P * H * D, dtype=torch.float32, device=dev),
        "gpart": torch.empty(P * H * D * D, dtype=torch.float32, device=dev),
        "zwpart": torch.empty(P * H * D, dtype=torch.float32, device=dev),
        "cnt": torch.zeros(32, dtype=torch.int32, device=dev),
        "st": torch.empty(H * D * D, dtype=torch.bfloat16, device=dev),
        "zeff": torch.empty(H * D, dtype=torch.float32, device=dev),
        "carry_z": torch.zeros(H * D, dtype=torch.float32, device=dev),
        "z_new": torch.empty(H * D, dtype=torch.float32, device=dev),
        "grid_a": (triton.cdiv(N, cfg.block_a),),
        "grid_b": (P * H,),
        "grid_c": (triton.cdiv(N, cfg.block_c) * H,),
    }
    _scratch_cache[key] = ent
    return ent


def _launch(
    s: dict,
    out: torch.Tensor,
    qkv: torch.Tensor,
    beta_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    rotary_emb: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_norm_weight: torch.Tensor,
    carry_s: torch.Tensor,
    decay: torch.Tensor,
    key_scale: float,
    N: int,
    C: int,
    H: int,
    D: int,
    norm_eps: float,
    cfg: GdnLaunchConfig,
    s_new: torch.Tensor | None = None,
) -> None:
    """Launch the three kernels of one chunk on device tensors only; with ``s_new`` the second kernel also writes the state after the chunk."""
    rope_real = torch.view_as_real(rotary_emb).reshape(N, D // 2, 2)
    if rope_real.stride(2) != 1 or rope_real.stride(1) != 2:
        rope_real = rope_real.contiguous()
    emit = s_new is not None
    sana_gdn_causal_sm110_prep[s["grid_a"]](
        qkv,
        beta_logits,
        rope_real,
        s["rstd_q"],
        s["rstd_k"],
        s["beta_s"],
        s["cs32"],
        s["cnt"],
        N,
        s["Np"],
        qkv.stride(0),
        beta_logits.stride(0),
        rope_real.stride(0),
        norm_eps,
        H=H,
        D=D,
        C=C,
        BLOCK_N=cfg.block_a,
        BLOCK_C=_BLOCK_AC,
        num_warps=cfg.warps_a,
    )
    sana_gdn_causal_sm110_kv_state[s["grid_b"]](
        qkv,
        s["rstd_k"],
        s["beta_s"],
        s["cs32"],
        k_norm_weight,
        carry_s,
        s["carry_z"],
        decay,
        s["spart"],
        s["zpart"],
        s["cnt"],
        s["st"],
        s["zeff"],
        s["gpart"],
        s["zwpart"],
        s_new if emit else s["gpart"],
        s["z_new"],
        N,
        s["Np"],
        qkv.stride(0),
        s["rpc"],
        key_scale,
        H=H,
        D=D,
        C=C,
        P=cfg.chunks_b,
        BLOCK_N=cfg.block_b,
        EMIT_STATE=emit,
        FOLD_KB=_FOLD_KB,
        num_warps=cfg.warps_b_emit if emit else cfg.warps_b,
        num_stages=cfg.stages_b,
    )
    sana_gdn_causal_sm110_q_out[s["grid_c"]](
        qkv,
        gate_logits,
        s["rstd_q"],
        s["cs32"],
        q_norm_weight,
        o_norm_weight,
        s["st"],
        s["zeff"],
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
        BLOCK_N=cfg.block_c,
        NORMALIZER=False,
        num_warps=cfg.warps_c,
    )


def seeded_gdn_read(
    qkv: torch.Tensor,
    beta_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    rotary_emb: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_norm_weight: torch.Tensor,
    carry_s: torch.Tensor,
    decay: torch.Tensor,
    key_scale: float,
    num_heads: int,
    norm_eps: float,
) -> torch.Tensor:
    """Return the output [N, C] (bf16, before the output projection) of one chunk that reads the state ``carry_s`` [H, D, D] (float32) scaled by ``decay`` [H] together with its own tokens. ``qkv`` [N, 3C], ``beta_logits`` [N, H] and ``gate_logits`` [N, C] are bf16 projections of the chunk's tokens and ``rotary_emb`` their complex128 phases."""
    tokens, packed = qkv.shape
    channels = packed // 3
    head_dim = channels // num_heads
    cfg = READ_CONFIGS.get(tokens, DEFAULT_CONFIG)
    s = _get_scratch(tokens, channels, num_heads, head_dim, qkv.device, cfg)
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
        decay.reshape(-1).contiguous(),
        float(key_scale),
        tokens,
        channels,
        num_heads,
        head_dim,
        float(norm_eps),
        cfg,
    )
    return out


def seeded_gdn_read_commit(
    qkv: torch.Tensor,
    beta_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    rotary_emb: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    o_norm_weight: torch.Tensor,
    carry_s: torch.Tensor,
    decay: torch.Tensor,
    key_scale: float,
    num_heads: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the output of ``seeded_gdn_read`` and the state [H, D, D] (float32, a new tensor) after the chunk has written its tokens into the decayed ``carry_s`` by the delta rule."""
    tokens, packed = qkv.shape
    channels = packed // 3
    head_dim = channels // num_heads
    cfg = COMMIT_CONFIGS.get(tokens, DEFAULT_CONFIG)
    s = _get_scratch(tokens, channels, num_heads, head_dim, qkv.device, cfg)
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
        decay.reshape(-1).contiguous(),
        float(key_scale),
        tokens,
        channels,
        num_heads,
        head_dim,
        float(norm_eps),
        cfg,
        s_new=s_new,
    )
    return out, s_new
