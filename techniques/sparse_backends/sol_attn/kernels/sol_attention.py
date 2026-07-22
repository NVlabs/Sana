# Copyright (c) 2025-2026, Haopeng Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""qc-diag online sparse attention with a real threshold parameter.

This is copied from the previous qc-diag prototype, but removes the hard-coded
GLOBAL_TAU path. The Python API's ``thresh`` value is now passed to the
diag-threshold kernel as ``TAU`` and directly controls exact-block density.
"""

import functools
import math
import os

import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
        )

    return wrapper


GROUP_SIZE = 32


def _validate_tau(thresh):
    thresh = float(thresh)
    if not math.isfinite(thresh):
        raise ValueError(f"SOL Attention thresh/tau must be finite, got {thresh!r}")
    if 0.0 <= thresh < 1.0 and os.environ.get("SOL_ATTN_ALLOW_LOW_TAU", "0") != "1":
        raise ValueError(
            f"SOL Attention thresh/tau={thresh} looks like a target density, not a calibrated tau. "
            "Run calibrate_density and pass its threshold, or set SOL_ATTN_ALLOW_LOW_TAU=1 "
            "for an intentional debug sweep."
        )
    return thresh


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def quant_qk_reduce_kc_kernel(
    q_desc,
    k_desc,
    q_int8_desc,
    k_int8_desc,
    q_scale,
    k_scale,
    kc_int8,
    kc_scale,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    block_size = tl.minimum(BT, T - i_t * BT)

    b_q = q_desc.load([i_bh, i_t * BT, i_k * BK]).reshape([BT, BK])
    b_k = k_desc.load([i_bh, i_t * BT, i_k * BK]).reshape([BT, BK])

    bq_scale = tl.max(tl.abs(b_q)) / 127.0 + 1e-7
    bk_scale = tl.max(tl.abs(b_k)) / 127.0 + 1e-7

    b_q_int8 = b_q / bq_scale
    b_q_int8 += 0.5 * tl.where(b_q_int8 >= 0, 1, -1)
    b_q_int8 = b_q_int8.to(tl.int8)

    b_k_int8 = b_k / bk_scale
    b_k_int8 += 0.5 * tl.where(b_k_int8 >= 0, 1, -1)
    b_k_int8 = b_k_int8.to(tl.int8)

    b_kc = tl.sum(b_k, axis=0) / block_size
    bkc_scale = tl.max(tl.abs(b_kc)) / 127.0 + 1e-7
    b_kc_int8 = b_kc / bkc_scale
    b_kc_int8 += 0.5 * tl.where(b_kc_int8 >= 0, 1, -1)
    b_kc_int8 = b_kc_int8.to(tl.int8)

    p_kc_int8 = tl.make_block_ptr(
        kc_int8 + i_bh * N * K + i_t * K,
        (K,),
        (1,),
        (i_k * BK,),
        (BK,),
        (0,),
    )
    q_int8_desc.store([i_bh, i_t * BT, i_k * BK], b_q_int8[None, :, :])
    k_int8_desc.store([i_bh, i_t * BT, i_k * BK], b_k_int8[None, :, :])
    tl.store(q_scale + i_bh * N + i_t, bq_scale.to(tl.float32))
    tl.store(k_scale + i_bh * N + i_t, bk_scale.to(tl.float32))
    tl.store(kc_scale + i_bh * N + i_t, bkc_scale.to(tl.float32))
    tl.store(p_kc_int8, b_kc_int8, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2]
        for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def reduce_vc_kernel(
    v_desc,
    vc,
    N: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_v = v_desc.load([i_bh, i_t * BT, i_v * BV]).reshape([BT, BV])
    b_vc = tl.sum(b_v, axis=0)

    p_vc = tl.make_block_ptr(
        vc + i_bh * N * V + i_t * V,
        (V,),
        (1,),
        (i_v * BV,),
        (BV,),
        (0,),
    )
    tl.store(p_vc, b_vc, boundary_check=(0,))


def preprocess_qkv(q, k, v, block_size):
    B, H, T, K, V = *k.shape, v.shape[-1]
    N = triton.cdiv(T, block_size)

    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    q_int8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    k_int8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    kc_int8 = torch.empty(B, H, N, K, device=k.device, dtype=torch.int8)
    vc = torch.empty(B, H, N, V, device=v.device, dtype=v.dtype)

    q_scale = torch.empty(B, H, N, 1, device=q.device, dtype=torch.float32)
    k_scale = torch.empty(B, H, N, 1, device=k.device, dtype=torch.float32)
    kc_scale = torch.empty(B, H, N, 1, device=k.device, dtype=torch.float32)

    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T, K), [1, block_size, BK])
    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T, K), [1, block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T, V), [1, block_size, BV])

    q_int8_desc = TensorDescriptor.from_tensor(q_int8.reshape(B * H, T, K), [1, block_size, BK])
    k_int8_desc = TensorDescriptor.from_tensor(k_int8.reshape(B * H, T, K), [1, block_size, BK])

    grid_qk = (triton.cdiv(K, BK), N, B * H)
    quant_qk_reduce_kc_kernel[grid_qk](
        q_desc=q_desc,
        k_desc=k_desc,
        q_int8_desc=q_int8_desc,
        k_int8_desc=k_int8_desc,
        q_scale=q_scale,
        k_scale=k_scale,
        kc_int8=kc_int8,
        kc_scale=kc_scale,
        T=T,
        N=N,
        K=K,
        BT=block_size,
        BK=BK,
    )

    grid_v = (triton.cdiv(V, BV), N, B * H)
    reduce_vc_kernel[grid_v](
        v_desc=v_desc,
        vc=vc,
        N=N,
        V=V,
        BT=block_size,
        BV=BV,
    )

    return q_int8, q_scale, k_int8, k_scale, kc_int8, kc_scale, vc


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2]
    ],
    key=["N"],
)
@triton.jit
def reduce_kc_diag_stats_kernel(
    kc_desc,
    kc_scale,
    kc_mean,
    kc_var_diag,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)

    offsets = tl.arange(0, GROUP_SIZE)
    offsets = tl.max_contiguous(offsets, GROUP_SIZE)
    k_offsets = i_k * BK + tl.arange(0, BK)

    total = tl.zeros((BK,), dtype=tl.float32)
    total_sq = tl.zeros((BK,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)

    for start_n in range(0, N, GROUP_SIZE):
        chunk_indices = start_n + offsets
        valid = chunk_indices < N
        b_kc = kc_desc.load([i_bh, start_n, i_k * BK]).reshape([GROUP_SIZE, BK]).to(tl.float32)
        b_scale = tl.load(kc_scale + i_bh * N + chunk_indices, mask=valid, other=0.0)
        b_kc = b_kc * b_scale[:, None]
        b_kc = tl.where(valid[:, None], b_kc, 0.0)
        total += tl.sum(b_kc, axis=0)
        total_sq += tl.sum(b_kc * b_kc, axis=0)
        count += tl.sum(valid.to(tl.float32), axis=0)

    mean = total / count
    var = tl.maximum(total_sq / count - mean * mean, 0.0)
    mask = k_offsets < K
    tl.store(kc_mean + i_bh * K + k_offsets, mean, mask=mask)
    tl.store(kc_var_diag + i_bh * K + k_offsets, var, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def global_qck_diag_threshold_kernel(
    q_desc,
    q_scale,
    kc_mean,
    kc_var_diag,
    global_thresh,
    scale,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    TAU: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    q_start = i_t * BT
    block_size = tl.minimum(BT, T - q_start).to(tl.float32)

    k_offsets = tl.arange(0, BK)
    valid_k = k_offsets < K

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    b_qc = tl.sum(b_q.to(tl.float32), axis=0) / block_size
    b_q_scale = tl.load(q_scale + i_bh * N + i_t)

    b_mean = tl.load(kc_mean + i_bh * K + k_offsets, mask=valid_k, other=0.0)
    b_var = tl.load(kc_var_diag + i_bh * K + k_offsets, mask=valid_k, other=0.0)

    log2_scale = scale * 1.44269504
    mean = tl.sum(b_qc * b_mean, axis=0) * (b_q_scale * log2_scale)
    q_actual = b_qc * b_q_scale
    var = tl.sum(q_actual * q_actual * b_var, axis=0) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(var, 0.0) + 1e-6)
    tl.store(global_thresh + i_bh * N + i_t, mean + TAU * std)


def compute_global_qck_threshold(q_int8, q_scale, kc_int8, kc_scale, scale, block_size, tau):
    tau = _validate_tau(tau)
    B, H, T, K = q_int8.shape
    N = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    kc_mean = torch.empty(B, H, K, device=q_int8.device, dtype=torch.float32)
    kc_var_diag = torch.empty(B, H, K, device=q_int8.device, dtype=torch.float32)
    global_thresh = torch.empty(B, H, N, device=q_int8.device, dtype=torch.float32)
    q_desc = TensorDescriptor.from_tensor(q_int8.reshape(B * H, T, K), [1, block_size, BK])
    kc_desc = TensorDescriptor.from_tensor(kc_int8.reshape(B * H, N, K), [1, GROUP_SIZE, BK])

    reduce_kc_diag_stats_kernel[(triton.cdiv(K, BK), B * H)](
        kc_desc=kc_desc,
        kc_scale=kc_scale,
        kc_mean=kc_mean,
        kc_var_diag=kc_var_diag,
        N=N,
        K=K,
        BK=BK,
        GROUP_SIZE=GROUP_SIZE,
    )
    global_qck_diag_threshold_kernel[(N, B * H)](
        q_desc=q_desc,
        q_scale=q_scale,
        kc_mean=kc_mean,
        kc_var_diag=kc_var_diag,
        global_thresh=global_thresh,
        scale=scale,
        T=T,
        N=N,
        K=K,
        BT=block_size,
        BK=BK,
        TAU=float(tau),
    )
    return global_thresh


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def single_pass_dynamic_routing_kernel(
    q_desc,
    q_scale,
    k_desc,
    k_scale,
    v_desc,
    kc_desc,
    kc_scale,
    global_thresh,
    vc_desc,
    o_desc,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)

    token_offsets = tl.arange(0, BT)
    token_offsets = tl.max_contiguous(token_offsets, BT)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    b_q_scale = tl.load(q_scale + i_bh * NT + i_t)

    q_len = tl.minimum(BT, T - q_start).to(tl.float32)

    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")
    sm_scale = b_q_scale * scale * 1.44269504

    for start_n in range(0, NT, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)
        chunk_indices = start_n + chunk_offsets

        remaining = NT - start_n
        valid_mask = chunk_offsets < remaining

        b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
        b_kc_scale = tl.load(kc_scale + i_bh * NT + chunk_indices, mask=valid_mask, other=0.0)
        b_vc = vc_desc.load([i_bh, start_n, i_v * BV]).reshape([GROUP_SIZE, BV])

        b_s = tl.dot(b_q, b_kc.T).to(tl.float32)
        b_s = b_s * (sm_scale * b_kc_scale[None, :])

        col_mean = tl.sum(b_s, axis=0) / q_len
        thresh = tl.load(global_thresh + i_bh * NT + i_t)
        local_mask = tl.abs(i_t - chunk_indices) <= 1
        is_exact = ((col_mean > thresh) | local_mask) & valid_mask

        approx_mask = valid_mask & (~is_exact)
        b_s = tl.where(approx_mask[None, :], b_s, float("-inf"))
        has_approx = tl.sum(approx_mask.to(tl.int32), axis=0) > 0

        safe_b_s = tl.where(has_approx, b_s, 0.0)
        candidate_m = tl.maximum(m_i, tl.max(safe_b_s, axis=1))
        new_m = tl.where(has_approx, candidate_m, m_i)
        alpha = tl.math.exp2(tl.where(has_approx, m_i - new_m, 0.0))
        prob = tl.math.exp2(safe_b_s - tl.where(has_approx, new_m, 0.0)[:, None])
        prob = tl.where(has_approx, prob, 0.0)

        acc = acc * alpha[:, None] + tl.dot(prob.to(b_vc.dtype), b_vc)
        current_lens = tl.where(chunk_indices == NT - 1, T - (NT - 1) * BT, BT).to(tl.float32)
        l_i = l_i * alpha + tl.sum(prob * current_lens[None, :], axis=1)
        m_i = new_m

        exact_offsets = tl.where(is_exact, chunk_offsets, GROUP_SIZE)
        num_exact = tl.sum(is_exact.to(tl.int32))

        for _ in range(num_exact):
            next_offset = tl.min(exact_offsets)
            n_idx = start_n + next_offset
            exact_offsets = tl.where(chunk_offsets == next_offset, GROUP_SIZE, exact_offsets)
            kv_start = n_idx * BT
            tl.multiple_of(kv_start, BT)

            b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])
            b_k_scale = tl.load(k_scale + i_bh * NT + n_idx) * sm_scale

            b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32)
            b_s_exact = b_s_exact * b_k_scale

            valid_mask_ex = (kv_start + token_offsets)[None, :] < T
            b_s_exact += tl.where(valid_mask_ex, 0, float("-inf"))

            new_m = tl.maximum(m_i, tl.max(b_s_exact, axis=1))
            alpha = tl.math.exp2(m_i - new_m)
            prob_exact = tl.math.exp2(b_s_exact - new_m[:, None])

            l_i = l_i * alpha + tl.sum(prob_exact, axis=1)

            b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])
            exact_dot = tl.dot(prob_exact.to(b_v.dtype), b_v)
            acc = acc * alpha[:, None] + exact_dot
            m_i = new_m

    acc /= l_i[:, None]
    o_desc.store([i_bh, i_t * BT, i_v * BV], acc.to(tl.bfloat16)[None, :, :])


@contiguous
@torch.compiler.disable
def sol_attention(q, k, v, thresh=1.0, block_size=64, scale=None, density=None):
    if density is not None:
        raise ValueError(
            "sol_attention no longer accepts density= as an alias for thresh. "
            "SOL Attention density is controlled by runtime tau calibration: call calibrate_density(...), "
            "then pass cal['threshold'] as thresh."
        )
    thresh = _validate_tau(thresh)

    B, H, T, K, V = *k.shape, v.shape[-1]
    scale = K**-0.5 if scale is None else scale

    NT = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    q_int8, q_scale, k_int8, k_scale, kc_int8, kc_scale, vc = preprocess_qkv(q, k, v, block_size)
    global_thresh = compute_global_qck_threshold(q_int8, q_scale, kc_int8, kc_scale, scale, block_size, thresh)
    o = torch.empty_like(v)

    q_desc = TensorDescriptor.from_tensor(q_int8.reshape(B * H, T, K), [1, block_size, BK])
    k_desc = TensorDescriptor.from_tensor(k_int8.reshape(B * H, T, K), [1, block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T, V), [1, block_size, BV])
    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T, V), [1, block_size, BV])

    kc_desc = TensorDescriptor.from_tensor(kc_int8.reshape(B * H, NT, K), [1, GROUP_SIZE, BK])
    vc_desc = TensorDescriptor.from_tensor(vc.reshape(B * H, NT, V), [1, GROUP_SIZE, BV])

    grid = (triton.cdiv(V, BV), NT, B * H)
    single_pass_dynamic_routing_kernel[grid](
        q_desc=q_desc,
        q_scale=q_scale,
        k_desc=k_desc,
        k_scale=k_scale,
        v_desc=v_desc,
        kc_desc=kc_desc,
        kc_scale=kc_scale,
        global_thresh=global_thresh,
        vc_desc=vc_desc,
        o_desc=o_desc,
        scale=scale,
        T=T,
        K=K,
        V=V,
        BT=block_size,
        BK=BK,
        BV=BV,
        NT=NT,
        GROUP_SIZE=GROUP_SIZE,
    )

    return o
