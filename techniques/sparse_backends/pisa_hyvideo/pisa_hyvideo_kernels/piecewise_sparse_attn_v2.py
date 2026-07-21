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


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _resolve_group_size(group_size=None):
    group_size = _env_int("PISA2_GROUP_SIZE", GROUP_SIZE) if group_size is None else int(group_size)
    if group_size <= 0 or group_size & (group_size - 1):
        raise ValueError(f"PISA2 group_size must be a positive power of two, got {group_size}")
    return group_size


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def quant_q_kernel(
    q_desc,
    q_int8_desc,
    q_scale,
    T,
    N_Q: tl.constexpr,
    K: tl.constexpr,
    BT_Q: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_q = q_desc.load([i_bh, i_t * BT_Q, i_k * BK]).reshape([BT_Q, BK])
    bq_scale = tl.max(tl.abs(b_q)) / 127.0 + 1e-7

    b_q_int8 = b_q / bq_scale
    b_q_int8 += 0.5 * tl.where(b_q_int8 >= 0, 1, -1)
    b_q_int8 = b_q_int8.to(tl.int8)

    q_int8_desc.store([i_bh, i_t * BT_Q, i_k * BK], b_q_int8[None, :, :])
    tl.store(q_scale + i_bh * N_Q + i_t, bq_scale.to(tl.float32))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2]
    ],
    key=["T"],
)
@triton.jit
def quant_k_reduce_kc_kernel(
    k_desc,
    k_int8_desc,
    k_scale,
    kc_int8,
    kc_scale,
    T,
    N_KV: tl.constexpr,
    K: tl.constexpr,
    BT_KV: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    block_size = tl.minimum(BT_KV, T - i_t * BT_KV)
    b_k = k_desc.load([i_bh, i_t * BT_KV, i_k * BK]).reshape([BT_KV, BK])
    bk_scale = tl.max(tl.abs(b_k)) / 127.0 + 1e-7

    b_k_int8 = b_k / bk_scale
    b_k_int8 += 0.5 * tl.where(b_k_int8 >= 0, 1, -1)
    b_k_int8 = b_k_int8.to(tl.int8)

    b_kc = tl.sum(b_k, axis=0) / block_size
    bkc_scale = tl.max(tl.abs(b_kc)) / 127.0 + 1e-7
    b_kc_int8 = b_kc / bkc_scale
    b_kc_int8 += 0.5 * tl.where(b_kc_int8 >= 0, 1, -1)
    b_kc_int8 = b_kc_int8.to(tl.int8)

    p_kc_int8 = tl.make_block_ptr(
        kc_int8 + i_bh * N_KV * K + i_t * K,
        (K,),
        (1,),
        (i_k * BK,),
        (BK,),
        (0,),
    )
    k_int8_desc.store([i_bh, i_t * BT_KV, i_k * BK], b_k_int8[None, :, :])
    tl.store(k_scale + i_bh * N_KV + i_t, bk_scale.to(tl.float32))
    tl.store(kc_scale + i_bh * N_KV + i_t, bkc_scale.to(tl.float32))
    tl.store(p_kc_int8, b_kc_int8, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2]
        for num_stages in [2]
    ],
    key=["N_KV", "V", "BT_KV", "BV"],
)
@triton.jit
def reduce_vc_kernel(
    v_desc,
    vc,
    N_KV: tl.constexpr,
    V: tl.constexpr,
    BT_KV: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_v = v_desc.load([i_bh, i_t * BT_KV, i_v * BV]).reshape([BT_KV, BV])
    b_vc = tl.sum(b_v, axis=0)

    p_vc = tl.make_block_ptr(
        vc + i_bh * N_KV * V + i_t * V,
        (V,),
        (1,),
        (i_v * BV,),
        (BV,),
        (0,),
    )
    tl.store(p_vc, b_vc, boundary_check=(0,))


def preprocess_qkv(q, k, v, block_size=64, q_block_size=None, kv_block_size=None):
    q_block_size = block_size if q_block_size is None else q_block_size
    kv_block_size = block_size if kv_block_size is None else kv_block_size

    B, H, T_Q, K = q.shape
    T_KV = k.shape[-2]
    V = v.shape[-1]
    N_Q = triton.cdiv(T_Q, q_block_size)
    N_KV = triton.cdiv(T_KV, kv_block_size)

    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    q_int8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    k_int8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    kc_int8 = torch.empty(B, H, N_KV, K, device=k.device, dtype=torch.int8)
    vc = torch.empty(B, H, N_KV, V, device=v.device, dtype=v.dtype)

    q_scale = torch.empty(B, H, N_Q, 1, device=q.device, dtype=torch.float32)
    k_scale = torch.empty(B, H, N_KV, 1, device=k.device, dtype=torch.float32)
    kc_scale = torch.empty(B, H, N_KV, 1, device=k.device, dtype=torch.float32)

    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T_Q, K), [1, q_block_size, BK])
    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T_KV, K), [1, kv_block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, kv_block_size, BV])

    q_int8_desc = TensorDescriptor.from_tensor(q_int8.reshape(B * H, T_Q, K), [1, q_block_size, BK])
    k_int8_desc = TensorDescriptor.from_tensor(k_int8.reshape(B * H, T_KV, K), [1, kv_block_size, BK])

    grid_q = (triton.cdiv(K, BK), N_Q, B * H)
    quant_q_kernel[grid_q](
        q_desc=q_desc,
        q_int8_desc=q_int8_desc,
        q_scale=q_scale,
        T=T_Q,
        N_Q=N_Q,
        K=K,
        BT_Q=q_block_size,
        BK=BK,
    )

    grid_k = (triton.cdiv(K, BK), N_KV, B * H)
    quant_k_reduce_kc_kernel[grid_k](
        k_desc=k_desc,
        k_int8_desc=k_int8_desc,
        k_scale=k_scale,
        kc_int8=kc_int8,
        kc_scale=kc_scale,
        T=T_KV,
        N_KV=N_KV,
        K=K,
        BT_KV=kv_block_size,
        BK=BK,
    )

    grid_v = (triton.cdiv(V, BV), N_KV, B * H)
    reduce_vc_kernel[grid_v](
        v_desc=v_desc,
        vc=vc,
        N_KV=N_KV,
        V=V,
        BT_KV=kv_block_size,
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
    T_Q,
    N_Q: tl.constexpr,
    K: tl.constexpr,
    BT_Q: tl.constexpr,
    BK: tl.constexpr,
    TAU: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    q_start = i_t * BT_Q
    block_size = tl.minimum(BT_Q, T_Q - q_start).to(tl.float32)

    k_offsets = tl.arange(0, BK)
    valid_k = k_offsets < K

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT_Q, BK])
    b_qc = tl.sum(b_q.to(tl.float32), axis=0) / block_size
    b_q_scale = tl.load(q_scale + i_bh * N_Q + i_t)

    b_mean = tl.load(kc_mean + i_bh * K + k_offsets, mask=valid_k, other=0.0)
    b_var = tl.load(kc_var_diag + i_bh * K + k_offsets, mask=valid_k, other=0.0)

    log2_scale = scale * 1.44269504
    mean = tl.sum(b_qc * b_mean, axis=0) * (b_q_scale * log2_scale)
    q_actual = b_qc * b_q_scale
    var = tl.sum(q_actual * q_actual * b_var, axis=0) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(var, 0.0) + 1e-6)
    tl.store(global_thresh + i_bh * N_Q + i_t, mean + TAU * std)


def compute_global_qck_threshold(
    q_int8,
    q_scale,
    kc_int8,
    kc_scale,
    scale,
    block_size=64,
    tau=1.0,
    q_block_size=None,
    group_size=None,
):
    q_block_size = block_size if q_block_size is None else q_block_size
    group_size = _resolve_group_size(group_size)
    B, H, T_Q, K = q_int8.shape
    N_Q = q_scale.shape[2]
    N_KV = kc_int8.shape[2]
    BK = min(128, triton.next_power_of_2(K))
    kc_mean = torch.empty(B, H, K, device=q_int8.device, dtype=torch.float32)
    kc_var_diag = torch.empty(B, H, K, device=q_int8.device, dtype=torch.float32)
    global_thresh = torch.empty(B, H, N_Q, device=q_int8.device, dtype=torch.float32)
    q_desc = TensorDescriptor.from_tensor(q_int8.reshape(B * H, T_Q, K), [1, q_block_size, BK])
    kc_desc = TensorDescriptor.from_tensor(kc_int8.reshape(B * H, N_KV, K), [1, group_size, BK])

    reduce_kc_diag_stats_kernel[(triton.cdiv(K, BK), B * H)](
        kc_desc=kc_desc,
        kc_scale=kc_scale,
        kc_mean=kc_mean,
        kc_var_diag=kc_var_diag,
        N=N_KV,
        K=K,
        BK=BK,
        GROUP_SIZE=group_size,
    )
    global_qck_diag_threshold_kernel[(N_Q, B * H)](
        q_desc=q_desc,
        q_scale=q_scale,
        kc_mean=kc_mean,
        kc_var_diag=kc_var_diag,
        global_thresh=global_thresh,
        scale=scale,
        T_Q=T_Q,
        N_Q=N_Q,
        K=K,
        BT_Q=q_block_size,
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
    key=["T_Q", "T_KV", "K", "V", "BT_Q", "BT_KV", "BK", "BV", "NT_Q", "NT_KV"],
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
    T_Q,
    T_KV,
    Q_START_OFFSET: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT_Q: tl.constexpr,
    BT_KV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SINK_BLOCKS: tl.constexpr,
    SINK_HISTORY: tl.constexpr,
    SINK_NEAR_TOKENS: tl.constexpr,
    SINK_STRIDE_TOKENS: tl.constexpr,
    SINK_STRIDE_WIDTH_TOKENS: tl.constexpr,
    ROUTE_SELECTED_ONLY: tl.constexpr,
    ROUTE_SINK_BLOCKS: tl.constexpr,
    ROUTE_NEAR_TOKENS: tl.constexpr,
    ROUTE_STRIDE_TOKENS: tl.constexpr,
    ROUTE_STRIDE_WIDTH_TOKENS: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)

    q_token_offsets = tl.arange(0, BT_Q)
    q_token_offsets = tl.max_contiguous(q_token_offsets, BT_Q)
    kv_token_offsets = tl.arange(0, BT_KV)
    kv_token_offsets = tl.max_contiguous(kv_token_offsets, BT_KV)

    q_start = i_t * BT_Q
    tl.multiple_of(q_start, BT_Q)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT_Q, BK])
    b_q_scale = tl.load(q_scale + i_bh * NT_Q + i_t)

    q_valid_len = tl.minimum(BT_Q, T_Q - q_start)
    q_len = q_valid_len.to(tl.float32)
    q_abs_start = Q_START_OFFSET + q_start
    q_abs_end = q_abs_start + q_valid_len

    acc = tl.zeros([BT_Q, BV], dtype=tl.float32)
    l_i = tl.zeros((BT_Q,), dtype=tl.float32)
    m_i = tl.zeros((BT_Q,), dtype=tl.float32) - float("inf")
    sm_scale = b_q_scale * scale * 1.44269504
    inv_q_len = 1.0 / q_len

    for start_n in range(0, NT_KV, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)
        chunk_indices = start_n + chunk_offsets

        remaining = NT_KV - start_n
        valid_mask = chunk_offsets < remaining
        kv_start_tokens = chunk_indices * BT_KV
        kv_end_tokens = kv_start_tokens + BT_KV
        local_mask = (kv_start_tokens < q_abs_end + BT_KV) & (kv_end_tokens > q_abs_start - BT_KV)
        sink_mask = chunk_indices < SINK_BLOCKS
        history_sink_mask = SINK_HISTORY & (kv_start_tokens < Q_START_OFFSET)
        near_start = tl.maximum(Q_START_OFFSET - SINK_NEAR_TOKENS, 0)
        near_sink_mask = (SINK_NEAR_TOKENS > 0) & (kv_start_tokens < Q_START_OFFSET) & (kv_end_tokens > near_start)
        stride_sink_mask = (SINK_STRIDE_TOKENS > 0) & (
            (kv_start_tokens % SINK_STRIDE_TOKENS) < SINK_STRIDE_WIDTH_TOKENS
        )
        force_exact_mask = local_mask | sink_mask | history_sink_mask | near_sink_mask | stride_sink_mask

        route_sink_mask = chunk_indices < ROUTE_SINK_BLOCKS
        route_near_start = tl.maximum(Q_START_OFFSET - ROUTE_NEAR_TOKENS, 0)
        route_near_mask = (ROUTE_NEAR_TOKENS > 0) & (kv_start_tokens < Q_START_OFFSET) & (kv_end_tokens > route_near_start)
        route_stride_mask = (ROUTE_STRIDE_TOKENS > 0) & (
            (kv_start_tokens % ROUTE_STRIDE_TOKENS) < ROUTE_STRIDE_WIDTH_TOKENS
        )
        selected_route_mask = route_sink_mask | route_near_mask | route_stride_mask
        if ROUTE_SELECTED_ONLY:
            route_mask = selected_route_mask
        else:
            route_mask = valid_mask
        candidate_mask = valid_mask & (force_exact_mask | route_mask)
        has_candidate = tl.sum(candidate_mask.to(tl.int32), axis=0) > 0

        if has_candidate:
            b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
            b_kc_scale = tl.load(kc_scale + i_bh * NT_KV + chunk_indices, mask=valid_mask, other=0.0)
            b_vc = vc_desc.load([i_bh, start_n, i_v * BV]).reshape([GROUP_SIZE, BV])

            b_s = tl.dot(b_q, b_kc.T).to(tl.float32)
            b_s = b_s * (sm_scale * b_kc_scale[None, :])

            col_mean = tl.sum(b_s, axis=0) * inv_q_len
            thresh = tl.load(global_thresh + i_bh * NT_Q + i_t)
            is_exact = (((col_mean > thresh) & route_mask) | force_exact_mask) & valid_mask

            approx_mask = valid_mask & route_mask & (~is_exact)
            b_s = tl.where(approx_mask[None, :], b_s, float("-inf"))
            has_approx = tl.sum(approx_mask.to(tl.int32), axis=0) > 0

            safe_b_s = tl.where(has_approx, b_s, 0.0)
            candidate_m = tl.maximum(m_i, tl.max(safe_b_s, axis=1))
            new_m = tl.where(has_approx, candidate_m, m_i)
            alpha = tl.math.exp2(tl.where(has_approx, m_i - new_m, 0.0))
            prob = tl.math.exp2(safe_b_s - tl.where(has_approx, new_m, 0.0)[:, None])
            prob = tl.where(has_approx, prob, 0.0)

            acc = acc * alpha[:, None] + tl.dot(prob.to(b_vc.dtype), b_vc)
            current_lens = tl.minimum(BT_KV, tl.maximum(0, T_KV - chunk_indices * BT_KV)).to(tl.float32)
            l_i = l_i * alpha + tl.sum(prob * current_lens[None, :], axis=1)
            m_i = new_m

            exact_offsets = tl.where(is_exact, chunk_offsets, GROUP_SIZE)
            num_exact = tl.sum(is_exact.to(tl.int32))

            for _ in range(num_exact):
                next_offset = tl.min(exact_offsets)
                n_idx = start_n + next_offset
                exact_offsets = tl.where(chunk_offsets == next_offset, GROUP_SIZE, exact_offsets)
                kv_start = n_idx * BT_KV
                tl.multiple_of(kv_start, BT_KV)

                b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT_KV, BK])
                b_k_scale = tl.load(k_scale + i_bh * NT_KV + n_idx) * sm_scale

                b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32)
                b_s_exact = b_s_exact * b_k_scale

                valid_mask_ex = (kv_start + kv_token_offsets)[None, :] < T_KV
                b_s_exact += tl.where(valid_mask_ex, 0, float("-inf"))

                new_m = tl.maximum(m_i, tl.max(b_s_exact, axis=1))
                alpha = tl.math.exp2(m_i - new_m)
                prob_exact = tl.math.exp2(b_s_exact - new_m[:, None])

                l_i = l_i * alpha + tl.sum(prob_exact, axis=1)

                b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT_KV, BV])
                exact_dot = tl.dot(prob_exact.to(b_v.dtype), b_v)
                acc = acc * alpha[:, None] + exact_dot
                m_i = new_m

    acc /= l_i[:, None]
    acc = tl.where((q_start + q_token_offsets)[:, None] < T_Q, acc, 0.0)
    o_desc.store([i_bh, i_t * BT_Q, i_v * BV], acc.to(tl.bfloat16)[None, :, :])


@triton.jit
def single_pass_dynamic_routing_fast_kernel(
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
    T_Q,
    T_KV,
    Q_START_OFFSET: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT_Q: tl.constexpr,
    BT_KV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)
    q_token_offsets = tl.arange(0, BT_Q)
    q_token_offsets = tl.max_contiguous(q_token_offsets, BT_Q)
    kv_token_offsets = tl.arange(0, BT_KV)
    kv_token_offsets = tl.max_contiguous(kv_token_offsets, BT_KV)

    q_start = i_t * BT_Q
    tl.multiple_of(q_start, BT_Q)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT_Q, BK])
    b_q_scale = tl.load(q_scale + i_bh * NT_Q + i_t)

    q_valid_len = tl.minimum(BT_Q, T_Q - q_start)
    q_len = q_valid_len.to(tl.float32)
    q_abs_start = Q_START_OFFSET + q_start
    q_abs_end = q_abs_start + q_valid_len

    acc = tl.zeros([BT_Q, BV], dtype=tl.float32)
    l_i = tl.zeros((BT_Q,), dtype=tl.float32)
    m_i = tl.zeros((BT_Q,), dtype=tl.float32) - float("inf")
    sm_scale = b_q_scale * scale * 1.44269504
    inv_q_len = 1.0 / q_len

    for start_n in range(0, NT_KV, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)
        chunk_indices = start_n + chunk_offsets
        remaining = NT_KV - start_n
        valid_mask = chunk_offsets < remaining

        b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
        b_kc_scale = tl.load(kc_scale + i_bh * NT_KV + chunk_indices, mask=valid_mask, other=0.0)
        b_vc = vc_desc.load([i_bh, start_n, i_v * BV]).reshape([GROUP_SIZE, BV])

        b_s = tl.dot(b_q, b_kc.T).to(tl.float32)
        b_s = b_s * (sm_scale * b_kc_scale[None, :])

        col_mean = tl.sum(b_s, axis=0) * inv_q_len
        thresh = tl.load(global_thresh + i_bh * NT_Q + i_t)
        kv_start_tokens = chunk_indices * BT_KV
        kv_end_tokens = kv_start_tokens + BT_KV
        local_mask = (kv_start_tokens < q_abs_end + BT_KV) & (kv_end_tokens > q_abs_start - BT_KV)
        is_exact = ((col_mean > thresh) | local_mask) & valid_mask

        approx_mask = valid_mask & (~is_exact)
        b_s = tl.where(approx_mask[None, :], b_s, -1.0e20)
        new_m = tl.maximum(m_i, tl.max(b_s, axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        prob = tl.math.exp2(b_s - new_m[:, None])
        prob = tl.where(approx_mask[None, :], prob, 0.0)

        acc = acc * alpha[:, None] + tl.dot(prob.to(b_vc.dtype), b_vc)
        current_lens = tl.minimum(BT_KV, tl.maximum(0, T_KV - chunk_indices * BT_KV)).to(tl.float32)
        l_i = l_i * alpha + tl.sum(prob * current_lens[None, :], axis=1)
        m_i = new_m

        exact_offsets = tl.where(is_exact, chunk_offsets, GROUP_SIZE)
        num_exact = tl.sum(is_exact.to(tl.int32))

        for _ in range(num_exact):
            next_offset = tl.min(exact_offsets)
            n_idx = start_n + next_offset
            exact_offsets = tl.where(chunk_offsets == next_offset, GROUP_SIZE, exact_offsets)
            kv_start = n_idx * BT_KV
            tl.multiple_of(kv_start, BT_KV)

            b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT_KV, BK])
            b_k_scale = tl.load(k_scale + i_bh * NT_KV + n_idx) * sm_scale

            b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32)
            b_s_exact = b_s_exact * b_k_scale

            valid_mask_ex = (kv_start + kv_token_offsets)[None, :] < T_KV
            b_s_exact += tl.where(valid_mask_ex, 0, float("-inf"))

            new_m = tl.maximum(m_i, tl.max(b_s_exact, axis=1))
            alpha = tl.math.exp2(m_i - new_m)
            prob_exact = tl.math.exp2(b_s_exact - new_m[:, None])

            l_i = l_i * alpha + tl.sum(prob_exact, axis=1)

            b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT_KV, BV])
            exact_dot = tl.dot(prob_exact.to(b_v.dtype), b_v)
            acc = acc * alpha[:, None] + exact_dot
            m_i = new_m

    acc /= l_i[:, None]
    acc = tl.where((q_start + q_token_offsets)[:, None] < T_Q, acc, 0.0)
    o_desc.store([i_bh, i_t * BT_Q, i_v * BV], acc.to(tl.bfloat16)[None, :, :])


@contiguous
@torch.compiler.disable
def online_piecewise_sparse_attention(
    q,
    k,
    v,
    thresh=1.0,
    block_size=64,
    scale=None,
    density=None,
    q_block_size=None,
    kv_block_size=None,
    q_start_offset=None,
    sink_blocks=0,
    sink_tokens=0,
    sink_history=False,
    sink_near_tokens=0,
    sink_near_frames=0,
    frame_tokens=0,
    sink_stride_tokens=0,
    sink_stride_width_tokens=0,
    route_sink_blocks=0,
    route_sink_tokens=0,
    route_near_tokens=0,
    route_near_frames=0,
    route_stride_tokens=0,
    route_stride_width_tokens=0,
    group_size=None,
    num_warps=None,
    num_stages=None,
):
    if density is not None:
        thresh = density

    q_block_size = block_size if q_block_size is None else q_block_size
    kv_block_size = block_size if kv_block_size is None else kv_block_size
    group_size = _resolve_group_size(group_size)
    num_warps = _env_int("PISA2_NUM_WARPS", 4) if num_warps is None else int(num_warps)
    num_stages = _env_int("PISA2_NUM_STAGES", 2) if num_stages is None else int(num_stages)

    B, H, T_Q, K = q.shape
    T_KV = k.shape[-2]
    V = v.shape[-1]
    q_start_offset = T_KV - T_Q if q_start_offset is None else q_start_offset
    sink_blocks = max(int(sink_blocks), 0)
    if sink_tokens:
        sink_blocks = max(sink_blocks, triton.cdiv(int(sink_tokens), kv_block_size))
    sink_near_tokens = max(int(sink_near_tokens), 0)
    sink_near_frames = max(int(sink_near_frames), 0)
    frame_tokens = max(int(frame_tokens), 0)
    if sink_near_frames:
        if frame_tokens == 0:
            raise ValueError("frame_tokens must be positive when sink_near_frames is set")
        sink_near_tokens = max(sink_near_tokens, sink_near_frames * frame_tokens)
    sink_stride_tokens = max(int(sink_stride_tokens), 0)
    sink_stride_width_tokens = max(int(sink_stride_width_tokens), 0)
    if sink_stride_width_tokens and sink_stride_tokens == 0:
        raise ValueError("sink_stride_tokens must be positive when sink_stride_width_tokens is set")
    route_sink_blocks = max(int(route_sink_blocks), 0)
    if route_sink_tokens:
        route_sink_blocks = max(route_sink_blocks, triton.cdiv(int(route_sink_tokens), kv_block_size))
    route_near_tokens = max(int(route_near_tokens), 0)
    route_near_frames = max(int(route_near_frames), 0)
    if route_near_frames:
        if frame_tokens == 0:
            raise ValueError("frame_tokens must be positive when route_near_frames is set")
        route_near_tokens = max(route_near_tokens, route_near_frames * frame_tokens)
    route_stride_tokens = max(int(route_stride_tokens), 0)
    route_stride_width_tokens = max(int(route_stride_width_tokens), 0)
    if route_stride_width_tokens and route_stride_tokens == 0:
        raise ValueError("route_stride_tokens must be positive when route_stride_width_tokens is set")
    route_selected_only = bool(route_sink_blocks or route_near_tokens or route_stride_width_tokens)
    scale = K**-0.5 if scale is None else scale

    NT_Q = triton.cdiv(T_Q, q_block_size)
    NT_KV = triton.cdiv(T_KV, kv_block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    q_int8, q_scale, k_int8, k_scale, kc_int8, kc_scale, vc = preprocess_qkv(
        q,
        k,
        v,
        block_size=block_size,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    global_thresh = compute_global_qck_threshold(
        q_int8,
        q_scale,
        kc_int8,
        kc_scale,
        scale,
        block_size=block_size,
        tau=thresh,
        q_block_size=q_block_size,
        group_size=group_size,
    )
    o = torch.empty(B, H, T_Q, V, device=q.device, dtype=v.dtype)

    q_desc = TensorDescriptor.from_tensor(q_int8.reshape(B * H, T_Q, K), [1, q_block_size, BK])
    k_desc = TensorDescriptor.from_tensor(k_int8.reshape(B * H, T_KV, K), [1, kv_block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, kv_block_size, BV])
    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T_Q, V), [1, q_block_size, BV])

    kc_desc = TensorDescriptor.from_tensor(kc_int8.reshape(B * H, NT_KV, K), [1, group_size, BK])
    vc_desc = TensorDescriptor.from_tensor(vc.reshape(B * H, NT_KV, V), [1, group_size, BV])

    grid = (triton.cdiv(V, BV), NT_Q, B * H)
    use_fast_path = not (
        sink_blocks
        or sink_history
        or sink_near_tokens
        or sink_stride_width_tokens
        or route_selected_only
    )
    if use_fast_path:
        single_pass_dynamic_routing_fast_kernel[grid](
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
            T_Q=T_Q,
            T_KV=T_KV,
            Q_START_OFFSET=int(q_start_offset),
            K=K,
            V=V,
            BT_Q=q_block_size,
            BT_KV=kv_block_size,
            BK=BK,
            BV=BV,
            NT_Q=NT_Q,
            NT_KV=NT_KV,
            GROUP_SIZE=group_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
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
            T_Q=T_Q,
            T_KV=T_KV,
            Q_START_OFFSET=int(q_start_offset),
            K=K,
            V=V,
            BT_Q=q_block_size,
            BT_KV=kv_block_size,
            BK=BK,
            BV=BV,
            NT_Q=NT_Q,
            NT_KV=NT_KV,
            GROUP_SIZE=group_size,
            SINK_BLOCKS=sink_blocks,
            SINK_HISTORY=bool(sink_history),
            SINK_NEAR_TOKENS=sink_near_tokens,
            SINK_STRIDE_TOKENS=sink_stride_tokens,
            SINK_STRIDE_WIDTH_TOKENS=sink_stride_width_tokens,
            ROUTE_SELECTED_ONLY=route_selected_only,
            ROUTE_SINK_BLOCKS=route_sink_blocks,
            ROUTE_NEAR_TOKENS=route_near_tokens,
            ROUTE_STRIDE_TOKENS=route_stride_tokens,
            ROUTE_STRIDE_WIDTH_TOKENS=route_stride_width_tokens,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return o
