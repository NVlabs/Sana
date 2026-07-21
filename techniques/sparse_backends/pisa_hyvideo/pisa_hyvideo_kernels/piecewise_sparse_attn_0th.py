# Copyright (c) 2025-2026, Haopeng Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor


def _make_tma_allocator():
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    return alloc_fn


def build_block_map(indices, nt_kv):
    block_map = torch.zeros(*indices.shape[:-1], nt_kv, device=indices.device, dtype=torch.int8)
    block_map.scatter_(-1, indices.to(torch.long), 1)
    return block_map.contiguous()


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"],
)
@triton.jit
def chunk_reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    k_var,   # [B*H, N]
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))

    b_k = p_k.load([i_t * BT, 0])
    b_v = p_v.load([i_t * BT, 0])

    b_kc = tl.sum(b_k, axis=0) / block_size
    b_vc = tl.sum(b_v, axis=0)

    # Var(K_j) = E[||k||^2] - ||E[k]||^2
    mean_norm = tl.sum(b_k * b_k) / block_size

    kc_norm = tl.sum(b_kc * b_kc, axis=0)
    b_k_var = tl.maximum(mean_norm - kc_norm, 0.0)

    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (0,), (BV,), (0,))

    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))
    tl.store(k_var + i_bh * N + i_t, b_k_var)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"],
)
@triton.jit
def chunk_reduce_q_kernel(
    q,
    qc,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_q = p_q.load([i_t * BT, 0])
    
    b_qc = tl.sum(b_q, axis=0) / block_size

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))


def chunk_reduce_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]

    N_Q = triton.cdiv(T_Q, block_size)
    N_KV = triton.cdiv(T_KV, block_size)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    qc = torch.empty(B, H, N_Q, K, device=q.device, dtype=q.dtype)
    kc = torch.empty(B, H, N_KV, K, device=k.device, dtype=k.dtype)
    vc = torch.empty(B, H, N_KV, V, device=v.device, dtype=v.dtype)

    # scalar variance proxy per KV block
    k_var = torch.empty(B, H, N_KV, device=k.device, dtype=k.dtype)

    chunk_reduce_q_kernel[(N_Q, B * H)](
        q=q,
        qc=qc,
        T=T_Q,
        N=N_Q,
        K=K,
        BT=block_size,
        BK=BK,
    )

    chunk_reduce_kv_kernel[(N_KV, B * H)](
        k=k,
        v=v,
        kc=kc,
        vc=vc,
        k_var=k_var,
        T=T_KV,
        N=N_KV,
        K=K,
        V=V,
        BT=block_size,
        BK=BK,
        BV=BV,
    )

    return qc, kc, vc, k_var


@triton.autotune(
    configs=[
        triton.Config({"GROUP_SIZE": GROUP_SIZE}, num_warps=num_warps, num_stages=num_stages)
        for GROUP_SIZE in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"],
)
@triton.jit
def piecewise_attn_fwd_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    kc,
    vc,
    lse,
    indices,
    scale,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    token_offsets = tl.arange(0, BT)
    token_offsets = tl.max_contiguous(token_offsets, BT)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    
    sm_scale = scale * 1.44269504
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        kv_start = i_n * BT
        tl.multiple_of(kv_start, BT)

        b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])

        b_s = tl.dot(b_q, b_k.T) * sm_scale
        b_s += tl.where((kv_start + token_offsets)[None, :] < T_KV, 0, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(b_s,  axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        score = tl.math.exp2(b_s - new_m[:, None])

        b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])

        l_i = l_i * alpha + tl.sum(score, axis=1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m

    B_NS: tl.constexpr = triton.next_power_of_2(NS)
    offs_n_idx = tl.arange(0, B_NS)
    selected = tl.load(
        indices + i_bh * NT_Q * NS + i_t * NS + offs_n_idx,
        mask=offs_n_idx < NS,
        other=-1,
    )

    for start_n in range(0, NT_KV, GROUP_SIZE):
        p_kc = tl.make_tensor_descriptor(kc + i_bh * NT_KV * K, (NT_KV, K), (K, 1), (GROUP_SIZE, BK))
        b_kc = p_kc.load([start_n, 0])
        
        chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
        is_selected = chunk_indices[:, None] == selected[None, :]
        selected_mask = tl.max(is_selected, axis=1)
        valid_mask = (chunk_indices < NT_KV) & (selected_mask == 0)

        current_lens = tl.minimum(BT, tl.maximum(0, T_KV - chunk_indices * BT)).to(tl.float32)
        
        b_s_mean = tl.dot(b_q, b_kc.T) * sm_scale
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(b_s_mean, axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        
        prob_chunk = tl.math.exp2(b_s_mean - new_m[:, None])

        p_vc = tl.make_tensor_descriptor(vc + i_bh * NT_KV * V, (NT_KV, V), (V, 1), (GROUP_SIZE, BV))
        b_vc = p_vc.load([start_n, i_v * BV])

        acc = acc * alpha[:, None] + tl.dot(prob_chunk.to(b_vc.dtype), b_vc)
        l_i = l_i * alpha + tl.sum(prob_chunk * current_lens[None, :], axis=1)
        m_i = new_m

    acc = acc / l_i[:, None]
    m_i += tl.math.log2(l_i)

    p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
    tl.store(p_lse, m_i, boundary_check=(0,))

    o_desc.store([i_bh, q_start, i_v * BV], acc[None, :, :])


@triton.jit
def attn_bwd_preprocess(
    o_desc,
    do_desc,
    delta,
    T,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    t_start = i_t * BT
    tl.multiple_of(t_start, BT)

    b_o = o_desc.load([i_bh, t_start, 0]).reshape([BT, BV])
    b_do = do_desc.load([i_bh, t_start, 0]).reshape([BT, BV])
    b_delta = tl.sum(b_o.to(tl.float32) * b_do.to(tl.float32), axis=1)

    p_delta = tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (t_start,), (BT,), (0,))
    tl.store(p_delta, b_delta, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"GROUP_SIZE": GROUP_SIZE}, num_warps=num_warps, num_stages=num_stages)
        for GROUP_SIZE in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=["T_Q", "T_KV", "K", "V", "BT", "NS"],
)
@triton.jit
def piecewise_attn_bwd_dq_kernel(
    do_desc,
    q_desc,
    k_desc,
    v_desc,
    kc,
    vc,
    lse,
    delta,
    dq,
    indices,
    block_map,
    scale,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q,
    NT_KV,
    NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    b_do = do_desc.load([i_bh, q_start, i_v * BV]).reshape([BT, BV])
    
    p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))

    b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
    b_D = tl.load(p_delta, boundary_check=(0,), padding_option="zero")
    
    sm_scale = scale * 1.44269504
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    offs_bt = tl.arange(0, BT)
    offs_bt = tl.max_contiguous(offs_bt, BT)

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT
        tl.multiple_of(bos, BT)
        offs_n = bos + offs_bt

        b_k = k_desc.load([i_bh, bos, 0]).reshape([BT, BK])
        b_v = v_desc.load([i_bh, bos, i_v * BV]).reshape([BT, BV])

        b_s = tl.dot(b_q, tl.trans(b_k)) * sm_scale
        b_s += tl.where(offs_n[None, :] < T_KV, 0, float("-inf"))
        b_p = tl.math.exp2(b_s - b_lse[:, None])

        b_term1 = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_term1 - b_D[:, None])

        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k) * scale

    p_kc = tl.make_tensor_descriptor(kc + i_bh * NT_KV * K, (NT_KV, K), (K, 1), (GROUP_SIZE, BK))
    p_vc = tl.make_tensor_descriptor(vc + i_bh * NT_KV * V, (NT_KV, V), (V, 1), (GROUP_SIZE, BV))

    for start_n in range(0, NT_KV, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)

        chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
        b_kc = p_kc.load([start_n, 0])
        selected = tl.load(
            block_map + (i_bh * NT_Q + i_t) * NT_KV + chunk_indices,
            mask=chunk_indices < NT_KV,
            other=1,
        )
        valid_mask = (chunk_indices < NT_KV) & (selected == 0)

        current_lens = tl.minimum(BT, tl.maximum(0, T_KV - chunk_indices * BT)).to(tl.float32)
        b_s_mean = tl.dot(b_q, tl.trans(b_kc)) * sm_scale
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))
        b_p = tl.math.exp2(b_s_mean - b_lse[:, None])

        b_vc = p_vc.load([start_n, i_v * BV])

        b_term1 = tl.dot(b_do, tl.trans(b_vc))
        b_ds = b_p * (b_term1 - b_D[:, None] * current_lens[None, :])

        b_dq += tl.dot(b_ds.to(b_kc.dtype), b_kc) * scale

    offs_q_n = tl.arange(0, BK)
    offs_q = q_start + tl.arange(0, BT)
    ptr_dq = dq + (i_bh * T_Q * K + offs_q[:, None] * K + offs_q_n[None, :])
    tl.store(ptr_dq, b_dq, mask=(offs_q[:, None] < T_Q) & (offs_q_n[None, :] < K))


@triton.autotune(
    configs=[
        triton.Config({"BN": BN}, num_warps=num_warps, num_stages=num_stages)
        for BN in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T_Q", "T_KV", "K", "V", "BT", "BN"],
)
@triton.jit
def piecewise_attn_bwd_approx_dkdv_kernel(
    do_desc,
    q_desc,
    kc,
    vc,
    lse,
    delta,
    dkc_grad,
    dvc_grad,
    block_map,
    scale: tl.constexpr,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    BN: tl.constexpr,
):
    i_v, i_kv_group, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    offs_c = i_kv_group * BN + tl.arange(0, BN)

    p_kc = tl.make_tensor_descriptor(kc + i_bh * NT_KV * K, (NT_KV, K), (K, 1), (BN, BK))
    p_vc = tl.make_tensor_descriptor(vc + i_bh * NT_KV * V, (NT_KV, V), (V, 1), (BN, BV))

    b_kc = p_kc.load([i_kv_group * BN, 0])
    b_vc = p_vc.load([i_kv_group * BN, i_v * BV])

    sm_scale: tl.constexpr = scale * 1.44269504
    b_dkc = tl.zeros([BN, BK], dtype=tl.float32)
    b_dvc = tl.zeros([BN, BV], dtype=tl.float32)

    for i_q in tl.range(0, NT_Q, 1, num_stages=1):
        q_start = i_q * BT
        tl.multiple_of(q_start, BT)

        q_offs = q_start + tl.arange(0, BT)
        selected = tl.load(block_map + (i_bh * NT_Q + i_q) * NT_KV + offs_c, mask=offs_c < NT_KV, other=1)
        valid_chunk = (offs_c < NT_KV) & (selected == 0)

        b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
        b_do = do_desc.load([i_bh, q_start, i_v * BV]).reshape([BT, BV])
        
        p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))

        b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
        b_lse = tl.where(q_offs < T_Q, b_lse, float("inf"))
        b_D = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

        current_lens = tl.minimum(BT, tl.maximum(0, T_KV - offs_c * BT)).to(tl.float32)
        safe_lens = tl.maximum(current_lens, 1.0)
        b_s_t = tl.dot(b_kc, tl.trans(b_q)) * sm_scale
        b_p_t = tl.math.exp2(b_s_t - b_lse[None, :])
        b_p_t = tl.where(valid_chunk[:, None] & (q_offs[None, :] < T_Q), b_p_t, 0.0)

        b_dvc += tl.dot(b_p_t.to(b_do.dtype), b_do)
        b_dp_t = tl.dot(b_vc, tl.trans(b_do))
        b_ds_t = b_p_t * (b_dp_t - current_lens[:, None] * b_D[None, :])
        b_dkc += tl.dot(b_ds_t.to(b_q.dtype), b_q) * scale / safe_lens[:, None]

    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)

    tl.store(
        dkc_grad + i_bh * NT_KV * K + offs_c[:, None] * K + offs_k[None, :],
        b_dkc,
        mask=(offs_c[:, None] < NT_KV) & (offs_k[None, :] < K),
    )
    tl.store(
        dvc_grad + i_bh * NT_KV * V + offs_c[:, None] * V + offs_v[None, :],
        b_dvc,
        mask=(offs_c[:, None] < NT_KV) & (offs_v[None, :] < V),
    )


@triton.jit
def piecewise_attn_bwd_exact_dkdv_kernel(
    do_desc,
    q_desc,
    k_desc,
    v_desc,
    lse,
    delta,
    dkc_grad,
    dvc_grad,
    dk,
    dv,
    block_map,
    scale: tl.constexpr,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
):
    i_v, i_kv, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    kv_start = i_kv * BT
    tl.multiple_of(kv_start, BT)

    offs_n = kv_start + tl.arange(0, BT)

    b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])
    b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])

    sm_scale = scale * 1.44269504
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    for i_q in tl.range(0, NT_Q, 1, num_stages=1):
        selected = tl.load(block_map + (i_bh * NT_Q + i_q) * NT_KV + i_kv)
        if selected == 1:
            q_start = i_q * BT
            tl.multiple_of(q_start, BT)

            q_offs = q_start + tl.arange(0, BT)
            b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
            b_do = do_desc.load([i_bh, q_start, i_v * BV]).reshape([BT, BV])

            p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
            p_delta = tl.make_block_ptr(delta + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))

            b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
            b_lse = tl.where(q_offs < T_Q, b_lse, float("inf"))
            b_D = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

            b_s_t = tl.dot(b_k, tl.trans(b_q)) * sm_scale
            b_p_t = tl.math.exp2(b_s_t - b_lse[None, :])
            b_p_t = tl.where(offs_n[:, None] < T_KV, b_p_t, 0.0)

            b_dv += tl.dot(b_p_t.to(b_do.dtype), b_do)
            b_dp_t = tl.dot(b_v, tl.trans(b_do))
            b_ds_t = b_p_t * (b_dp_t - b_D[None, :])
            b_dk += tl.dot(b_ds_t.to(b_q.dtype), b_q)

    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)

    b_dkc = tl.load(
        dkc_grad + i_bh * NT_KV * K + i_kv * K + offs_k,
        mask=offs_k < K,
        other=0.0,
    )
    tl.store(
        dk + i_bh * T_KV * K + offs_n[:, None] * K + offs_k[None, :],
        b_dk * scale + b_dkc[None, :],
        mask=(offs_n[:, None] < T_KV) & (offs_k[None, :] < K),
    )

    b_dvc = tl.load(
        dvc_grad + i_bh * NT_KV * V + i_kv * V + offs_v,
        mask=offs_v < V,
        other=0.0,
    )
    tl.store(
        dv + i_bh * T_KV * V + offs_n[:, None] * V + offs_v[None, :],
        b_dv + b_dvc[None, :],
        mask=(offs_n[:, None] < T_KV) & (offs_v[None, :] < V),
    )


def piecewise_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    scale: float,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]
    BT, NS = block_size, block_indices.shape[-1]
    
    o = torch.empty(B, H, T_Q, V, device=q.device, dtype=v.dtype)
    lse = torch.empty(B, H, T_Q, device=q.device, dtype=torch.float)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    
    NT_Q = triton.cdiv(T_Q, BT)
    NT_KV = triton.cdiv(T_KV, BT)
    
    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T_Q, K), [1, block_size, BK])
    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T_Q, V), [1, block_size, BV])

    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T_KV, K), [1, block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, block_size, BV])
    
    grid = (triton.cdiv(V, BV), NT_Q, B * H)
    piecewise_attn_fwd_kernel[grid](
        q_desc=q_desc,
        k_desc=k_desc,
        v_desc=v_desc,
        o_desc=o_desc,
        kc=kc,
        vc=vc,
        lse=lse,
        indices=block_indices,
        scale=scale,
        T_Q=T_Q,
        T_KV=T_KV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NS=NS,
        NT_Q=NT_Q,
        NT_KV=NT_KV,
    )
    return o, lse


def piecewise_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    scale: float,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]
    BT, NS = block_size, block_indices.shape[-1]

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    
    NT_Q = triton.cdiv(T_Q, BT)
    NT_KV = triton.cdiv(T_KV, BT)

    delta = torch.empty_like(lse)

    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T_Q, V), [1, BT, BV])
    do_desc = TensorDescriptor.from_tensor(do.reshape(B * H, T_Q, V), [1, BT, BV])

    attn_bwd_preprocess[(NT_Q, B * H)](
        o_desc=o_desc,
        do_desc=do_desc,
        delta=delta,
        T=T_Q,
        BT=BT,
        BV=BV,
    )

    block_map = build_block_map(block_indices, NT_KV)

    dq = torch.empty_like(q)

    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T_Q, K), [1, BT, BK])
    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T_KV, K), [1, BT, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, BT, BV])
    
    grid = (triton.cdiv(V, BV), NT_Q, B * H)
    piecewise_attn_bwd_dq_kernel[grid](
        do_desc=do_desc,
        q_desc=q_desc,
        k_desc=k_desc,
        v_desc=v_desc,
        kc=kc,
        vc=vc,
        lse=lse,
        delta=delta,
        dq=dq,
        indices=block_indices,
        block_map=block_map,
        scale=scale,
        T_Q=T_Q,
        T_KV=T_KV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT_Q=NT_Q,
        NT_KV=NT_KV,
        NS=NS,
    )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    dkc_grad = torch.empty(B, H, NT_KV, K, device=k.device, dtype=kc.dtype)
    dvc_grad = torch.empty(B, H, NT_KV, V, device=v.device, dtype=vc.dtype)

    # BNC = 64
    # grid = (triton.cdiv(V, BV), triton.cdiv(NT_KV, BNC), B * H)
    grid = lambda META: (triton.cdiv(V, BV), triton.cdiv(NT_KV, META["BN"]), B * H)
    piecewise_attn_bwd_approx_dkdv_kernel[grid](
        do_desc,
        q_desc,
        kc,
        vc,
        lse,
        delta,
        dkc_grad,
        dvc_grad,
        block_map,
        scale,
        T_Q,
        T_KV,
        K,
        V,
        BT,
        BK,
        BV,
        NT_Q,
        NT_KV,
        # BNC,
    )

    grid = (triton.cdiv(V, BV), NT_KV, B * H)
    piecewise_attn_bwd_exact_dkdv_kernel[grid](
        do_desc,
        q_desc,
        k_desc,
        v_desc,
        lse,
        delta,
        dkc_grad,
        dvc_grad,
        dk,
        dv,
        block_map,
        scale,
        T_Q,
        T_KV,
        K,
        V,
        BT,
        BK,
        BV,
        NT_Q,
        NT_KV,
    )
    return dq, dk, dv


@torch.no_grad()
def taylor_error_block_indices(
    qc: torch.Tensor,       # [B, H, NT_Q, K]
    kc: torch.Tensor,       # [B, H, NT_KV, K]
    k_var: torch.Tensor,    # [B, H, NT_KV]
    density: float,
    scale: float,
    eps: float = 1e-8,
):
    NT_KV = kc.shape[2]
    top_k = max(1, int(density * NT_KV))
    top_k = min(top_k, NT_KV)

    # [B, H, NT_Q, NT_KV]
    mean_logits = torch.einsum("bhik,bhjk->bhij", qc, kc) * scale

    # Use log-domain score to avoid exp overflow.
    # Equivalent to topk(exp(2*logit) * k_var).
    log_k_var = torch.log(k_var.clamp_min(eps)).unsqueeze(-2)  # [B,H,1,NT_KV]

    route_score = 1.0 * mean_logits + log_k_var

    block_indices = torch.topk(route_score, k=top_k, dim=-1).indices.to(torch.int32)
    return block_indices


class PiecewiseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, density, block_size, scale):

        triton.set_allocator(_make_tma_allocator())

        qc, kc, vc, k_var = chunk_reduce_qkv(q=q, k=k, v=v, block_size=block_size)

        block_indices = taylor_error_block_indices(
            qc=qc,
            kc=kc,
            k_var=k_var,
            density=density,
            scale=scale,
        )
        o, lse = piecewise_attn_fwd(
            q=q,
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale
        )

        ctx.save_for_backward(q, k, v, kc, vc, o, lse, block_indices)
        ctx.scale = scale
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, kc, vc, o, lse, block_indices = ctx.saved_tensors

        scale = ctx.scale
        block_size = ctx.block_size

        triton.set_allocator(_make_tma_allocator())

        dq, dk, dv = piecewise_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            kc=kc,
            vc=vc,
            do=do,
            lse=lse,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
        )

        return dq, dk, dv, None, None, None


def piecewise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    density: float = 0.1,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        density (float):
            The ratio of blocks computed exactly.
        block_size (int):
            The tile size
    """
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    return PiecewiseAttentionFunction.apply(
        q, k, v, density, block_size, scale
    )
