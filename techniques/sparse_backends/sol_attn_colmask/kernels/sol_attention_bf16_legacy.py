# Copyright (c) 2025-2026, Haopeng Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Legacy pure-BF16 Triton SOL Attention with group-local thresholding.

This is an importable copy of
``sol_attn-improve/remote_project_copy/sol_attention_bf16.py``.
The algorithm and threshold semantics are intentionally preserved.  The only
functional extension is an explicit, validated ``group_size`` argument; its
default remains the source implementation's G32.
"""

import functools

import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(
            *(x.contiguous() if isinstance(x, torch.Tensor) else x for x in args),
            **{
                k: (v.contiguous() if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            },
        )

    return wrapper


GROUP_SIZE = 32
VALID_GROUP_SIZES = (16, 32, 64, 128)
ORIGINAL_SOURCE_SHA256 = (
    "109ec4f237edebd51d6160d98c0832c703e6b7c99a76461db74d344506debf06"
)


def validate_group_size(group_size: int) -> int:
    """Return a supported route-group size or fail before JIT compilation."""

    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError(
            f"group_size must be an integer in {VALID_GROUP_SIZES}, "
            f"got {group_size!r}"
        )
    if group_size not in VALID_GROUP_SIZES:
        raise ValueError(
            f"group_size must be one of {VALID_GROUP_SIZES}, got {group_size}"
        )
    return group_size


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["T"],
)
@triton.jit
def reduce_kc_kernel(
    k_desc,
    kc,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    block_size = tl.minimum(BT, T - i_t * BT)
    b_k = k_desc.load([i_bh, i_t * BT, i_k * BK]).reshape([BT, BK])
    b_kc = tl.sum(b_k, axis=0) / block_size

    p_kc = tl.make_block_ptr(
        kc + i_bh * N * K + i_t * K,
        (K,),
        (1,),
        (i_k * BK,),
        (BK,),
        (0,),
    )
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3, 4]
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
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))


def preprocess_kv(k, v, block_size):
    B, H, T, K, V = *k.shape, v.shape[-1]
    N = triton.cdiv(T, block_size)

    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    kc = torch.empty(B, H, N, K, device=k.device, dtype=k.dtype)
    vc = torch.empty(B, H, N, V, device=v.device, dtype=v.dtype)

    k_desc = TensorDescriptor.from_tensor(
        k.reshape(B * H, T, K), [1, block_size, BK]
    )
    v_desc = TensorDescriptor.from_tensor(
        v.reshape(B * H, T, V), [1, block_size, BV]
    )

    grid_k = (triton.cdiv(K, BK), N, B * H)
    reduce_kc_kernel[grid_k](
        k_desc=k_desc,
        kc=kc,
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

    return kc, vc


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    # GROUP_SIZE changes both the compiled loop and resource footprint.
    key=["T", "GROUP_SIZE"],
)
@triton.jit
def single_pass_dynamic_routing_kernel(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
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
    BETA: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)

    token_offsets = tl.arange(0, BT)
    token_offsets = tl.max_contiguous(token_offsets, BT)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")
    sm_scale = scale * 1.44269504

    last_chunk_len = T - (NT - 1) * BT

    for start_n in range(0, NT, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)
        chunk_indices = start_n + chunk_offsets

        remaining = NT - start_n
        valid_mask = chunk_offsets < remaining

        b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
        b_vc = vc_desc.load([i_bh, start_n, i_v * BV]).reshape(
            [GROUP_SIZE, BV]
        )

        b_s_mean = tl.dot(b_q, b_kc.T).to(tl.float32)
        b_s_mean = b_s_mean * sm_scale

        # Deliberately preserve the source's old group-local threshold:
        # every route group computes its own mean + beta * std cutoff.
        group_scores = tl.sum(b_s_mean, axis=0) / BT
        group_mean = tl.sum(group_scores) / GROUP_SIZE
        group_sq_mean = tl.sum(group_scores * group_scores) / GROUP_SIZE
        group_var = tl.maximum(group_sq_mean - group_mean * group_mean, 0.0)
        group_std = tl.sqrt(group_var + 1e-6)

        thresh = group_mean + BETA * group_std
        is_exact = (group_scores > thresh) & valid_mask

        approx_mask = valid_mask & (~is_exact)
        b_s_approx = tl.where(
            approx_mask[None, :], b_s_mean, float("-inf")
        )

        new_m = tl.maximum(m_i, tl.max(b_s_approx, axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        prob = tl.math.exp2(b_s_approx - new_m[:, None])

        acc = acc * alpha[:, None] + tl.dot(prob.to(b_vc.dtype), b_vc)
        current_lens = tl.where(
            chunk_indices == NT - 1, last_chunk_len, BT
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(prob * current_lens[None, :], axis=1)
        m_i = new_m

        exact_offsets = tl.where(is_exact, chunk_offsets, GROUP_SIZE)
        num_exact = tl.sum(is_exact.to(tl.int32))

        for _ in range(num_exact):
            next_offset = tl.min(exact_offsets)
            n_idx = start_n + next_offset
            exact_offsets = tl.where(
                chunk_offsets == next_offset, GROUP_SIZE, exact_offsets
            )
            kv_start = n_idx * BT
            tl.multiple_of(kv_start, BT)

            b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])
            b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32)
            b_s_exact = b_s_exact * sm_scale

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
    o_desc.store(
        [i_bh, q_start, i_v * BV], acc.to(tl.bfloat16)[None, :, :]
    )


@contiguous
@torch.compiler.disable
def sol_attention(
    q,
    k,
    v,
    thresh=1.0,
    block_size=64,
    scale=None,
    use_bias=False,
    group_size=GROUP_SIZE,
):
    group_size = validate_group_size(group_size)
    B, H, T, K, V = *k.shape, v.shape[-1]
    scale = K**-0.5 if scale is None else scale

    NT = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    kc, vc = preprocess_kv(k, v, block_size)
    o = torch.empty_like(v)

    q_desc = TensorDescriptor.from_tensor(
        q.reshape(B * H, T, K), [1, block_size, BK]
    )
    k_desc = TensorDescriptor.from_tensor(
        k.reshape(B * H, T, K), [1, block_size, BK]
    )
    v_desc = TensorDescriptor.from_tensor(
        v.reshape(B * H, T, V), [1, block_size, BV]
    )
    o_desc = TensorDescriptor.from_tensor(
        o.reshape(B * H, T, V), [1, block_size, BV]
    )

    kc_desc = TensorDescriptor.from_tensor(
        kc.reshape(B * H, NT, K), [1, group_size, BK]
    )
    vc_desc = TensorDescriptor.from_tensor(
        vc.reshape(B * H, NT, V), [1, group_size, BV]
    )

    grid = (triton.cdiv(V, BV), NT, B * H)
    single_pass_dynamic_routing_kernel[grid](
        q_desc=q_desc,
        k_desc=k_desc,
        v_desc=v_desc,
        kc_desc=kc_desc,
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
        GROUP_SIZE=group_size,
        BETA=thresh,
    )

    return o


__all__ = [
    "GROUP_SIZE",
    "ORIGINAL_SOURCE_SHA256",
    "VALID_GROUP_SIZES",
    "sol_attention",
    "preprocess_kv",
    "validate_group_size",
]
