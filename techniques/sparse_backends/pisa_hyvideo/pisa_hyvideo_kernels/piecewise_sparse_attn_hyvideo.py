# Copyright (c) 2025-2026, Haopeng Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PISA kernels for bidirectional HunyuanVideo/MMDiT attention.

HunyuanVideo uses a bidirectional sequence where video tokens are followed by
text/context tokens. The text suffix must remain exact/dense. This file keeps
that text-suffix sink separate from the chunk-AR kernels, whose sinks are about
history prefixes, frame windows, and route masks.
"""

import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor

from pisa_hyvideo_kernels.piecewise_sparse_attn_0th import (
    _make_tma_allocator,
    chunk_reduce_qkv,
    piecewise_attn_fwd,
)
from pisa_hyvideo_kernels.sol_attention import (
    GROUP_SIZE,
    compute_global_qck_threshold,
    contiguous,
    preprocess_qkv,
)


def _suffix_sink_blocks(kv_len: int, sink_tokens: int, block_size: int) -> int:
    """Return suffix KV blocks touched by the last sink_tokens tokens."""
    kv_len = max(int(kv_len), 0)
    sink_tokens = min(max(int(sink_tokens), 0), kv_len)
    block_size = int(block_size)
    if kv_len == 0 or sink_tokens == 0:
        return 0
    first_sink_token = kv_len - sink_tokens
    return triton.cdiv(kv_len, block_size) - (first_sink_token // block_size)


@torch.no_grad()
def _hyvideo_topk_indices(
    qc: torch.Tensor,
    kc: torch.Tensor,
    k_var: torch.Tensor,
    density: float,
    scale: float,
    text_sink_blocks: int,
    eps: float = 1e-8,
):
    nt_kv = kc.shape[2]
    text_sink_blocks = min(max(int(text_sink_blocks), 0), nt_kv)
    top_k = max(1, int(density * nt_kv))
    top_k = max(top_k, text_sink_blocks)
    top_k = min(top_k, nt_kv)

    mean_logits = torch.einsum("bhik,bhjk->bhij", qc, kc) * scale
    route_score = mean_logits + torch.log(k_var.clamp_min(eps)).unsqueeze(-2)
    if text_sink_blocks:
        route_score[..., nt_kv - text_sink_blocks :] = float("inf")
    return torch.topk(route_score, k=top_k, dim=-1).indices.to(torch.int32)


class HyvideoPiecewiseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, density, block_size, scale, text_sink_blocks):
        triton.set_allocator(_make_tma_allocator())
        qc, kc, vc, k_var = chunk_reduce_qkv(q=q, k=k, v=v, block_size=block_size)
        block_indices = _hyvideo_topk_indices(
            qc=qc,
            kc=kc,
            k_var=k_var,
            density=density,
            scale=scale,
            text_sink_blocks=text_sink_blocks,
        )
        o, _ = piecewise_attn_fwd(
            q=q,
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        raise RuntimeError("hyvideo_piecewise_attention is inference-only")


def hyvideo_piecewise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    density: float = 0.1,
    block_size: int = 64,
    text_sink_blocks: int = 0,
    text_sink_tokens: int = 0,
) -> torch.Tensor:
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    if text_sink_tokens:
        text_sink_blocks = max(
            int(text_sink_blocks),
            _suffix_sink_blocks(k.shape[-2], int(text_sink_tokens), int(block_size)),
        )
    return HyvideoPiecewiseAttentionFunction.apply(
        q, k, v, density, block_size, scale, int(text_sink_blocks)
    )


@torch.no_grad()
def calibrate_hyvideo_text_sink_density(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    target_density: float,
    block_size: int,
    sample_heads: int,
    q_stride_blocks: int,
    coarse_start: float,
    coarse_end: float,
    coarse_step: float,
    fine_radius: float,
    fine_step: float,
    text_sink_tokens: int = 0,
) -> dict[str, object]:
    """Calibrate SOL Attention threshold for HYVideo with a dense text suffix."""
    from pisa_hyvideo_kernels.utils import calibrate_density

    return calibrate_density(
        q,
        k,
        "diag",
        float(target_density),
        int(block_size),
        GROUP_SIZE,
        int(sample_heads),
        int(q_stride_blocks),
        float(coarse_start),
        float(coarse_end),
        float(coarse_step),
        float(fine_radius),
        float(fine_step),
        q_block_size=int(block_size),
        kv_block_size=int(block_size),
        q_start_offset=0,
        sink_last_tokens=int(text_sink_tokens),
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=[
        "T_Q",
        "T_KV",
        "K",
        "V",
        "BT_Q",
        "BT_KV",
        "BK",
        "BV",
        "NT_Q",
        "NT_KV",
        "NT_VIDEO_KV",
        "TEXT_SINK_BLOCKS",
    ],
)
@triton.jit
def hyvideo_dynamic_routing_kernel(
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
    K: tl.constexpr,
    V: tl.constexpr,
    BT_Q: tl.constexpr,
    BT_KV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    NT_VIDEO_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TEXT_SINK_BLOCKS: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)
    q_token_offsets = tl.arange(0, BT_Q)
    q_token_offsets = tl.max_contiguous(q_token_offsets, BT_Q)
    kv_token_offsets = tl.arange(0, BT_KV)
    kv_token_offsets = tl.max_contiguous(kv_token_offsets, BT_KV)

    q_start = i_t * BT_Q
    q_valid_len = tl.minimum(BT_Q, T_Q - q_start)
    q_len = q_valid_len.to(tl.float32)
    q_abs_start = q_start
    q_abs_end = q_abs_start + q_valid_len

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT_Q, BK])
    b_q_scale = tl.load(q_scale + i_bh * NT_Q + i_t)
    sm_scale = b_q_scale * scale * 1.44269504

    acc = tl.zeros([BT_Q, BV], dtype=tl.float32)
    l_i = tl.zeros((BT_Q,), dtype=tl.float32)
    m_i = tl.zeros((BT_Q,), dtype=tl.float32) - float("inf")

    for start_n in range(0, NT_VIDEO_KV, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)
        chunk_indices = start_n + chunk_offsets
        remaining = NT_VIDEO_KV - start_n
        valid_mask = chunk_offsets < remaining

        kv_start_tokens = chunk_indices * BT_KV
        kv_end_tokens = kv_start_tokens + BT_KV
        local_mask = (kv_start_tokens < q_abs_end + BT_KV) & (kv_end_tokens > q_abs_start - BT_KV)

        b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
        b_kc_scale = tl.load(kc_scale + i_bh * NT_KV + chunk_indices, mask=valid_mask, other=0.0)
        b_vc = vc_desc.load([i_bh, start_n, i_v * BV]).reshape([GROUP_SIZE, BV])

        b_s = tl.dot(b_q, b_kc.T).to(tl.float32)
        b_s = b_s * (sm_scale * b_kc_scale[None, :])
        col_mean = tl.sum(b_s, axis=0) / q_len
        thresh = tl.load(global_thresh + i_bh * NT_Q + i_t)
        is_exact = (((col_mean > thresh) | local_mask) & valid_mask)

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
        current_lens = tl.minimum(BT_KV, tl.maximum(0, T_KV - chunk_indices * BT_KV)).to(tl.float32)
        l_i = l_i * alpha + tl.sum(prob * current_lens[None, :], axis=1)
        m_i = new_m

        exact_offsets = tl.where(is_exact, chunk_offsets, GROUP_SIZE)
        num_exact = tl.sum(is_exact.to(tl.int32), axis=0)
        for _ in range(num_exact):
            rel = tl.min(exact_offsets, axis=0)
            exact_offsets = tl.where(exact_offsets == rel, GROUP_SIZE, exact_offsets)
            kv_start = (start_n + rel) * BT_KV
            b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT_KV, BK])
            b_k_scale = tl.load(k_scale + i_bh * NT_KV + start_n + rel)
            b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32)
            b_s_exact = b_s_exact * (sm_scale * b_k_scale)
            valid_mask_ex = (kv_start + kv_token_offsets)[None, :] < T_KV
            b_s_exact += tl.where(valid_mask_ex, 0, float("-inf"))

            new_m = tl.maximum(m_i, tl.max(b_s_exact, axis=1))
            alpha = tl.math.exp2(m_i - new_m)
            prob_exact = tl.math.exp2(b_s_exact - new_m[:, None])
            l_i = l_i * alpha + tl.sum(prob_exact, axis=1)

            b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT_KV, BV])
            acc = acc * alpha[:, None] + tl.dot(prob_exact.to(b_v.dtype), b_v)
            m_i = new_m

    for sink_block in range(0, TEXT_SINK_BLOCKS):
        kv_block = NT_KV - TEXT_SINK_BLOCKS + sink_block
        kv_start = kv_block * BT_KV
        b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT_KV, BK])
        b_k_scale = tl.load(k_scale + i_bh * NT_KV + kv_block)
        b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32)
        b_s_exact = b_s_exact * (sm_scale * b_k_scale)
        valid_mask_ex = (kv_start + kv_token_offsets)[None, :] < T_KV
        b_s_exact += tl.where(valid_mask_ex, 0, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(b_s_exact, axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        prob_exact = tl.math.exp2(b_s_exact - new_m[:, None])
        l_i = l_i * alpha + tl.sum(prob_exact, axis=1)

        b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT_KV, BV])
        acc = acc * alpha[:, None] + tl.dot(prob_exact.to(b_v.dtype), b_v)
        m_i = new_m

    acc /= l_i[:, None]
    acc = tl.where((q_start + q_token_offsets)[:, None] < T_Q, acc, 0.0)
    o_desc.store([i_bh, i_t * BT_Q, i_v * BV], acc.to(tl.bfloat16)[None, :, :])


@contiguous
@torch.compiler.disable
def hyvideo_online_piecewise_sparse_attention(
    q,
    k,
    v,
    thresh=1.0,
    block_size=64,
    scale=None,
    density=None,
    text_sink_blocks=0,
    text_sink_tokens=0,
):
    if density is not None:
        thresh = density

    bsz, heads, t_q, dim = q.shape
    t_kv = k.shape[-2]
    value_dim = v.shape[-1]
    scale = dim**-0.5 if scale is None else scale

    text_sink_blocks = max(int(text_sink_blocks), 0)
    if text_sink_tokens:
        text_sink_blocks = max(
            text_sink_blocks,
            _suffix_sink_blocks(t_kv, int(text_sink_tokens), int(block_size)),
        )

    nt_q = triton.cdiv(t_q, block_size)
    nt_kv = triton.cdiv(t_kv, block_size)
    bk = min(128, triton.next_power_of_2(dim))
    bv = min(128, triton.next_power_of_2(value_dim))

    q_int8, q_scale, k_int8, k_scale, kc_int8, kc_scale, vc = preprocess_qkv(
        q,
        k,
        v,
        block_size=block_size,
        q_block_size=block_size,
        kv_block_size=block_size,
    )
    global_thresh = compute_global_qck_threshold(
        q_int8,
        q_scale,
        kc_int8,
        kc_scale,
        scale,
        block_size=block_size,
        tau=thresh,
        q_block_size=block_size,
    )
    o = torch.empty(bsz, heads, t_q, value_dim, device=q.device, dtype=v.dtype)

    q_desc = TensorDescriptor.from_tensor(q_int8.reshape(bsz * heads, t_q, dim), [1, block_size, bk])
    k_desc = TensorDescriptor.from_tensor(k_int8.reshape(bsz * heads, t_kv, dim), [1, block_size, bk])
    v_desc = TensorDescriptor.from_tensor(v.reshape(bsz * heads, t_kv, value_dim), [1, block_size, bv])
    o_desc = TensorDescriptor.from_tensor(o.reshape(bsz * heads, t_q, value_dim), [1, block_size, bv])
    kc_desc = TensorDescriptor.from_tensor(kc_int8.reshape(bsz * heads, nt_kv, dim), [1, GROUP_SIZE, bk])
    vc_desc = TensorDescriptor.from_tensor(vc.reshape(bsz * heads, nt_kv, value_dim), [1, GROUP_SIZE, bv])

    text_sink_blocks = min(text_sink_blocks, nt_kv)
    grid = (triton.cdiv(value_dim, bv), nt_q, bsz * heads)
    hyvideo_dynamic_routing_kernel[grid](
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
        T_Q=t_q,
        T_KV=t_kv,
        K=dim,
        V=value_dim,
        BT_Q=block_size,
        BT_KV=block_size,
        BK=bk,
        BV=bv,
        NT_Q=nt_q,
        NT_KV=nt_kv,
        NT_VIDEO_KV=max(nt_kv - text_sink_blocks, 0),
        GROUP_SIZE=GROUP_SIZE,
        TEXT_SINK_BLOCKS=text_sink_blocks,
    )
    return o
