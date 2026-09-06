"""SOL omitted-block centroid residual for the cuDNN BSA backend."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


BLOCK = 64


@triton.jit
def _merge_residual_kernel(
    q_ptr,
    kc_ptr,
    vc_ptr,
    exact_mask_ptr,
    exact_o_ptr,
    exact_lse_ptr,
    out_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_kc_block,
    stride_kc_head,
    stride_kc_dim,
    stride_vc_block,
    stride_vc_head,
    stride_vc_dim,
    stride_mask_head,
    stride_mask_query,
    stride_mask_key,
    stride_o_head,
    stride_o_token,
    stride_o_dim,
    stride_lse_head,
    stride_lse_token,
    tokens: tl.constexpr,
    blocks: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale_log2: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_dv: tl.constexpr,
):
    q_block = tl.program_id(0)
    head = tl.program_id(1)
    value_tile = tl.program_id(2)

    q_offsets = q_block * block_m + tl.arange(0, block_m)
    d_offsets = tl.arange(0, head_dim)
    v_offsets = value_tile * block_dv + tl.arange(0, block_dv)
    q_valid = q_offsets < tokens
    v_valid = v_offsets < head_dim

    q_indices = (
        q_offsets[:, None] * stride_q_token
        + head * stride_q_head
        + d_offsets[None, :] * stride_q_dim
    )
    q = tl.load(q_ptr + q_indices, mask=q_valid[:, None], other=0.0)

    numerator = tl.zeros((block_m, block_dv), dtype=tl.float32)
    row_sum = tl.zeros((block_m,), dtype=tl.float32)
    row_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)

    for block_start in range(0, blocks, block_n):
        block_offsets = block_start + tl.arange(0, block_n)
        block_valid = block_offsets < blocks
        kc_indices = (
            block_offsets[:, None] * stride_kc_block
            + head * stride_kc_head
            + d_offsets[None, :] * stride_kc_dim
        )
        kc = tl.load(kc_ptr + kc_indices, mask=block_valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(kc)).to(tl.float32) * scale_log2

        mask_indices = (
            head * stride_mask_head
            + q_block * stride_mask_query
            + block_offsets * stride_mask_key
        )
        is_exact = tl.load(exact_mask_ptr + mask_indices, mask=block_valid, other=1).to(tl.int1)
        approximate = block_valid & ~is_exact
        has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0

        masked_scores = tl.where(approximate[None, :], scores, -float("inf"))
        candidate_max = tl.max(masked_scores, axis=1)
        new_max = tl.where(has_approximate, tl.maximum(row_max, candidate_max), row_max)
        alpha = tl.where(has_approximate, tl.exp2(row_max - new_max), 1.0)
        probability = tl.where(
            approximate[None, :],
            tl.exp2(scores - new_max[:, None]),
            0.0,
        )

        vc_indices = (
            block_offsets[:, None] * stride_vc_block
            + head * stride_vc_head
            + v_offsets[None, :] * stride_vc_dim
        )
        vc = tl.load(
            vc_ptr + vc_indices,
            mask=block_valid[:, None] & v_valid[None, :],
            other=0.0,
        )
        numerator = numerator * alpha[:, None] + tl.dot(probability.to(tl.bfloat16), vc)

        block_lengths = tl.minimum(64, tl.maximum(0, tokens - block_offsets * 64)).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(probability * block_lengths[None, :], axis=1)
        row_max = new_max

    # cuDNN BSA returns natural-log LSE in [B, H, S].  Sol's residual path
    # above is accumulated in base-2, matching the released Sol kernel.
    exact_lse = tl.load(
        exact_lse_ptr + head * stride_lse_head + q_offsets * stride_lse_token,
        mask=q_valid,
        other=-float("inf"),
    )
    exact_lse_log2 = exact_lse * 1.4426950408889634
    joint_max = tl.maximum(row_max, exact_lse_log2)
    residual_scale = tl.exp2(row_max - joint_max)
    exact_scale = tl.exp2(exact_lse_log2 - joint_max)
    denominator = row_sum * residual_scale + exact_scale

    # BSA output is BHSD.  The hybrid output is BSHD for H3's packed path.
    exact_o_indices = (
        head * stride_o_head
        + q_offsets[:, None] * stride_o_token
        + v_offsets[None, :] * stride_o_dim
    )
    exact_o = tl.load(
        exact_o_ptr + exact_o_indices,
        mask=q_valid[:, None] & v_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = (numerator * residual_scale[:, None] + exact_o * exact_scale[:, None]) / denominator[:, None]
    out_indices = (q_offsets[:, None] * heads + head) * head_dim + v_offsets[None, :]
    tl.store(out_ptr + out_indices, merged, mask=q_valid[:, None] & v_valid[None, :])


@torch.no_grad()
def merge_sol_residual(
    q: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    exact_mask: torch.Tensor,
    exact_o_bhsd: torch.Tensor,
    exact_lse: torch.Tensor,
) -> torch.Tensor:
    """Merge BSA's exact state with Sol's centroid approximation of omitted blocks.

    All tensors have batch size one. ``q``, ``kc`` and ``vc`` are BSHD;
    ``exact_o_bhsd`` and ``exact_lse`` use BSA's BHSD convention.
    """
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("SOL/BSA requires q=[1,T,H,128]")
    tokens, heads, head_dim = q.shape[1:]
    blocks = math.ceil(tokens / BLOCK)
    expected_mask = (1, heads, blocks, blocks)
    if tuple(exact_mask.shape) != expected_mask:
        raise ValueError(f"expected exact_mask={expected_mask}, got {tuple(exact_mask.shape)}")
    if tuple(exact_o_bhsd.shape) != (1, heads, tokens, head_dim):
        raise ValueError("unexpected BSA output shape")
    if tuple(exact_lse.shape) != (1, heads, tokens):
        raise ValueError(f"unexpected BSA LSE shape: {tuple(exact_lse.shape)}")

    # Q is an interleaved view of the decoded [Q|K|V] allocation, while BSA's
    # output and LSE are logical-token slices of compile-bucket allocations.
    # Reading their real strides here avoids materializing roughly 130 MiB per
    # sparse call.  The output stays contiguous for the return collective.
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    # Keep the complete V head in one program. Splitting D=128 into four
    # programs repeats the expensive Q @ KC centroid scores four times.
    block_m, block_n, block_dv = 64, 32, 128
    grid = (triton.cdiv(tokens, block_m), heads, triton.cdiv(head_dim, block_dv))
    _merge_residual_kernel[grid](
        q,
        kc,
        vc,
        exact_mask,
        exact_o_bhsd,
        exact_lse,
        out,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kc.stride(1),
        kc.stride(2),
        kc.stride(3),
        vc.stride(1),
        vc.stride(2),
        vc.stride(3),
        exact_mask.stride(1),
        exact_mask.stride(2),
        exact_mask.stride(3),
        exact_o_bhsd.stride(1),
        exact_o_bhsd.stride(2),
        exact_o_bhsd.stride(3),
        exact_lse.stride(1),
        exact_lse.stride(2),
        tokens=tokens,
        blocks=blocks,
        heads=heads,
        head_dim=head_dim,
        scale_log2=head_dim**-0.5 * math.log2(math.e),
        block_m=block_m,
        block_n=block_n,
        block_dv=block_dv,
        num_warps=8,
        num_stages=2,
    )
    return out
