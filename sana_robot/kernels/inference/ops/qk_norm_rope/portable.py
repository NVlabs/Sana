# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.dispatch import SM110, SM120, device_architecture

GDN_LAYOUT = "gdn_bhdn"
SOFTMAX_LAYOUT = "softmax_bnhd"


@dataclass(frozen=True)
class QKNormRopeResult:
    """Outputs of one fused q / k RMSNorm + RoPE launch: the rotated queries and keys and, for the Gated DeltaNet layout, the normalized ones before the rotation."""

    q_rotated: torch.Tensor
    k_rotated: torch.Tensor
    q_normalized: torch.Tensor | None
    k_normalized: torch.Tensor | None


def _eager_inverse_rms(x, eps):
    """Return the inverse RMS of every token row of ``x`` [B, N, C] as the policy's RMSNorm computes it, [B, N] in float32."""
    x_float = x.float()
    return torch.rsqrt(x_float.pow(2).mean(-1) + eps)


def _allocate_bnhd(batch, tokens, heads, head_dim, dtype, device):
    """Return an empty [B, N, heads, head_dim] tensor with the strides the policy's own rotation gives its output."""
    token_stride = heads * head_dim
    batch_stride = token_stride if batch == 1 else tokens * token_stride
    return torch.empty_strided(
        (batch, tokens, heads, head_dim),
        (batch_stride, token_stride, head_dim, 1),
        dtype=dtype,
        device=device,
    )


@triton.jit
def sana_qk_norm_rope(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freq_ptr,
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    q_rot_ptr,
    k_rot_ptr,
    q_norm_ptr,
    k_norm_ptr,
    num_tokens,
    eps,
    q_stride_b,
    q_stride_n,
    k_stride_b,
    k_stride_n,
    freq_stride_b,
    freq_stride_n,
    freq_stride_pair,
    freq_stride_comp,
    out_stride_b,
    out_stride_n,
    inv_stride_b,
    inv_stride_n,
    C_TOTAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    FREQ_BATCHED: tl.constexpr,
    FUSED_REDUCTION: tl.constexpr,
    OUTPUT_NORMALIZED: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    """One token row of q and of k: the RMSNorm in float32 rounded to the input dtype, then the rotation of every channel pair of every head in float64."""
    pid = tl.program_id(0)
    b = pid // num_tokens
    n = pid % num_tokens

    q_row = q_ptr + b * q_stride_b + n * q_stride_n
    k_row = k_ptr + b * k_stride_b + n * k_stride_n
    out_off = b * out_stride_b + n * out_stride_n

    if FUSED_REDUCTION:
        dim = tl.arange(0, HEAD_DIM)
        q_sumsq = 0.0
        k_sumsq = 0.0
        for h in range(NUM_HEADS):
            x_q = tl.load(q_row + h * HEAD_DIM + dim).to(tl.float32)
            q_sumsq += tl.sum(x_q * x_q)
            x_k = tl.load(k_row + h * HEAD_DIM + dim).to(tl.float32)
            k_sumsq += tl.sum(x_k * x_k)
        q_inv = tl.math.rsqrt(q_sumsq / C_TOTAL + eps)
        k_inv = tl.math.rsqrt(k_sumsq / C_TOTAL + eps)
    else:
        inv_off = b * inv_stride_b + n * inv_stride_n
        q_inv = tl.load(q_inv_rms_ptr + inv_off)
        k_inv = tl.load(k_inv_rms_ptr + inv_off)

    pair = tl.arange(0, HEAD_DIM // 2)
    even = 2 * pair
    odd = even + 1
    if FREQ_BATCHED:
        freq_row = freq_ptr + b * freq_stride_b + n * freq_stride_n
    else:
        freq_row = freq_ptr + n * freq_stride_n
    cos = tl.load(freq_row + pair * freq_stride_pair)
    sin = tl.load(freq_row + pair * freq_stride_pair + freq_stride_comp)

    for h in range(NUM_HEADS):
        channel = h * HEAD_DIM

        x0 = tl.load(q_row + channel + even).to(tl.float32)
        x1 = tl.load(q_row + channel + odd).to(tl.float32)
        w0 = tl.load(q_weight_ptr + channel + even).to(tl.float32)
        w1 = tl.load(q_weight_ptr + channel + odd).to(tl.float32)
        n0 = w0 * (x0 * q_inv)
        n1 = w1 * (x1 * q_inv)
        if ROUND_BF16:
            n0 = n0.to(tl.bfloat16)
            n1 = n1.to(tl.bfloat16)
        if OUTPUT_NORMALIZED:
            tl.store(q_norm_ptr + out_off + channel + even, n0)
            tl.store(q_norm_ptr + out_off + channel + odd, n1)
        r0 = n0.to(tl.float64)
        r1 = n1.to(tl.float64)
        y0 = (r0 * cos - r1 * sin).to(tl.float32)
        y1 = (r0 * sin + r1 * cos).to(tl.float32)
        tl.store(q_rot_ptr + out_off + channel + even, y0)
        tl.store(q_rot_ptr + out_off + channel + odd, y1)

        x0 = tl.load(k_row + channel + even).to(tl.float32)
        x1 = tl.load(k_row + channel + odd).to(tl.float32)
        w0 = tl.load(k_weight_ptr + channel + even).to(tl.float32)
        w1 = tl.load(k_weight_ptr + channel + odd).to(tl.float32)
        n0 = w0 * (x0 * k_inv)
        n1 = w1 * (x1 * k_inv)
        if ROUND_BF16:
            n0 = n0.to(tl.bfloat16)
            n1 = n1.to(tl.bfloat16)
        if OUTPUT_NORMALIZED:
            tl.store(k_norm_ptr + out_off + channel + even, n0)
            tl.store(k_norm_ptr + out_off + channel + odd, n1)
        r0 = n0.to(tl.float64)
        r1 = n1.to(tl.float64)
        y0 = (r0 * cos - r1 * sin).to(tl.float32)
        y1 = (r0 * sin + r1 * cos).to(tl.float32)
        tl.store(k_rot_ptr + out_off + channel + even, y0)
        tl.store(k_rot_ptr + out_off + channel + odd, y1)


def fused_qk_norm_rope(q, k, q_weight, k_weight, freqs, layout, head_dim, eps) -> QKNormRopeResult:
    """Normalize the query and key projections [B, N, C] and rotate them by the complex128 phases ``freqs`` [B or 1, 1, N, head_dim / 2] in one launch of the kernel of their device; the result equals the policy's RMSNorm followed by its float64 rotation.

    ``SOFTMAX_LAYOUT`` returns the rotated tensors as [B, N, heads, head_dim]; ``GDN_LAYOUT`` returns the rotated and the normalized tensors as [B, heads, head_dim, N].
    """
    (batch, tokens, channels) = q.shape
    architecture = device_architecture(q.device)
    if architecture is not None:
        from sana_robot.kernels.inference.ops.qk_norm_rope import sm110, sm120

        kernel = sm110 if architecture == SM110 else sm120
        if kernel.serves_shape(channels, head_dim, batch * tokens):
            return kernel.fused_qk_norm_rope(q, k, q_weight, k_weight, freqs, layout, head_dim, eps)

    heads = channels // head_dim
    is_gdn = layout == GDN_LAYOUT
    frequency_components = torch.view_as_real(freqs)
    q_rotated = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
    k_rotated = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
    if is_gdn:
        q_normalized = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
        k_normalized = _allocate_bnhd(batch, tokens, heads, head_dim, q.dtype, q.device)
    else:
        q_normalized = q_rotated
        k_normalized = k_rotated
    q_inv_rms = _eager_inverse_rms(q, eps)
    k_inv_rms = _eager_inverse_rms(k, eps)

    grid = (batch * tokens,)
    sana_qk_norm_rope[grid](
        q,
        k,
        q_weight,
        k_weight,
        frequency_components,
        q_inv_rms,
        k_inv_rms,
        q_rotated,
        k_rotated,
        q_normalized,
        k_normalized,
        tokens,
        float(eps),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        frequency_components.stride(0),
        frequency_components.stride(2),
        frequency_components.stride(3),
        frequency_components.stride(4),
        q_rotated.stride(0),
        q_rotated.stride(1),
        q_inv_rms.stride(0),
        q_inv_rms.stride(1),
        C_TOTAL=channels,
        HEAD_DIM=head_dim,
        NUM_HEADS=heads,
        FREQ_BATCHED=freqs.shape[0] > 1,
        FUSED_REDUCTION=False,
        OUTPUT_NORMALIZED=is_gdn,
        ROUND_BF16=q.dtype == torch.bfloat16,
        num_warps=4 if head_dim <= 128 else 8,
    )

    if is_gdn:
        return QKNormRopeResult(
            q_rotated.permute(0, 2, 3, 1),
            k_rotated.permute(0, 2, 3, 1),
            q_normalized.permute(0, 2, 3, 1),
            k_normalized.permute(0, 2, 3, 1),
        )
    return QKNormRopeResult(q_rotated, k_rotated, None, None)
