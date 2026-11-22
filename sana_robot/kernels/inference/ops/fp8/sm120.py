# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.fp8.portable import quantize_fp8

_BM = 128
_BN = 64
_BK = 64
_GROUP = 8
_STAGES = 3
_WARPS = 4


@triton.jit
def sana_fp8_swiglu_sm120_gemm(
    a_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    sa_ptr,
    sb_ptr,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """One [BM, BN] tile of y [M, N] = silu(g) * u in bf16 with g = (a @ Wg^T) * sa * sb + bg and u = (a @ Wu^T) * sa * sb + bu, for E4M3 a [M, K] and the packed E4M3 weight [2N, K] (gate rows [0, N), up rows [N, 2N)): two float32 accumulators over the same a tile, the scales, the biases and the activation in float32, one rounding to bf16."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_in_group = GROUP * num_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP
    group_size = tl.minimum(num_m - first_m, GROUP)
    pid_m = first_m + ((pid % num_in_group) % group_size)
    pid_n = (pid % num_in_group) // group_size

    offs_m = (pid_m * BM + tl.arange(0, BM)) % M
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    g_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    u_ptrs = g_ptrs + N * stride_wn

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        g = tl.load(g_ptrs)
        u = tl.load(u_ptrs)
        acc_g = tl.dot(a, g, acc_g)
        acc_u = tl.dot(a, u, acc_u)
        a_ptrs += BK * stride_ak
        g_ptrs += BK * stride_wk
        u_ptrs += BK * stride_wk

    scale = tl.load(sa_ptr) * tl.load(sb_ptr)
    h_g = acc_g * scale
    h_u = acc_u * scale
    if HAS_BIAS:
        h_g += tl.load(b_ptr + offs_n).to(tl.float32)[None, :]
        h_u += tl.load(b_ptr + N + offs_n).to(tl.float32)[None, :]
    y = h_g * tl.sigmoid(h_g) * h_u

    offs_ym = pid_m * BM + tl.arange(0, BM)
    tl.store(
        y_ptr + offs_ym[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y.to(tl.bfloat16),
        mask=offs_ym[:, None] < M,
    )


def _launch(x_fp8, x_scale, w_fp8, w_scale, bias, out) -> None:
    """Launch the kernel over contiguous ``x_fp8`` [M, K] and ``w_fp8`` [2F, K] into ``out`` [M, F]."""
    (M, K) = x_fp8.shape
    F = out.shape[1]
    grid = (triton.cdiv(M, _BM) * (F // _BN),)
    sana_fp8_swiglu_sm120_gemm[grid](
        x_fp8,
        w_fp8,
        bias if bias is not None else x_scale,
        out,
        M,
        F,
        K,
        x_scale,
        w_scale,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BM=_BM,
        BN=_BN,
        BK=_BK,
        GROUP=_GROUP,
        num_stages=_STAGES,
        num_warps=_WARPS,
    )


def supports(in_features: int, hidden_features: int) -> bool:
    """Return whether the kernel serves a SwiGLU of these sizes: the input features and the hidden features both positive multiples of 64, the steps of its loops."""
    return in_features >= _BK and in_features % _BK == 0 and hidden_features >= _BN and hidden_features % _BN == 0


def fp8_swiglu_gate_up(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Return ``silu(x @ Wg^T + bg) * (x @ Wu^T + bu)`` [rows, F] in bf16 for contiguous ``x`` [rows, K], the packed E4M3 weight [2F, K] (gate rows, then up rows) with its scale and the bf16 bias [2F]: ``x`` is quantized to E4M3, then one kernel computes both products and the activation without a bf16 intermediate."""
    (x_fp8, x_scale) = quantize_fp8(x)
    out = torch.empty((x.shape[0], weight.shape[0] // 2), dtype=torch.bfloat16, device=x.device)
    _launch(x_fp8, x_scale, weight, weight_scale, bias, out)
    return out
