# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_K_ALIGN = 64
_BM = 256
_BN = 32
_BK = 64
_GROUP = 8
_STAGES = 2
_WARPS = 4
_DEVICE_RECIPES = (("RTX PRO 6000", (128, 64, 64, 8, 3, 8)),)
_DEVICE = {}


def launch_recipe(device: torch.device) -> tuple:
    """Return the launch constants (BM, BN, BK, GROUP, stages, warps) of a device: the RTX PRO 6000 has its own, every other device the default."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    entry = _DEVICE.get(index)
    if entry is None:
        entry = _DEVICE[index] = (tuple(torch.cuda.get_device_capability(index)), torch.cuda.get_device_name(index))
    (capability, name) = entry
    if capability == (12, 0):
        for key, recipe in _DEVICE_RECIPES:
            if key in name:
                return recipe
    return (_BM, _BN, _BK, _GROUP, _STAGES, _WARPS)


@triton.jit
def sana_swiglu_sm120_gemm(
    x_ptr,
    w_ptr,
    b_ptr,
    h_ptr,
    N,
    F,
    K,
    stride_xn,
    stride_xk,
    stride_wf,
    stride_wk,
    stride_hn,
    stride_hf,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """One [BM, BN] tile of h [N, F] = silu(x @ Wg^T + bg) * (x @ Wu^T + bu) for bf16 x [N, K] and the packed bf16 weight [2F, K] (gate rows [0, F), up rows [F, 2F)) and bias [2F]: two float32 accumulators over the same x tile, then gate, up, silu and the product each rounded to bf16 as the separate layers round them."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(N, BM)
    num_n = tl.cdiv(F, BN)
    num_in_group = GROUP * num_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP
    group_size = tl.minimum(num_m - first_m, GROUP)
    pid_m = first_m + ((pid % num_in_group) % group_size)
    pid_n = (pid % num_in_group) // group_size

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = rm < N
    n_mask = rn < F

    x_ptrs = x_ptr + rm[:, None] * stride_xn + rk[None, :] * stride_xk
    g_ptrs = w_ptr + rn[None, :] * stride_wf + rk[:, None] * stride_wk
    u_ptrs = w_ptr + (F + rn)[None, :] * stride_wf + rk[:, None] * stride_wk

    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for _k in range(0, tl.cdiv(K, BK)):
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        g = tl.load(g_ptrs, mask=n_mask[None, :], other=0.0)
        u = tl.load(u_ptrs, mask=n_mask[None, :], other=0.0)
        acc_g = tl.dot(a, g, acc_g)
        acc_u = tl.dot(a, u, acc_u)
        x_ptrs += BK * stride_xk
        g_ptrs += BK * stride_wk
        u_ptrs += BK * stride_wk

    b_g = tl.load(b_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)
    b_u = tl.load(b_ptr + F + rn, mask=n_mask, other=0.0).to(tl.float32)
    gate = (acc_g + b_g[None, :]).to(tl.bfloat16).to(tl.float32)
    up = (acc_u + b_u[None, :]).to(tl.bfloat16).to(tl.float32)
    s = tl.fdiv(gate, 1.0 + libdevice.exp(-gate), ieee_rounding=True)
    s = s.to(tl.bfloat16).to(tl.float32)
    h = (s * up).to(tl.bfloat16)
    tl.store(
        h_ptr + rm[:, None] * stride_hn + rn[None, :] * stride_hf,
        h,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _launch(x2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, h2d: torch.Tensor) -> None:
    """Launch the kernel over x2d [N, K] into h2d [N, F] with the launch constants of the device."""
    (tokens, in_features) = x2d.shape
    hidden = weight.shape[0] // 2
    (bm, bn, bk, group, stages, warps) = launch_recipe(x2d.device)
    grid = (triton.cdiv(tokens, bm) * triton.cdiv(hidden, bn),)
    sana_swiglu_sm120_gemm[grid](
        x2d,
        weight,
        bias,
        h2d,
        tokens,
        hidden,
        in_features,
        x2d.stride(0),
        x2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        h2d.stride(0),
        h2d.stride(1),
        BM=bm,
        BN=bn,
        BK=bk,
        GROUP=group,
        num_stages=stages,
        num_warps=warps,
    )


def supports(in_features: int, hidden_features: int) -> bool:
    """Return whether the kernel serves a SwiGLU of these sizes: the input features a multiple of 64."""
    return in_features >= _K_ALIGN and in_features % _K_ALIGN == 0


def swiglu_gate_up(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return ``silu(x @ Wg^T + bg) * (x @ Wu^T + bu)`` as a new bf16 tensor [1, M, F] for contiguous bf16 ``x`` [1, M, K], the packed weight [2F, K] (gate rows, then up rows) and bias [2F]; bitwise equal to the separate layers."""
    out = torch.empty((1, x.shape[1], weight.shape[0] // 2), dtype=torch.bfloat16, device=x.device)
    _launch(x[0], weight, bias, out[0])
    return out
