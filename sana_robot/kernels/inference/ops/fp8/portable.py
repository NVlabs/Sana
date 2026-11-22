# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

E4M3 = torch.float8_e4m3fn
E4M3_MAX_C = tl.constexpr(448.0)
SIZE_MULTIPLE = 16


@triton.jit
def _amax_atomic_kernel(x_ptr, amax_ptr, numel, BLOCK: tl.constexpr):
    """One block of a flat tensor: its largest absolute value in float32, merged into ``amax_ptr`` by an atomic maximum."""
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + idx, mask=idx < numel, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(v), axis=0))


@triton.jit
def _scale_cast_kernel(x_ptr, out_ptr, amax_ptr, scale_ptr, numel, BLOCK: tl.constexpr):
    """One block of a flat tensor divided by ``amax / 448`` and rounded to E4M3; the first block also writes that scale."""
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    amax = tl.maximum(tl.load(amax_ptr), 1e-12)
    if pid == 0:
        tl.store(scale_ptr, amax / E4M3_MAX_C)
    inv = E4M3_MAX_C / amax
    v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    q = tl.minimum(tl.maximum(v * inv, -E4M3_MAX_C), E4M3_MAX_C)
    tl.store(out_ptr + idx, q.to(tl.float8e4nv), mask=mask)


@triton.jit
def _swiglu_epilogue_kernel(h_ptr, out_ptr, F, BLOCK: tl.constexpr):
    """One block of one row of the packed [rows, 2F] gate and up projections: ``silu(gate) * up`` in float32, rounded to bf16."""
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = idx < F
    g = tl.load(h_ptr + row * 2 * F + idx, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(h_ptr + row * 2 * F + F + idx, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + row * F + idx, y.to(tl.bfloat16), mask=mask)


def supports(out_features: int, in_features: int) -> bool:
    """Return whether the FP8 matrix product serves a weight [out_features, in_features]: both sizes multiples of 16."""
    return out_features % SIZE_MULTIPLE == 0 and in_features % SIZE_MULTIPLE == 0


def absolute_maximum(x: torch.Tensor) -> torch.Tensor:
    """Return the largest absolute value of a contiguous tensor as a float32 scalar on its device; the value stays on the device."""
    amax = torch.zeros((), device=x.device, dtype=torch.float32)
    numel = x.numel()
    _amax_atomic_kernel[(triton.cdiv(numel, 4096),)](x.view(-1), amax, numel, BLOCK=4096, num_warps=8)
    return amax


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a contiguous tensor [rows, columns] in E4M3 and its float32 scale [1]: the scale is the largest absolute value of ``x`` over 448, and ``x`` is about the E4M3 tensor times the scale."""
    numel = x.numel()
    amax = absolute_maximum(x)
    scale = torch.empty((1,), device=x.device, dtype=torch.float32)
    out = torch.empty_like(x, dtype=E4M3)
    _scale_cast_kernel[(triton.cdiv(numel, 4096),)](
        x.view(-1), out.view(-1), amax, scale, numel, BLOCK=4096, num_warps=8
    )
    return (out, scale)


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return ``x @ weight^T + bias`` [rows, out] in bf16 as one FP8 matrix product: contiguous ``x`` [rows, in] is quantized here, ``weight`` [out, in] is E4M3 with its scale [1] (``quantize_fp8``) and ``bias`` [out] is bf16."""
    (x_fp8, x_scale) = quantize_fp8(x)
    return torch._scaled_mm(
        x_fp8,
        weight.t(),
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=bias,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


def swiglu_epilogue(h: torch.Tensor) -> torch.Tensor:
    """Return ``silu(h[:, :F]) * h[:, F:]`` [rows, F] in bf16 for the contiguous packed gate and up projections ``h`` [rows, 2F]."""
    (rows, twof) = h.shape
    hidden = twof // 2
    out = torch.empty((rows, hidden), device=h.device, dtype=torch.bfloat16)
    grid = (rows, triton.cdiv(hidden, 2048))
    _swiglu_epilogue_kernel[grid](h, out, hidden, BLOCK=2048, num_warps=8)
    return out


def fp8_swiglu_gate_up(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Return ``silu(x @ Wg^T + bg) * (x @ Wu^T + bu)`` [rows, F] in bf16 for contiguous ``x`` [rows, K], the packed E4M3 weight [2F, K] (gate rows, then up rows) with its scale and the bf16 bias [2F]: one FP8 matrix product into bf16, then the activation kernel."""
    return swiglu_epilogue(fp8_linear(x, weight, weight_scale, bias))
