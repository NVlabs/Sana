# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

_CA = 2048


@triton.jit
def sana_softmax_output_gate(
    OUT_IN,
    GATE,
    OUT,
    gate_stride,
    C: tl.constexpr,
    CA: tl.constexpr,
    CB: tl.constexpr,
):
    """One token row: the sigmoid of the gate logits in float32 rounded to bf16, times the attention output in float32 rounded to bf16."""
    t = tl.program_id(0)
    ca = tl.arange(0, CA)
    cb = CA + tl.arange(0, CB)

    ga = tl.sigmoid(tl.load(GATE + t * gate_stride + ca).to(tl.float32)).to(tl.bfloat16)
    gb = tl.sigmoid(tl.load(GATE + t * gate_stride + cb).to(tl.float32)).to(tl.bfloat16)
    xa = tl.load(OUT_IN + t * C + ca)
    xb = tl.load(OUT_IN + t * C + cb)
    tl.store(OUT + t * C + ca, (xa.to(tl.float32) * ga.to(tl.float32)).to(tl.bfloat16))
    tl.store(OUT + t * C + cb, (xb.to(tl.float32) * gb.to(tl.float32)).to(tl.bfloat16))


def serves_channels(channels: int) -> bool:
    """Return whether the kernel serves a width: 2048 channels plus a power of two, the two spans it loads a row in."""
    tail = channels - _CA
    return tail > 0 and tail & (tail - 1) == 0


def fused_sigmoid_gate(out: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """Return ``out * sigmoid(gate_logits)`` in one launch on every architecture, bitwise the policy's sigmoid and product in bf16.

    ``out`` is a contiguous [1, N, C] bf16 tensor; ``gate_logits`` [1, N, C] bf16 may be a column slice of a wider projection, read through its row stride.
    """
    tokens, channels = out.shape[1], out.shape[2]
    gated = torch.empty_like(out)
    sana_softmax_output_gate[(tokens,)](
        out,
        gate_logits,
        gated,
        gate_logits.stride(1),
        C=channels,
        CA=_CA,
        CB=channels - _CA,
        num_warps=2,
    )
    return gated
