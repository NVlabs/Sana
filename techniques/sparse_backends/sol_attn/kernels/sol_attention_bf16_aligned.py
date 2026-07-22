# Copyright (c) 2025-2026, Haopeng Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pure-BF16 Triton SOL Attention with canonical global-threshold routing.

The user-provided BF16 prototype uses a separate ``mean + beta * std``
threshold inside every route group.  That makes the selected route depend on
``GROUP_SIZE`` and does not match the current SOL Attention selector.  This module
keeps the prototype's BF16 Q/K/Kc tensor-core dataplane, but uses the current
SOL Attention global diagonal threshold and mandatory local-neighbour exact mask.

No INT8/FP8 Q, K, or K centroid participates in this implementation.  Route
group size is a scheduling parameter only: for a fixed threshold, every
supported group size must produce the same device-side route mask.
"""

from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from kernels import sol_attention_bf16_legacy as legacy
from kernels import sol_attention as canonical_sol_attn


GROUP_SIZE = 32
VALID_GROUP_SIZES = legacy.VALID_GROUP_SIZES


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> None:
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError(
            "pure-BF16 SOL Attention requires BF16 Q/K, got "
            f"q={q.dtype} k={k.dtype}"
        )
    if v.dtype != torch.bfloat16:
        raise TypeError(f"pure-BF16 SOL Attention requires BF16 V, got {v.dtype}")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("pure-BF16 SOL Attention requires Q/K/V on one CUDA device")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "pure-BF16 SOL Attention currently requires equal [B,H,T,D] Q/K/V, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.shape[-1] != 128:
        raise ValueError(f"pure-BF16 SOL Attention currently requires D=128, got {q.shape[-1]}")
    if isinstance(block_size, bool) or block_size != 64:
        raise ValueError(f"pure-BF16 SOL Attention currently requires block_size=64, got {block_size!r}")


def prepare_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    block_size: int = 64,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build BF16 centroids and the canonical per-query global threshold."""

    _validate_inputs(q, k, v, block_size)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    kc, vc = legacy.preprocess_kv(k, v, block_size)
    b, h, t, _ = q.shape
    nt = triton.cdiv(t, block_size)
    unit_scale = torch.ones(
        (b, h, nt, 1), device=q.device, dtype=torch.float32
    )
    global_thresh = canonical_sol_attn.compute_global_qck_threshold(
        q,
        unit_scale,
        kc,
        unit_scale,
        scale,
        block_size,
        tau,
    )
    return kc, vc, global_thresh, unit_scale, unit_scale


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (4, 8)
        for num_stages in (1, 2, 3, 4)
    ],
    key=["T", "GROUP_SIZE"],
)
@triton.jit
def single_pass_dynamic_routing_kernel(
    q_desc,
    k_desc,
    v_desc,
    kc_desc,
    vc_desc,
    global_thresh,
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
):
    """Fused BF16 attention main with canonical SOL Attention selector semantics."""

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)
    token_offsets = tl.arange(0, BT)
    token_offsets = tl.max_contiguous(token_offsets, BT)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)
    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    q_len = tl.minimum(BT, T - q_start).to(tl.float32)

    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")
    sm_scale = scale * 1.44269504
    last_chunk_len = T - (NT - 1) * BT
    thresh = tl.load(global_thresh + i_bh * NT + i_t)

    for start_n in range(0, NT, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)
        chunk_indices = start_n + chunk_offsets
        valid_mask = chunk_offsets < NT - start_n

        b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
        b_vc = vc_desc.load([i_bh, start_n, i_v * BV]).reshape(
            [GROUP_SIZE, BV]
        )
        b_s = tl.dot(b_q, b_kc.T).to(tl.float32) * sm_scale

        col_mean = tl.sum(b_s, axis=0) / q_len
        local_mask = tl.abs(i_t - chunk_indices) <= 1
        is_exact = ((col_mean > thresh) | local_mask) & valid_mask

        # Preserve the compiling user-provided BF16 prototype's accumulator
        # dataflow.  A scalar ``has_approx`` select around this dot triggers
        # Triton 3.7's SM100 OptimizeTMemLayoutsPass failure when combined
        # with the runtime exact loop below.  Per-row equality and the final
        # predicate keep an all-exact group neutral without adding that scalar
        # control edge: alpha=1 and every approximate probability is zero.
        approx_mask = valid_mask & (~is_exact)
        b_s_approx = tl.where(
            approx_mask[None, :], b_s, float("-inf")
        )
        new_m = tl.maximum(m_i, tl.max(b_s_approx, axis=1))
        alpha = tl.math.exp2(tl.where(m_i == new_m, 0.0, m_i - new_m))
        prob = tl.where(
            approx_mask[None, :],
            tl.math.exp2(b_s_approx - new_m[:, None]),
            0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(prob.to(b_vc.dtype), b_vc)
        current_lens = tl.where(
            chunk_indices == NT - 1, last_chunk_len, BT
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(
            prob * current_lens[None, :], axis=1
        )
        m_i = new_m

        # Keep the compiling BF16 prototype's vector-min enumerator.  The
        # earlier mixed-mode experiment failed SM100 TMEM layout optimization
        # after adding scale loads and a bitset/ffs loop around this dataplane.
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
            b_s_exact = tl.dot(b_q, b_k.T).to(tl.float32) * sm_scale
            valid_mask_ex = (kv_start + token_offsets)[None, :] < T
            b_s_exact += tl.where(valid_mask_ex, 0.0, float("-inf"))

            new_m = tl.maximum(m_i, tl.max(b_s_exact, axis=1))
            alpha = tl.math.exp2(m_i - new_m)
            prob_exact = tl.math.exp2(b_s_exact - new_m[:, None])
            l_i = l_i * alpha + tl.sum(prob_exact, axis=1)
            b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])
            acc = acc * alpha[:, None] + tl.dot(
                prob_exact.to(b_v.dtype), b_v
            )
            m_i = new_m

    acc /= l_i[:, None]
    o_desc.store(
        [i_bh, q_start, i_v * BV], acc.to(tl.bfloat16)[None, :, :]
    )


@triton.jit
def route_mask_kernel(
    q_desc,
    kc_desc,
    global_thresh,
    route_mask,
    scale,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Materialize the exact predicate used by the fused main kernel."""

    i_group, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start_n = i_group * GROUP_SIZE
    chunk_offsets = tl.arange(0, GROUP_SIZE)
    chunk_offsets = tl.max_contiguous(chunk_offsets, GROUP_SIZE)
    chunk_indices = start_n + chunk_offsets
    valid_mask = chunk_indices < NT

    q_start = i_t * BT
    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    b_kc = kc_desc.load([i_bh, start_n, 0]).reshape([GROUP_SIZE, BK])
    b_s = tl.dot(b_q, b_kc.T).to(tl.float32) * (scale * 1.44269504)
    q_len = tl.minimum(BT, T - q_start).to(tl.float32)
    col_mean = tl.sum(b_s, axis=0) / q_len
    thresh = tl.load(global_thresh + i_bh * NT + i_t)
    local_mask = tl.abs(i_t - chunk_indices) <= 1
    is_exact = ((col_mean > thresh) | local_mask) & valid_mask

    output_offsets = (i_bh * NT + i_t) * NT + chunk_indices
    tl.store(
        route_mask + output_offsets,
        is_exact.to(tl.uint8),
        mask=valid_mask,
    )


def make_prepared_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    global_thresh: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    block_size: int = 64,
    scale: float | None = None,
) -> Callable[[], torch.Tensor]:
    """Return a reusable main-kernel-only callable with stable output storage."""

    _validate_inputs(q, k, v, block_size)
    group_size = legacy.validate_group_size(group_size)
    b, h, t, d = q.shape
    nt = triton.cdiv(t, block_size)
    if kc.dtype != torch.bfloat16 or tuple(kc.shape) != (b, h, nt, d):
        raise ValueError(f"invalid BF16 Kc shape/dtype: {kc.shape} {kc.dtype}")
    if vc.dtype != torch.bfloat16 or tuple(vc.shape) != (b, h, nt, d):
        raise ValueError(f"invalid BF16 Vc shape/dtype: {vc.shape} {vc.dtype}")
    if global_thresh.dtype != torch.float32 or tuple(global_thresh.shape) != (
        b,
        h,
        nt,
    ):
        raise ValueError(
            "invalid global threshold shape/dtype: "
            f"{global_thresh.shape} {global_thresh.dtype}"
        )

    scale = d**-0.5 if scale is None else float(scale)
    bk = min(128, triton.next_power_of_2(d))
    bv = bk
    output = torch.empty_like(v)
    q_desc = TensorDescriptor.from_tensor(
        q.contiguous().reshape(b * h, t, d), [1, block_size, bk]
    )
    k_desc = TensorDescriptor.from_tensor(
        k.contiguous().reshape(b * h, t, d), [1, block_size, bk]
    )
    v_desc = TensorDescriptor.from_tensor(
        v.contiguous().reshape(b * h, t, d), [1, block_size, bv]
    )
    kc_desc = TensorDescriptor.from_tensor(
        kc.contiguous().reshape(b * h, nt, d), [1, group_size, bk]
    )
    vc_desc = TensorDescriptor.from_tensor(
        vc.contiguous().reshape(b * h, nt, d), [1, group_size, bv]
    )
    o_desc = TensorDescriptor.from_tensor(
        output.reshape(b * h, t, d), [1, block_size, bv]
    )
    threshold = global_thresh.contiguous()
    grid = (triton.cdiv(d, bv), nt, b * h)

    def run() -> torch.Tensor:
        single_pass_dynamic_routing_kernel[grid](
            q_desc=q_desc,
            k_desc=k_desc,
            v_desc=v_desc,
            kc_desc=kc_desc,
            vc_desc=vc_desc,
            global_thresh=threshold,
            o_desc=o_desc,
            scale=scale,
            T=t,
            K=d,
            V=d,
            BT=block_size,
            BK=bk,
            BV=bv,
            NT=nt,
            GROUP_SIZE=group_size,
        )
        return output

    run.output = output  # type: ignore[attr-defined]
    run.group_size = group_size  # type: ignore[attr-defined]
    return run


@torch.no_grad()
def materialize_route_mask(
    q: torch.Tensor,
    kc: torch.Tensor,
    global_thresh: torch.Tensor,
    *,
    group_size: int = GROUP_SIZE,
    block_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    """Return the device predicate as ``[B,H,Nq,Nk]`` uint8 values."""

    if q.dtype != torch.bfloat16 or kc.dtype != torch.bfloat16:
        raise TypeError(f"route census requires BF16 Q/Kc, got {q.dtype}/{kc.dtype}")
    group_size = legacy.validate_group_size(group_size)
    b, h, t, d = q.shape
    nt = triton.cdiv(t, block_size)
    if tuple(kc.shape) != (b, h, nt, d):
        raise ValueError(f"invalid Kc shape {kc.shape}, expected {(b, h, nt, d)}")
    if tuple(global_thresh.shape) != (b, h, nt):
        raise ValueError(
            f"invalid threshold shape {global_thresh.shape}, expected {(b, h, nt)}"
        )
    scale = d**-0.5 if scale is None else float(scale)
    bk = min(128, triton.next_power_of_2(d))
    output = torch.zeros((b, h, nt, nt), device=q.device, dtype=torch.uint8)
    q_desc = TensorDescriptor.from_tensor(
        q.contiguous().reshape(b * h, t, d), [1, block_size, bk]
    )
    kc_desc = TensorDescriptor.from_tensor(
        kc.contiguous().reshape(b * h, nt, d), [1, group_size, bk]
    )
    route_mask_kernel[(triton.cdiv(nt, group_size), nt, b * h)](
        q_desc=q_desc,
        kc_desc=kc_desc,
        global_thresh=global_thresh.contiguous(),
        route_mask=output,
        scale=scale,
        T=t,
        K=d,
        BT=block_size,
        BK=bk,
        NT=nt,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=2,
    )
    return output


@torch.no_grad()
def sol_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    block_size: int = 64,
    scale: float | None = None,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Run complete pure-BF16, routing-aligned SOL Attention forward."""

    kc, vc, global_thresh, _, _ = prepare_qkv(
        q, k, v, tau=tau, block_size=block_size, scale=scale
    )
    return make_prepared_runner(
        q,
        k,
        v,
        kc,
        vc,
        global_thresh,
        group_size=group_size,
        block_size=block_size,
        scale=scale,
    )()


__all__ = [
    "GROUP_SIZE",
    "VALID_GROUP_SIZES",
    "make_prepared_runner",
    "materialize_route_mask",
    "sol_attention",
    "prepare_qkv",
]
