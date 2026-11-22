# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from sana_robot.policy.attention.causal_softmax import teacher_forcing_reads
from sana_robot.policy.layout.teacher_forcing import teacher_forcing_observation_index

_CSR_BLOCK = 512


@triton.jit
def _csr_sum_kernel(G, PTR, POS, OUT, W, BLOCK_W: tl.constexpr):
    """OUT[r] = the sum of G[POS[i]] for PTR[r] <= i < PTR[r + 1], accumulated in float32; rows of W elements."""
    r = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    inside = offs < W
    lo = tl.load(PTR + r)
    hi = tl.load(PTR + r + 1)
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for i in range(lo, hi):
        src = tl.load(POS + i).to(tl.int64)
        acc += tl.load(G + src * W + offs, mask=inside, other=0.0).to(tl.float32)
    tl.store(OUT + r * W + offs, acc.to(OUT.dtype.element_ty), mask=inside)


class PermuteRows(torch.autograd.Function):
    """x.index_select(0, perm); the backward gathers with the inverse permutation."""

    @staticmethod
    def forward(ctx, x, perm, inverse):
        ctx.save_for_backward(inverse)
        return x.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return (grad.index_select(0, inverse), None, None)


class GatherRows(torch.autograd.Function):
    """``src.index_select(0, index)`` whose backward sums the copies of every source row through the CSR table (ptr, pos) of ``index``."""

    @staticmethod
    def forward(ctx, src, index, ptr, pos):
        ctx.save_for_backward(index, ptr, pos)
        ctx.rows = src.shape[0]
        return src.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (index, ptr, pos) = ctx.saved_tensors
        grad = grad.contiguous()
        shape = (ctx.rows, *grad.shape[1:])
        if grad.is_cuda and grad.dtype in (torch.float16, torch.bfloat16):
            out = torch.empty(shape, dtype=grad.dtype, device=grad.device)
            width = grad[0].numel()
            _csr_sum_kernel[ctx.rows, triton.cdiv(width, _CSR_BLOCK)](grad, ptr, pos, out, width, BLOCK_W=_CSR_BLOCK)
            return (out, None, None, None)
        sum_dtype = torch.promote_types(grad.dtype, torch.float32)
        out = torch.zeros(shape, dtype=sum_dtype, device=grad.device).index_add_(0, index, grad.to(sum_dtype))
        return (out.to(grad.dtype), None, None, None)


class VarlenPlan(NamedTuple):
    """The teacher-forcing attention of a batch packed into variable-length groups. ``q_index`` / ``q_inverse`` permute the B * N token rows into group order and back; ``k_index`` and ``v_index`` pick the keys and values of every group, with their CSR tables (``*_ptr``, ``*_pos``) for the backward; ``cu_q`` / ``cu_k`` and ``max_q`` / ``max_k`` are the group bounds; ``sink_starts`` are the window starts of the rotated sink copies."""

    q_index: torch.Tensor
    q_inverse: torch.Tensor
    k_index: torch.Tensor
    k_ptr: torch.Tensor
    k_pos: torch.Tensor
    v_index: torch.Tensor
    v_ptr: torch.Tensor
    v_pos: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    sink_starts: torch.Tensor


def teacher_forcing_varlen_groups(
    bounds: tuple,
    clean_segments: int,
    window: int,
    sink: int,
    observation_spans: tuple = (),
) -> tuple:
    """Return (groups, sink_starts) of one teacher-forcing sequence: ``groups`` lists (query rows, key rows) per segment, a noisy chunk as two groups (its prediction tokens, then its observation tokens); key rows address [sequence | one rotated sink copy per start in ``sink_starts``]."""
    reads = teacher_forcing_reads(bounds, clean_segments, window, sink)
    starts = sorted({start for (_visible, start) in reads if start is not None})
    (total_tokens, sink_end) = (bounds[-1], bounds[sink])
    observation = torch.zeros(total_tokens, dtype=torch.bool)
    if observation_spans:
        observation[teacher_forcing_observation_index(bounds, clean_segments, observation_spans)[0].cpu()] = True
    groups = []
    for segment, (visible, start) in enumerate(reads):
        tokens = torch.arange(bounds[segment], bounds[segment + 1])
        others = [torch.arange(bounds[other], bounds[other + 1]) for other in visible if other != segment]
        copy = [total_tokens + starts.index(start) * sink_end + torch.arange(sink_end)] if start is not None else []
        own = observation[tokens]
        if bool((~own).any()):
            groups.append((tokens[~own], torch.cat(others + [tokens] + copy)))
        if bool(own.any()):
            groups.append((tokens[own], torch.cat(others + [tokens[own]] + copy)))
    return (groups, starts)


def csr_table(index: torch.Tensor, rows: int):
    """Return the CSR table of ``index``: the row pointers [rows + 1] and the positions of ``index`` sorted by source row, both int32."""
    order = torch.argsort(index, stable=True)
    ptr = torch.zeros(rows + 1, dtype=torch.int64)
    ptr[1:] = torch.cumsum(torch.bincount(index, minlength=rows), 0)
    return (ptr.to(torch.int32), order.to(torch.int32))


@lru_cache(maxsize=128)
def teacher_forcing_varlen_plan(
    bounds: tuple,
    clean_segments: int,
    window: int,
    sink: int,
    observation_spans: tuple,
    batch: int,
    device: torch.device,
) -> VarlenPlan:
    """Return the ``VarlenPlan`` of ``batch`` teacher-forcing sequences with these bounds, cached per layout, batch size and device."""
    (groups, starts) = teacher_forcing_varlen_groups(bounds, clean_segments, window, sink, observation_spans)
    (tokens, sink_end) = (bounds[-1], bounds[sink])
    extended_tokens = tokens + len(starts) * sink_end
    q_one = torch.cat([q for (q, _) in groups])
    k_one = torch.cat([k for (_, k) in groups])
    v_one = torch.where(k_one < tokens, k_one, (k_one - tokens) % max(sink_end, 1))
    sample = torch.arange(batch)[:, None]
    q_index = (q_one + sample * tokens).reshape(-1)
    k_index = (k_one + sample * extended_tokens).reshape(-1)
    v_index = (v_one + sample * tokens).reshape(-1)
    q_inverse = torch.empty_like(q_index)
    q_inverse[q_index] = torch.arange(q_index.numel())

    def cumulative(lengths):
        return torch.tensor([0] + lengths * batch).cumsum(0).to(torch.int32)

    q_lengths = [q.numel() for (q, _) in groups]
    k_lengths = [k.numel() for (_, k) in groups]
    (k_ptr, k_pos) = csr_table(k_index, batch * extended_tokens)
    (v_ptr, v_pos) = csr_table(v_index, batch * tokens)
    tensors = [
        q_index,
        q_inverse,
        k_index,
        k_ptr,
        k_pos,
        v_index,
        v_ptr,
        v_pos,
        cumulative(q_lengths),
        cumulative(k_lengths),
    ]
    tensors = [tensor.to(device) for tensor in tensors]
    return VarlenPlan(*tensors, max(q_lengths), max(k_lengths), torch.tensor(starts, dtype=torch.long, device=device))


def varlen_teacher_forcing_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bounds,
    clean_segments: int,
    window: int,
    sink: int,
    sink_phases: torch.Tensor,
    observation_spans: tuple,
    rotated_sink_keys: Callable,
    deterministic: bool = False,
) -> torch.Tensor:
    """Run the teacher-forcing attention as packed variable-length groups through FlashAttention."""
    sink = int(sink)
    (batch_size, tokens, heads, dim) = q.shape
    plan = teacher_forcing_varlen_plan(
        tuple((int(bound) for bound in bounds)),
        int(clean_segments),
        int(window),
        sink,
        tuple(observation_spans),
        batch_size,
        q.device,
    )
    if plan.sink_starts.numel():
        sink_end = int(bounds[sink])
        k = torch.cat([k, rotated_sink_keys(k[:, :sink_end], sink_phases, plan.sink_starts)], dim=1)
    q_packed = PermuteRows.apply(q.reshape(batch_size * tokens, heads, dim), plan.q_index, plan.q_inverse)
    k_packed = GatherRows.apply(k.reshape(-1, heads, dim), plan.k_index, plan.k_ptr, plan.k_pos)
    v_packed = GatherRows.apply(v.reshape(batch_size * tokens, heads, dim), plan.v_index, plan.v_ptr, plan.v_pos)
    from flash_attn import flash_attn_varlen_func

    extra = {"deterministic": True} if deterministic else {}
    out = flash_attn_varlen_func(q_packed, k_packed, v_packed, plan.cu_q, plan.cu_k, plan.max_q, plan.max_k, **extra)
    return PermuteRows.apply(out, plan.q_inverse, plan.q_index).reshape(batch_size, tokens, heads, dim)


@lru_cache(maxsize=64)
def packed_row_varlen_plan(windows: tuple, window: int, sink: int, observation_spans: tuple, device: torch.device):
    """Return the ``VarlenPlan`` of a packed row of teacher-forcing windows, ``windows`` = ((first token, bounds, clean segments), ...), and every window's sink copy starts. The key rows are [row tokens | the rotated sink copies of window 1 | of window 2 | ...]; a window's groups address its own tokens and copies only."""
    row_tokens = windows[-1][0] + windows[-1][1][-1]
    (q_rows, k_rows, v_rows, q_lengths, k_lengths, sink_starts, copy_rows) = ([], [], [], [], [], [], 0)
    for first_token, bounds, clean_segments in windows:
        (groups, starts) = teacher_forcing_varlen_groups(bounds, clean_segments, window, sink, observation_spans)
        (tokens, sink_end) = (bounds[-1], bounds[sink])
        for queries, keys in groups:
            own = keys < tokens
            q_rows.append(queries + first_token)
            k_rows.append(torch.where(own, keys + first_token, keys - tokens + row_tokens + copy_rows))
            v_rows.append(torch.where(own, keys, (keys - tokens) % max(sink_end, 1)) + first_token)
            q_lengths.append(queries.numel())
            k_lengths.append(keys.numel())
        copy_rows += len(starts) * sink_end
        sink_starts.append(torch.tensor(starts, dtype=torch.long, device=device))
    (q_index, k_index, v_index) = (torch.cat(q_rows), torch.cat(k_rows), torch.cat(v_rows))
    q_inverse = torch.empty_like(q_index)
    q_inverse[q_index] = torch.arange(q_index.numel())
    (k_ptr, k_pos) = csr_table(k_index, row_tokens + copy_rows)
    (v_ptr, v_pos) = csr_table(v_index, row_tokens)
    tensors = [
        q_index,
        q_inverse,
        k_index,
        k_ptr,
        k_pos,
        v_index,
        v_ptr,
        v_pos,
        torch.tensor([0] + q_lengths).cumsum(0).to(torch.int32),
        torch.tensor([0] + k_lengths).cumsum(0).to(torch.int32),
    ]
    tensors = [tensor.to(device) for tensor in tensors]
    plan = VarlenPlan(*tensors, max(q_lengths), max(k_lengths), torch.cat(sink_starts))
    return (plan, tuple(sink_starts))


def varlen_packed_row_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    windows,
    window: int,
    sink: int,
    observation_spans: tuple,
    rotated_sink_keys: Callable,
) -> torch.Tensor:
    """Run the teacher-forcing attention of a packed row of windows [1, N, heads, head_dim] as one FlashAttention call over variable-length groups: every window reads its own tokens and its own rotated sink copies."""
    (_, tokens, heads, dim) = q.shape
    (plan, sink_starts) = packed_row_varlen_plan(
        tuple((row.start, row.bounds, row.num_chunks) for row in windows),
        int(window),
        int(sink),
        tuple(observation_spans),
        q.device,
    )
    copies = [
        rotated_sink_keys(k[:, row.start : row.start + row.bounds[sink]], row.sink_phases, starts)
        for (row, starts) in zip(windows, sink_starts)
        if starts.numel()
    ]
    k = torch.cat([k] + copies, dim=1)
    q_packed = PermuteRows.apply(q.reshape(tokens, heads, dim), plan.q_index, plan.q_inverse)
    k_packed = GatherRows.apply(k.reshape(-1, heads, dim), plan.k_index, plan.k_ptr, plan.k_pos)
    v_packed = GatherRows.apply(v.reshape(tokens, heads, dim), plan.v_index, plan.v_ptr, plan.v_pos)
    from flash_attn import flash_attn_varlen_func

    out = flash_attn_varlen_func(
        q_packed, k_packed, v_packed, plan.cu_q, plan.cu_k, plan.max_q, plan.max_k, deterministic=True
    )
    return PermuteRows.apply(out, plan.q_inverse, plan.q_index).reshape(1, tokens, heads, dim)
