"""Block summaries and routing thresholds shared by both CuTe kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK_SIZE = 64
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    # Tune once per padded descriptor capacity.  ``T`` is the logical prompt-dependent length;
    # keying on it puts autotune back on the first request after every prompt-length change.
    key=["N"],
)
@triton.jit
def _reduce_kc_kernel(
    k_desc,
    kc,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    values = k_desc.load(
        [batch, block * BLOCK, head, d_tile * TILE_D]
    ).reshape([BLOCK, TILE_D])
    summary = tl.sum(values, axis=0) / block_len
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    tl.store(
        kc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["N"],
)
@triton.jit
def _reduce_vc_kernel(
    v_desc,
    vc,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    values = v_desc.load(
        [batch, block * BLOCK, head, d_tile * TILE_D]
    ).reshape([BLOCK, TILE_D])
    summary = tl.sum(values, axis=0)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    tl.store(
        vc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2, 3, 4)
    ],
    key=["N"],
)
@triton.jit
def _reduce_kv_kernel(
    k_desc,
    v_desc,
    kc,
    vc,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """Compute the independent K centroid and V sum in one launch."""
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    k_values = k_desc.load(
        [batch, block * BLOCK, head, d_tile * TILE_D]
    ).reshape([BLOCK, TILE_D])
    v_values = v_desc.load(
        [batch, block * BLOCK, head, d_tile * TILE_D]
    ).reshape([BLOCK, TILE_D])
    k_summary = tl.sum(k_values, axis=0) / block_len
    v_summary = tl.sum(v_values, axis=0)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    output_offsets = ((batch * N + block) * H + head) * D + offsets
    valid = offsets < D
    tl.store(kc + output_offsets, k_summary, mask=valid)
    tl.store(vc + output_offsets, v_summary, mask=valid)


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["N"],
)
@triton.jit
def _reduce_kc_stats_kernel(
    kc_desc,
    kc_mean,
    kc_var_diag,
    valid_blocks,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    TILE_D: tl.constexpr,
    GROUP: tl.constexpr,
):
    d_tile, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    block_offsets = tl.arange(0, GROUP)
    block_offsets = tl.max_contiguous(block_offsets, GROUP)
    d_offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    total = tl.zeros((TILE_D,), dtype=tl.float32)
    total_sq = tl.zeros((TILE_D,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, N, GROUP):
        valid = start + block_offsets < valid_blocks
        values = kc_desc.load(
            [batch, start, head, d_tile * TILE_D]
        ).reshape([GROUP, TILE_D]).to(tl.float32)
        values = tl.where(valid[:, None], values, 0.0)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        count += tl.sum(valid.to(tl.float32), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    valid_d = d_offsets < D
    tl.store(
        kc_mean + batch_head * D + d_offsets,
        mean,
        mask=valid_d,
    )
    tl.store(
        kc_var_diag + batch_head * D + d_offsets,
        variance,
        mask=valid_d,
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["N"],
)
@triton.jit
def _diag_threshold_kernel(
    q_desc,
    kc_mean,
    kc_var_diag,
    global_threshold,
    q_bar,
    softmax_scale,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
    TAU: tl.constexpr,
    STORE_Q_BAR: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    d_offsets = tl.arange(0, TILE_D)
    valid_d = d_offsets < D
    q_values = q_desc.load(
        [batch, q_start, head, 0]
    ).reshape([BLOCK, TILE_D])
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    if STORE_Q_BAR:
        tl.store(
            q_bar + ((batch * N + q_block) * H + head) * D + d_offsets,
            q_centroid,
            mask=valid_d,
        )
    mean_kc = tl.load(
        kc_mean + batch_head * D + d_offsets,
        mask=valid_d,
        other=0.0,
    )
    var_kc = tl.load(
        kc_var_diag + batch_head * D + d_offsets,
        mask=valid_d,
        other=0.0,
    )
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc, axis=0
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(
        global_threshold + (batch * N + q_block) * H + head,
        mean + TAU * std,
    )


@triton.jit
def _pool_query_kernel(
    q_desc,
    q_bar,
    T,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    offsets = tl.arange(0, TILE_D)
    values = q_desc.load([batch, q_start, head, 0]).reshape(
        [BLOCK, TILE_D]
    )
    centroid = tl.sum(values.to(tl.float32), axis=0) / q_len
    tl.store(
        q_bar + (batch_head * N + q_block) * D + offsets,
        centroid,
        mask=offsets < D,
    )


@triton.jit
def _exact_fused_threshold_kernel(
    q_bar,
    kc_mean,
    kc_second_moment,
    global_threshold,
    softmax_scale,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_D: tl.constexpr,
    TAU: tl.constexpr,
):
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets = tl.arange(0, TILE_D)
    valid_rows = rows < N
    valid_d = offsets < D

    q_centroid = tl.load(
        q_bar + (batch_head * N + rows[:, None]) * D + offsets[None, :],
        mask=valid_rows[:, None] & valid_d[None, :],
        other=0.0,
    )
    mean_kc = tl.load(
        kc_mean + batch_head * D + offsets,
        mask=valid_d,
        other=0.0,
    )
    second_moment = tl.load(
        kc_second_moment
        + batch_head * D * D
        + offsets[:, None] * D
        + offsets[None, :],
        mask=valid_d[:, None] & valid_d[None, :],
        other=0.0,
    )

    raw_mean = tl.sum(q_centroid.to(tl.float32) * mean_kc[None, :], axis=1)
    projected = tl.dot(
        q_centroid,
        second_moment,
        out_dtype=tl.float32,
    )
    raw_second_moment = tl.sum(
        projected * q_centroid.to(tl.float32),
        axis=1,
    )
    log2_scale = softmax_scale * 1.4426950408889634
    mean = raw_mean * log2_scale
    variance = tl.maximum(
        raw_second_moment - raw_mean * raw_mean,
        0.0,
    ) * (log2_scale * log2_scale)
    threshold = mean + TAU * tl.sqrt(variance + 1.0e-6)
    batch, head = batch_head // H, batch_head % H
    tl.store(
        global_threshold + (batch * N + rows) * H + head,
        threshold,
        mask=valid_rows,
    )


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    valid_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, capacity_tokens, heads, head_dim = k.shape
    tokens = capacity_tokens if valid_tokens is None else int(valid_tokens)
    if not 0 < tokens <= capacity_tokens:
        raise ValueError(f"valid_tokens must be in [1, {capacity_tokens}], got {tokens}")
    capacity_blocks = triton.cdiv(capacity_tokens, BLOCK_SIZE)
    valid_blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    kc = torch.empty(
        (batch, capacity_blocks, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_like(kc)
    k_desc = TensorDescriptor.from_tensor(
        k,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    v_desc = TensorDescriptor.from_tensor(
        v,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    grid = (triton.cdiv(head_dim, tile_d), valid_blocks, batch * heads)
    _reduce_kv_kernel[grid](
        k_desc,
        v_desc,
        kc,
        vc,
        tokens,
        heads,
        capacity_blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    # The SM100 route mainloop TMA-loads a complete 64-block summary tile and
    # masks lanes beyond ``valid_blocks`` afterwards.  With a compile bucket,
    # those lanes are still in-bounds for KC/VC.  Leaving them uninitialized
    # permits NaNs to enter the score tile before the logical mask is applied
    # (NaN + -inf remains NaN).  Only the bucket tail is cleared; real block
    # summaries and their arithmetic remain bit-for-bit unchanged.
    if valid_blocks < capacity_blocks:
        kc[:, valid_blocks:].zero_()
        vc[:, valid_blocks:].zero_()
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    valid_tokens: int | None = None,
    return_q_bar: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch, capacity_tokens, heads, head_dim = q.shape
    tokens = capacity_tokens if valid_tokens is None else int(valid_tokens)
    capacity_blocks = triton.cdiv(capacity_tokens, BLOCK_SIZE)
    valid_blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    kc_mean = torch.empty(
        (batch, heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    kc_var_diag = torch.empty_like(kc_mean)
    global_threshold = torch.empty(
        (batch, capacity_blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    q_bar = (
        torch.empty(
            (batch, capacity_blocks, heads, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        if return_q_bar
        else global_threshold
    )
    q_desc = TensorDescriptor.from_tensor(
        q,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    kc_desc = TensorDescriptor.from_tensor(
        kc,
        [1, THRESHOLD_GROUP_SIZE, 1, tile_d],
    )
    _reduce_kc_stats_kernel[
        (triton.cdiv(head_dim, tile_d), batch * heads)
    ](
        kc_desc,
        kc_mean,
        kc_var_diag,
        valid_blocks,
        heads,
        capacity_blocks,
        head_dim,
        tile_d,
        THRESHOLD_GROUP_SIZE,
    )
    _diag_threshold_kernel[(valid_blocks, batch * heads)](
        q_desc,
        kc_mean,
        kc_var_diag,
        global_threshold,
        q_bar,
        scale,
        tokens,
        heads,
        capacity_blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        tau,
        return_q_bar,
    )
    if return_q_bar:
        return global_threshold, q_bar
    return global_threshold


def _compute_exact_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    valid_tokens: int | None = None,
) -> torch.Tensor:
    batch, capacity_tokens, heads, head_dim = q.shape
    tokens = capacity_tokens if valid_tokens is None else int(valid_tokens)
    capacity_blocks = triton.cdiv(capacity_tokens, BLOCK_SIZE)
    valid_blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    batch_heads = batch * heads
    kc_bh = kc[:, :valid_blocks].permute(0, 2, 1, 3)
    kc_mean = kc_bh.mean(dim=2, dtype=torch.float32)
    kc_second_moment = torch.matmul(
        kc_bh.transpose(-1, -2),
        kc_bh,
    )
    kc_second_moment.div_(valid_blocks)
    q_bar = torch.empty(
        (batch_heads, capacity_blocks, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    global_threshold = torch.empty(
        (batch, capacity_blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    q_desc = TensorDescriptor.from_tensor(
        q,
        [1, BLOCK_SIZE, 1, tile_d],
    )
    _pool_query_kernel[(valid_blocks, batch_heads)](
        q_desc,
        q_bar,
        tokens,
        heads,
        capacity_blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        num_warps=4,
        num_stages=1,
    )
    block_m = 64
    _exact_fused_threshold_kernel[(triton.cdiv(valid_blocks, block_m), batch_heads)](
        q_bar,
        kc_mean,
        kc_second_moment,
        global_threshold,
        scale,
        heads,
        capacity_blocks,
        head_dim,
        block_m,
        tile_d,
        tau,
        num_warps=4,
        num_stages=1,
    )
    return global_threshold


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    thresh_type: str = "diag",
    valid_tokens: int | None = None,
    return_q_bar: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
):
    capacity_tokens = q.shape[1]
    tokens = capacity_tokens if valid_tokens is None else int(valid_tokens)
    if not 0 < tokens <= capacity_tokens:
        raise ValueError(f"valid_tokens must be in [1, {capacity_tokens}], got {tokens}")
    kc, vc = _reduce_kv(k, v, valid_tokens=tokens)
    if thresh_type == "exact":
        threshold = _compute_exact_threshold(
            q, kc, tau=tau, scale=scale, valid_tokens=tokens
        )
        q_bar = None
    else:
        threshold_result = _compute_diag_threshold(
            q,
            kc,
            tau=tau,
            scale=scale,
            valid_tokens=tokens,
            return_q_bar=return_q_bar,
        )
        if return_q_bar:
            threshold, q_bar = threshold_result
        else:
            threshold = threshold_result
            q_bar = None
    if return_q_bar:
        return kc, vc, threshold, q_bar
    return kc, vc, threshold


__all__ = ["prepare"]
