"""GPU compaction helpers for cuDNN block-sparse metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _route_and_compact_kernel(
    scores_ptr,
    threshold_ptr,
    route_ptr,
    indices_ptr,
    nums_ptr,
    logical_blocks,
    scale_log2,
    sink_first,
    sink_last,
    sink_blocks_ptr,
    score_stride_b,
    score_stride_q,
    score_stride_k,
    score_stride_h,
    threshold_stride_b,
    threshold_stride_q,
    threshold_stride_h,
    HEADS: tl.constexpr,
    METADATA_BLOCKS: tl.constexpr,
    CHUNK: tl.constexpr,
    HAS_SINK_BLOCKS: tl.constexpr,
):
    """Apply the route policy and emit both residual mask and cuDNN lists."""
    row = tl.program_id(0)
    query_block = row % METADATA_BLOCKS
    batch_head = row // METADATA_BLOCKS
    head = batch_head % HEADS
    batch = batch_head // HEADS
    is_logical = query_block < logical_blocks

    route_base = (batch_head * logical_blocks + query_block) * logical_blocks
    output_base = row * METADATA_BLOCKS
    count = tl.zeros((), dtype=tl.int32)
    threshold_offset = (
        batch * threshold_stride_b
        + query_block * threshold_stride_q
        + head * threshold_stride_h
    )
    row_threshold = tl.load(
        threshold_ptr + threshold_offset,
        mask=is_logical,
        other=float("inf"),
    )
    if HAS_SINK_BLOCKS:
        query_is_sink = tl.load(
            sink_blocks_ptr + query_block,
            mask=is_logical,
            other=0,
        ).to(tl.int1)
    else:
        query_is_sink = False

    for start in range(0, METADATA_BLOCKS, CHUNK):
        key_blocks = start + tl.arange(0, CHUNK)
        valid = is_logical & (key_blocks < logical_blocks)
        score_offsets = (
            batch * score_stride_b
            + query_block * score_stride_q
            + key_blocks * score_stride_k
            + head * score_stride_h
        )
        score = tl.load(scores_ptr + score_offsets, mask=valid, other=0.0)
        if HAS_SINK_BLOCKS:
            key_is_sink = tl.load(
                sink_blocks_ptr + key_blocks,
                mask=valid,
                other=0,
            ).to(tl.int1)
        else:
            key_is_sink = False
        selected = (
            (score * scale_log2 > row_threshold)
            | (tl.abs(query_block - key_blocks) <= 1)
            | ((key_blocks >= sink_first) & (key_blocks < sink_last))
            | ((query_block >= sink_first) & (query_block < sink_last))
            | key_is_sink
            | query_is_sink
        ) & valid
        selected_i32 = selected.to(tl.int32)

        tl.store(route_ptr + route_base + key_blocks, selected, mask=valid)
        positions = count + tl.cumsum(selected_i32, axis=0) - 1
        tl.store(
            indices_ptr + output_base + positions,
            key_blocks,
            mask=selected,
        )
        count += tl.sum(selected_i32, axis=0)

    tl.store(indices_ptr + output_base, 0, mask=~is_logical)
    tl.store(nums_ptr + row, tl.where(is_logical, count, 1))


@torch.no_grad()
def route_and_compact(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    *,
    scale_log2: float,
    sink_first: int,
    sink_last: int,
    metadata_blocks: int,
    sink_block_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse thresholding, policy masks, layout conversion, and list packing."""
    if scores.ndim != 4 or scores.dtype != torch.float32:
        raise ValueError("scores must be a float32 tensor with shape [B,Q,K,H]")
    batch, query_blocks, key_blocks, heads = scores.shape
    if query_blocks != key_blocks:
        raise ValueError("scores must be square in its block dimensions")
    if tuple(threshold.shape) != (batch, key_blocks, heads):
        raise ValueError(
            f"expected threshold={(batch, key_blocks, heads)}, got {tuple(threshold.shape)}"
        )
    if metadata_blocks < query_blocks:
        raise ValueError(
            f"metadata_blocks={metadata_blocks} is smaller than logical blocks={query_blocks}"
        )
    if sink_block_mask is not None:
        if (
            sink_block_mask.dtype != torch.bool
            or sink_block_mask.device != scores.device
            or not sink_block_mask.is_contiguous()
            or tuple(sink_block_mask.shape) != (key_blocks,)
        ):
            raise ValueError(
                "sink_block_mask must be a contiguous bool tensor on the scores device "
                f"with shape {(key_blocks,)}, got dtype={sink_block_mask.dtype}, "
                f"device={sink_block_mask.device}, shape={tuple(sink_block_mask.shape)}"
            )

    route = torch.empty(
        (batch, heads, query_blocks, key_blocks),
        device=scores.device,
        dtype=torch.bool,
    )
    indices = torch.empty(
        (batch, heads, metadata_blocks, metadata_blocks),
        device=scores.device,
        dtype=torch.int32,
    )
    nums = torch.empty(
        (batch, heads, metadata_blocks),
        device=scores.device,
        dtype=torch.int32,
    )
    _route_and_compact_kernel[(batch * heads * metadata_blocks,)](
        scores,
        threshold,
        route,
        indices,
        nums,
        query_blocks,
        scale_log2,
        sink_first,
        sink_last,
        sink_block_mask if sink_block_mask is not None else threshold,
        *scores.stride(),
        *threshold.stride(),
        HEADS=heads,
        METADATA_BLOCKS=metadata_blocks,
        CHUNK=128,
        HAS_SINK_BLOCKS=sink_block_mask is not None,
        num_warps=4,
    )
    return route, indices, nums
