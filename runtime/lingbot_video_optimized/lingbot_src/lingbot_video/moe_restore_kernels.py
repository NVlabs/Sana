from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional outside GPU deployments.
    triton = None
    tl = None


if triton is not None and tl is not None:
    @triton.jit
    def _moe_restore_weighted_sum_kernel(
        expert_output,
        sorted_scores,
        route_for_position,
        output,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        hidden_block = tl.program_id(1)
        offsets_h = hidden_block * block_h + tl.arange(0, block_h)
        hidden_mask = offsets_h < hidden_size
        acc = tl.zeros((block_h,), dtype=tl.float32)

        for route_slot in tl.range(0, top_k):
            route_pos = token_idx * top_k + route_slot
            route_idx = tl.load(route_for_position + route_pos)
            active = route_idx >= 0
            safe_route_idx = tl.maximum(route_idx, 0)
            values = tl.load(
                expert_output + safe_route_idx * hidden_size + offsets_h,
                mask=active & hidden_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.load(sorted_scores + safe_route_idx, mask=active, other=0.0).to(tl.float32)
            acc += values * score

        tl.store(output + token_idx * hidden_size + offsets_h, acc, mask=hidden_mask)
else:
    _moe_restore_weighted_sum_kernel = None


def restore_tokens_triton(
    expert_output: torch.Tensor,
    sorted_positions: torch.Tensor,
    sorted_scores: torch.Tensor,
    num_tokens: int,
    top_k: int,
    block_h: int = 64,
) -> torch.Tensor:
    if _moe_restore_weighted_sum_kernel is None:
        raise RuntimeError("LINGBOT_MOE_RESTORE_BACKEND=triton requires Triton")
    if expert_output.ndim != 2:
        raise ValueError(f"Expected 2D expert_output, got {expert_output.ndim}D")
    if sorted_positions.numel() != sorted_scores.numel():
        raise ValueError("sorted_positions and sorted_scores must have the same length")
    if sorted_positions.numel() == 0:
        return expert_output.new_zeros((num_tokens, expert_output.shape[-1]))

    hidden_size = expert_output.shape[-1]
    route_for_position = torch.full(
        (num_tokens * top_k,),
        -1,
        dtype=torch.int32,
        device=expert_output.device,
    )
    route_for_position[sorted_positions] = torch.arange(
        sorted_positions.numel(),
        dtype=torch.int32,
        device=expert_output.device,
    )
    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=expert_output.dtype,
        device=expert_output.device,
    )
    grid = (num_tokens, triton.cdiv(hidden_size, block_h))
    _moe_restore_weighted_sum_kernel[grid](
        expert_output.contiguous(),
        sorted_scores.contiguous(),
        route_for_position,
        output,
        hidden_size,
        top_k,
        block_h,
    )
    return output
