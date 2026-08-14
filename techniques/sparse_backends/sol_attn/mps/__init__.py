"""Apple Silicon Metal backend for Sol-Attn."""

from __future__ import annotations

import torch

from .metal import sol_attn_tiled_mps


BLOCK_SIZE = 64


def _sink_block_range(tokens: int, sink_start: int | None, sink_tokens: int):
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return (
        start // BLOCK_SIZE,
        (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    tau: float,
    thresh_type: str,
    kv_splits: int,
    sink_tokens: int,
    sink_start: int | None,
) -> torch.Tensor:
    """Run Sol-Attn through the tiled Metal forward kernel."""

    if kv_splits != 1:
        raise ValueError("kv_splits=2/4 is currently available on SM90 only")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("the Metal backend requires torch.mps.compile_shader")

    return sol_attn_tiled_mps(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        sink_blocks=_sink_block_range(q.shape[1], sink_start, sink_tokens),
    )


__all__ = ["sol_attn"]