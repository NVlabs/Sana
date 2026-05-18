"""Causal / sliding-window video self-attention mask for LTX transformer.

Supports three modes controlled by ``attention_mode``:

  - **Per-frame causal** (``attention_mode="causal"``, ``chunk_sizes=None``):
    frame t can only attend to frames <= t.

  - **Chunk causal** (``attention_mode="causal"``, ``chunk_sizes=[5, 3, 3]``):
    Latent frames are grouped into chunks.  Within a chunk, attention is
    bidirectional.  Across chunks, only earlier chunks are visible (causal).

  - **Sliding window** (``attention_mode="sliding_window"``, ``window_radius=W``):
    Each frame attends to frames within ±W frames.  Bidirectional local
    attention with no hard chunk boundaries.  Supports streaming with
    delay = W frames.

In all modes:
  - spatial tokens inside a visible frame are fully connected
  - text cross-attention is unchanged (handled by a separate attn layer)
  - audio is interface-compatible only

The mask is injected into ``LatentState.attention_mask`` *before* the
transformer sees it.  The existing ``_prepare_self_attention_mask`` in
``TransformerArgsPreprocessor`` converts ``[0, 1]`` values to additive
log-space bias, so no changes to the transformer are needed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from diffusion.refiner.vendor.ltx_core.types import LatentState


def _frame_time_to_index(frame_time: torch.Tensor) -> torch.Tensor:
    """Map per-token frame times (B, T) to frame indices (B, T).

    Tokens with the same frame_time get the same index (0, 1, 2, ...).
    """
    B, T = frame_time.shape
    frame_indices = torch.zeros_like(frame_time, dtype=torch.long)
    for b in range(B):
        unique_times, inverse = frame_time[b].unique(sorted=True, return_inverse=True)
        frame_indices[b] = inverse
    return frame_indices


def _frame_index_to_chunk_index(frame_indices: torch.Tensor, chunk_sizes: Sequence[int]) -> torch.Tensor:
    """Map frame indices to chunk indices based on chunk_sizes.

    E.g. chunk_sizes=[5, 3, 3] means frames 0-4 → chunk 0, 5-7 → chunk 1, 8-10 → chunk 2.
    """
    # Build a lookup: frame_idx → chunk_idx
    total_frames = sum(chunk_sizes)
    frame_to_chunk = torch.zeros(total_frames, dtype=torch.long, device=frame_indices.device)
    offset = 0
    for chunk_idx, size in enumerate(chunk_sizes):
        frame_to_chunk[offset : offset + size] = chunk_idx
        offset += size
    # Clamp frame indices to valid range (safety)
    clamped = frame_indices.clamp(0, total_frames - 1)
    return frame_to_chunk[clamped]


def build_causal_video_mask(
    positions: torch.Tensor,
    chunk_sizes: Sequence[int] | None = None,
) -> torch.Tensor:
    """Build a causal self-attention mask from patchified positions.

    Parameters
    ----------
    positions : torch.Tensor
        Shape ``(B, 3, T, 2)`` — patchified position coordinates.
        ``positions[:, 0, :, 0]`` is the temporal axis (frame time).
        Tokens from the same frame share the same value.
    chunk_sizes : list[int] or None
        If None, per-frame causal (default).
        If provided, chunk-level causal.  Must sum to num latent frames.

    Returns
    -------
    torch.Tensor
        Shape ``(B, T, T)`` with values in {0.0, 1.0}.
    """
    if positions.ndim != 4 or positions.shape[1] != 3 or positions.shape[3] != 2:
        raise ValueError(f"Expected positions (B, 3, T, 2), got {tuple(positions.shape)}")

    # (B, T) — frame start-time per token
    frame_time = positions[:, 0, :, 0]

    if chunk_sizes is None:
        # Per-frame causal: query i can attend to key j when key's frame <= query's frame
        mask = (frame_time.unsqueeze(1) <= frame_time.unsqueeze(2)).float()
    else:
        # Chunk causal: map frame_time → frame_index → chunk_index
        frame_indices = _frame_time_to_index(frame_time)
        num_frames = int(frame_indices.max().item()) + 1
        chunk_total = sum(chunk_sizes)
        if chunk_total != num_frames:
            raise ValueError(
                f"sum(chunk_sizes)={chunk_total} != num latent frames={num_frames}. "
                f"chunk_sizes={list(chunk_sizes)} must sum to the number of latent frames."
            )
        chunk_indices = _frame_index_to_chunk_index(frame_indices, chunk_sizes)  # (B, T)
        # mask[b, i, j] = 1 iff key j's chunk <= query i's chunk
        # unsqueeze(1) = (B,1,T) = key broadcast, unsqueeze(2) = (B,T,1) = query broadcast
        mask = (chunk_indices.unsqueeze(1) <= chunk_indices.unsqueeze(2)).float()

    return mask


def build_sliding_window_mask(
    positions: torch.Tensor,
    window_radius: int,
) -> torch.Tensor:
    """Build a sliding-window self-attention mask from patchified positions.

    Each frame can attend to frames within ±window_radius (bidirectional local).

    Parameters
    ----------
    positions : torch.Tensor
        Shape ``(B, 3, T, 2)``.  ``positions[:, 0, :, 0]`` is the temporal axis.
    window_radius : int
        Number of frames each frame can see in each direction.
        E.g. window_radius=3 means each frame sees ±3 frames (window size 7).

    Returns
    -------
    torch.Tensor
        Shape ``(B, T, T)`` with values in {0.0, 1.0}.
    """
    if positions.ndim != 4 or positions.shape[1] != 3 or positions.shape[3] != 2:
        raise ValueError(f"Expected positions (B, 3, T, 2), got {tuple(positions.shape)}")

    frame_time = positions[:, 0, :, 0]  # (B, T)
    frame_indices = _frame_time_to_index(frame_time)  # (B, T) — integer frame indices

    # mask[b, i, j] = 1 iff |frame_i - frame_j| <= window_radius
    qi = frame_indices.unsqueeze(2)  # (B, T, 1) — query
    kj = frame_indices.unsqueeze(1)  # (B, 1, T) — key
    mask = ((qi - kj).abs() <= window_radius).float()

    return mask


def apply_causal_mask(
    state: LatentState,
    chunk_sizes: Sequence[int] | None = None,
    attention_mode: str = "causal",
    window_radius: int | None = None,
) -> LatentState:
    """Return a new LatentState with an attention mask applied.

    Parameters
    ----------
    state : LatentState
    chunk_sizes : list[int] or None
        For causal mode: None = per-frame, List = chunk-level.
    attention_mode : str
        ``"causal"`` — chunk or per-frame causal (default).
        ``"sliding_window"`` — bidirectional sliding window.
    window_radius : int or None
        For sliding_window mode: each frame sees ±window_radius frames.

    If the state already has an ``attention_mask``, the new mask is
    combined with it (element-wise multiply).
    """
    if attention_mode == "sliding_window":
        if window_radius is None:
            raise ValueError("window_radius is required for sliding_window mode")
        mask = build_sliding_window_mask(state.positions, window_radius=window_radius)
    else:
        mask = build_causal_video_mask(state.positions, chunk_sizes=chunk_sizes)

    if state.attention_mask is not None:
        mask = mask * state.attention_mask

    return replace(state, attention_mask=mask)
