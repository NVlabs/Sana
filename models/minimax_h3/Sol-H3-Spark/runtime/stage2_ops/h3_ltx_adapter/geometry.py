"""Exact frozen H3-to-LTX temporal and spatial alignment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .constants import H3_CLIP_LENGTH, H3_TEMPORAL_COMPRESSION, H3_TOKEN_DROP, LTX_TEMPORAL_COMPRESSION


def h3_temporal_positions(num_pixel_frames: int) -> list[float]:
    if num_pixel_frames <= 0:
        raise ValueError("num_pixel_frames must be positive")
    tokens_per_chunk = (H3_CLIP_LENGTH + H3_TEMPORAL_COMPRESSION - 1) // H3_TEMPORAL_COMPRESSION
    num_chunks = (num_pixel_frames + H3_CLIP_LENGTH - 1) // H3_CLIP_LENGTH
    positions = [
        min(chunk * H3_CLIP_LENGTH + token * H3_TEMPORAL_COMPRESSION, num_pixel_frames - 1)
        for chunk in range(num_chunks)
        for token in range(tokens_per_chunk)
    ]
    if H3_TOKEN_DROP:
        positions = positions[:-H3_TOKEN_DROP]
    return [float(position) for position in positions]


def padded_ltx_pixel_frames(num_pixel_frames: int) -> int:
    if num_pixel_frames <= 0:
        raise ValueError("num_pixel_frames must be positive")
    return ((num_pixel_frames - 1 + LTX_TEMPORAL_COMPRESSION - 1) // LTX_TEMPORAL_COMPRESSION) * LTX_TEMPORAL_COMPRESSION + 1


def ltx_temporal_positions(num_pixel_frames: int) -> list[float]:
    latent_frames = (num_pixel_frames - 1) // LTX_TEMPORAL_COMPRESSION + 1
    return [float(min(index * LTX_TEMPORAL_COMPRESSION, num_pixel_frames - 1)) for index in range(latent_frames)]


def temporal_resample(
    latent: torch.Tensor,
    source_positions: Sequence[float],
    target_positions: Sequence[float],
) -> torch.Tensor:
    source = torch.as_tensor(source_positions, device=latent.device, dtype=torch.float32)
    target = torch.as_tensor(target_positions, device=latent.device, dtype=torch.float32)
    right = torch.searchsorted(source, target, right=False).clamp(max=source.numel() - 1)
    left = (right - 1).clamp(min=0)
    denominator = source[right] - source[left]
    weight = torch.where(denominator > 0, (target - source[left]) / denominator, torch.zeros_like(target))
    weight = weight.clamp(0, 1).to(dtype=latent.dtype).view(1, 1, -1, 1, 1)
    return latent.index_select(2, left).lerp(latent.index_select(2, right), weight)


def temporal_nearest_pack(
    latent: torch.Tensor,
    source_positions: Sequence[float],
    target_positions: Sequence[float],
    slots: int,
) -> torch.Tensor:
    source = torch.as_tensor(source_positions, device=latent.device, dtype=torch.float32)
    target = torch.as_tensor(target_positions, device=latent.device, dtype=torch.float32)
    assignment = torch.argmin(torch.abs(source[:, None] - target[None, :]), dim=1)
    counts = torch.bincount(assignment, minlength=target.numel())
    required_slots = int(counts.max().item())
    if required_slots > slots:
        raise ValueError(f"temporal packing needs {required_slots} slots, checkpoint provides {slots}")
    indices = torch.zeros((target.numel(), slots), dtype=torch.long, device=latent.device)
    occupied = torch.zeros((target.numel(), slots), dtype=torch.bool, device=latent.device)
    for target_index in range(target.numel()):
        source_indices = torch.nonzero(assignment == target_index, as_tuple=False).flatten()
        count = source_indices.numel()
        if count:
            indices[target_index, :count] = source_indices
            occupied[target_index, :count] = True
    selected = latent.index_select(2, indices.flatten())
    batch, channels, _, height, width = selected.shape
    selected = selected.reshape(batch, channels, target.numel(), slots, height, width)
    selected = selected * occupied.view(1, 1, target.numel(), slots, 1, 1).to(latent.dtype)
    return selected.permute(0, 3, 1, 2, 4, 5).reshape(batch, slots * channels, target.numel(), height, width)


def align_h3_to_ltx(
    latent: torch.Tensor,
    *,
    pixel_frames: int,
    target_height: int,
    target_width: int,
    slots: int = 3,
) -> torch.Tensor:
    if latent.ndim != 5:
        raise ValueError(f"expected B,C,T,H,W, got {tuple(latent.shape)}")
    source_positions = h3_temporal_positions(pixel_frames)
    if len(source_positions) != latent.shape[2]:
        raise ValueError(
            f"H3 latent T={latent.shape[2]} does not match pixel_frames={pixel_frames}; expected T={len(source_positions)}"
        )
    padded_frames = padded_ltx_pixel_frames(pixel_frames)
    target_positions = ltx_temporal_positions(padded_frames)
    linear = temporal_resample(latent, source_positions, target_positions)
    packed = temporal_nearest_pack(latent, source_positions, target_positions, slots)
    aligned = torch.cat((linear, packed), dim=1)
    source_height, source_width = aligned.shape[-2:]
    scale_h, remainder_h = divmod(source_height, target_height)
    scale_w, remainder_w = divmod(source_width, target_width)
    if remainder_h or remainder_w or scale_h != 2 or scale_w != 2:
        raise ValueError(
            "frozen adapter requires 2x pixel-unshuffle: "
            f"source={(source_height, source_width)}, target={(target_height, target_width)}"
        )
    batch, channels, frames, height, width = aligned.shape
    aligned_2d = aligned.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    aligned_2d = F.pixel_unshuffle(aligned_2d, downscale_factor=2)
    return aligned_2d.reshape(batch, frames, channels * 4, target_height, target_width).permute(0, 2, 1, 3, 4).contiguous()
