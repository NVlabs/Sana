# SPDX-License-Identifier: Apache-2.0
"""Qwen presentations restricted to the reference rows of a streaming window."""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.data.streaming import streaming_window_time_shift
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
    Ref2VAPresentation,
    Ref2VAPresentationMedia,
)


def streaming_reference_rgb_indices(plan: dict[str, Any]) -> torch.Tensor:
    """Expand the actual reference selection onto its decoded 24 FPS frame grid."""
    indices = plan["reference_video_indices"]
    if indices is None:
        raise ValueError("Qwen visual context requires a video reference")
    mapping = VideoTemporalMapping.from_dict(plan["video_temporal_mapping"])
    boundary = mapping.decode_timeline.boundary
    return torch.tensor([
        frame for index in indices.tolist()
        for frame in range(boundary(index), boundary(index + 1))
    ], dtype=torch.long)


def _reference_frame_metadata(
    plan: dict[str, Any], *, fixed_window_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = streaming_reference_rgb_indices(plan)
    mapping = VideoTemporalMapping.from_dict(plan["video_temporal_mapping"])
    shift = streaming_window_time_shift(
        plan, video_temporal_mapping=mapping, fixed_window_rope=fixed_window_rope,
    )
    frame_positions = indices.clone()
    recent = indices >= mapping.decode_timeline.boundary(int(plan["sink_history_stop"]))
    frame_positions[recent] += round(shift * 24.0 / 40.0)
    return indices, frame_positions.to(torch.float64) / 24.0


def build_streaming_ref_presentation(
    processor: MiniMaxH3Ref2VAPresentationProcessor,
    prompt: str,
    reference_pixels: torch.Tensor | None,
    plan: dict[str, Any],
    *,
    reference_frame_indices: torch.Tensor | None = None,
    fixed_window_rope: bool = False,
    picture: Any = None,
) -> Ref2VAPresentation:
    """Build one Video 1 condition from retained CTHW RGB in [0, 1].

    Without frame indices, pixels represent a complete clip starting at frame
    zero. Live callers supply the global indices of their retained RGB rows.
    Timestamps use the reference's DiT time coordinates without text or target
    domain offsets. Sampling never fills gaps in the reference selection.
    An optional prepared RGB ``picture`` precedes it as Picture 1. Without
    reference pixels, the picture alone conditions the caption.
    """
    pictures = [] if picture is None else [Ref2VAPresentationMedia(kind="image", media=picture)]
    if reference_pixels is None:
        return processor.build(prompt, pictures)
    if reference_pixels.ndim != 4 or reference_pixels.shape[0] != 3:
        raise ValueError("reference pixels must have shape [3,T,H,W]")
    indices, timestamps = _reference_frame_metadata(plan, fixed_window_rope=fixed_window_rope)
    available = (
        torch.arange(reference_pixels.shape[1]) if reference_frame_indices is None
        else reference_frame_indices.detach().cpu().to(torch.long)
    )
    if available.ndim != 1 or available.numel() != reference_pixels.shape[1]:
        raise ValueError("reference frame indices must describe every supplied RGB row")
    if available.numel() == 0 or available[0] < 0 or not bool(torch.all(available[1:] > available[:-1])):
        raise ValueError("reference frame indices must be strictly increasing and nonnegative")
    positions = torch.searchsorted(available, indices)
    if bool(torch.any(positions >= available.numel())) or not torch.equal(available[positions], indices):
        raise ValueError("reference pixels do not cover the selected streaming window")
    pixels = reference_pixels.index_select(1, positions.to(reference_pixels.device))
    media = pixels.detach().cpu().permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).contiguous()
    return processor.build(prompt, [*pictures, Ref2VAPresentationMedia(
        kind="video", media=media, has_audio=False,
        video_frame_indices=indices.tolist(), video_frame_timestamps=timestamps.tolist(),
    )])


def streaming_ref_presentation_length(
    processor: MiniMaxH3Ref2VAPresentationProcessor,
    prompt: str,
    plan: dict[str, Any],
    *,
    height: int,
    width: int,
    fixed_window_rope: bool = False,
) -> int:
    """Budget the same reference presentation from geometry and language alone."""
    indices, timestamps = _reference_frame_metadata(plan, fixed_window_rope=fixed_window_rope)
    return processor.video_presentation_length(
        prompt, frame_indices=indices.tolist(), frame_timestamps=timestamps.tolist(),
        height=height, width=width,
    )


__all__ = [
    "build_streaming_ref_presentation",
    "streaming_ref_presentation_length",
    "streaming_reference_rgb_indices",
]
