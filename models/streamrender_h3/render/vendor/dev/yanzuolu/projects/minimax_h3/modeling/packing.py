# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-sequence materialization from the validated workspace
builder, covering fl2va and t2va layouts.

Layout: [text L | imgvid_cond C | audio A(=t*2ch) | video_target V | pad P].
Builder rules:
- block-derived position infos, update masks, token tags, and cu_seqlens
- img_position_ids fp64 grid: text rows (row_idx,0,0); video/cond t counter
  continues text_len with temporal interp spans (frame_rescale 5/3 x
  frame_per_token (1,4,4,4,4)); each spatial sqrt_area axis uses evenly spaced
  coordinates excluding the right endpoint, then scales them by INTERP;
  audio channel-major blocks pinned to the w-grid extremes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .time_request import FrameTimeline, H3_VIDEO_TEMPORAL_MAPPING

# ===== JARVIS PATCH BEGIN(sglang runtime coupling: sglang is not a dependency, so the DiT config resolves to its vendored copy and task_profiles — serving-side request validation, not vendored — contributes its one constant inline, verbatim) =====
# from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
#     MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
# )
# from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
#     MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
# )
from .transformer.config import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)

# verbatim from sglang .../minimax_h3/task_profiles.py
MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES: tuple[tuple[int, ...], ...] = (
    (0,),
    (-1,),
    (0, -1),
)
# ===== JARVIS PATCH END =====

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


def _keyframe_cond_frame_indices(
    *,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if not include_keyframe_cond:
        if keyframe_frame_indices is not None:
            raise ValueError(
                "keyframe_frame_indices must be omitted when keyframe cond is not included"
            )
        return []
    if keyframe_frame_indices is None:
        raise ValueError("strict fl2va packed layout requires keyframe_frame_indices")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in keyframe_frame_indices
    ):
        raise ValueError(
            "strict fl2va packed layout requires integer keyframe_frame_indices"
        )
    out = list(keyframe_frame_indices)
    if tuple(out) not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "strict fl2va packed layout requires keyframe_frame_indices in "
            f"{MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got {out!r}"
        )
    return out


def _resolve_keyframe_frame_indices(
    frame_indices: Sequence[int],
    *,
    frame_count: int | None,
) -> list[int]:
    if frame_indices and frame_count is None:
        raise ValueError(
            "frame_count is required when keyframe_frame_indices are provided"
        )
    if frame_count is None:
        return []
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    seen: dict[int, int] = {}
    resolved: list[int] = []
    for block_index, semantic_index in enumerate(frame_indices):
        if semantic_index == -1:
            resolved_index = frame_count - 1
        elif 0 <= semantic_index < frame_count:
            resolved_index = semantic_index
        else:
            raise ValueError(
                f"keyframe frame index {semantic_index} must be -1 or in "
                f"[0, {frame_count})"
            )
        previous = seen.get(resolved_index)
        if previous is not None:
            raise ValueError(
                f"keyframe frame index at block {block_index} resolves to "
                f"{resolved_index}, already bound by block {previous}"
            )
        seen[resolved_index] = block_index
        resolved.append(resolved_index)
    return resolved


def _temporal_position_span(temporal_length: int) -> float:
    """Temporal position span for patch_t=1, in fp64."""
    spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
    for token_index in range(_T_GROUP):
        spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
    return float(spans.sum())


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None = None,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Build the packed-sequence structural fields for one CFG branch.

    The used length is padded up to a multiple of 64.
    """
    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    cond_frame_indices = _keyframe_cond_frame_indices(
        include_keyframe_cond=include_keyframe_cond,
        keyframe_frame_indices=keyframe_frame_indices,
    )
    resolved_cond_frame_indices = _resolve_keyframe_frame_indices(
        cond_frame_indices,
        frame_count=frame_count,
    )
    cond_rows = len(cond_frame_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )

    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    target_img_pos = torch.arange(video_sl.start, video_sl.stop)
    img_pos = (
        torch.cat([torch.arange(cond_sl.start, cond_sl.stop), target_img_pos])
        if cond_rows
        else target_img_pos
    )
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    text_pos = torch.arange(0, text_len)

    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    t_grid = _video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = t_grid[:, None]
    video_g[:, :, 1:] = frame[None]
    for block_index, pixel_index in enumerate(resolved_cond_frame_indices):
        sl = slice(
            cond_sl.start + block_index * frame_rows,
            cond_sl.start + (block_index + 1) * frame_rows,
        )
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = (
                float(text_len) + _temporal_position_span(latent_t) - _FRAME_RESCALE
            )
        else:
            raise ValueError(
                "fl2va packed layout only supports first/last keyframe anchors, "
                f"got resolved frame index {pixel_index}"
            )
        g[sl, 0] = cond_t
        g[sl, 1:] = frame
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    g[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    g[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)  # PADDING
    token_tags[text_sl] = 1  # TEXT (fl2va image-segment override happens upstream)
    token_tags[audio_sl] = 2  # AUDIO
    token_tags[img_pos] = 0  # VIDEO

    cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
        "cu_seqlens": cu,
    }


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 1.0 / 2.0
    right = left + ratio * 1.0
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(
    n: int, origin: float, *, timeline: FrameTimeline = H3_VIDEO_TEMPORAL_MAPPING.decode_timeline,
) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * span for span in timeline.spans(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


__all__ = ["minimax_h3_packed_sequence"]
