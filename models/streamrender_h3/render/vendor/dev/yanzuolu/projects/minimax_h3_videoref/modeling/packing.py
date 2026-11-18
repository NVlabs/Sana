# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence materialization for MiniMax H3 reference blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from dev.yanzuolu.projects.minimax_h3.modeling.packing import (
    _axis_from_sqrt_area,
    _video_t_grid,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.config import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)

_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


def _positive_int(
    block: Mapping[str, object],
    key: str,
    path: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        predicate = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path}.{key} must be {predicate}")
    return int(value)


def _video_t_span(
    n: int, *, video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> float:
    # Sequential fp64 summation on purpose — see _temporal_position_span for
    # why the two span implementations must not be unified.
    return sum(_FRAME_RESCALE * span for span in video_temporal_mapping.decode_timeline.spans(n))


def causal_video_ref_rope_offsets(
    reference_spans: Sequence[float],
) -> list[tuple[float, float]]:
    """Return reference and target time shifts for each logical causal chunk."""
    offsets = []
    reference_offset = 0.0
    for span in reference_spans:
        target_offset = reference_offset + float(span)
        offsets.append((reference_offset, target_offset))
        reference_offset = target_offset
    return offsets


def _range_for_slice(sl: slice) -> torch.Tensor:
    return torch.arange(sl.start, sl.stop, dtype=torch.long)


def _cat_ranges(parts: list[torch.Tensor]) -> torch.Tensor:
    if len(parts) == 1:
        return parts[0]
    if parts:
        return torch.cat(parts)
    return torch.empty(0, dtype=torch.long)


def minimax_h3_packed_sequence_ref2va_blocks(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    ref_blocks: Sequence[Mapping[str, object]],
    audio_channel: int = 2,
    seq_len: int | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> dict[str, Any]:
    """General ref2va-family packed layout.

    ``ref_blocks`` are consumed in request/plan order:
    - ``{"kind": "image", "latent_h": H, "latent_w": W}``
    - ``{"kind": "audio", "ref_audio_t": T}``
    - ``{"kind": "video"|"video_audio", "ref_audio_t": T,
       "latent_t": RT, "latent_h": RH, "latent_w": RW}``

    Video-bearing blocks pack their audio rows immediately before their video
    rows; both share the same temporal origin and advance by the longer of the
    audio and video spans. Standalone audio advances the target origin by its
    own T, and image blocks advance it by one integer slot.
    A video block may carry a serialized ``video_temporal_mapping`` for its
    reference coordinates. Target coordinates use the function argument.
    """
    if not isinstance(ref_blocks, Sequence) or isinstance(ref_blocks, (str, bytes)):
        raise ValueError("ref_blocks must be a sequence")

    parsed: list[dict[str, object]] = []
    ref_visual_rows = 0
    ref_audio_rows = 0
    for index, raw in enumerate(ref_blocks):
        path = f"ref_blocks[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be an object")
        kind = raw.get("kind", raw.get("type"))
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{path}.kind must be a non-empty string")
        if kind == "image":
            rh = _positive_int(raw, "latent_h", path)
            rw = _positive_int(raw, "latent_w", path)
            rows = (rh // _PATCH_H) * (rw // _PATCH_W)
            item = {"kind": kind, "latent_h": rh, "latent_w": rw, "rows": rows}
            ref_visual_rows += rows
        elif kind == "audio":
            rt = _positive_int(raw, "ref_audio_t", path, allow_zero=True)
            rows = rt * audio_channel
            item = {"kind": kind, "ref_audio_t": rt, "audio_rows": rows}
            ref_audio_rows += rows
        elif kind in ("video", "video_audio"):
            rt = _positive_int(raw, "ref_audio_t", path, allow_zero=True)
            vt = _positive_int(raw, "latent_t", path)
            vh = _positive_int(raw, "latent_h", path)
            vw = _positive_int(raw, "latent_w", path)
            frame_rows = (vh // _PATCH_H) * (vw // _PATCH_W)
            audio_rows = rt * audio_channel
            video_rows = vt * frame_rows
            item = {
                "kind": kind,
                "ref_audio_t": rt,
                "latent_t": vt,
                "latent_h": vh,
                "latent_w": vw,
                "frame_rows": frame_rows,
                "audio_rows": audio_rows,
                "video_rows": video_rows,
                "video_temporal_mapping": VideoTemporalMapping.from_dict(
                    raw.get("video_temporal_mapping", H3_VIDEO_TEMPORAL_MAPPING.to_dict())
                ),
            }
            ref_audio_rows += audio_rows
            ref_visual_rows += video_rows
        else:
            raise ValueError(f"{path}.kind unsupported for ref2va: {kind!r}")
        parsed.append(item)

    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    ref_rows = ref_visual_rows + ref_audio_rows
    used = text_len + ref_rows + audio_rows + video_rows
    if seq_len is None:
        seq_len = (
            (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
            // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
            * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        )
    if seq_len < used:
        raise ValueError(f"seq_len {seq_len} < used rows {used}")

    text_sl = slice(0, text_len)
    cursor = text_len
    block_slices: list[dict[str, object]] = []
    for item in parsed:
        kind = str(item["kind"])
        if kind == "image":
            rows = int(item["rows"])
            visual_sl = slice(cursor, cursor + rows)
            cursor = visual_sl.stop
            block_slices.append({**item, "visual_sl": visual_sl})
        elif kind == "audio":
            rows = int(item["audio_rows"])
            audio_sl = slice(cursor, cursor + rows)
            cursor = audio_sl.stop
            block_slices.append({**item, "audio_sl": audio_sl})
        else:
            a_rows = int(item["audio_rows"])
            v_rows = int(item["video_rows"])
            audio_sl = slice(cursor, cursor + a_rows)
            visual_sl = slice(audio_sl.stop, audio_sl.stop + v_rows)
            cursor = visual_sl.stop
            block_slices.append({**item, "audio_sl": audio_sl, "visual_sl": visual_sl})

    audio_sl = slice(cursor, cursor + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    ref_img_pos_parts: list[torch.Tensor] = []
    ref_audio_pos_parts: list[torch.Tensor] = []
    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    target_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, target_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, target_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    target_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    t_cursor = float(text_len)
    for item in block_slices:
        kind = str(item["kind"])
        if kind == "image":
            visual_sl = item["visual_sl"]
            assert isinstance(visual_sl, slice)
            ref_img_pos_parts.append(_range_for_slice(visual_sl))
            rh = int(item["latent_h"])
            rw = int(item["latent_w"])
            area = np.sqrt(rh * rw)
            ref_hh, ref_ww = torch.meshgrid(
                _axis_from_sqrt_area(rh, _PATCH_H, area),
                _axis_from_sqrt_area(rw, _PATCH_W, area),
                indexing="ij",
            )
            g[visual_sl, 0] = t_cursor
            g[visual_sl, 1] = ref_hh.reshape(-1)
            g[visual_sl, 2] = ref_ww.reshape(-1)
            t_cursor += 1.0
        elif kind == "audio":
            audio_ref_sl = item["audio_sl"]
            assert isinstance(audio_ref_sl, slice)
            ref_t = int(item["ref_audio_t"])
            ref_audio_pos_parts.append(_range_for_slice(audio_ref_sl))
            ref_t_grid = t_cursor + torch.arange(ref_t, dtype=torch.float64)
            g[audio_ref_sl, 0] = ref_t_grid.repeat(audio_channel)
            if ref_t:
                g[audio_ref_sl.start : audio_ref_sl.start + ref_t, 2] = float(w_grid[0])
                g[audio_ref_sl.start + ref_t : audio_ref_sl.stop, 2] = float(w_grid[-1])
            t_cursor += float(ref_t)
        else:
            audio_ref_sl = item["audio_sl"]
            visual_sl = item["visual_sl"]
            assert isinstance(audio_ref_sl, slice)
            assert isinstance(visual_sl, slice)
            ref_t = int(item["ref_audio_t"])
            vt = int(item["latent_t"])
            vh = int(item["latent_h"])
            vw = int(item["latent_w"])
            reference_mapping = item["video_temporal_mapping"]
            ref_audio_pos_parts.append(_range_for_slice(audio_ref_sl))
            ref_img_pos_parts.append(_range_for_slice(visual_sl))

            ref_area = np.sqrt(vh * vw)
            rv_h_grid = _axis_from_sqrt_area(vh, _PATCH_H, ref_area)
            rv_w_grid = _axis_from_sqrt_area(vw, _PATCH_W, ref_area)
            rv_hh, rv_ww = torch.meshgrid(rv_h_grid, rv_w_grid, indexing="ij")

            ref_t_grid = t_cursor + torch.arange(ref_t, dtype=torch.float64)
            g[audio_ref_sl, 0] = ref_t_grid.repeat(audio_channel)
            if ref_t:
                g[audio_ref_sl.start : audio_ref_sl.start + ref_t, 2] = float(
                    rv_w_grid[0]
                )
                g[audio_ref_sl.start + ref_t : audio_ref_sl.stop, 2] = float(
                    rv_w_grid[-1]
                )

            rv_frame = torch.stack([rv_hh.reshape(-1), rv_ww.reshape(-1)], dim=-1)
            rv_g = g[visual_sl].view(vt, int(item["frame_rows"]), 3)
            rv_g[:, :, 0] = _video_t_grid(
                vt, t_cursor, timeline=reference_mapping.decode_timeline
            )[:, None]
            rv_g[:, :, 1:] = rv_frame[None]
            t_cursor += max(float(ref_t), _video_t_span(vt, video_temporal_mapping=reference_mapping))

    audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    g[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    g[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    video_g = g[video_sl].view(latent_t, frame_rows, 3)
    video_g[:, :, 0] = _video_t_grid(
        latent_t, t_cursor, timeline=video_temporal_mapping.decode_timeline
    )[:, None]
    video_g[:, :, 1:] = target_frame[None]

    target_img_pos = _range_for_slice(video_sl)
    target_audio_pos = _range_for_slice(audio_sl)
    img_pos = _cat_ranges(ref_img_pos_parts + [target_img_pos])
    audio_pos = _cat_ranges(ref_audio_pos_parts + [target_audio_pos])

    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[ref_visual_rows:] = True
    audio_update_mask = torch.zeros(audio_pos.shape[0], dtype=torch.bool)
    audio_update_mask[ref_audio_rows:] = True
    text_pos = torch.arange(0, text_len)

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)  # PADDING
    token_tags[text_sl] = 1  # TEXT
    token_tags[audio_pos] = 2  # AUDIO (refs + target)
    token_tags[img_pos] = 0  # VIDEO (refs + target)

    cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": g,
        "token_tags": token_tags,
        "cu_seqlens": cu,
    }


__all__ = [
    "causal_video_ref_rope_offsets",
    "minimax_h3_packed_sequence_ref2va_blocks",
]
