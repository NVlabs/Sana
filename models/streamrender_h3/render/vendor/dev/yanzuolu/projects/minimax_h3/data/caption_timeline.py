# SPDX-License-Identifier: Apache-2.0
"""Caption intervals on the source RGB frame clock, independent of model RoPE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping


def normalize_caption_segments(
    segments: Sequence[Mapping[str, Any]], *, num_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Validate ordered half-open RGB intervals without rewriting their captions."""
    result = []
    previous = -1
    for segment in segments:
        start, end, prompt = segment["start_frame"], segment["end_frame"], segment["prompt"]
        if (isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or not 0 <= start < end or start < previous):
            raise ValueError("caption segments require ordered nonnegative integer frame intervals")
        if num_frames is not None and end > num_frames:
            raise ValueError("caption segment exceeds the video frame count")
        if not isinstance(prompt, str):
            raise TypeError("caption prompt must be a string")
        result.append(dict(start_frame=start, end_frame=end, prompt=prompt))
        previous = start
    if not result:
        raise ValueError("caption timeline must contain at least one segment")
    return result


def crop_caption_segments(
    segments: Sequence[Mapping[str, Any]], start_frame: int, num_frames: int,
) -> list[dict[str, Any]]:
    """Intersect a mother-clip timeline with a crop and rebase its frame clock."""
    end_frame = start_frame + num_frames
    cropped = [
        dict(start_frame=max(start_frame, segment["start_frame"]) - start_frame,
             end_frame=min(end_frame, segment["end_frame"]) - start_frame,
             prompt=segment["prompt"])
        for segment in normalize_caption_segments(segments)
        if segment["start_frame"] < end_frame and segment["end_frame"] > start_frame
    ]
    return normalize_caption_segments(cropped, num_frames=num_frames)


def caption_index_for_interval(
    segments: Sequence[Mapping[str, Any]], start_frame: int, end_frame: int,
) -> int:
    """Choose maximum overlap with the noisy interval, preferring the later caption on ties."""
    if not 0 <= start_frame < end_frame:
        raise ValueError("noisy caption interval must be nonempty")
    scores = [(max(0, min(end_frame, item["end_frame"]) - max(start_frame, item["start_frame"])),
               item["start_frame"], index) for index, item in enumerate(segments)]
    if not scores or max(scores)[0] == 0:
        raise ValueError("no caption overlaps the noisy video interval")
    return max(scores)[2]


def caption_index_for_plan(
    segments: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
    video_temporal_mapping: VideoTemporalMapping,
) -> int:
    """Use the real current target interval, excluding sink and recent history."""
    boundary = video_temporal_mapping.decode_timeline.boundary
    return caption_index_for_interval(segments, boundary(plan["start"]), boundary(plan["stop"]))


__all__ = ["normalize_caption_segments", "crop_caption_segments", "caption_index_for_interval", "caption_index_for_plan"]
