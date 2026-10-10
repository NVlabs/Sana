# SPDX-License-Identifier: Apache-2.0
"""Fixed-window plans and normalized latent data for bidirectional AV streaming."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from dev.yanzuolu.common.data import WorkerResumeContext
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    FrameTimeline,
    VideoTemporalMapping,
    minimax_h3_audio_latent_t,
)
from dev.yanzuolu.projects.minimax_h3.data.causal_latent import CausalLatentT2AVDataset
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS


def _nonnegative_chunk_count(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _bootstrap_units(window_size: int, chunk_size: int, bootstrap_size: int | None) -> int:
    """Return the first window's unit count, W + C unless configured explicitly."""
    if bootstrap_size is None:
        return window_size + chunk_size
    if _nonnegative_chunk_count(bootstrap_size, "bootstrap_size") == 0:
        raise ValueError("bootstrap_size must be positive")
    return bootstrap_size


_LATENT_UNITS = FrameTimeline((), (1,))
_MERGED_SINGLE_FRAME_UNITS = FrameTimeline((), (2, 1, 1, 1))


def _streaming_window_time_origins(
    plan: Mapping[str, Any], *,
    video_temporal_mapping: VideoTemporalMapping | None = None,
    fixed_window_rope: bool = False,
) -> tuple[float, float]:
    """Return destination and source origins on the same 40 Hz clock."""
    if not fixed_window_rope or plan["is_bootstrap"]:
        return 0.0, 0.0
    mapping = video_temporal_mapping or VideoTemporalMapping.from_dict(plan["video_temporal_mapping"])
    sink_stop, recent_start = int(plan["sink_stop"]), int(plan["recent_start"])
    return mapping.position_start(sink_stop), mapping.position_start(max(sink_stop, recent_start))


def streaming_window_time_shift(
    plan: Mapping[str, Any], *,
    video_temporal_mapping: VideoTemporalMapping | None = None,
    fixed_window_rope: bool = False,
) -> float:
    """Return the recent/current window's temporal translation in 40 Hz units."""
    destination, source = _streaming_window_time_origins(
        plan, video_temporal_mapping=video_temporal_mapping,
        fixed_window_rope=fixed_window_rope,
    )
    return destination - source


def streaming_unit_timeline(
    *,
    merge_single_frame_units: bool = False,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> FrameTimeline:
    """Map scheduling units to native latent counts without changing the codec."""
    if not isinstance(merge_single_frame_units, bool):
        raise ValueError("merge_single_frame_units must be a boolean")
    if merge_single_frame_units and video_temporal_mapping != H3_VIDEO_TEMPORAL_MAPPING:
        raise ValueError("merge_single_frame_units requires the native H3 temporal mapping")
    return _MERGED_SINGLE_FRAME_UNITS if merge_single_frame_units else _LATENT_UNITS


def streaming_window_stop(
    start: int,
    policy: Mapping[str, Any],
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> int:
    """Return a full window's native stop, before clipping a final short piece."""
    start = _nonnegative_chunk_count(start, "start")
    window_size = _nonnegative_chunk_count(policy["window_size"], "window_size")
    chunk_size = _nonnegative_chunk_count(policy["chunk_size"], "chunk_size")
    if chunk_size == 0:
        raise ValueError("chunk_size must be positive")
    units = streaming_unit_timeline(
        merge_single_frame_units=policy.get("merge_single_frame_units", False),
        video_temporal_mapping=video_temporal_mapping,
    )
    start_unit = units.count_to_cover(start)
    bootstrap_units = _bootstrap_units(window_size, chunk_size, policy.get("bootstrap_size"))
    if units.boundary(start_unit) != start or (
        start != 0 and (start_unit < bootstrap_units or (start_unit - bootstrap_units) % chunk_size)
    ):
        raise ValueError("start must select a complete streaming unit at a window boundary")
    return units.boundary(bootstrap_units if start == 0 else start_unit + chunk_size)


def streaming_target_shapes(
    *,
    height: int,
    width: int,
    num_frames: int,
    video_latent_channels: int = 24,
    audio_latent_channels: int = 32,
    spatial_vae_stride: int = 16,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Return normalized AV shapes on the selected codec's complete timeline."""
    height, width, num_frames = int(height), int(width), int(num_frames)
    spatial_vae_stride = int(spatial_vae_stride)
    if height % spatial_vae_stride or width % spatial_vae_stride:
        raise ValueError("height and width must be divisible by spatial_vae_stride")
    latent_h, latent_w = height // spatial_vae_stride, width // spatial_vae_stride
    if latent_h % 2 or latent_w % 2:
        raise ValueError("target latent height and width must be divisible by patch (2, 2)")
    latent_t = video_temporal_mapping.target_latent_t(num_frames)
    audio_t = minimax_h3_audio_latent_t(num_frames / float(MINIMAX_H3_SUPPORTED_FPS))
    return (
        (int(video_latent_channels), latent_t, latent_h, latent_w),
        (2, int(audio_latent_channels), audio_t),
    )


def _latent_shapes(
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    reference_video_shape: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]:
    target = tuple(int(value) for value in target_video_shape)
    audio = tuple(int(value) for value in target_audio_shape)
    reference = (
        None if reference_video_shape is None
        else tuple(int(value) for value in reference_video_shape)
    )
    if len(target) != 4 or len(audio) != 3 or (reference is not None and len(reference) != 4):
        raise ValueError("video latents must be [C,T,H,W] and audio must be [2,C,T]")
    if reference is not None and target[:2] != reference[:2]:
        raise ValueError("reference and target video latent C/T must match")
    if audio[0] != 2:
        raise ValueError("target audio latent must be stereo")
    shapes = (target,) if reference is None else (target, reference)
    if any(shape[2] % 2 or shape[3] % 2 for shape in shapes):
        raise ValueError("video latent H/W must be divisible by patch (2, 2)")
    return target, audio, reference


def streaming_window_starts(
    latent_t: int, *, sink_size: int, window_size: int, chunk_size: int,
    audio_lookahead_latents: int = 17,
    audio_right_lookahead_latents: int | None = None,
    sink_switch_at: int = 1,
    sink_size_after_switch: int | None = None,
    merge_single_frame_units: bool = False,
    bootstrap_size: int | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> list[int]:
    """Return native starts for the bootstrap units and C-unit continuations.

    The bootstrap spans ``bootstrap_size`` units, W + C when omitted.
    """
    latent_t = _nonnegative_chunk_count(latent_t, "latent_t")
    sink_size = _nonnegative_chunk_count(sink_size, "sink_size")
    window_size = _nonnegative_chunk_count(window_size, "window_size")
    chunk_size = _nonnegative_chunk_count(chunk_size, "chunk_size")
    _nonnegative_chunk_count(audio_lookahead_latents, "audio_lookahead_latents")
    if audio_right_lookahead_latents is not None:
        _nonnegative_chunk_count(audio_right_lookahead_latents, "audio_right_lookahead_latents")
    sink_switch_at = _nonnegative_chunk_count(sink_switch_at, "sink_switch_at")
    if sink_switch_at == 0:
        raise ValueError("sink_switch_at must be a positive continuation index")
    if sink_size_after_switch is not None:
        sink_size_after_switch = _nonnegative_chunk_count(sink_size_after_switch, "sink_size_after_switch")
        if sink_size_after_switch > sink_size:
            raise ValueError("sink_size_after_switch must not exceed sink_size")
    if latent_t == 0 or chunk_size == 0:
        raise ValueError("latent_t and chunk_size must be positive")
    units = streaming_unit_timeline(
        merge_single_frame_units=merge_single_frame_units,
        video_temporal_mapping=video_temporal_mapping,
    )
    return [0, *(units.boundary(index) for index in range(
        _bootstrap_units(window_size, chunk_size, bootstrap_size), units.count_to_cover(latent_t), chunk_size
    ))]


def build_streaming_plan(
    *,
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    reference_video_shape: Sequence[int] | None = None,
    start: int,
    sink_size: int,
    window_size: int,
    chunk_size: int,
    audio_lookahead_latents: int = 17,
    audio_right_lookahead_latents: int | None = None,
    sink_switch_at: int = 1,
    sink_size_after_switch: int | None = None,
    merge_single_frame_units: bool = False,
    bootstrap_size: int | None = None,
    audio_previous_prediction_stop: int | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
    text_len: int = 0,
    fixed_window_rope: bool = False,
    keep_sink_reference: bool = True,
) -> dict[str, Any]:
    """Select a target window and optional reference using global latent indices.

    The bootstrap generates up to ``bootstrap_size`` units, W + C when
    omitted. Each continuation reads the available sink prefix and most
    recent W target units, then generates up to C new units, and its fixed
    RoPE template spans S + W + C units regardless of the bootstrap. Units
    normally contain one native latent. Enabling merge_single_frame_units
    groups the native H3 cadence into 2/1/1/1 latent units while preserving
    every latent and its original index. Overlapping
    history indices occur only once. Sink size
    can decrease at a 1-based continuation index, with bootstrap numbered 0.
    The plan records the current and next sink sizes so a completed window
    can retain exactly the history its successor needs. Optional references
    share the target selection, or only recent/current frames when
    keep_sink_reference is disabled. Audio
    indices address one channel's full timeline, preserving stereo ordering
    when the same selection is applied to both channels. Noisy masks align
    with the selected video and audio indices, not the complete source clip.
    Audio retains up to R generated future latents for decoder context, where
    audio_right_lookahead_latents selects R and defaults to audio_lookahead_latents.
    The latter configures the decoder's left context. Only
    the previously ungenerated right tail is noisy and supervised. Current
    audio is committed for publication independently of whether it is newly
    generated. The audio source shape may extend beyond the current video
    prefix in a live stream. Known clips supply their true complete audio
    shape to truncate the tail. An explicit previous prediction boundary
    records a live stream's actual generated extent.
    The complete selected window is one bidirectional attention document.
    """
    target_shape, audio_shape, reference_shape = _latent_shapes(
        target_video_shape, target_audio_shape, reference_video_shape
    )
    if min(target_shape) <= 0 or (reference_shape is not None and min(reference_shape) <= 0):
        raise ValueError("video latent dimensions must be positive")
    if audio_shape[1] <= 0 or audio_shape[2] < 0:
        raise ValueError("audio latent C must be positive and T nonnegative")
    latent_t = target_shape[1]
    starts = streaming_window_starts(
        latent_t, sink_size=sink_size, window_size=window_size, chunk_size=chunk_size,
        audio_lookahead_latents=audio_lookahead_latents,
        audio_right_lookahead_latents=audio_right_lookahead_latents,
        sink_switch_at=sink_switch_at, sink_size_after_switch=sink_size_after_switch,
        merge_single_frame_units=merge_single_frame_units, bootstrap_size=bootstrap_size,
        video_temporal_mapping=video_temporal_mapping,
    )
    start = _nonnegative_chunk_count(start, "start")
    text_len = _nonnegative_chunk_count(text_len, "text_len")
    if start not in starts:
        raise ValueError("start must select a bootstrap or continuation window")
    is_bootstrap = start == 0
    units = streaming_unit_timeline(
        merge_single_frame_units=merge_single_frame_units,
        video_temporal_mapping=video_temporal_mapping,
    )
    start_unit = units.count_to_cover(start)
    continuation_index = starts.index(start)
    reduced_sink_size = sink_size if sink_size_after_switch is None else sink_size_after_switch
    effective_sink_size = sink_size if continuation_index < sink_switch_at else reduced_sink_size
    next_sink_size = sink_size if continuation_index + 1 < sink_switch_at else reduced_sink_size
    stop = min(streaming_window_stop(start, {
        "window_size": window_size,
        "chunk_size": chunk_size,
        "merge_single_frame_units": merge_single_frame_units,
        "bootstrap_size": bootstrap_size,
    }, video_temporal_mapping), latent_t)
    stop_unit = units.count_to_cover(stop)
    sink_stop = units.boundary(effective_sink_size)
    sink_history_stop = min(sink_stop, start)
    recent_start = units.boundary(max(0, start_unit - window_size))
    next_sink_stop = min(units.boundary(next_sink_size), stop)
    next_recent_start = min(units.boundary(max(0, stop_unit - window_size)), stop)
    rope_window_stop = units.boundary(effective_sink_size + window_size + chunk_size)
    selected = (
        list(range(stop))
        if is_bootstrap
        else sorted(set(range(sink_history_stop)) | set(range(recent_start, stop)))
    )
    video_indices = torch.tensor(selected, dtype=torch.long)
    reference_video_indices = None
    if reference_shape is not None:
        reference_video_indices = (
            torch.arange(recent_start, stop, dtype=torch.long)
            if not keep_sink_reference and not is_bootstrap
            else video_indices
        )
    video_noisy_mask = video_indices >= start
    boundary = video_temporal_mapping.decode_timeline.clock_boundary_ceil
    audio_t = audio_shape[2]
    audio_indices_list, audio_commit_list = [], []
    for index in selected:
        audio_start = min(boundary(index), audio_t)
        audio_stop = min(boundary(index + 1), audio_t)
        audio_indices_list.extend(range(audio_start, audio_stop))
        audio_commit_list.extend([index >= start] * (audio_stop - audio_start))
    audio_commit_stop = min(boundary(stop), audio_t)
    right_lookahead = audio_lookahead_latents if audio_right_lookahead_latents is None else audio_right_lookahead_latents
    audio_prediction_stop = min(audio_commit_stop + right_lookahead, audio_t)
    if audio_previous_prediction_stop is None:
        audio_previous_prediction_stop = (
            0 if is_bootstrap else min(boundary(start) + right_lookahead, audio_t)
        )
    else:
        audio_previous_prediction_stop = min(
            _nonnegative_chunk_count(audio_previous_prediction_stop, "audio_previous_prediction_stop"),
            audio_t,
        )
    if is_bootstrap and audio_previous_prediction_stop != 0:
        raise ValueError("bootstrap audio must begin without generated history")
    if audio_previous_prediction_stop < min(boundary(start), audio_t):
        raise ValueError("generated audio must cover all previously committed audio")
    future_count = audio_prediction_stop - audio_commit_stop
    audio_commit_mask = torch.tensor(audio_commit_list + [False] * future_count, dtype=torch.bool)
    audio_lookahead_mask = torch.tensor([False] * len(audio_indices_list) + [True] * future_count, dtype=torch.bool)
    audio_indices_list.extend(range(audio_commit_stop, audio_prediction_stop))
    audio_indices = torch.tensor(audio_indices_list, dtype=torch.long)
    audio_noisy_mask = audio_indices >= audio_previous_prediction_stop
    target_frame_rows = (target_shape[2] // 2) * (target_shape[3] // 2)
    reference_frame_rows = (
        0 if reference_shape is None
        else (reference_shape[2] // 2) * (reference_shape[3] // 2)
    )
    packing_rows = (
        text_len
        + len(selected) * target_frame_rows
        + (0 if reference_video_indices is None else reference_video_indices.numel()) * reference_frame_rows
        + 2 * len(audio_indices_list)
    )
    plan = {
        "target_video_shape": target_shape,
        "target_audio_shape": audio_shape,
        "reference_video_shape": reference_shape,
        "video_temporal_mapping": video_temporal_mapping.to_dict(),
        "sink_size": sink_size,
        "sink_switch_at": sink_switch_at,
        "sink_size_after_switch": sink_size_after_switch,
        "continuation_index": continuation_index,
        "effective_sink_size": effective_sink_size,
        "next_sink_size": next_sink_size,
        "sink_stop": sink_stop,
        "sink_history_stop": sink_history_stop,
        "recent_start": recent_start,
        "next_sink_stop": next_sink_stop,
        "next_recent_start": next_recent_start,
        "rope_window_stop": rope_window_stop,
        "window_size": window_size,
        "chunk_size": chunk_size,
        "audio_lookahead_latents": audio_lookahead_latents,
        "merge_single_frame_units": merge_single_frame_units,
        "start": start,
        "stop": stop,
        "start_unit": start_unit,
        "stop_unit": stop_unit,
        "is_bootstrap": is_bootstrap,
        "fixed_window_rope": fixed_window_rope,
        "keep_sink_reference": keep_sink_reference,
        "video_indices": video_indices,
        "reference_video_indices": reference_video_indices,
        "audio_indices": audio_indices,
        "video_noisy_mask": video_noisy_mask,
        "audio_noisy_mask": audio_noisy_mask,
        "audio_retained_mask": ~audio_noisy_mask,
        "audio_commit_mask": audio_commit_mask,
        "audio_lookahead_mask": audio_lookahead_mask,
        "audio_commit_indices": audio_indices[audio_commit_mask],
        "audio_commit_stop": audio_commit_stop,
        "audio_prediction_stop": audio_prediction_stop,
        "audio_previous_prediction_stop": audio_previous_prediction_stop,
        "text_len": text_len,
        "packing_rows": packing_rows,
    }
    if audio_right_lookahead_latents is not None:
        plan["audio_right_lookahead_latents"] = audio_right_lookahead_latents
    if bootstrap_size is not None:
        plan["bootstrap_size"] = bootstrap_size
    return plan


def build_history_resample_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Predict a window's history alone while preserving its RoPE template.

    Current target AV rows are absent. References keep only the selected
    history frames, and text and picture conditions retain their original
    positions. Every selected AV row is predicted.
    """
    indices = {
        modality: plan[f"{modality}_indices"][~plan[f"{modality}_noisy_mask"]].cpu()
        for modality in ("video", "audio")
    }
    reference_indices = plan["reference_video_indices"]
    if reference_indices is not None:
        reference_indices = reference_indices.cpu()
        reference_indices = reference_indices[torch.isin(reference_indices, indices["video"])]
    target_shape, reference_shape = plan["target_video_shape"], plan["reference_video_shape"]
    picture_shape = plan.get("picture_shape")
    rows = (
        plan["text_len"] + indices["video"].numel() * (target_shape[2] // 2) * (target_shape[3] // 2)
        + 2 * indices["audio"].numel()
        + (0 if reference_shape is None else reference_indices.numel() * (reference_shape[2] // 2) * (reference_shape[3] // 2))
        + (0 if picture_shape is None else (picture_shape[2] // 2) * (picture_shape[3] // 2))
    )
    audio_indices = indices["audio"]
    return {
        **plan,
        "video_indices": indices["video"], "audio_indices": audio_indices,
        "reference_video_indices": reference_indices,
        "video_noisy_mask": torch.ones_like(indices["video"], dtype=torch.bool),
        "audio_noisy_mask": torch.ones_like(audio_indices, dtype=torch.bool),
        "audio_retained_mask": torch.zeros_like(audio_indices, dtype=torch.bool),
        "audio_commit_mask": torch.zeros_like(audio_indices, dtype=torch.bool),
        "audio_commit_indices": audio_indices[:0],
        "audio_lookahead_mask": plan["audio_lookahead_mask"][~plan["audio_noisy_mask"]].cpu(),
        "packing_rows": int(rows),
    }


class StreamingLatentT2AVDataset(CausalLatentT2AVDataset):
    """Reuse latent-corpus reading and resume while budgeting streaming windows.

    Workers return complete normalized clips. Window selection happens after
    SP input synchronization, and token budgets cover the largest legal window.
    """

    _STATE_SCHEMA = "minimax_h3_streaming_latent_worker"
    _STATE_VERSION = 1
    _MIN_SINK = 0

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        sink_size: int,
        window_size: int,
        chunk_size: int,
        audio_lookahead_latents: int = 17,
        audio_right_lookahead_latents: int | None = None,
        sink_switch_at: int = 1,
        sink_size_after_switch: int | None = None,
        merge_single_frame_units: bool = False,
        bootstrap_size: int | None = None,
        video_temporal_mapping: VideoTemporalMapping | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sink_size = _nonnegative_chunk_count(sink_size, "sink_size")
        self.chunk_size = _nonnegative_chunk_count(chunk_size, "chunk_size")
        self.audio_lookahead_latents = _nonnegative_chunk_count(
            audio_lookahead_latents, "audio_lookahead_latents"
        )
        self.audio_right_lookahead_latents = (
            None if audio_right_lookahead_latents is None
            else _nonnegative_chunk_count(audio_right_lookahead_latents, "audio_right_lookahead_latents")
        )
        self.video_temporal_mapping = (
            H3_VIDEO_TEMPORAL_MAPPING if video_temporal_mapping is None
            else video_temporal_mapping if isinstance(video_temporal_mapping, VideoTemporalMapping)
            else VideoTemporalMapping.from_dict(video_temporal_mapping)
        )
        streaming_window_starts(
            1, sink_size=sink_size, window_size=window_size, chunk_size=chunk_size,
            audio_lookahead_latents=audio_lookahead_latents,
            audio_right_lookahead_latents=audio_right_lookahead_latents,
            sink_switch_at=sink_switch_at, sink_size_after_switch=sink_size_after_switch,
            merge_single_frame_units=merge_single_frame_units, bootstrap_size=bootstrap_size,
            video_temporal_mapping=self.video_temporal_mapping,
        )
        if "sink" in kwargs or "independent_first_chunk" in kwargs:
            raise ValueError("streaming geometry uses sink_size, window_size and chunk_size")
        super().__init__(seed, resume_context, sink=0, window_size=window_size, **kwargs)
        self.streaming_config = {
            "sink_size": self.sink_size,
            "window_size": self.window_size,
            "chunk_size": self.chunk_size,
            "audio_lookahead_latents": self.audio_lookahead_latents,
            "sink_switch_at": sink_switch_at,
            "sink_size_after_switch": sink_size_after_switch,
            "merge_single_frame_units": merge_single_frame_units,
        }
        if self.audio_right_lookahead_latents is not None:
            self.streaming_config["audio_right_lookahead_latents"] = self.audio_right_lookahead_latents
        if bootstrap_size is not None:
            self.streaming_config["bootstrap_size"] = bootstrap_size
        self._max_window_media_rows = max(
            build_streaming_plan(
                target_video_shape=self.latent_shape,
                target_audio_shape=self.audio_shape,
                start=start,
                video_temporal_mapping=self.video_temporal_mapping,
                **self.streaming_config,
            )["packing_rows"]
            for start in streaming_window_starts(
                self.latent_t, **self.streaming_config,
                video_temporal_mapping=self.video_temporal_mapping,
            )
        )

    def _target_video_latent_t(self, frame_count: int) -> int:
        return self.video_temporal_mapping.target_latent_t(frame_count)

    def _pack_sample(self, *, prompt: str) -> dict[str, Any]:
        text_input_ids, text_len = self.tokenizer.encode(prompt)
        packing_rows = text_len + self._max_window_media_rows
        return {
            "prompts": prompt,
            "text_input_ids": text_input_ids,
            "text_lens": text_len,
            "latent_shapes": self.latent_shape,
            "audio_shapes": self.audio_shape,
            "video_temporal_mapping": self.video_temporal_mapping.to_dict(),
            "streaming_config": dict(self.streaming_config),
            "packing_rows": packing_rows,
            "seqlens": packing_rows,
        }


EntryClass = StreamingLatentT2AVDataset

__all__ = [
    "StreamingLatentT2AVDataset",
    "build_streaming_plan",
    "build_history_resample_plan",
    "streaming_target_shapes",
    "streaming_unit_timeline",
    "streaming_window_starts",
    "streaming_window_stop",
    "streaming_window_time_shift",
    "EntryClass",
]
