# SPDX-License-Identifier: Apache-2.0
"""Raw reference-prefix teacher forcing with exact Qwen packing budgets."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dev.yanzuolu.common.data import WorkerResumeContext
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_raw import (
    CausalVideoRefRawTeacherForcingT2AVDataset,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
    Ref2VAPresentationMedia,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.video_ref_conditions import (
    _decoded_video,
)


def _chunk_ranges(
    values: Sequence[Sequence[int]],
    total: int,
    name: str,
    *,
    allow_empty: bool,
) -> list[tuple[int, int]]:
    ranges = [(int(start), int(stop)) for start, stop in values]
    cursor = 0
    for start, stop in ranges:
        if start != cursor or stop < start or stop > total:
            raise ValueError(f"{name} must partition [0, {total}) in order")
        if stop == start and not allow_empty:
            raise ValueError(f"{name} must contain non-empty chunks")
        cursor = stop
    if cursor != total:
        raise ValueError(f"{name} must cover all {total} latent steps")
    return ranges


def _positive_chunk_size(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _audio_chunk_ranges_for_video(
    video_ranges: Sequence[Sequence[int]],
    audio_t: int,
    video_time: Callable[[int], float],
) -> list[tuple[int, int]]:
    """Assign audio starts to video chunks on the supplied audio-latent clock."""
    return [
        (
            min(math.ceil(video_time(int(start))), audio_t),
            min(math.ceil(video_time(int(stop))), audio_t),
        )
        for start, stop in video_ranges
    ]


def build_video_ref_prefix_tf_plan(
    *,
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    chunk_size: int = 5,
    independent_first_chunk: int | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
    video_chunk_ranges: Sequence[Sequence[int]] | None = None,
    audio_chunk_ranges: Sequence[Sequence[int]] | None = None,
    target_chunk_index: int | None = None,
) -> dict[str, Any]:
    """Define the reference prefix and target visibility before physical packing.

    Training uses ``P, N0, C0, ..., Nlast``. Validation of chunk ``k`` uses
    ``P, C0, ..., C(k-1), Nk``. The complete prefix is one bidirectional block
    that cannot read targets. Each target reads the prefix and every earlier
    clean chunk. Block roles are 0 for prefix, 1 for clean and 2 for noisy.
    Chunk sizes count video latents. The first chunk defaults to chunk_size,
    and a final chunk may be shorter. Explicit ranges take precedence.
    """
    chunk_size = _positive_chunk_size(chunk_size, "chunk_size")
    first_chunk = (
        chunk_size
        if independent_first_chunk is None
        else _positive_chunk_size(independent_first_chunk, "independent_first_chunk")
    )
    video_shape = tuple(int(value) for value in target_video_shape)
    audio_shape = tuple(int(value) for value in target_audio_shape)
    if len(video_shape) != 4 or len(audio_shape) != 3:
        raise ValueError("video latents must be [C,T,H,W] and audio must be [2,C,T]")
    _, latent_t, latent_h, latent_w = video_shape
    if min(video_shape) <= 0 or latent_h % 2 or latent_w % 2:
        raise ValueError("video latent dimensions must be positive with even H/W")
    if audio_shape[0] != 2 or audio_shape[1] <= 0 or audio_shape[2] < 0:
        raise ValueError("audio latents must have shape [2,C,T] with C > 0 and T >= 0")
    audio_t = audio_shape[2]
    if video_chunk_ranges is None:
        first_stop = min(first_chunk, latent_t)
        video_chunk_ranges = [(0, first_stop)] + [
            (start, min(start + chunk_size, latent_t))
            for start in range(first_stop, latent_t, chunk_size)
        ]
    video_ranges = _chunk_ranges(
        video_chunk_ranges,
        latent_t,
        "video_chunk_ranges",
        allow_empty=False,
    )
    audio_ranges = _chunk_ranges(
        _audio_chunk_ranges_for_video(
            video_ranges, audio_t, video_temporal_mapping.decode_timeline.clock_boundary_ceil
        )
        if audio_chunk_ranges is None
        else audio_chunk_ranges,
        audio_t,
        "audio_chunk_ranges",
        allow_empty=True,
    )
    if len(video_ranges) != len(audio_ranges):
        raise ValueError("video and audio must have the same number of chunks")
    if target_chunk_index is not None and not 0 <= target_chunk_index < len(video_ranges):
        raise ValueError("target_chunk_index must select an existing target chunk")

    block_roles = [0]
    block_chunk_indices: list[int | None] = [None]
    attn_modes = ["full"]
    target_packing_rows = 0
    frame_rows = (latent_h // 2) * (latent_w // 2)
    num_chunks = len(video_ranges) if target_chunk_index is None else target_chunk_index + 1
    for index in range(num_chunks):
        if target_chunk_index is None:
            roles = (2, 1) if index + 1 < num_chunks else (2,)
        else:
            roles = (2,) if index == target_chunk_index else (1,)
        video_start, video_stop = video_ranges[index]
        audio_start, audio_stop = audio_ranges[index]
        chunk_rows = (video_stop - video_start) * frame_rows + 2 * (audio_stop - audio_start)
        for role in roles:
            block_roles.append(role)
            block_chunk_indices.append(index)
            attn_modes.append("noise" if role == 2 else "full")
            target_packing_rows += chunk_rows
    plan = {
        "target_video_shape": video_shape,
        "target_audio_shape": audio_shape,
        "video_chunk_ranges": video_ranges,
        "audio_chunk_ranges": audio_ranges,
        "block_roles": block_roles,
        "block_chunk_indices": block_chunk_indices,
        "attn_modes": attn_modes,
        "sink": 1,
        "window_size": None,
        "target_packing_rows": target_packing_rows,
    }
    if video_temporal_mapping != H3_VIDEO_TEMPORAL_MAPPING:
        plan["video_temporal_mapping"] = video_temporal_mapping.to_dict()
    return plan


class VideoRefPrefixRawTeacherForcingT2AVDataset(CausalVideoRefRawTeacherForcingT2AVDataset):
    """Own prefix-TF plans and budget complete Qwen presentations in each batch."""

    _STATE_SCHEMA = "minimax_h3_videoref_prefix_tf_raw_worker"
    _prefix_plan_factory = staticmethod(build_video_ref_prefix_tf_plan)

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        processor_path: str,
        chunk_size: int = 5,
        independent_first_chunk: int | None = None,
        video_temporal_mapping: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if int(kwargs.get("sink", 1)) != 1 or kwargs.get("window_size") is not None:
            raise ValueError("reference prefix requires sink=1 and window_size=null")
        self.video_temporal_mapping = (
            H3_VIDEO_TEMPORAL_MAPPING
            if video_temporal_mapping is None
            else VideoTemporalMapping.from_dict(video_temporal_mapping)
        )
        self.chunk_size = _positive_chunk_size(chunk_size, "chunk_size")
        self.independent_first_chunk = (
            None
            if independent_first_chunk is None
            else _positive_chunk_size(independent_first_chunk, "independent_first_chunk")
        )
        kwargs.setdefault("tokenizer_path", str(processor_path))
        super().__init__(
            seed, resume_context,
            chunk_size=self.chunk_size,
            independent_first_chunk=self.independent_first_chunk,
            video_temporal_mapping=self.video_temporal_mapping.to_dict(),
            **kwargs,
        )
        self.processor_path = str(processor_path)
        self._processor: MiniMaxH3Ref2VAPresentationProcessor | None = None

    def _target_video_latent_t(self, frame_count: int) -> int:
        return self.video_temporal_mapping.target_latent_t(frame_count)

    def _presentation_processor(self) -> MiniMaxH3Ref2VAPresentationProcessor:
        if self._processor is None:
            self._processor = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(
                self.processor_path
            )
        return self._processor

    def _pack_video_ref_sample(self, *, prompt: str) -> dict[str, Any]:
        plan = self._prefix_plan_factory(
            target_video_shape=self.latent_shape,
            target_audio_shape=self.audio_shape,
            chunk_size=self.chunk_size,
            independent_first_chunk=self.independent_first_chunk,
            video_temporal_mapping=self.video_temporal_mapping,
        )
        _, ref_t, ref_h, ref_w = self.reference_latent_shape
        media_rows = plan["target_packing_rows"] + ref_t * (ref_h // 2) * (ref_w // 2)
        return {
            "prompts": prompt,
            "prefix_plan": plan,
            "latent_shapes": self.latent_shape,
            "audio_shapes": self.audio_shape,
            "reference_latent_shapes": self.reference_latent_shape,
            "media_packing_rows": media_rows,
            "packing_rows": media_rows,
        }

    def _finalize_video_ref_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        media, _ = _decoded_video(
            sample["reference_video_pixels"],
            video_temporal_mapping=self.video_temporal_mapping,
        )
        presentation = self._presentation_processor().build(
            sample["prompts"],
            [Ref2VAPresentationMedia(kind="video", media=media, has_audio=False)],
        )
        sample["ref_presentation"] = presentation
        sample["packing_rows"] = sample["media_packing_rows"] + int(presentation.input_ids.numel())
        return sample


EntryClass = VideoRefPrefixRawTeacherForcingT2AVDataset

__all__ = [
    "VideoRefPrefixRawTeacherForcingT2AVDataset",
    "build_video_ref_prefix_tf_plan",
    "EntryClass",
]
