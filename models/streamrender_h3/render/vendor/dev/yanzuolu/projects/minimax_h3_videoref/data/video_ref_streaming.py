# SPDX-License-Identifier: Apache-2.0
"""Paired raw-media streaming backed by the shared AV window planner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dev.yanzuolu.common.data import WorkerResumeContext
from dev.yanzuolu.projects.minimax_h3.data.streaming import (
    build_streaming_plan,
    streaming_window_starts,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_latent import (
    _latent_shapes,
    _nonnegative_chunk_count,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_raw import (
    CausalVideoRefRawT2AVDataset,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_qwen import (
    streaming_ref_presentation_length,
)


def build_video_ref_streaming_plan(
    *,
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    reference_video_shape: Sequence[int] | None,
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
    """Plan paired target/reference rows with the shared streaming semantics."""
    return build_streaming_plan(
        target_video_shape=target_video_shape,
        target_audio_shape=target_audio_shape,
        reference_video_shape=reference_video_shape,
        start=start,
        sink_size=sink_size,
        window_size=window_size,
        chunk_size=chunk_size,
        audio_lookahead_latents=audio_lookahead_latents,
        audio_right_lookahead_latents=audio_right_lookahead_latents,
        sink_switch_at=sink_switch_at,
        sink_size_after_switch=sink_size_after_switch,
        merge_single_frame_units=merge_single_frame_units,
        bootstrap_size=bootstrap_size,
        audio_previous_prediction_stop=audio_previous_prediction_stop,
        video_temporal_mapping=video_temporal_mapping,
        text_len=text_len,
        fixed_window_rope=fixed_window_rope,
        keep_sink_reference=keep_sink_reference,
    )


class _VideoRefStreamingLayout:
    """Shared window budgets for raw and normalized-latent video references.

    ``teacher_streaming_config`` overrides the streaming policy for a
    distillation teacher that reads its own window at every student start.
    The budget then covers the larger of the two windows at each start.
    ``reference_video_rows`` false budgets windows without reference rows,
    for hosts that read the reference latents but never place them in the window.
    ``teacher_reference_video_rows`` sets the same for the teacher's windows,
    following ``reference_video_rows`` when None, so a teacher that reads the
    reference rows is budgeted with them beside a student that has none.
    """

    teacher_streaming_config: dict[str, Any] | None = None
    teacher_reference_video_rows: bool | None = None

    def _configure_streaming_conditioning(
        self, *, qwen_visual_context: bool, qwen_processor_path: str | None,
        fixed_window_rope: bool, keep_sink_reference: bool, qwen_reference_video: bool = True,
        reference_video_rows: bool = True,
    ) -> None:
        if qwen_visual_context and not qwen_processor_path:
            raise ValueError("Qwen visual context requires qwen_processor_path")
        self.qwen_visual_context = qwen_visual_context
        self.qwen_reference_video = qwen_reference_video
        self.qwen_processor_path = qwen_processor_path
        self.fixed_window_rope = fixed_window_rope
        self._qwen_processor: MiniMaxH3Ref2VAPresentationProcessor | None = None
        self._qwen_media_prefix_rows: int | None = None
        self.keep_sink_reference = keep_sink_reference
        self.reference_video_rows = reference_video_rows

    def _set_streaming_window_budget(self) -> None:
        policies = [(self.streaming_config, self.reference_video_rows)]
        if self.teacher_streaming_config is not None:
            policies.append(({**self.streaming_config, **self.teacher_streaming_config},
                             self.reference_video_rows if self.teacher_reference_video_rows is None
                             else self.teacher_reference_video_rows))
        plans = [
            build_video_ref_streaming_plan(
                target_video_shape=self.latent_shape,
                target_audio_shape=self.audio_shape,
                reference_video_shape=self.reference_latent_shape if reference_rows else None,
                start=start,
                video_temporal_mapping=self.video_temporal_mapping,
                keep_sink_reference=self.keep_sink_reference if self.qwen_visual_context else True,
                fixed_window_rope=self.fixed_window_rope if self.qwen_visual_context else False,
                **policy,
            )
            for start in streaming_window_starts(
                self.latent_t, **self.streaming_config,
                video_temporal_mapping=self.video_temporal_mapping,
            )
            for policy, reference_rows in policies
        ]
        if self.qwen_visual_context:
            self._qwen_window_plans = plans
        self._max_window_media_rows = max(plan["packing_rows"] for plan in plans)

    def _picture_prefix_rows(self, processor: MiniMaxH3Ref2VAPresentationProcessor) -> int:
        """Rows a leading picture adds to every window, none without picture conditions."""
        return 0

    def _qwen_packing_rows(self, prompt: str) -> int:
        if self._qwen_processor is None:
            self._qwen_processor = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(self.qwen_processor_path)
        processor = self._qwen_processor
        if self._qwen_media_prefix_rows is None:
            self._qwen_media_prefix_rows = self._picture_prefix_rows(processor) + max(
                plan["packing_rows"] + (streaming_ref_presentation_length(
                    processor, "", plan, height=self.reference_height, width=self.reference_width,
                    fixed_window_rope=self.fixed_window_rope,
                ) if self.qwen_reference_video else 0)
                for plan in self._qwen_window_plans
            )
        text_len = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return self._qwen_media_prefix_rows + text_len

    def _pack_video_ref_sample(self, *, prompt: str) -> dict[str, Any]:
        text_input_ids, text_len = self.tokenizer.encode(prompt)
        return {
            "prompts": prompt,
            "text_input_ids": text_input_ids,
            "text_lens": text_len,
            "latent_shapes": self.latent_shape,
            "audio_shapes": self.audio_shape,
            "reference_latent_shapes": self.reference_latent_shape,
            "video_temporal_mapping": self.video_temporal_mapping.to_dict(),
            "streaming_config": dict(self.streaming_config),
            "packing_rows": (
                self._qwen_packing_rows(prompt) if self.qwen_visual_context
                else text_len + self._max_window_media_rows
            ),
        }


class VideoRefStreamingRawT2AVDataset(_VideoRefStreamingLayout, CausalVideoRefRawT2AVDataset):
    """Read paired full clips and budget the largest selected streaming window.

    Window selection belongs to the meta model. Dataset workers only sample the shared raw-media crop
    and language dropout. Optional Qwen budgets cover every legal reference
    window using processor geometry, without encoding pixels in the worker.
    """

    _STATE_SCHEMA = "minimax_h3_videoref_streaming_raw_worker"
    _STATE_VERSION = 1

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
        qwen_visual_context: bool = False,
        qwen_processor_path: str | None = None,
        fixed_window_rope: bool = False,
        keep_sink_reference: bool = True,
        qwen_reference_video: bool = True,
        **kwargs: Any,
    ) -> None:
        self._configure_streaming_conditioning(
            qwen_visual_context=qwen_visual_context, qwen_processor_path=qwen_processor_path,
            fixed_window_rope=fixed_window_rope, keep_sink_reference=keep_sink_reference,
            qwen_reference_video=qwen_reference_video,
        )
        self.sink_size = _nonnegative_chunk_count(sink_size, "sink_size")
        self.audio_lookahead_latents = _nonnegative_chunk_count(
            audio_lookahead_latents, "audio_lookahead_latents"
        )
        self.audio_right_lookahead_latents = (
            None if audio_right_lookahead_latents is None
            else _nonnegative_chunk_count(audio_right_lookahead_latents, "audio_right_lookahead_latents")
        )
        if "sink" in kwargs or "independent_first_chunk" in kwargs:
            raise ValueError("streaming geometry uses sink_size, window_size and chunk_size")
        super().__init__(
            seed, resume_context, chunk_size=chunk_size, window_size=window_size, **kwargs
        )
        streaming_window_starts(
            1, sink_size=sink_size, window_size=window_size, chunk_size=chunk_size,
            audio_lookahead_latents=audio_lookahead_latents,
            audio_right_lookahead_latents=audio_right_lookahead_latents,
            sink_switch_at=sink_switch_at, sink_size_after_switch=sink_size_after_switch,
            merge_single_frame_units=merge_single_frame_units, bootstrap_size=bootstrap_size,
            video_temporal_mapping=self.video_temporal_mapping,
        )
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
        self._set_streaming_window_budget()


EntryClass = VideoRefStreamingRawT2AVDataset

__all__ = [
    "VideoRefStreamingRawT2AVDataset",
    "build_video_ref_streaming_plan",
    "streaming_window_starts",
    "EntryClass",
]
