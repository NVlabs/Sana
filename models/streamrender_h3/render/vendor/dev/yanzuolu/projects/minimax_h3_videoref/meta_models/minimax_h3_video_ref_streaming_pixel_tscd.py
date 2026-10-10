# SPDX-License-Identifier: Apache-2.0
"""Streaming TSCD from a video-reference teacher into a pixel-conditioned text-to-audio-video student.

The student, and with it the EMA target, is ``MiniMaxH3VideoRefStreamingPixelRF``:
its windows carry no reference rows, the picture is placed by ``picture_mode``,
normally as the native first-frame keyframe, and every target video row reads
the condition video of its own latent through the backbone's input condition
encoder. The frozen teacher is a video-reference model. At every student start
it reads its own window, planned with ``meta_model.teacher_streaming`` (the
student's policy when unset) and with the clip's reference latents, the
payload's ``source_latents``, as reference rows anchored at the clean time.
``meta_model.teacher_picture_mode``, ``rows`` for a Ref2VA teacher, places the
picture in its own time slot before those rows. The teacher's plans carry no
condition frames, so its forward receives no input condition.

Both networks read one Qwen context. Without ``qwen_reference_video`` it is the
picture as Picture 1 followed by the caption for every picture mode, and the
captionless context is the picture alone, so their text rows are equal. The
teacher's noisy target rows are the student's x_t in the same order at the
same video and audio times, and its history rows read the chain's working
copies, the student's own write-back, at its indices, as in
``MiniMaxH3VideoRefStreamingTSCD``. With ``keep_negative_reference`` each
network's captionless branch drops only its caption: the teacher keeps its
reference rows and picture, the student its keyframe and condition video.

The dataset loads the reference latents for the teacher and decodes
``data.args.pixel_condition`` for the student. It budgets the teacher's
windows with their reference rows through ``teacher_reference_video_rows``,
which this meta sets, beside student windows without them.

Expected configuration additions::

    entry:
      module: dev.yanzuolu.engines.tscd
      class_name: TrajectorySegmentedConsistencyDistillation
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_tscd
      class_name: MiniMaxH3VideoRefStreamingPixelTSCD
      picture_mode: keyframe
      teacher_picture_mode: rows
      teacher_streaming: {sink_size: 4, window_size: 5}
      qwen_reference_video: false
    data:
      args:
        pixel_condition: {root: /path/to/clips, filename: semantics.mp4}
    models:
      backbone:
        args:
          input_condition_channels: 772
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_rf import PixelChainMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_rf import SemanticChainMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_tscd import (
    MiniMaxH3VideoRefStreamingTSCD,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_pixel import PixelConditionMixin


class SemanticReferenceTeacherMixin:
    """Give a teacher without the student's reference rows its own window with the clip's semantic latents as references.

    The teacher's windows follow ``meta_model.teacher_streaming``, the student's
    policy when it is empty, and the dataset budgets them with their reference
    rows through ``data.args.teacher_reference_video_rows``.
    """

    def __init__(self, config: Any) -> None:
        if config.meta_model.get("teacher_streaming") is None:
            config.meta_model.teacher_streaming = {}
        configured = config.data.args.get("teacher_reference_video_rows")
        if configured is not None and not bool(configured):
            raise ValueError("the teacher reads reference rows, so data.args.teacher_reference_video_rows must be true")
        config.data.args.teacher_reference_video_rows = True
        super().__init__(config)

    def _teacher_chain_geometry(self, batch: dict[str, Any]) -> StreamingBatch:
        """Plan the teacher's windows with the clip's reference latents."""
        return replace(super()._teacher_chain_geometry(batch), reference_latents=list(batch["reference_video_latents"]))

    def _teacher_references(self, ctx: dict[str, Any]) -> list[torch.Tensor]:
        return ctx["encoded_batch"]["source_latents"]


class MiniMaxH3VideoRefStreamingPixelTSCD(SemanticReferenceTeacherMixin, PixelConditionMixin, PixelChainMixin,
                                          SemanticChainMixin, MiniMaxH3VideoRefStreamingTSCD):
    """Distill a video-reference teacher into a pixel-conditioned text-to-audio-video student."""


EntryClass = MiniMaxH3VideoRefStreamingPixelTSCD

__all__ = ["MiniMaxH3VideoRefStreamingPixelTSCD", "SemanticReferenceTeacherMixin", "EntryClass"]
