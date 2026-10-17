# SPDX-License-Identifier: Apache-2.0
"""Streaming DMD of a pixel-conditioned text-to-audio-video generator against Ref2VA scores.

The generator ``backbone`` is the student of ``MiniMaxH3VideoRefStreamingPixelTSCD``:
its windows carry no reference rows, the picture is placed by
``picture_mode``, normally as the native first-frame keyframe, and every
target video row reads the condition video of its own latent through the
backbone's input condition encoder, from the frames each window's payload
carries. It rolls out, writes back and trains on its own window, as in
``MiniMaxH3VideoRefStreamingDMD``.

The trainable fake score ``fake_model`` and the frozen real score
``tea_model`` are Ref2VA networks. At every student start they read their
own window, planned with ``meta_model.teacher_streaming`` (the student's
policy when unset) and with the clip's semantic latents, the payload's
``source_latents``, as reference rows. ``meta_model.teacher_picture_mode``,
``rows`` for Ref2VA scores, places the student's anchored picture in its own
time slot before those rows. Their noisy rows are the generator's in the same
order at the same noise and timesteps, and their history the generator's
write-back at their indices. The dataset budgets the score windows with their
reference rows through ``teacher_reference_video_rows``, which this meta
sets.

Both windows read one Qwen context, the picture as Picture 1 followed by the
caption, because ``qwen_reference_video`` is false and Qwen never reads the
condition video. With ``cfg_fitting_guidance`` the generator's branches drop
its condition video for ``reference`` and keep its keyframe, and the fake
score's drop its reference rows and keep its picture.

``student_cfg_fitting`` with ``cfg_fitting_guidance`` makes the generator read
its full-condition fit everywhere it predicts: at every rollout step, so the
written-back x0 is the fit, at GEN with gradients through the full-condition
prediction only, and in validation. For the chain ``0 --1.5--> T --3.0--> TS``
with guided anchors that is ``(f(TS) + f(T) + f(0)) / 3``, three forwards per
denoising step: T drops the condition video and keeps the keyframe and
caption, and 0 also drops the caption, reading the picture-only Qwen context.
Validation samples the three states as branches weighted 1/3 each over the
same noisy rows and history, each with its own Qwen context and condition
video, and requires ``guidance_scale`` 1. Without the switch validation
samples the raw generator as before.

Expected configuration additions::

    entry:
      module: dev.yanzuolu.engines.dmd
      class_name: DistributionMatchingDistillation
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_dmd
      class_name: MiniMaxH3VideoRefStreamingPixelDMD
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

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from dev.yanzuolu.projects.minimax_h3.meta_models.streaming_guidance import StreamingGuidanceBranch
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_dmd import (
    MiniMaxH3VideoRefStreamingDMD,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_rf import PixelChainMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_pixel_tscd import (
    SemanticReferenceTeacherMixin,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_rf import SemanticChainMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_pixel import (
    PixelConditionMixin,
    PixelStreamingBatch,
    condition_frames_of,
)


class MiniMaxH3VideoRefStreamingPixelDMD(SemanticReferenceTeacherMixin, PixelConditionMixin, PixelChainMixin,
                                         SemanticChainMixin, MiniMaxH3VideoRefStreamingDMD):
    """DMD of a pixel-conditioned generator whose Ref2VA scores read their own window."""

    _samples_guided_student = True

    def _dmd_student_plans(self, payload: dict[str, Any], plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach every sample's condition frames to its window plan."""
        return [dict(plan, condition_frames=value) for plan, value in zip(plans, condition_frames_of(payload), strict=True)]

    def _validation_rollout(self, models: dict[str, Any], backbone: Any, branch_inputs: Any,
                            rngs: Sequence[Any], *, guidance_scale: float, previous_guidance_scale: float) -> Any:
        """Sample a guided student through its full-condition fit."""
        if self.cfg_fitting_guidance is None or not self.student_cfg_fitting:
            return super()._validation_rollout(models, backbone, branch_inputs, rngs, guidance_scale=guidance_scale,
                                               previous_guidance_scale=previous_guidance_scale)
        if guidance_scale != 1.0 or previous_guidance_scale != 1.0:
            raise ValueError("a guided student samples through its fit, so validation guidance_scale must be 1")
        inputs = branch_inputs[0][1]
        texts = inputs.prompts if self.qwen_visual_context else [value.shape[0] for value in inputs.prompt_embeds]
        available = [frozenset({"reference", "text"} if text else {"reference"}) for text in texts]
        branches = self._guided_sampling_branches(inputs, available, lambda branch: self._student_sampling_batch(inputs, branch))
        return self._guided_rollout_latents(backbone, branches, rngs,
                                            **({"models": models} if self.qwen_visual_context else {}))

    def _student_sampling_batch(self, inputs: PixelStreamingBatch, branch: StreamingGuidanceBranch) -> PixelStreamingBatch:
        """One reduced-condition sampling batch: ``text`` drops the caption, ``reference`` the condition video."""
        if "text" in branch.drop:
            inputs = self._captionless_batch(inputs)
        return replace(inputs, condition_frames=None) if "reference" in branch.drop else inputs


EntryClass = MiniMaxH3VideoRefStreamingPixelDMD

__all__ = ["MiniMaxH3VideoRefStreamingPixelDMD", "EntryClass"]
