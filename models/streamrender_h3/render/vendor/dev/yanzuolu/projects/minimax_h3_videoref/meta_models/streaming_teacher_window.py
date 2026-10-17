# SPDX-License-Identifier: Apache-2.0
"""A teacher's own streaming window at every student start of a chain.

``TeacherWindowChainMixin`` serves distillation hosts of the videoref chain
pool whose teacher networks read another window policy than the student.
Optional ``meta_model.teacher_streaming`` overrides the student's policy for
the teacher with any of ``sink_size``, ``window_size``, ``chunk_size``,
``sink_switch_at``, ``sink_size_after_switch``, ``bootstrap_size``,
``audio_lookahead_latents`` and ``audio_right_lookahead_latents``. The head
then plans the teacher's window at every student start, and every window's
payload carries the teacher's plans conditioned like the student's. Both
policies must share C, the bootstrap stop and the effective audio right
lookahead, so every window has the same noisy video and audio rows in the
same order, which the head checks window by window. A student window other
than the teacher's therefore sets ``data.args.bootstrap_size`` to the
teacher's W + C. Its S may exceed the bootstrap because the planner reads only
available history. Layout flags stay meta-level and shared, and a teacher
policy other than the student's requires ``qwen_reference_video: false`` so
that both read the same Qwen context. ``meta_model.teacher_picture_mode``, the
student's ``picture_mode`` by default, places the picture in the teacher's
window. The meta sets ``data.args.teacher_streaming_config``, so the dataset
budgets the larger of the two windows at every start.

``_teacher_window`` builds the teacher's window: history from the chain's
working copies at the teacher's indices, anchored from a fork of each sample
stream, retained audio clean, the student's noisy rows, the student's
anchored picture and the teacher's reference rows. ``_teacher_chain_geometry``
supplies the reference latents the teacher's windows are planned with and
``_teacher_references`` the latents its window reads, the student's by
default, so hosts whose student reads no reference rows may still give the
teacher its reference rows.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.data.streaming import streaming_window_starts
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import _ChainState

_TEACHER_POLICY_KEYS = frozenset({
    "sink_size", "window_size", "chunk_size", "sink_switch_at", "sink_size_after_switch",
    "bootstrap_size", "audio_lookahead_latents", "audio_right_lookahead_latents",
})


class TeacherWindowChainMixin:
    """Plan and build the teacher's own window beside every student window of a chain."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.teacher_picture_mode = str(config.meta_model.get("teacher_picture_mode", self.picture_mode))
        if self.teacher_picture_mode not in {"rows", "keyframe"}:
            raise ValueError("meta_model.teacher_picture_mode must be rows or keyframe")
        overrides = config.meta_model.get("teacher_streaming")
        self.teacher_streaming = None if overrides is None else dict(overrides)
        if self.teacher_streaming is None:
            if self.teacher_picture_mode != self.picture_mode:
                raise ValueError("a teacher_picture_mode other than picture_mode requires meta_model.teacher_streaming")
            return
        assert set(self.teacher_streaming) <= _TEACHER_POLICY_KEYS, (
            f"meta_model.teacher_streaming accepts only {sorted(_TEACHER_POLICY_KEYS)}"
        )
        student, teacher = self.streaming_config, self._teacher_policy(self.streaming_config)
        streaming_window_starts(1, **teacher, video_temporal_mapping=self.video_temporal_mapping)
        assert teacher["chunk_size"] == student["chunk_size"], "teacher and student must share chunk_size"
        assert self._bootstrap_size(teacher) == self._bootstrap_size(student), (
            "the student's bootstrap must span the teacher's: set data.args.bootstrap_size to the teacher's W + C"
        )
        assert self._audio_right_lookahead(teacher) == self._audio_right_lookahead(student), (
            "teacher and student must share the effective audio right lookahead"
        )
        assert teacher == student or not (self.qwen_visual_context and self.qwen_reference_video), (
            "a teacher window policy other than the student's requires qwen_reference_video false"
        )
        configured = config.data.args.get("teacher_streaming_config")
        if configured is not None and dict(configured) != self.teacher_streaming:
            raise ValueError("data.args.teacher_streaming_config must match meta_model.teacher_streaming")
        config.data.args.teacher_streaming_config = dict(self.teacher_streaming)

    def _teacher_policy(self, policy: dict[str, Any]) -> dict[str, Any]:
        return {**policy, **self.teacher_streaming}

    def _teacher_chain_geometry(self, batch: dict[str, Any]) -> StreamingBatch:
        """The chain head's geometry the teacher's windows are planned with, the student's by default."""
        geometry, _ = self._chain_head_inputs(batch)
        return geometry

    def _teacher_references(self, ctx: dict[str, Any]) -> list[torch.Tensor] | None:
        """Complete clean reference latents of the teacher's window, the student's by default."""
        return ctx["inputs"].reference_latents

    # -------------------------------------------------------------- chain --
    def _start_chain(self, ctx: dict[str, Any]) -> _ChainState:
        """Also plan the teacher's window at every student start of the chain."""
        entry = super()._start_chain(ctx)
        if self.teacher_streaming is None:
            return entry
        geometry = self._teacher_chain_geometry(ctx["batch"])
        teacher = replace(geometry, streaming_configs=[self._teacher_policy(policy) for policy in geometry.streaming_configs])
        entry.teacher_plans = []
        for index, plans in enumerate(entry.plans):
            latent_t = geometry.video_shapes[index][1]
            assert streaming_window_starts(
                latent_t, **teacher.streaming_configs[index], video_temporal_mapping=self.video_temporal_mapping,
            ) == streaming_window_starts(
                latent_t, **geometry.streaming_configs[index], video_temporal_mapping=self.video_temporal_mapping,
            ), "teacher and student streaming windows must start together"
            windows = [self._plan(teacher, index, plan["start"]) for plan in plans]
            for mine, student in zip(windows, plans, strict=True):
                assert all(torch.equal(self._chunk_rows(mine, audio=audio), self._chunk_rows(student, audio=audio))
                           for audio in (False, True)), "teacher and student windows must share their noisy rows"
                assert (mine["audio_commit_stop"], mine["audio_prediction_stop"]) == (
                    student["audio_commit_stop"], student["audio_prediction_stop"]
                ), "teacher and student windows must commit and predict the same audio"
            entry.teacher_plans.append(windows)
        return entry

    def _chain_window_conditioning(
        self, ctx: dict[str, Any], entry: _ChainState, window: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Condition the teacher's plans like the student's and carry them in the payload."""
        plans, conditioning = super()._chain_window_conditioning(ctx, entry, window)
        if entry.teacher_plans is None:
            return plans, conditioning
        teachers = []
        for windows, student, budget in zip(entry.teacher_plans, plans, entry.packing_rows, strict=True):
            plan = self._conditioned_plan(windows[window], student["text_len"])
            if student.get("picture_shape") is not None:
                plan = self._with_picture(plan, student["picture_clean_latents"],
                                          keyframe=self.teacher_picture_mode == "keyframe")
            assert plan["packing_rows"] <= budget, "teacher window exceeds the dataset packing budget"
            teachers.append(plan)
        return plans, dict(conditioning, teacher_plans=teachers)

    def _teacher_window(
        self, ctx: dict[str, Any], students: list[dict[str, Any]], rngs: list[Any],
        noisy: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Build the teacher's window and its complete noisy windows from the payload and the working copies.

        ``students`` are the student's plans with their anchored pictures and
        ``rngs`` the sample streams. ``noisy`` holds the student's noisy rows
        in order, or None to leave those rows zero for later values.
        """
        inputs, anchor = ctx["inputs"], MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
        plans, xts = [], ([], [])
        sources = self._teacher_references(ctx)
        references = None if sources is None else []
        for index, (payload_plan, student, sample_rng) in enumerate(zip(
            ctx["encoded_batch"]["teacher_plans"], students, rngs, strict=True,
        )):
            plan = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in payload_plan.items()}
            if plan.get("picture_shape") is not None:
                plan["picture_latents"] = student["picture_latents"]
            rng = sample_rng.fork("teacher_window")
            for modality, is_audio in enumerate((False, True)):
                prefix = "audio" if is_audio else "video"
                values = self._select(ctx["clean_latents"][modality][index], plan[f"{prefix}_indices"],
                                      audio=is_audio).detach().float()
                if not is_audio:
                    values = anchor * values + (1 - anchor) * self._noise(values, rng)
                rows = plan[f"{prefix}_noisy_mask"].nonzero().flatten().to(values.device)
                if noisy is None:
                    values.index_fill_(2 if is_audio else 1, rows, 0)
                else:
                    values.index_copy_(2 if is_audio else 1, rows, noisy[modality][index].to(values.dtype))
                xts[modality].append(values)
            if references is not None:
                reference = self._select(sources[index], plan["reference_video_indices"]).detach().float()
                references.append(anchor * reference + (1 - anchor) * self._noise(reference, rng))
            plans.append(plan)
        return self._window_inputs(inputs, plans, references), xts


__all__ = ["TeacherWindowChainMixin"]
