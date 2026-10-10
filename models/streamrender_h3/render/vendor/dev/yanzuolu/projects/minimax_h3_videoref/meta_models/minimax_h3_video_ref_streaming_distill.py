# SPDX-License-Identifier: Apache-2.0
"""Streaming distillation over resampling-forcing chains.

``MiniMaxH3StreamingDistill`` regresses a streaming student's prediction onto
a frozen teacher's inside windows whose history is corpus latents. This meta
trains the same objective over the chains of ``MiniMaxH3VideoRefStreamingRF``:
a pack's windows run in order, one per engine iteration, and every window's
history rows are the student's one-step x0 reconstructions of the earlier
windows. The reconstruction is the RF write-back unchanged, a no-grad
``eval()`` student forward at the clip's ``t_s`` from ``resample_timesteps``.
Timesteps cover the full ``training_timesteps`` range, as in RF.

Per iteration the resample forward runs first, then the student's training
forward and the frozen teacher's no-grad forwards on the same noisy rows at
the same timesteps. The write-back follows the loss, so both networks read
the same history in one iteration, and the next window reads the new values.
Ranks at different chain phases issue the same collectives.

The student's window policy comes from ``data.args`` as usual. Optional
``meta_model.teacher_streaming`` gives the teacher its own window at every
student start, as ``TeacherWindowChainMixin`` in ``streaming_teacher_window``
plans and builds it: history from the working copies at the teacher's
indices, the student's noisy rows unchanged, the teacher's reference rows and
fixed-RoPE template, and the student's picture placed by
``teacher_picture_mode``. Without ``teacher_streaming`` the teacher reads the
student's window exactly.

The student always reads its raw forward. The teacher target is the
teacher's raw forward by default. ``meta_model.teacher_cfg_fitting`` makes
it the full-condition fit of the combination the teacher was trained with,
which ``meta_model.cfg_fitting_guidance`` names with the SFT keys, as
``GuidedDistillationFittingMixin`` in ``streaming_guided_fitting``
describes. Every branch drops its conditions on the teacher's own window
and every sample compiles its own fit from the conditions it holds. For a
teacher trained along ``0 --1.5--> T --3.0--> TS`` with guided anchors the
target is ``(f(TS) + f(T) + f(0)) / 3``, three teacher forwards per window,
where T drops the reference rows, 0 also drops the caption, and the picture
with its Qwen context stays in every branch. The student then learns the
guidance-free prediction and samples it with guidance 1 and one forward per
step. Validation samples the student and its EMA as RF does.

Expected configuration additions::

    entry:
      module: dev.yanzuolu.engines.diffusion_distillation
      class_name: DiffusionDistillation
    data:
      args: {sink_size: 2, window_size: 2, chunk_size: 2, bootstrap_size: 7}
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_distill
      class_name: MiniMaxH3VideoRefStreamingDistill
      teacher_streaming: {sink_size: 4, window_size: 5}
      teacher_cfg_fitting: true
      cfg_fitting_guidance:
        anchors: guided
        branches:
          - {name: "null", drop: [text, reference]}
          - {name: text_only, drop: [reference]}
        stages:
          - {branch: "null", scale: 1.5}
          - {branch: text_only, scale: 3.0}
      resample_timesteps: {...}
"""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_distill import MiniMaxH3StreamingDistill
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import MiniMaxH3VideoRefStreamingRF
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_guided_fitting import GuidedDistillationFittingMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_teacher_window import TeacherWindowChainMixin


class MiniMaxH3VideoRefStreamingDistill(GuidedDistillationFittingMixin, TeacherWindowChainMixin,
                                        MiniMaxH3VideoRefStreamingRF, MiniMaxH3StreamingDistill):
    """Streaming distillation over the student's chain-mode resampled history."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if self.resample_mode != "chain":
            raise ValueError("streaming distillation currently requires resample_mode chain")
        self.teacher_cfg_fitting = bool(config.meta_model.get("teacher_cfg_fitting", False))
        if self.teacher_cfg_fitting and self.cfg_fitting_guidance is None:
            raise ValueError("teacher_cfg_fitting reads the teacher through its trained combination, "
                             "so it requires meta_model.cfg_fitting_guidance")

    def _teacher_target(
        self, ctx: dict[str, Any], inputs: StreamingInputs, kwargs: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The teacher's raw forward, or with ``teacher_cfg_fitting`` its full-condition fit on its own window."""
        if not self.teacher_cfg_fitting:
            return super()._teacher_target(ctx, inputs, kwargs)
        return self._guidance_fit(
            ctx["models"]["tea_model"], inputs, self._guidance_available(inputs, ctx["inputs"]),
            lambda branch: self._teacher_branch_inputs(inputs, ctx["inputs"], branch), **kwargs,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def student_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads the parent's keys plus chain and rng. Writes student_pred, resample_prediction and chain_metrics.

        The resample forward runs before the student's training forward and
        draws only forked RNGs.
        """
        prediction = self._resample_window(ctx)
        ctx["resample_prediction"] = prediction
        ctx["chain_metrics"] = self._resample_metrics(ctx, prediction)
        return super().student_forward(ctx)

    def _teacher_inputs(
        self, ctx: dict[str, Any],
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Build the teacher's own window from the payload's plans and the working copies.

        History reads the same working copies as the student at the teacher's
        indices and is anchored from a fork of each sample stream. Retained
        audio stays clean, noisy rows are the student's x_t, and the picture
        is the student's anchored latent.
        """
        if self.teacher_streaming is None:
            return super()._teacher_inputs(ctx)
        return self._teacher_window(ctx, ctx["plans"], ctx["sample_rngs"],
                                    self._noisy_rows(ctx["noisy_latents"], ctx["plans"]))

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Write the reconstruction back after both networks have read this window."""
        ctx = super().compute_loss(ctx)
        self._commit_resampled_window(ctx, ctx["resample_prediction"])
        return ctx


EntryClass = MiniMaxH3VideoRefStreamingDistill

__all__ = ["MiniMaxH3VideoRefStreamingDistill", "EntryClass"]
