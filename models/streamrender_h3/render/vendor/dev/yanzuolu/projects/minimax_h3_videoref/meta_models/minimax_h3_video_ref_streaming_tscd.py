# SPDX-License-Identifier: Apache-2.0
"""Trajectory-segmented consistency distillation over resampling-forcing chains.

``MiniMaxH3StreamingTSCD`` distills a streaming student inside windows whose
history is corpus latents. This meta trains the same objective over the
chains of ``MiniMaxH3VideoRefStreamingRF``. With the default
``resample_mode: chain``, a pack's windows run in order,
one per engine iteration, and every window's history rows are the student's
one-step x0 reconstructions of the earlier windows. The reconstruction is the
RF write-back unchanged, a no-grad ``eval()`` student forward at the clip's
``t_s`` from ``resample_timesteps``, so the history distribution does not
depend on which trajectory segment the window distills. The TSCD grid index
is drawn per window as in the parent.

In chain mode the resample forward runs first, then the student forward at
the grid time t, the frozen teacher forward and its one DDIM step to t', and
the EMA forward at t'. The write-back follows the loss, so every network reads
the same history in one iteration, and the next window reads the new values.
Ranks at different chain phases issue the same collectives.

``resample_mode: history_only`` instead reconstructs this step's history
from GT before the student keeps its training graph. Current target AV rows
are excluded from every resample pass. ``history_resample_source: student``
uses the current student on its own window for student and EMA history. If
the teacher's window policy differs, a second student pass reconstructs that
window's history with the student's conditions and picture mode. Equal
window policies reuse the first reconstruction, even when teacher conditions
differ. ``history_resample_source: per_model`` runs student, EMA and teacher
separately on their own windows and conditions. The three passes use
independent noise streams and share only the clip's existing resample time.
Every configured pass executes even for an empty bootstrap, so collective
counts do not depend on the rank's chain phase. The teacher advances the same
current noisy rows as before; the EMA replaces only history around those
advanced rows. Reconstructed history is local to this step and never written
to the GT chain pool. These are local reconstruction distributions, not
replays of sequential generation.

The student's window policy comes from ``data.args`` as usual. Optional
``meta_model.teacher_streaming`` gives the teacher its own window at every
student start, as ``TeacherWindowChainMixin`` in ``streaming_teacher_window``
plans and builds it: history from the working copies at the teacher's
indices, the student's noisy rows unchanged, the teacher's reference rows and
fixed-RoPE template, and the student's picture placed by
``teacher_picture_mode``. Without ``teacher_streaming`` the teacher reads the
student's window exactly like the parent.

``meta_model.cfg_fitting_guidance`` replaces the single captionless edge of
every enabled ``*_cfg_fitting`` switch with the network's trained sequential
guidance combination, as ``GuidedDistillationFittingMixin`` in
``streaming_guided_fitting`` describes: the student and its EMA on the
student's window, and the teacher on its own window without external CFG.

Expected configuration additions::

    entry:
      module: dev.yanzuolu.engines.tscd
      class_name: TrajectorySegmentedConsistencyDistillation
    data:
      args: {sink_size: 1, window_size: 3, chunk_size: 2, bootstrap_size: 7}
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_tscd
      class_name: MiniMaxH3VideoRefStreamingTSCD
      resample_mode: chain
      history_resample_source: student
      num_segments: 4
      loss_type: x_t
      teacher_streaming: {sink_size: 4, window_size: 5, chunk_size: 2}
      resample_timesteps: {...}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_tscd import MiniMaxH3StreamingTSCD
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import MiniMaxH3VideoRefStreamingRF
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_guided_fitting import GuidedDistillationFittingMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_teacher_window import TeacherWindowChainMixin

class MiniMaxH3VideoRefStreamingTSCD(GuidedDistillationFittingMixin, TeacherWindowChainMixin, MiniMaxH3VideoRefStreamingRF,
                                     MiniMaxH3StreamingTSCD):
    """Streaming TSCD with chain history or per-step, role-specific history reconstruction."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.history_resample_source = str(config.meta_model.get("history_resample_source", "student"))
        if self.history_resample_source not in {"student", "per_model"}:
            raise ValueError("meta_model.history_resample_source must be student or per_model")
        self._share_teacher_history = (
            self.teacher_streaming is None
            or self._teacher_policy(self.streaming_config) == self.streaming_config
        )
        if (self.cfg_fitting_guidance is not None and self.teacher_cfg_fitting
                and self.teacher_guidance_scale != (1.0, 1.0)):
            raise ValueError("a guided teacher fit carries its trained guidance, so teacher_guidance_scale must stay 1")

    def _cfg_fitting_forward(
        self, ctx: dict[str, Any], model_name: str, *, xts: Any, timesteps: Any, enabled: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fit the student or the EMA through its guided combination on the student's window."""
        if not enabled or self.cfg_fitting_guidance is None:
            return super()._cfg_fitting_forward(ctx, model_name, xts=xts, timesteps=timesteps, enabled=enabled)
        view = dict(ctx, noisy_latents=xts)
        return self._guidance_fit(
            ctx["models"][model_name], ctx["window_inputs"], self._guidance_available(ctx["window_inputs"], ctx["inputs"]),
            lambda branch: self._student_branch_inputs(view, branch),
            video_xts=xts[0], audio_xts=xts[1], video_timesteps=timesteps[0], audio_timesteps=timesteps[1],
        )

    def _teacher_prediction(
        self, ctx: dict[str, Any], inputs: StreamingInputs, negative: Callable[[], StreamingInputs],
        kwargs: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fit the teacher through its guided combination on its own window."""
        if not self.teacher_cfg_fitting or self.cfg_fitting_guidance is None:
            return super()._teacher_prediction(ctx, inputs, negative, kwargs)
        return self._guidance_fit(
            ctx["models"]["tea_model"], inputs, self._guidance_available(inputs, ctx["inputs"]),
            lambda branch: self._teacher_branch_inputs(inputs, ctx["inputs"], branch), **kwargs,
        )

    # ----------------------------------------------------------- training --
    def _student_resample_plans_for_teacher(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Use the teacher's window policy with the student's conditions and picture mode."""
        inputs = replace(ctx["inputs"], streaming_configs=[
            self._teacher_policy(policy) for policy in ctx["inputs"].streaming_configs
        ])
        plans = []
        for index, original in enumerate(ctx["plans"]):
            plan = self._plan(inputs, index, original["start"])
            if original.get("picture_shape") is not None:
                plan = self._with_picture(plan, original["picture_clean_latents"], keyframe=self.picture_mode == "keyframe")
            plans.append(plan)
        return plans

    @torch.no_grad()
    def _prepare_history_roles(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Build each role from the same immutable GT before the student keeps a graph."""
        student = self._resample_history(ctx, ctx["plans"], model_name="backbone", role="student")
        teacher_plans = ctx["encoded_batch"].get("teacher_plans", ctx["plans"])
        roles = {"student": student["clean_latents"]}
        if self.history_resample_source == "per_model":
            roles["ema"] = self._resample_history(
                ctx, ctx["plans"], model_name="backbone_ema", role="ema",
            )["clean_latents"]
            roles["teacher"] = self._resample_history(
                ctx, teacher_plans, model_name="tea_model", role="teacher", teacher_conditions=True,
            )["clean_latents"]
        else:
            roles["ema"] = roles["student"]
            if self._share_teacher_history:
                for mine, teacher in zip(ctx["plans"], teacher_plans, strict=True):
                    for prefix in ("video", "audio"):
                        left = mine[f"{prefix}_indices"][~mine[f"{prefix}_noisy_mask"]].cpu()
                        right = teacher[f"{prefix}_indices"][~teacher[f"{prefix}_noisy_mask"]].cpu()
                        assert torch.equal(left, right), "shared history requires equal student and teacher history indices"
                roles["teacher"] = roles["student"]
            else:
                roles["teacher"] = self._resample_history(
                    ctx, self._student_resample_plans_for_teacher(ctx), model_name="backbone", role="teacher",
                )["clean_latents"]
        ctx.update(student)
        ctx["history_resample_clean"] = roles
        ctx["noisy_latents"] = self._history_conditioned_values(ctx, roles["student"], role="student")
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def student_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads the parent's keys plus rng. Writes student_pred, resample_prediction and chain_metrics.

        Resampling precedes the student's training forward and uses forked
        RNGs. History-only mode builds the configured role histories from GT;
        chain mode prepares the current chunk's later write-back.
        """
        ctx = (self._prepare_history_roles(ctx) if self.resample_mode == "history_only"
               else self._prepare_resampling(ctx))
        return super().student_forward(ctx)

    def _solver_window(
        self, ctx: dict[str, Any],
    ) -> tuple[StreamingInputs, Callable[[], StreamingInputs], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Build the teacher's own window from the payload's plans and the synchronized working copies.

        History comes from the chain or this step's teacher-role reconstruction
        and is anchored from a fork of each sample stream. Retained audio
        stays clean and current noisy rows remain the student's x_t.
        """
        teacher_ctx = ctx
        if self.resample_mode == "history_only":
            clean = ctx["history_resample_clean"]["teacher"]
            teacher_ctx = dict(ctx, clean_latents=clean)
            if clean is not ctx["history_resample_clean"]["student"]:
                teacher_ctx["noisy_latents"] = self._history_conditioned_values(ctx, clean, role="teacher")
        if self.teacher_streaming is None:
            return super()._solver_window(teacher_ctx)
        window, xts = self._teacher_window(teacher_ctx, ctx["plans"], ctx["sample_rngs"],
                                           self._noisy_rows(ctx["noisy_latents"], ctx["plans"]))
        inputs = ctx["inputs"]
        view = {"window_inputs": window, "inputs": inputs}
        return window, lambda: self._negative_window_inputs(view), xts

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    @torch.no_grad()
    def target_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Replace only the EMA's history, retaining the solver's current noisy rows."""
        if self.resample_mode != "history_only" or self.history_resample_source == "student":
            return super().target_forward(ctx)
        view = dict(ctx, solver_xts=self._history_conditioned_values(
            ctx, ctx["history_resample_clean"]["ema"], role="ema", values=ctx["solver_xts"],
        ))
        view = super().target_forward(view)
        ctx["target_pred"] = view["target_pred"]
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Advance after all roles read the window, writing predictions only in chain mode."""
        ctx = super().compute_loss(ctx)
        self._commit_resampled_window(ctx, ctx["resample_prediction"])
        return ctx

    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("streaming TSCD trains through student_forward, solver_step and target_forward")


EntryClass = MiniMaxH3VideoRefStreamingTSCD

__all__ = ["MiniMaxH3VideoRefStreamingTSCD", "EntryClass"]
