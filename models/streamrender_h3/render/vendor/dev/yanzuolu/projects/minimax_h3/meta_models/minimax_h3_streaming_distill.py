# SPDX-License-Identifier: Apache-2.0
"""Distillation of a streaming student's prediction onto a frozen streaming teacher's."""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import MiniMaxH3StreamingSFT
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs

_Pair = tuple[list[torch.Tensor], list[torch.Tensor]]


class MiniMaxH3StreamingDistill(MiniMaxH3StreamingSFT):
    """Regress the student's noisy-row prediction onto a frozen teacher's on the same window.

    Windows, timesteps over the full ``training_timesteps`` range, noise,
    history and validation are the SFT's. The student ``backbone`` and the
    teacher ``tea_model`` read the same noisy rows at the same timesteps over
    the same history. ``_teacher_inputs`` gives the teacher the student's
    window, and a host may give it another layout whose noisy rows equal the
    student's in the same order. ``_student_prediction`` and
    ``_teacher_target`` are one raw forward each. A host distilling a
    combination of forwards, such as a CFG-guided or CFG-fitted teacher,
    overrides them.

    The loss is the SFT's window loss of the student's prediction against the
    teacher's, both read in ``loss_prediction_type``: noisy rows only, with the
    bootstrap, first-frame, continuation and audio weights. The SFT's own
    guidance objectives do not apply, so ``guidance_branches``,
    ``guidance_fitting``, a ``guidance_scale`` other than 1 and a positive
    ``negative_loss_weight`` are rejected.
    """

    def __init__(self, config: Any) -> None:
        options = config.meta_model
        if (options.get("guidance_branches") is not None or options.get("guidance_fitting") is not None
                or float(options.get("guidance_scale", 1.0)) != 1.0
                or float(options.get("negative_loss_weight", 0.0)) != 0.0):
            raise ValueError("streaming distillation regresses onto the teacher: remove guidance_branches, "
                             "guidance_fitting, guidance_scale and negative_loss_weight")
        super().__init__(config)

    @staticmethod
    def _forward_kwargs(xts: _Pair, timesteps: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        return dict(video_xts=xts[0], audio_xts=xts[1], video_timesteps=timesteps[0], audio_timesteps=timesteps[1])

    def _student_prediction(self, ctx: dict[str, Any], inputs: StreamingInputs, kwargs: dict[str, Any]) -> _Pair:
        """The student's noisy-row prediction on ``inputs``, one raw forward with its graph attached."""
        return self._streaming_forward(ctx["models"]["backbone"], inputs, **kwargs)

    def _teacher_inputs(self, ctx: dict[str, Any]) -> tuple[StreamingInputs, _Pair]:
        """The teacher's window and its complete noisy windows, the student's by default."""
        return ctx["window_inputs"], ctx["noisy_latents"]

    def _teacher_target(self, ctx: dict[str, Any], inputs: StreamingInputs, kwargs: dict[str, Any]) -> _Pair:
        """The teacher's noisy-row prediction on its window ``inputs``, one raw forward."""
        return self._streaming_forward(ctx["models"]["tea_model"], inputs, **kwargs)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def student_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads models, window_inputs, noisy_latents and train_timesteps. Writes student_pred."""
        ctx["student_pred"] = self._student_prediction(
            ctx, ctx["window_inputs"], self._forward_kwargs(ctx["noisy_latents"], ctx["train_timesteps"]),
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    @torch.no_grad()
    def teacher_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads the student's keys through ``_teacher_inputs``. Writes teacher_pred at the student's timesteps."""
        inputs, xts = self._teacher_inputs(ctx)
        ctx["teacher_pred"] = self._teacher_target(ctx, inputs, self._forward_kwargs(xts, ctx["train_timesteps"]))
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads student_pred, teacher_pred and the window keys. Writes loss and metrics."""
        student, teacher = (self._supervised_pairs(ctx, ctx[key])[0] for key in ("student_pred", "teacher_pred"))
        return self._compute_window_loss(ctx, student, teacher)


EntryClass = MiniMaxH3StreamingDistill

__all__ = ["MiniMaxH3StreamingDistill", "EntryClass"]
