# SPDX-License-Identifier: Apache-2.0
"""Sequential guided CFG fitting for the networks of streaming distillation.

A network trained with ``guidance_branches`` and ``guidance_fitting`` does
not predict the data mean with its conditional output alone. Its trained
prediction combines the conditional output with its reduced-condition
branches, and distillation must read every network through the same
combination it was trained with. ``meta_model.cfg_fitting_guidance`` names
that combination with the SFT keys::

    cfg_fitting_guidance:
      anchors: guided
      branches:
        - {name: "null", drop: [text, reference]}
        - {name: text_only, drop: [reference]}
      stages:
        - {branch: "null", scale: 1.5}
        - {branch: text_only, scale: 3.0}

With it, every network whose ``*_cfg_fitting`` switch is on reads
``(P - sum(c_i * stopgrad(N_i))) / s``, the full-condition fit of
``compile_guidance`` for each sample's available conditions, in place of the
single captionless edge ``(P + (s - 1) U) / s`` at ``cfg_fitting_scale``. For
the chain ``0 --1.5--> T --3.0--> TS`` with guided anchors that is
``(f(TS) + f(T) + f(0)) / 3``. Gradients reach only the positive prediction,
and every branch forward runs without gradients. ``history`` is never
dropped. Branches carry no supervision unless the host trains one network's
branches, as ``fake_branch_loss`` of the videoref DMD does. Only then may a
branch set ``loss_weight``, and that network's supervised branches run with
gradients and carry the SFT's branch fits of ``compile_guidance``.

``_guidance_state_fit`` reads a network at a reduced state on the chain
instead: the fit ``compile_guidance`` gives a branch supervised at that
state, the same fit the SFT trains that branch with. For ``text_only`` above
that is ``(2 f(T) + f(0)) / 3``, with gradients only through ``f(T)``.

Streaming sampling reads the full-condition fit as weighted branches, since
it is linear in the branch predictions: ``_guided_sampling_branches`` gives
the full batch weight ``1 / s`` and each branch batch ``-c_i / s``, 1/3 each
on the chain above. Every branch denoises the same noisy rows over the same
history and reads its own conditions.

Each network drops conditions on its own layout. On a Ref2VA window
``reference`` removes the reference rows and ``text`` the caption, and the
picture with its Qwen context stays in every branch that does not drop
``picture``, as ``reference_drop_keeps_picture`` trains it. ``picture``
removes the picture rows and Picture 1, so Qwen reads the caption's text-only
encoding, or nothing without the caption. ``_teacher_branch_inputs`` builds
these for teacher and score windows and ``_student_branch_inputs`` for the
student's, where a pixel-conditioned host instead drops its condition video
for ``reference`` and keeps its keyframe, which no branch may drop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.meta_models.streaming_guidance import (
    CompiledStreamingGuidance,
    StreamingGuidanceBranch,
    compile_guidance,
    parse_guidance_branches,
    parse_guidance_fitting,
    possible_guidance_uses,
)
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs

_Pair = tuple[list[torch.Tensor], list[torch.Tensor]]


class GuidedDistillationFittingMixin:
    """Read fitted networks through their trained sequential guidance combination."""

    cfg_fitting_guidance: tuple[tuple[StreamingGuidanceBranch, ...], tuple[Any, ...], str] | None = None

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        options = config.meta_model.get("cfg_fitting_guidance")
        if options is None:
            return
        if set(options) - {"branches", "stages", "anchors"} or "branches" not in options or "stages" not in options:
            raise ValueError("cfg_fitting_guidance requires branches and stages and accepts only anchors besides them")
        branches = parse_guidance_branches({"guidance_branches": options["branches"]}, self.guidance_conditions)
        fitting = {"stages": options["stages"], "anchors": options.get("anchors", "posterior")}
        stages, anchors = parse_guidance_fitting({"guidance_fitting": fitting}, branches)
        if not stages:
            raise ValueError("cfg_fitting_guidance requires at least one stage")
        if any(branch.loss_weight for branch in branches) and not self._supervises_guidance_branches(config):
            raise ValueError("distillation fits carry no branch supervision here, so cfg_fitting_guidance loss_weight must be 0")
        if any("history" in branch.drop for branch in branches):
            raise ValueError("distillation windows keep their history in every cfg_fitting_guidance branch")
        self.cfg_fitting_guidance = (branches, stages, anchors)
        self._cfg_fitting_uses = possible_guidance_uses(branches, stages, anchors=anchors,
                                                        conditions=self.guidance_conditions)

    def _supervises_guidance_branches(self, config: Any) -> bool:
        """Whether a network of this host trains its branches with their ``loss_weight``."""
        return False

    def _student_branch_inputs(
        self, view: dict[str, Any], branch: StreamingGuidanceBranch,
    ) -> StreamingInputs:
        """One reduced-condition student window. ``view`` holds window_inputs, inputs and noisy_latents."""
        return self._picture_keeping_branch_inputs(view["window_inputs"], view["inputs"], branch.drop)

    def _teacher_branch_inputs(
        self, window: StreamingInputs, source: Any, branch: StreamingGuidanceBranch,
    ) -> StreamingInputs:
        """One reduced-condition Ref2VA window that keeps its picture."""
        return self._picture_keeping_branch_inputs(window, source, branch.drop)

    def _guidance_available(self, window: StreamingInputs, source: Any) -> tuple[frozenset[str], ...]:
        return self._guidance_available_conditions({"window_inputs": window, "inputs": source})

    def _guided_sampling_branches(
        self, inputs: Any, available: Sequence[frozenset[str]], branch_batch: Callable[[StreamingGuidanceBranch], Any],
    ) -> list[tuple[float, Any]]:
        """Weighted sampling batches whose sum is the full-condition fit of ``inputs``.

        Sampling weighs every sample of a batch alike, so every sample must
        compile the same fit.
        """
        branches, stages, anchors = self.cfg_fitting_guidance
        compiled = {compile_guidance(branches, stages, conditions, anchors=anchors) for conditions in available}
        if len(compiled) != 1:
            raise ValueError("guided sampling requires every sample of a batch to hold the same conditions")
        item = compiled.pop()
        return [(1.0 / item.full_scale, inputs)] + [
            (-coefficient / item.full_scale, branch_batch(branch))
            for branch, coefficient in zip(branches, item.coefficients, strict=True) if coefficient != 0.0
        ]

    def _guidance_predictions(
        self, model: Any, window: StreamingInputs, available: Sequence[frozenset[str]],
        branch_inputs: Callable[[StreamingGuidanceBranch], StreamingInputs], *, supervised: bool = False, **kwargs: Any,
    ) -> tuple[list[CompiledStreamingGuidance], _Pair, dict[int, _Pair]]:
        """Each sample's compiled guidance, the positive prediction and every useful branch's prediction by index.

        Every branch any availability may use runs on every rank, so ranks
        issue the same forwards whatever their samples hold. A branch runs
        with gradients only when ``supervised`` and its ``loss_weight`` is
        positive.
        """
        branches, stages, anchors = self.cfg_fitting_guidance
        compiled = [compile_guidance(branches, stages, conditions, anchors=anchors) for conditions in available]
        prediction = self._streaming_forward(model, window, **kwargs)
        anchored = {}
        for index, branch in enumerate(branches):
            if self._cfg_fitting_uses[index]:
                with torch.set_grad_enabled(torch.is_grad_enabled() and supervised and branch.loss_weight > 0.0):
                    anchored[index] = self._streaming_forward(model, branch_inputs(branch), **kwargs)
        return compiled, prediction, anchored

    @staticmethod
    def _guided_combination(
        prediction: _Pair, anchored: dict[int, _Pair], fits: Sequence[tuple[float, tuple[float, ...]]],
    ) -> _Pair:
        """``(P - sum(c_i * stopgrad(N_i))) / s`` per sample, where ``fits`` holds each sample's s and c."""
        fitted: _Pair = ([], [])
        for modality, values in enumerate(prediction):
            for sample, (positive, (scale, coefficients)) in enumerate(zip(values, fits, strict=True)):
                value = positive
                for index, coefficient in enumerate(coefficients):
                    if coefficient != 0.0:
                        value = value - coefficient * anchored[index][modality][sample].detach()
                fitted[modality].append(value / scale if scale != 1.0 else value)
        return fitted

    def _state_guidance(self, state: str) -> tuple[int, tuple[StreamingGuidanceBranch, ...]]:
        """The ``state`` branch's index and the branches with only that one supervised."""
        branches = self.cfg_fitting_guidance[0]
        index = [branch.name for branch in branches].index(state)
        return index, tuple(replace(branch, loss_weight=float(position == index)) for position, branch in enumerate(branches))

    def _state_fit_uses(self, state: str) -> tuple[bool, ...]:
        """Branches the fit of ``state`` reads for any availability that keeps the state its own."""
        _, stages, anchors = self.cfg_fitting_guidance
        index, supervised = self._state_guidance(state)
        used = [False] * len(supervised)
        for bits in range(1 << len(self.guidance_conditions)):
            available = frozenset(name for bit, name in enumerate(self.guidance_conditions) if bits & (1 << bit))
            item = compile_guidance(supervised, stages, available, anchors=anchors)
            if item.loss_weights[index] > 0.0:
                used = [before or value != 0.0 for before, value in zip(used, item.branch_coefficients[index], strict=True)]
        return tuple(used)

    def _guidance_state_fit(
        self, model: Any, available: Sequence[frozenset[str]],
        branch_inputs: Callable[[StreamingGuidanceBranch], StreamingInputs], state: str, **kwargs: Any,
    ) -> _Pair:
        """The fit of the ``state`` branch's own state with gradients only through that branch's prediction.

        The coefficients are ``compile_guidance``'s for that branch supervised
        alone. Every sample must hold the conditions the branch drops, so the
        state is its own on every sample's chain. The lower states any
        availability may read run without gradients on every rank.
        """
        _, stages, anchors = self.cfg_fitting_guidance
        index, supervised = self._state_guidance(state)
        compiled = [compile_guidance(supervised, stages, conditions, anchors=anchors) for conditions in available]
        if not all(item.loss_weights[index] > 0.0 for item in compiled):
            raise ValueError(f"every sample must hold the conditions the {state!r} guidance state drops")
        prediction = self._streaming_forward(model, branch_inputs(supervised[index]), **kwargs)
        anchored = {index: prediction}
        with torch.no_grad():
            for position, (branch, used) in enumerate(zip(supervised, self._state_fit_uses(state), strict=True)):
                if used and position != index:
                    anchored[position] = self._streaming_forward(model, branch_inputs(branch), **kwargs)
        return self._guided_combination(prediction, anchored, [
            (item.branch_scales[index], item.branch_coefficients[index]) for item in compiled
        ])

    def _guidance_fit(
        self, model: Any, window: StreamingInputs, available: Sequence[frozenset[str]],
        branch_inputs: Callable[[StreamingGuidanceBranch], StreamingInputs], **kwargs: Any,
    ) -> _Pair:
        """The full-condition fit of ``window`` with gradients only through its positive prediction."""
        compiled, prediction, anchored = self._guidance_predictions(model, window, available, branch_inputs, **kwargs)
        return self._guided_combination(prediction, anchored, [(item.full_scale, item.coefficients) for item in compiled])


__all__ = ["GuidedDistillationFittingMixin"]
