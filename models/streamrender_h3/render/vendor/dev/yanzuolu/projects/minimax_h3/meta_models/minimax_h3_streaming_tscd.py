# SPDX-License-Identifier: Apache-2.0
"""Trajectory-segmented consistency distillation within streaming SFT windows."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch

from dev.yanzuolu.common.diffusion.sampler.ddim import DDIMSampler
from dev.yanzuolu.common.diffusion.schedule import PredictionType
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import CausalMiniMaxH3Base
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import MiniMaxH3StreamingSFT
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs


class MiniMaxH3StreamingTSCD(MiniMaxH3StreamingSFT):
    """Distill streaming students using a frozen streaming teacher and an EMA target.

    All three networks read the same selected window and prepared history,
    unless ``_solver_window`` gives the teacher its own layout around the
    same noisy rows. The teacher advances only noisy rows to the adjacent
    positive grid point. The EMA reads that updated window, and only the
    student retains gradients.
    ``loss_type`` selects direct x0 comparison or deterministic projection to
    a shared comparison time. Generation may use a different sampler.
    Training CFG fitting is independently enabled for each network. Their
    negative branch shares one prepared prefix and layout, while model hidden
    states remain private. Teacher fitting precedes optional external CFG.
    """

    def __init__(self, config: Any) -> None:
        if (config.meta_model.get("guidance_branches") is not None
                or config.meta_model.get("guidance_fitting") is not None):
            raise ValueError("streaming TSCD does not support guidance_branches or guidance_fitting")
        CausalMiniMaxH3Base.__init__(self, config)
        self._configure_streaming(config)
        options = config.meta_model
        self.num_segments = int(options.num_segments)
        self.loss_type = str(options.loss_type)
        assert self.loss_type in {"x_0", "x_t"}, "loss_type must be x_0 or x_t"
        assert float(options.get("guidance_scale", 1.0)) == 1.0, (
            "streaming TSCD CFG fitting uses per-network switches and cfg_fitting_scale, not guidance_scale"
        )
        assert float(options.get("negative_loss_weight", 0.0)) == 0.0, (
            "streaming TSCD has no captionless calibration loss, so negative_loss_weight must stay 0"
        )
        assert options.get("loss_prediction_type", "x0") == "x0", (
            "streaming TSCD uses loss_type x_0 or x_t, not SFT flow_v fitting"
        )
        self.student_cfg_fitting = bool(options.get("student_cfg_fitting", False))
        self.ema_cfg_fitting = bool(options.get("ema_cfg_fitting", False))
        self.teacher_cfg_fitting = bool(options.get("teacher_cfg_fitting", False))
        self.cfg_fitting_scale = float(options.get("cfg_fitting_scale", 3.0))
        if not math.isfinite(self.cfg_fitting_scale) or self.cfg_fitting_scale <= 0:
            raise ValueError("cfg_fitting_scale must be finite and positive")
        guidance = options.get("teacher_guidance_scale", [1.0, 1.0])
        if isinstance(guidance, (int, float)):
            guidance = [guidance, guidance]
        lower, upper = map(float, guidance)
        assert math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper, (
            "teacher_guidance_scale must be a finite nonnegative scalar or ordered pair"
        )
        self.teacher_guidance_scale = (lower, upper)
        self.tea_schedule = self._diffusion["tea_schedule"]
        self.tea_sampler = self._diffusion["tea_sampler"]
        assert self.tea_schedule.pred_type == PredictionType.x_0, "tea_schedule.pred_type must be x_0"
        assert isinstance(self.tea_sampler, DDIMSampler), "tea_sampler must supply a deterministic DDIM step"
        self._projection_sampler = DDIMSampler(schedule=self.schedule)
        assert (
            not self.sampling_timesteps.dynamic_shift
            and not self.audio_sampling_timesteps.dynamic_shift
            and self.sampling_timesteps.num_sampling_steps == self.audio_sampling_timesteps.num_sampling_steps
        ), "video and audio TSCD grids must be static and share one step count"
        steps = self.sampling_timesteps.num_sampling_steps
        assert 1 <= self.num_segments <= steps, "num_segments must be between 1 and the TSCD grid length"
        assert steps % self.num_segments == 0, "num_segments must evenly divide the TSCD grid length"
        assert (self.sampling_timesteps.sampling_skip_max or 0) >= 1, (
            "sampling_timesteps.sampling_skip_max must be >= 1 to keep the adjacent EMA timestep on-grid"
        )
        minimum = self.sampling_timesteps.sampling_skip_min or 0
        maximum = steps - self.sampling_timesteps.sampling_skip_max
        assert 0 <= minimum < maximum <= steps - 1, "TSCD start-index bounds must leave a positive adjacent step"

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_segment(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Write window plans and paired [B] start, adjacent, boundary and comparison times."""
        inputs, rng = ctx["inputs"], ctx["rng"]
        sample_rngs = ctx.get("sample_rngs")
        if sample_rngs is None:
            sample_rngs = [rng.fork("streaming_sample", index) for index in range(inputs.batch_size)]
        assert len(sample_rngs) == inputs.batch_size
        plans = self._prepared_streaming_plans(ctx)
        if plans is None:
            plans = [self._sample_streaming_plan(inputs, index, sample_rng)
                     for index, sample_rng in enumerate(sample_rngs)]
        if "encoded_batch" in ctx:
            assert all(plan["packing_rows"] <= int(budget) for plan, budget in zip(
                plans, ctx["encoded_batch"]["packing_rows"], strict=True
            )), "selected streaming window exceeds the dataset packing budget"
        steps = self.sampling_timesteps.num_sampling_steps
        minimum = self.sampling_timesteps.sampling_skip_min or 0
        maximum = steps - self.sampling_timesteps.sampling_skip_max
        with local_seed(rng.seed % 2**31):
            indices = torch.randint(minimum, maximum, (inputs.batch_size,), device=get_device())
        rng.seed = yield_seed(rng.seed)
        segment_length = steps // self.num_segments
        boundary_indices = (indices // segment_length + 1) * segment_length
        ratio = 1.0 - torch.rand((inputs.batch_size,), device=get_device(), generator=self._generator(rng))
        starts, adjacent, boundaries, comparisons = [], [], [], []
        for grid in (self.sampling_timesteps, self.audio_sampling_timesteps):
            start = grid.get_timesteps_by_index(indices)
            step = grid.get_timesteps_by_index(indices + 1)
            boundary = grid.get_timesteps_by_index(boundary_indices)
            assert bool((step > 0).all()), "the EMA must receive positive adjacent timesteps"
            starts.append(start.float())
            adjacent.append(step.float())
            boundaries.append(boundary.float())
            comparisons.append((boundary + ratio * (step - boundary)).float())
        ctx["plans"], ctx["sample_rngs"] = plans, list(sample_rngs)
        ctx["start_timesteps"] = tuple(starts)
        ctx["step_timesteps"] = tuple(adjacent)
        ctx["boundary_timesteps"] = tuple(boundaries)
        ctx["target_timesteps"] = tuple(comparisons)
        ctx["train_timesteps"] = ctx["start_timesteps"]
        return ctx

    def _cfg_fitting_forward(
        self, ctx: dict[str, Any], model_name: str, *, xts: Any, timesteps: Any, enabled: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply optional fitting with gradients only through the positive prediction."""
        return self._fitted_prediction(
            ctx["models"][model_name], ctx["window_inputs"], lambda: self._negative_window_inputs(ctx),
            scale=self.cfg_fitting_scale if enabled else 1.0,
            video_xts=xts[0], audio_xts=xts[1], video_timesteps=timesteps[0], audio_timesteps=timesteps[1],
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def student_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Write noisy-row student predictions with the streaming forward graph attached."""
        ctx["student_pred"] = self._cfg_fitting_forward(
            ctx, "backbone", xts=ctx["noisy_latents"], timesteps=ctx["start_timesteps"],
            enabled=self.student_cfg_fitting,
        )
        return ctx

    def _solver_window(
        self, ctx: dict[str, Any],
    ) -> tuple[StreamingInputs, Callable[[], StreamingInputs], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Return the teacher's window, its negative builder and its complete noisy windows.

        The teacher reads the student's window. A host may give it another
        layout whose noisy rows equal the student's in the same order.
        """
        return ctx["window_inputs"], lambda: self._negative_window_inputs(ctx), ctx["noisy_latents"]

    def _teacher_prediction(
        self, ctx: dict[str, Any], inputs: StreamingInputs, negative: Callable[[], StreamingInputs],
        kwargs: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The teacher's x0 on its window: optional fitting, then optional external CFG."""
        return self._guided_prediction(
            ctx["models"]["tea_model"], inputs, negative,
            guidance=self.teacher_guidance_scale,
            fitting_scale=self.cfg_fitting_scale if self.teacher_cfg_fitting else 1.0,
            rng=ctx["rng"], **kwargs,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    @torch.no_grad()
    def solver_step(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Advance noisy rows to the adjacent grid point and write complete solver windows."""
        inputs, negative, xts = self._solver_window(ctx)
        kwargs = dict(video_xts=xts[0], audio_xts=xts[1],
                      video_timesteps=ctx["start_timesteps"][0], audio_timesteps=ctx["start_timesteps"][1])
        predictions = self._teacher_prediction(ctx, inputs, negative, kwargs)
        noisy_rows = self._noisy_rows(ctx["noisy_latents"], ctx["plans"])
        solver = ([], [])
        for modality, (preds, values) in enumerate(zip(predictions, noisy_rows, strict=True)):
            stepped = self.tea_sampler.step_to(
                pred=preds, x_t=values, t=ctx["start_timesteps"][modality], s=ctx["step_timesteps"][modality],
            )
            for full, new, plan in zip(ctx["noisy_latents"][modality], stepped, ctx["plans"], strict=True):
                updated = full.clone()
                mask = plan["audio_noisy_mask" if modality else "video_noisy_mask"]
                updated.index_copy_(2 if modality else 1, mask.nonzero().flatten().to(updated.device), new)
                solver[modality].append(updated)
        ctx["solver_xts"] = solver
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    @torch.no_grad()
    def target_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Write EMA predictions at the adjacent point using the same prepared history."""
        ctx["target_pred"] = self._cfg_fitting_forward(
            ctx, "backbone_ema", xts=ctx["solver_xts"], timesteps=ctx["step_timesteps"],
            enabled=self.ema_cfg_fitting,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Compare noisy-row predictions at x0 or a shared deterministic comparison point."""
        student, target = ctx["student_pred"], ctx["target_pred"]
        if self.loss_type == "x_t":
            current = self._noisy_rows(ctx["noisy_latents"], ctx["plans"])
            adjacent = self._noisy_rows(ctx["solver_xts"], ctx["plans"])
            student = tuple(self._projection_sampler.step_to(
                pred=preds, x_t=values, t=t, s=s,
            ) for preds, values, t, s in zip(
                student, current, ctx["start_timesteps"], ctx["target_timesteps"], strict=True
            ))
            with torch.no_grad():
                target = tuple(self._projection_sampler.step_to(
                    pred=preds, x_t=values, t=t, s=s,
                ) for preds, values, t, s in zip(
                    target, adjacent, ctx["step_timesteps"], ctx["target_timesteps"], strict=True
                ))
        return self._compute_window_loss(ctx, student, target)


EntryClass = MiniMaxH3StreamingTSCD

__all__ = ["MiniMaxH3StreamingTSCD", "EntryClass"]
