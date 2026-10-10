# SPDX-License-Identifier: Apache-2.0
"""Distribution matching distillation over the videoref chain pool, scored on the teacher's window.

``MiniMaxH3StreamingDMD`` distills a few-step streaming generator over chains
of windows whose history is the generator's own output. This meta runs the
same primitives on paired-reference windows: reference rows, the picture
placed by ``picture_mode`` and the per-window Qwen context of
``MiniMaxH3VideoRefStreamingRF``, from the same chain pool. The pool keeps
every chain's working copies across checkpoints, so a resumed chain continues
at its window, and the meta sets ``data.args.resume_mid_chain``. With a
training UP size above 1 each rank still owns its own chains. The SP input
broadcast gives the group every source rank's window in turn, the group rolls
it out, trains FAKE and GEN on it, and only its source writes the rollout
back.

Every window's rollout is written back, so every window after a chain's
first reads the generator's own output as history. The parent's corpus
prefix does not apply: a nonzero ``gen_gt_ratio`` or a
``gt_prefix_window_range`` raises ``ValueError``.

The generator ``backbone`` rolls out, writes back and trains on the
student's window, the ``data.args`` policy. With ``meta_model.teacher_streaming``
the trainable ``fake_model`` and the frozen real score ``tea_model`` read the
teacher's own window at every student start, as ``TeacherWindowChainMixin``
plans and builds it: history from the chain's working copies at the
teacher's indices, the teacher's reference rows, fixed-RoPE template and
picture placement, and the generator's noisy rows. FAKE trains the fake score
on that window, and GEN scores the generator's x0 there, so both scores see
exactly the student's generated rows in the same order at the same noise and
timesteps. Every negative the scores need is that window's captionless
layout, which keeps the reference rows and the picture. The student's own
negative is built only when ``student_cfg_fitting`` needs it. Both windows
read one Qwen context because the teacher policy requires
``qwen_reference_video: false``.

``meta_model.cfg_fitting_guidance`` replaces the single captionless edge of
the enabled ``student_cfg_fitting`` and ``fake_cfg_fitting`` with the
network's trained sequential guidance combination, as
``GuidedDistillationFittingMixin`` in ``streaming_guided_fitting`` describes.
Each window then carries its reduced-condition branch windows instead of a
negative: the student's for the generator, the teacher layout's for the fake
score. The real score then runs raw, so ``teacher_cfg_fitting`` stays false
and ``teacher_guidance_scale`` 1. A guided ``student_cfg_fitting`` makes the
generator's prediction the fit at every rollout step, whose final x0 is
written back, and at GEN, and validation must sample through the same fit. The
pixel generator of ``MiniMaxH3VideoRefStreamingPixelDMD`` does, and this
meta rejects a guided student elsewhere.

``meta_model.fake_branch_loss``, off by default, trains the fake score as the
SFT trains a network with those ``guidance_branches`` and ``guidance_fitting``.
The fake score's loss then adds, to its full-condition fit's loss, each
branch's ``loss_weight`` times the loss of that branch's own fit from
``compile_guidance``, against the same target: the generator's sample at the
fake score's timestep. Each fit detaches every lower state it reads. With
``loss_weight`` 0.1 on ``null`` and 0.5 on ``text_only`` along
``0 --1.5--> T --3.0--> TS`` with guided anchors, the loss is
``L((f(TS) + f(T) + f(0)) / 3) + 0.5 L((2 f(T) + f(0)) / 3) + 0.1 L(f(0))``,
where each fit's gradient reaches only its own state's prediction. It
requires ``fake_cfg_fitting`` and a positive ``loss_weight`` in
``cfg_fitting_guidance``. The generator and the real score, and the fake
score's reading of the generator's sample, stay as without it.

``meta_model.fake_fitting_state`` names the ``cfg_fitting_guidance`` branch
whose state the fake score reads, both when FAKE trains it and when GEN
scores the generator's sample. Unset, the fake score reads the full-condition
fit. Set, it reads the fit ``compile_guidance`` gives that branch when the SFT
supervises it, on the score window's branch windows, with gradients only
through that branch's prediction. For ``text_only`` along
``0 --1.5--> T --3.0--> TS`` with guided anchors the fake score reads
``(2 f(T) + f(0)) / 3``: no reference rows, the picture and its Qwen context
kept, and the caption in ``f(T)`` only. The real score still reads its own
prediction, so the DMD direction compares the two states. It requires
``fake_cfg_fitting``, a branch on ``stages``, and ``fake_branch_loss`` off.

Every window is prepared once from its payload. The payload's
``window_seed``, derived from the chain's head seed and the window's dataset
index, seeds the anchored picture, reference and history rows of both
windows, so every rank of a UP group builds identical windows and a resumed
chain rebuilds them exactly.

``meta_model.history_refresh: true`` optionally denoises cached generated
history before each window's rollout. It defaults
to false. Like RF, it requires a separate ``meta_model.resample_timesteps``
plugin providing ``sample_pair``. Each clip draws one video/audio ``t_s``
pair at its chain head and reuses it across every window and both history
layouts. Each window and role receives fresh independent corruption noise.
One x0 prediction reconstructs each role's noised history. The paired times
are independent of the DMD rollout and score timesteps, and are reproduced
from the original head seed on resume. The sampler's ``T`` must match the
student schedule. Refresh requires ``engine.offline: 1`` and does not support
a per-step ``shift_schedule`` on its clip-level timestep sampler.

Only sink and retained recent AV rows enter refresh, including retained
audio lookahead. Their history-only layout keeps the window's RoPE template,
picture, text and matching reference rows. ``history_resample_source`` is
``student`` by default: different student and score policies execute separate
student passes with the student's own conditions and picture mode, and equal
policies share the reconstructed values. ``per_model`` makes the student,
fake score and real score each reconstruct its own history from the same old
generated values, with its own window, conditions and fitting or guidance.
The fake score uses ``fake_schedule`` for corruption; the student and real
score use ``schedule``. Both schedules must share the resample sampler's T.
All three passes use independent corruption noise even with equal windows.
Empty bootstrap windows still execute every configured pass. The detached values
replace history temporarily, preserving its original near-clean perturbation.
The current chunk is then generated and FAKE and GEN reuse these fixed inputs;
each score reads its corresponding refreshed history. The chain writes
back only the newly generated chunk, never the refreshed historical rows.
This local reconstruction conditions on the old generated history and does
not replay an entire clip with current weights. DMD2's real corpus samples
retain their original history.

``meta_model.score_history_refresh: true`` is a separate, default-off mode,
mutually exclusive with ``history_refresh``. GEN preparation, and FAKE
preparation when ``fake_history_refresh`` is off, use the frozen real teacher
to reconstruct only the score window's sink and retained recent AV from
Gaussian-corrupted generated history. It uses
the separate clip-level ``resample_timesteps`` sampler, the real teacher's
conditions and fitting/guidance, and fresh phase-local corruption noise.
Current target and reference rows are absent from this history-only RF
pass. Picture and text remain available; visual Qwen conditioning requires
``qwen_reference_video: false`` so its embeddings cannot carry current
reference frames into the reconstruction.

Both scores and all their branches share the phase's detached reconstruction,
preserving the score window's near-clean perturbation. The student rolls out
and replays GEN against its original history and fixed prefix. The canonical
rollout is never changed. FAKE and GEN reconstruct independently; each
selected full-rollout inner batch executes its own teacher pass. Empty
bootstrap histories still execute a collective-compatible forward. This
mode supports chain and full-rollout hosts, including fixed student prefixes.

``meta_model.fake_history_refresh: true`` independently reconstructs FAKE's
history with the fake model itself before the gradient-enabled training
forward. It defaults to false and is independent of ``fake_branch_loss``,
so either single-fit or three-fit supervision can use the original or
reconstructed history. The reconstruction runs in eval mode without gradients
and uses the fake model's effective prediction, including its configured
fitting or ``fake_fitting_state``. All supervised branches then share that
detached history while keeping the original student C, its noise, times and
targets. The sampler's T must match ``fake_schedule.T``.

With both phase-refresh switches enabled, FAKE performs only its own fake
reconstruction and GEN performs the real-teacher reconstruction. Both start
from the original student-generated history. Student rollout, GEN replay,
the fixed prefix and the canonical clip remain unchanged. The fake switch
shares the paired ``resample_timesteps`` and history-only layout rules above,
including bootstrap forwards and the Qwen restriction. It is mutually
exclusive with the rollout-time ``history_refresh`` mode.

Expected configuration additions::

    entry:
      module: dev.yanzuolu.engines.dmd
      class_name: DistributionMatchingDistillation
    data:
      module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_long_latent_chain
      class_name: VideoRefStreamingLongLatentChainDataset
      args: {sink_size: 2, window_size: 2, chunk_size: 2, bootstrap_size: 7}
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_dmd
      class_name: MiniMaxH3VideoRefStreamingDMD
      teacher_streaming: {sink_size: 4, window_size: 5}
      fake_cfg_fitting: true
      cfg_fitting_scale: 3.0
      qwen_reference_video: false

Optional history refresh::

    meta_model:
      history_refresh: true
      history_resample_source: student
      resample_timesteps:
        module: dev.yanzuolu.projects.minimax_h3.modeling.training_timesteps
        class_name: PairedLogitNormalTrainingTimesteps
        args: {T: 1.0, loc: 0.0, scale: 1.0, shift: 0.6, audio_shift: 0.6}
    engine:
      offline: 1

Alternatively, refresh only the shared score history::

    meta_model:
      score_history_refresh: true
      resample_timesteps:
        module: dev.yanzuolu.projects.minimax_h3.modeling.training_timesteps
        class_name: PairedLogitNormalTrainingTimesteps
        args: {T: 1.0, loc: 0.0, scale: 1.0, shift: 0.6, audio_shift: 0.6}

For fake-model reconstruction during FAKE, use ``fake_history_refresh: true``
with the same resample sampler configuration. ``score_history_refresh``
independently selects real-teacher reconstruction during GEN.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import get_unified_parallel_world_size, is_unified_parallel_initialized
from dev.yanzuolu.common.meter import get_running_average_meter
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import RandomState, combine_seed
from dev.yanzuolu.projects.minimax_h3.data.streaming import build_history_resample_plan
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_dmd import MiniMaxH3StreamingDMD, StreamingDMDInputs
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import (
    StreamingChainPoolMixin,
    VideoRefChainHostMixin,
    _ChainState,
    _build_resample_timesteps,
    _sample_chain_resample_timesteps,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import (
    MiniMaxH3VideoRefStreamingSFT,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_guided_fitting import GuidedDistillationFittingMixin
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_teacher_window import TeacherWindowChainMixin


@dataclass(frozen=True)
class VideoRefStreamingDMDInputs(StreamingDMDInputs):
    """A chain window with its pool entry and its guided branch windows.

    ``chain`` is the pool entry on the window's source rank and None on the
    rest of its UP group. ``branch_windows`` holds the reduced-condition
    windows of ``cfg_fitting_guidance`` by branch name, with each sample's
    available conditions in ``guidance_available``.
    ``history_refresh`` optionally caches an unperturbed history snapshot and
    the model's layout for reconstructing it when rollout consumes this window.
    ``real_score_window`` separates the real score's history in per-model mode.
    ``score_history_refresh`` holds raw history and the score-model layout
    used by the enabled FAKE and GEN history reconstructions.
    """

    chain: _ChainState | None = None
    branch_windows: dict[str, StreamingInputs] | None = None
    guidance_available: tuple[frozenset[str], ...] | None = None
    history_refresh: DMDHistoryRefreshInputs | None = None
    real_score_window: VideoRefStreamingDMDInputs | None = None
    score_history_refresh: DMDHistoryRefreshInputs | None = None


@dataclass(frozen=True)
class DMDHistoryRefreshInputs:
    """A history-only model window, unperturbed generated history and clip-level paired times."""

    window: VideoRefStreamingDMDInputs
    clean_x0s: tuple[list[torch.Tensor], list[torch.Tensor]]
    timesteps: tuple[torch.Tensor, torch.Tensor]


class MiniMaxH3VideoRefStreamingDMD(GuidedDistillationFittingMixin, TeacherWindowChainMixin, VideoRefChainHostMixin,
                                    StreamingChainPoolMixin, MiniMaxH3VideoRefStreamingSFT, MiniMaxH3StreamingDMD):
    """Streaming DMD with optional transient history refresh on student and score windows.

    The checkpointed chain always stores generated chunks. Refresh denoises
    their selected history at consumption time without overwriting the chain.
    """

    _samples_guided_student = False

    def __init__(self, config: Any) -> None:
        options = config.meta_model
        if options.get("gen_gt_ratio", 0.0) != 0.0 or options.get("gt_prefix_window_range") is not None:
            raise ValueError("videoref streaming DMD writes back every window; remove gen_gt_ratio and gt_prefix_window_range")
        super().__init__(config)
        if self.cfg_fitting_guidance is not None and (self.teacher_cfg_fitting or self.teacher_guidance_scale != (1.0, 1.0)):
            raise ValueError("with cfg_fitting_guidance the real score runs raw: set teacher_cfg_fitting false and "
                             "teacher_guidance_scale 1")
        self.fake_branch_loss = self._supervises_guidance_branches(config)
        if self.fake_branch_loss and not (self.fake_cfg_fitting and self.cfg_fitting_guidance is not None and any(
                branch.loss_weight > 0.0 for branch in self.cfg_fitting_guidance[0])):
            raise ValueError("fake_branch_loss requires fake_cfg_fitting and a cfg_fitting_guidance branch with a "
                             "positive loss_weight")
        if self.cfg_fitting_guidance is not None and self.student_cfg_fitting and not self._samples_guided_student:
            raise ValueError("a guided student_cfg_fitting needs a host whose validation samples through the fit, "
                             "such as MiniMaxH3VideoRefStreamingPixelDMD")
        self.fake_fitting_state = options.get("fake_fitting_state")
        if self.fake_fitting_state is not None:
            guidance = self.cfg_fitting_guidance
            if (guidance is None or not self.fake_cfg_fitting or self.fake_branch_loss
                    or self.fake_fitting_state not in {stage.branch for stage in guidance[1]}):
                raise ValueError("fake_fitting_state requires fake_cfg_fitting, a cfg_fitting_guidance branch on its stages "
                                 "and fake_branch_loss off")
        refresh = options.get("history_refresh", False)
        if refresh is not None and not isinstance(refresh, bool):
            raise ValueError("meta_model.history_refresh must be a boolean")
        self.history_refresh = bool(refresh)
        score_refresh = options.get("score_history_refresh", False)
        if not isinstance(score_refresh, bool):
            raise ValueError("meta_model.score_history_refresh must be a boolean")
        self.score_history_refresh = score_refresh
        fake_refresh = options.get("fake_history_refresh", False)
        if not isinstance(fake_refresh, bool):
            raise ValueError("meta_model.fake_history_refresh must be a boolean")
        self.fake_history_refresh = fake_refresh
        if self.history_refresh and self.score_history_refresh:
            raise ValueError("score_history_refresh and history_refresh are mutually exclusive")
        if self.history_refresh and self.fake_history_refresh:
            raise ValueError("fake_history_refresh and history_refresh are mutually exclusive")
        if (self.score_history_refresh or self.fake_history_refresh) and self.qwen_visual_context and self.qwen_reference_video:
            mode = "fake_history_refresh" if self.fake_history_refresh else "score_history_refresh"
            raise ValueError(f"{mode} requires qwen_reference_video false to exclude current reference "
                             "frames from the history-only Qwen context")
        self.history_resample_source = str(options.get("history_resample_source", "student"))
        if self.history_resample_source not in {"student", "per_model"}:
            raise ValueError("meta_model.history_resample_source must be student or per_model")
        if self.history_refresh or self.score_history_refresh or self.fake_history_refresh:
            self.resample_timesteps = _build_resample_timesteps(options)
            if (self.history_refresh or self.score_history_refresh) and float(self.resample_timesteps.T) != float(self.schedule.T):
                raise ValueError("resample_timesteps.T must match the student schedule.T")
            if self.fake_history_refresh and float(self.resample_timesteps.T) != float(self.fake_schedule.T):
                raise ValueError("fake_history_refresh requires resample_timesteps.T to match fake_schedule.T")
        if self.history_refresh:
            if self.history_resample_source == "per_model" and float(self.resample_timesteps.T) != float(self.fake_schedule.T):
                raise ValueError("per_model history refresh requires resample_timesteps.T to match fake_schedule.T")
            if int(config.get("engine", {}).get("offline", 1)) != 1:
                raise ValueError("history_refresh requires engine.offline 1")
        self._share_refreshed_score_history = (
            self.teacher_streaming is None or self._teacher_policy(self.streaming_config) == self.streaming_config
        )

    def _supervises_guidance_branches(self, config: Any) -> bool:
        """The fake score trains its branches under ``fake_branch_loss``."""
        return bool(config.meta_model.get("fake_branch_loss", False))

    def _configure_chains(self, config: Any) -> None:
        """The chain pool checkpoints its chains and broadcasts each source's window to its UP group."""

    def _chain_head_draws(self, entry: _ChainState, batch: dict[str, Any], rng: Any) -> None:
        """Draw RF-style paired history resampling times once per clip when enabled."""
        super()._chain_head_draws(entry, batch, rng)
        if self.history_refresh or self.score_history_refresh or self.fake_history_refresh:
            entry.resample_timesteps = _sample_chain_resample_timesteps(self.resample_timesteps, entry, batch, rng)

    def _chain_window_payload(self, entry: _ChainState, window: int) -> dict[str, Any]:
        payload = {
            **super()._chain_window_payload(entry, window), "chain_id": entry.chain_id,
            "window_seed": combine_seed(entry.seed, "dmd_window", entry.chain_id, entry.offset + window),
        }
        if self.history_refresh or self.score_history_refresh or self.fake_history_refresh:
            payload["resample_timesteps"] = entry.resample_timesteps
        return payload

    @property
    def _student_needs_negative(self) -> bool:
        if self.cfg_fitting_guidance is not None:
            return False
        if self.teacher_streaming is None:
            return self._needs_negative
        return self.student_cfg_fitting and self.cfg_fitting_scale != 1.0

    @property
    def _score_needs_negative(self) -> bool:
        if self.cfg_fitting_guidance is not None:
            return False
        lower, upper = self.teacher_guidance_scale
        teacher_scale = self.cfg_fitting_scale if self.teacher_cfg_fitting else 1.0
        return (self.fake_cfg_fitting and self.cfg_fitting_scale != 1.0) or not lower == upper == teacher_scale

    def _dmd_window(self, ctx: dict[str, Any], *, prepare_score_history: bool = True) -> VideoRefStreamingDMDInputs:
        """Build the student's window and the scores' window from one payload and its working copies."""
        payload = ctx["encoded_batch"]
        source = self._inputs_from_payload(payload)
        view = dict(ctx, inputs=source)
        window, anchor = int(payload["chain_window"]), MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
        rng = RandomState(int(payload["window_seed"]))
        rngs = [rng.fork("sample", index) for index in range(source.batch_size)]
        plans, references = [], None if source.reference_latents is None else []
        for index, (plan, sample_rng) in enumerate(zip(self._prepared_streaming_plans(view), rngs, strict=True)):
            if plan.get("picture_shape") is not None:
                clean = plan["picture_clean_latents"].to(device=get_device(), dtype=torch.float32)
                plan = dict(plan, picture_latents=anchor * clean + (1 - anchor) * self._noise(clean, sample_rng.fork("picture")))
            if references is not None:
                reference = self._select(source.reference_latents[index], plan["reference_video_indices"]).float()
                references.append(anchor * reference + (1 - anchor) * self._noise(reference, sample_rng.fork("reference")))
            plans.append(plan)
        plans = self._dmd_student_plans(payload, plans)
        window_inputs = self._window_inputs(source, plans, references)
        fields = dict(chain_id=str(payload["chain_id"]), window=window, commit=(True,) * source.batch_size,
                      self_history=(window > 0,) * source.batch_size)
        guided = self.cfg_fitting_guidance is not None
        score_window = None
        if self.teacher_streaming is not None:
            teacher, history = self._teacher_window(view, plans, rngs, None)
            negative = (self._negative_window_inputs({"window_inputs": teacher, "inputs": source})
                        if self._score_needs_negative else None)
            branches = None
            if guided and self.fake_cfg_fitting:
                branches = {branch.name: self._teacher_branch_inputs(teacher, source, branch)
                            for branch in self.cfg_fitting_guidance[0]}
            score_window = VideoRefStreamingDMDInputs(
                **fields, window_inputs=teacher, negative_window_inputs=negative, history=history,
                branch_windows=branches, guidance_available=self._guidance_available(teacher, source) if branches else None,
            )
        negative = (self._negative_window_inputs({"window_inputs": window_inputs, "inputs": source})
                    if self._student_needs_negative else None)
        history = self._window_history(ctx["clean_latents"], plans, rng)
        branches = None
        if guided and (self.student_cfg_fitting or self.fake_cfg_fitting and score_window is None):
            branch_view = dict(view, window_inputs=window_inputs, noisy_latents=history)
            branches = {branch.name: self._student_branch_inputs(branch_view, branch) for branch in self.cfg_fitting_guidance[0]}
        inputs = VideoRefStreamingDMDInputs(
            **fields, window_inputs=window_inputs, negative_window_inputs=negative, history=history,
            score_window=score_window, chain=ctx["chain"], branch_windows=branches,
            guidance_available=self._guidance_available(window_inputs, source) if branches else None,
        )
        inputs = self._finish_dmd_window(inputs, view=view, plans=plans, rngs=rngs, rng=rng)
        if self.history_refresh:
            refresh = self._history_refresh_inputs(view, inputs, score=False)
            score_inputs = inputs.score_window
            if self.history_resample_source == "per_model":
                if score_inputs is None:
                    score_inputs = replace(inputs, chain=None, score_window=None, real_score_window=None)
                score_inputs = replace(score_inputs, history_refresh=self._score_history_refresh_inputs(view, score_inputs))
                inputs = replace(inputs, real_score_window=score_inputs)
            elif score_inputs is not None and not self._share_refreshed_score_history:
                score_inputs = replace(score_inputs, history_refresh=self._history_refresh_inputs(view, inputs, score=True))
            inputs = replace(inputs, history_refresh=refresh, score_window=score_inputs)
        if (self.score_history_refresh or self.fake_history_refresh) and prepare_score_history:
            inputs = replace(inputs, score_history_refresh=self._score_history_refresh_inputs(view, self._score_inputs(inputs)))
        return inputs

    def _history_refresh_inputs(
        self, ctx: dict[str, Any], inputs: VideoRefStreamingDMDInputs, *, score: bool,
    ) -> DMDHistoryRefreshInputs:
        """Cache generated x0 and a history-only student layout, without a model forward."""
        source, plans = ctx["inputs"], inputs.plans
        target = inputs.score_window if score else inputs
        if score:
            source = replace(source, streaming_configs=[self._teacher_policy(policy) for policy in source.streaming_configs])
            student_plans = []
            for index, original in enumerate(plans):
                plan = self._plan(source, index, original["start"])
                if original.get("picture_shape") is not None:
                    plan = self._with_picture(plan, original["picture_clean_latents"], keyframe=self.picture_mode == "keyframe")
                    plan["picture_latents"] = original["picture_latents"]
                student_plans.append(plan)
            plans = self._dmd_student_plans(ctx["encoded_batch"], student_plans)
        plans = [build_history_resample_plan(plan) for plan in plans]
        clean = tuple([
            self._select(value, plan["audio_indices" if modality else "video_indices"], audio=bool(modality)).detach().float()
            for value, plan in zip(ctx["clean_latents"][modality], plans, strict=True)
        ] for modality in (0, 1))
        references = None
        if source.reference_latents is not None:
            references = []
            for plan, original, reference in zip(plans, target.plans, target.window_inputs.reference_latents, strict=True):
                indices = original["reference_video_indices"].cpu()
                selected = torch.isin(indices, plan["reference_video_indices"])
                assert torch.equal(indices[selected], plan["reference_video_indices"])
                references.append(self._select(reference, selected.nonzero().flatten()))
        window = self._window_inputs(source, plans, references)
        zeros = tuple([torch.zeros_like(value) for value in modality] for modality in clean)
        view = dict(ctx, inputs=source, window_inputs=window, noisy_latents=zeros)
        negative = (self._negative_window_inputs(view) if self.student_cfg_fitting and self._student_needs_negative else None)
        guided = self.student_cfg_fitting and self.cfg_fitting_guidance is not None
        branches = ({branch.name: self._student_branch_inputs(view, branch) for branch in self.cfg_fitting_guidance[0]}
                    if guided else None)
        available = self._guidance_available(window, source) if guided else None

        refresh_window = VideoRefStreamingDMDInputs(
            chain_id=inputs.chain_id, window=inputs.window, commit=(False,) * inputs.batch_size,
            self_history=inputs.self_history, window_inputs=self._nonempty_history_window(window),
            negative_window_inputs=None if negative is None else self._nonempty_history_window(negative), history=zeros,
            branch_windows=(None if branches is None else
                            {name: self._nonempty_history_window(value) for name, value in branches.items()}),
            guidance_available=available,
        )
        return DMDHistoryRefreshInputs(refresh_window, clean, ctx["encoded_batch"]["resample_timesteps"])

    def _nonempty_history_window(self, value: StreamingInputs) -> StreamingInputs:
        """Keep empty bootstrap branches in their model's forward collectives."""
        if all(int(length) for length in value.seqlens):
            return value
        embeddings, plans = list(value.prompt_embeds), list(value.plans)
        tags = None if value.text_token_tags is None else list(value.text_token_tags)
        for index, length in enumerate(value.seqlens):
            if int(length) == 0:
                # A constant text row carries no current target content.
                embeddings[index] = embeddings[index].new_zeros((1, embeddings[index].shape[1]))
                plans[index] = dict(plans[index], text_len=1, packing_rows=1)
                if tags is not None:
                    tags[index] = tags[index].new_ones(1)
        return self._streaming_inputs_from_payload(dict(
            plans=plans, prompt_embeds=embeddings, reference_latents=value.reference_latents, text_token_tags=tags,
        ))

    def _score_history_refresh_inputs(
        self, ctx: dict[str, Any], inputs: VideoRefStreamingDMDInputs,
    ) -> DMDHistoryRefreshInputs:
        """Restrict a score's own conditions and branch layouts to its generated history."""
        def history_only(window: StreamingInputs) -> StreamingInputs:
            plans = [build_history_resample_plan(plan) for plan in window.plans]
            references = None
            if window.reference_latents is not None:
                references = []
                for plan, original, reference in zip(plans, window.plans, window.reference_latents, strict=True):
                    indices = original["reference_video_indices"].cpu()
                    selected = torch.isin(indices, plan["reference_video_indices"])
                    assert torch.equal(indices[selected], plan["reference_video_indices"])
                    references.append(self._select(reference, selected.nonzero().flatten()))
            return self._nonempty_history_window(self._streaming_inputs_from_payload(dict(
                plans=plans, prompt_embeds=window.prompt_embeds, reference_latents=references,
                text_token_tags=window.text_token_tags,
            )))

        window = history_only(inputs.window_inputs)
        clean = tuple([
            self._select(value, plan["audio_indices" if modality else "video_indices"], audio=bool(modality)).detach().float()
            for value, plan in zip(ctx["clean_latents"][modality], window.plans, strict=True)
        ] for modality in (0, 1))
        refresh_window = replace(
            inputs, chain=None, score_window=None, real_score_window=None, history_refresh=None, score_history_refresh=None,
            commit=(False,) * inputs.batch_size, window_inputs=window,
            negative_window_inputs=(None if inputs.negative_window_inputs is None else
                                    history_only(inputs.negative_window_inputs)),
            branch_windows=(None if inputs.branch_windows is None else
                            {name: history_only(value) for name, value in inputs.branch_windows.items()}),
            history=tuple([torch.zeros_like(value) for value in modality] for modality in clean),
        )
        return DMDHistoryRefreshInputs(refresh_window, clean, ctx["encoded_batch"]["resample_timesteps"])

    def _refresh_history_prediction(
        self, ctx: dict[str, Any], refresh: DMDHistoryRefreshInputs, *, role: str,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Predict history x0 once at the clip's paired times with fresh role noise."""
        rng = ctx["rng"].fork("history_refresh", role)
        noises = self._noise_rows(refresh.clean_x0s, rng.fork("noise"))
        schedule = self.fake_schedule if role == "fake" else self.schedule
        xts = tuple(schedule.forward(clean, noise, time)
                    for clean, noise, time in zip(refresh.clean_x0s, noises, refresh.timesteps, strict=True))
        if role == "fake":
            return self._fake_window_prediction(ctx["models"]["fake_model"], refresh.window, xts, refresh.timesteps)
        if role == "real":
            return self._guided_prediction(
                ctx["models"]["tea_model"], refresh.window.window_inputs, lambda: self._negative(refresh.window),
                guidance=self.teacher_guidance_scale,
                fitting_scale=self.cfg_fitting_scale if self.teacher_cfg_fitting else 1.0,
                rng=rng, **self._window_kwargs(refresh.window, xts, refresh.timesteps),
            )
        return self._fitted_window_prediction(
            ctx["models"]["backbone"], refresh.window, xts, refresh.timesteps, fitting=self.student_cfg_fitting,
        )

    @staticmethod
    def _with_refreshed_history(
        inputs: VideoRefStreamingDMDInputs, refresh: DMDHistoryRefreshInputs,
        prediction: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> VideoRefStreamingDMDInputs:
        """Replace history while preserving each layout's original near-clean perturbation."""
        history = ([], [])
        for modality, prefix in enumerate(("video", "audio")):
            dim = 2 if modality else 1
            for conditioned, old, new, plan, refreshed in zip(
                inputs.history[modality], refresh.clean_x0s[modality], prediction[modality],
                inputs.plans, refresh.window.plans, strict=True,
            ):
                mask = ~plan[f"{prefix}_noisy_mask"]
                assert torch.equal(plan[f"{prefix}_indices"][mask], refreshed[f"{prefix}_indices"]), (
                    "shared refreshed history requires the same AV indices"
                )
                rows = mask.nonzero().flatten().to(conditioned.device)
                values = (conditioned.index_select(dim, rows) + MINIMAX_H3_VIDEO_CLEAN_TIMESTEP * (new - old)
                          if not modality else new)
                history[modality].append(conditioned.index_copy(dim, rows, values.to(conditioned)))
        return replace(inputs, history=history)

    @torch.no_grad()
    def _refresh_model_history(
        self, ctx: dict[str, Any], inputs: VideoRefStreamingDMDInputs, *, role: str,
    ) -> VideoRefStreamingDMDInputs:
        """Reconstruct one model's history and restore its caller's train/eval state."""
        model = ctx["models"][{"student": "backbone", "fake": "fake_model", "real": "tea_model"}[role]]
        was_training = model.training
        model.eval()
        try:
            prediction = self._refresh_history_prediction(ctx, inputs.history_refresh, role=role)
            return self._with_refreshed_history(inputs, inputs.history_refresh, prediction)
        finally:
            model.train(was_training)

    def _real_score_inputs(self, inputs: VideoRefStreamingDMDInputs) -> VideoRefStreamingDMDInputs:
        """Read the real model's own refreshed history when the window supplies one."""
        return super()._real_score_inputs(inputs) if inputs.real_score_window is None else inputs.real_score_window

    def _refreshed_score_inputs(
        self, ctx: dict[str, Any], inputs: VideoRefStreamingDMDInputs, *, role: str = "real",
    ) -> VideoRefStreamingDMDInputs:
        """Use one role's reconstruction for this phase's score history, preserving student inputs."""
        assert inputs.score_history_refresh is not None
        score = replace(self._score_inputs(inputs), history_refresh=inputs.score_history_refresh)
        refreshed = self._refresh_model_history(ctx, score, role=role)
        if role == "fake":
            return replace(inputs, score_window=refreshed)
        return replace(inputs, score_window=refreshed, real_score_window=refreshed)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().prepare_fake(ctx)
        if self.fake_history_refresh:
            ctx["fake_inputs"] = self._refreshed_score_inputs(ctx, ctx["fake_inputs"], role="fake")
        elif self.score_history_refresh:
            ctx["fake_inputs"] = self._refreshed_score_inputs(ctx, ctx["fake_inputs"])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().prepare_gen(ctx)
        if self.score_history_refresh:
            ctx["gen_inputs"] = self._refreshed_score_inputs(ctx, ctx["gen_inputs"])
        return ctx

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Refresh history at consumption time, then cache the ordinary current-chunk rollout."""
        if self.history_refresh and self.history_resample_source == "per_model":
            inputs = ctx["inputs"]
            student = self._refresh_model_history(ctx, inputs, role="student")
            fake = self._refresh_model_history(ctx, self._score_inputs(inputs), role="fake")
            real = self._refresh_model_history(ctx, self._real_score_inputs(inputs), role="real")
            ctx["inputs"] = replace(student, score_window=fake, real_score_window=real)
        elif self.history_refresh:
            inputs = ctx["inputs"]
            backbone = ctx["models"]["backbone"]
            was_training = backbone.training
            backbone.eval()
            try:
                prediction = self._refresh_history_prediction(ctx, inputs.history_refresh, role="student")
                refreshed = self._with_refreshed_history(inputs, inputs.history_refresh, prediction)
                score = inputs.score_window
                if score is not None:
                    source = inputs.history_refresh if self._share_refreshed_score_history else score.history_refresh
                    if not self._share_refreshed_score_history:
                        prediction = self._refresh_history_prediction(ctx, source, role="score")
                    refreshed = replace(refreshed, score_window=self._with_refreshed_history(score, source, prediction))
                ctx["inputs"] = refreshed
            finally:
                backbone.train(was_training)
        return super().rollout(ctx)

    def _finish_dmd_window(
        self, inputs: VideoRefStreamingDMDInputs, *, view: dict[str, Any], plans: list[dict[str, Any]],
        rngs: list[RandomState], rng: RandomState,
    ) -> VideoRefStreamingDMDInputs:
        """Return the window's inputs with what the host adds, unchanged by default.

        ``view`` is the payload context the windows are built from, ``plans``
        the student's plans with their anchored pictures, ``rngs`` the sample
        streams and ``rng`` the window stream, so a host can rebuild either
        window's history with the same draws.
        """
        return inputs

    def _dmd_student_plans(self, payload: dict[str, Any], plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The student's window plans after the anchored picture, unchanged by default."""
        return plans

    def _fitted_window_prediction(
        self, model: Any, inputs: StreamingDMDInputs, values: tuple[list[torch.Tensor], list[torch.Tensor]],
        timesteps: tuple[torch.Tensor, torch.Tensor], *, fitting: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fit through the window's guided branches when ``cfg_fitting_guidance`` is set."""
        if not fitting or self.cfg_fitting_guidance is None:
            return super()._fitted_window_prediction(model, inputs, values, timesteps, fitting=fitting)
        return self._guidance_fit(model, inputs.window_inputs, inputs.guidance_available,
                                  lambda branch: inputs.branch_windows[branch.name],
                                  **self._window_kwargs(inputs, values, timesteps))

    def _fake_window_prediction(
        self, model: Any, inputs: StreamingDMDInputs, values: tuple[list[torch.Tensor], list[torch.Tensor]],
        timesteps: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Read the fake score at ``fake_fitting_state`` when it is set."""
        if self.fake_fitting_state is None:
            return super()._fake_window_prediction(model, inputs, values, timesteps)
        return self._guidance_state_fit(model, inputs.guidance_available,
                                        lambda branch: inputs.branch_windows[branch.name], self.fake_fitting_state,
                                        **self._window_kwargs(inputs, values, timesteps))

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """With ``fake_branch_loss`` also writes fake_branch_fits: every supervised branch's fit and sample weights."""
        if not self.fake_branch_loss:
            return super().fake_forward(ctx)
        inputs = self._score_inputs(ctx["fake_inputs"])
        compiled, prediction, anchored = self._guidance_predictions(
            ctx["models"]["fake_model"], inputs.window_inputs, inputs.guidance_available,
            lambda branch: inputs.branch_windows[branch.name], supervised=True,
            **self._window_kwargs(inputs, ctx["fake_noisy_latents"], (ctx["fake_timesteps"], ctx["fake_audio_timesteps"])),
        )
        ctx["fake_pred"] = self._guided_combination(
            prediction, anchored, [(item.full_scale, item.coefficients) for item in compiled],
        )
        ctx["fake_branch_fits"] = {
            branch.name: (
                self._guided_combination(anchored[index], anchored, [
                    (item.branch_scales[index], item.branch_coefficients[index]) for item in compiled
                ]),
                tuple(item.loss_weights[index] for item in compiled),
            )
            for index, branch in enumerate(self.cfg_fitting_guidance[0]) if branch.loss_weight > 0.0
        }
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """With ``fake_branch_loss`` adds every supervised branch's weighted loss against the same targets."""
        ctx = super().fake_loss(ctx)
        if not self.fake_branch_loss:
            return ctx
        timesteps = (ctx["fake_timesteps"], ctx["fake_audio_timesteps"])
        targets = tuple(
            self.fake_schedule.convert_to_pred(x_0=x0, x_T=noise, t=t, pred_type=self.fake_loss_type)
            for x0, noise, t in zip(ctx["fake_x0s"], ctx["fake_noises"], timesteps, strict=True)
        )
        meter = get_running_average_meter()
        for name, (fitted, weights) in ctx["fake_branch_fits"].items():
            predictions = tuple(
                self._to_loss_pred(pred, xts, t)
                for pred, xts, t in zip(fitted, ctx["fake_noisy_latents"], timesteps, strict=True)
            )
            video, audio = self._reduce_window_losses(
                {"plans": ctx["fake_inputs"].plans}, predictions, targets, sample_weights=weights,
            )
            meter.put_scalar(f"running/fake_losses/branches/{name}/video", float(video.detach()))
            meter.put_scalar(f"running/fake_losses/branches/{name}/audio", float(audio.detach()))
            ctx["fake_loss"] = ctx["fake_loss"] + video + self.audio_loss_weight * audio
        return ctx

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads batch, models, rng. Writes the pool's keys, inputs and neg_inputs.

        Without a training UP group ``inputs`` is this rank's window. With
        one, ``sync_inputs`` builds every source rank's window after the
        broadcast.
        """
        ctx = super().prepare_inputs(ctx)
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            ctx["inputs"] = self._dmd_window(ctx)
        ctx["neg_inputs"] = None
        return ctx

    @torch.no_grad()
    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield every source rank's window. Only the source keeps its chain entry."""
        for sub_ctx in super().sync_inputs(ctx):
            yield dict(sub_ctx, inputs=self._dmd_window(sub_ctx), neg_inputs=None)

    def _commit_window(self, inputs: VideoRefStreamingDMDInputs, final: tuple[list[torch.Tensor], list[torch.Tensor]]) -> None:
        """Write the rollout back on the source rank only."""
        entry = inputs.chain
        if entry is None:
            return
        assert entry.next_index == inputs.window, "chain rollouts must follow their preparation order"
        self._write_back(entry, inputs.plans, final)


EntryClass = MiniMaxH3VideoRefStreamingDMD

__all__ = ["MiniMaxH3VideoRefStreamingDMD", "VideoRefStreamingDMDInputs", "DMDHistoryRefreshInputs", "EntryClass"]
