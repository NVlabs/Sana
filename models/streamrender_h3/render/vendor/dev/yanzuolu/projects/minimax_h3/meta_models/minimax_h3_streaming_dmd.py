# SPDX-License-Identifier: Apache-2.0
"""Distribution matching distillation over chains of bidirectional streaming windows.

The few-step streaming student is the DMD generator. Every engine iteration
trains exactly one window of each clip in a chain, reusing the streaming
SFT window plans, dense layouts, captionless branches and validation. The
three networks share the streaming architecture and all return x0: the
LoRA student ``backbone``, the trainable LoRA fake score ``fake_model`` and
the frozen real score ``tea_model``.

ctx contract (see engines/dmd.py for the calling sequence)::

    prepare_inputs   reads batch/models/rng      writes inputs, neg_inputs
    sync_inputs      reads inputs                yields ctx unchanged
    rollout          reads models/inputs/rng     writes rollout_x0s, trajectory_xts  (no-grad)
    prepare_fake     reads rollout_x0s/          writes fake_x0s, fake_timesteps,
                     trajectory_xts/rng          fake_audio_timesteps, fake_noises,
                                                 fake_noisy_latents, fake_inputs
    fake_forward     reads models/fake_*         writes fake_pred
    fake_loss        reads fake_pred/fake_*      writes fake_loss
    prepare_gen      reads trajectory_xts/rng    writes gen_timesteps, gen_audio_timesteps,
                                                 gen_index, gen_xts, score_timesteps,
                                                 audio_score_timesteps, gen_inputs
    gen_forward      reads models/gen_*          writes gen_pred                     (graph attached)
    score            reads models/gen_pred/...   writes gen_x0s, fake_score_x0s, real_score_x0s
    gen_loss         reads gen_x0s/*_score_x0s   writes gen_loss

Chains
------
``StreamingLatentChainDataset`` emits each pack as one head batch followed
by light batches, one per window in order. This meta keeps a per-rank pool
keyed by ``chain_id`` in process memory, never checkpointed, and sets
``data.args.resume_mid_chain`` false so that a resumed chain restarts at
window 0 for its remaining windows. Hosts with their own checkpointed pool
and input broadcast override ``_configure_chains``. The head plans
every trained window and takes float32 device working copies of the corpus
video and audio latents. ``prepare_inputs`` reads the current window's sink
and recent history rows from those copies, and ``rollout`` writes the
student's generated x0 for the window's noisy rows back into them, so the
next window of the chain reads the student's own output as history. Only
``rollout`` writes: with ``ga_steps`` or ``offline`` above one the engine
prepares and rolls out several consecutive windows of a chain before any
FAKE or GEN phase, and each of them must already read its predecessor's
output. FAKE and GEN read only the payload, whose tensors never alias the
working copies. The last window's write-back is discarded with the entry.

``gen_gt_ratio`` mixes corpus and self-generated history per clip. At the
head, each clip keeps corpus latents for its first ``k`` windows with that
probability, ``k`` drawn uniformly from the inclusive
``gt_prefix_window_range``, and otherwise ``k = 0``. Windows before ``k``
are still generated and trained, but not written back, so window ``w``
reads corpus history when ``w <= k`` and its own recent output afterwards,
while a sink inside the first ``k`` windows stays corpus. Every window of a
chain is trained exactly once, so ``bootstrap_probability`` is rejected.

Each window is prepared once. Video history rows carry the clean-condition
perturbation at timestep 0.001 and retained audio stays clean, exactly as in
streaming SFT, and rollout, FAKE and GEN all read that same window. A rank
therefore issues the same collectives every iteration regardless of chain
phase: one text-encoder call, a fixed-length rollout and fixed FAKE and GEN
forwards. Working copies are rank-local, so ``engine.up_size.train`` must be 1
and training never broadcasts inputs.

DMD
---
``rollout`` runs the student's static few-step grid over the noisy rows and
keeps every stage's input x_t and x0 prediction. ``score_path`` chooses
the points at which FAKE trains the fake score and both scores read the
generator. With the default ``renoise``, GEN re-runs the student with
gradients at a uniformly drawn stage, adds fresh noise to its x0 at paired
score timesteps, and scores it with the fake and real networks on the same
window. ``fake_use_trajectory`` trains FAKE on a uniformly drawn stage's
x0, otherwise on the final x0, noised afresh at paired fake timesteps.

With ``ode`` FAKE and GEN draw no noise. Beyond the window's conditioned
history, the rollout's initial noise is the only noise, and every later
state follows DDIM. GEN draws paired score timesteps and selects each
sample's stage k whose video interval
``t_{k+1} <= s < t_k`` holds the video time, the last interval ending at
0. Both modalities use stage k, and each time is held
inside stage k's interval of its own grid. That leaves the paired draw
unchanged whenever the paired distributions share the grids' shifts, and
moves a time drawn at ``t_0`` just below it. GEN re-runs the student with
gradients at stage k's cached x_t, whose x0 the DMD loss scores, and both
scores read the DDIM step from that x_t along the detached prediction to s.

With ``fake_use_trajectory`` true, FAKE applies that same interval selection
to its paired fake timesteps. It steps the selected stage's cached x_t
along its cached prediction to s with DDIM and regresses onto that stage's
x0 and the x_T it implies. With false, FAKE uses the last stage's cached
x_t and the rollout's final x0 to define its DDIM line. Its paired fake
timesteps stay as drawn, so points beyond the last interval extrapolate
that line, and its targets are the final x0 and that line's implied x_T.
``fake_use_trajectory`` does not change GEN. The ODE path requires the DDIM
sampler, a ``fake_schedule`` equal to ``schedule`` in T, A and B, student
grids strictly decreasing to 0, and both ``fake_grad_enabled`` and
``real_grad_enabled`` false.

A host may give both scores their own window through
``StreamingDMDInputs.score_window``, whose noisy rows are the generator's
in the same order. FAKE then trains and both networks score on it, while
rollout and the generator forward keep the student's window. Losses cover
only noisy rows. Each sample and modality is averaged over its own elements
before ``audio_loss_weight`` combines the modalities, and the DMD normalizer
is per sample and modality as in causal DMD.

``student_cfg_fitting``, ``fake_cfg_fitting`` and ``teacher_cfg_fitting``
switch CFG fitting independently at ``cfg_fitting_scale``, with the
captionless negative detached. The student's fitted prediction is the
generator throughout rollout, write-back and GEN. ``teacher_guidance_scale``
is the real score's external CFG, a static ``[g, g]`` or a per-sample
range ``[lo, hi]``, applied after teacher fitting as ``uncond + g (cond - uncond)``.
The fake score reads one prediction, ``_fake_window_prediction``, both when
FAKE trains it and when GEN scores the generator's sample.

Expected configuration shape::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_dmd
      class_name: MiniMaxH3StreamingDMD
      audio_loss_weight: 1.0
      fixed_window_rope: true
      fake_loss_type: x_0
      fake_grad_enabled: false
      real_grad_enabled: false
      score_path: renoise
      fake_use_trajectory: true
      gen_gt_ratio: 0.5
      gt_prefix_window_range: [1, 10]
      student_cfg_fitting: true
      fake_cfg_fitting: true
      teacher_cfg_fitting: false
      cfg_fitting_scale: 3.0
      teacher_guidance_scale: [1.0, 1.0]
      dmd_loss: {type: dmd, norm_clip_min: 1.0e-5, phuber_c: 0.001, alpha: 1.0}

``config.diffusion`` provides the causal DMD nodes: ``sampling_timesteps``,
``audio_sampling_timesteps``, ``schedule``, ``sampler``,
``fake_training_timesteps``, ``audio_fake_training_timesteps``,
``score_timesteps``, ``audio_score_timesteps`` and ``fake_schedule``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch

from dev.yanzuolu.common.diffusion.sampler.ddim import DDIMSampler
from dev.yanzuolu.common.distributed.ops import all_reduce_sum, get_device, get_world_size
from dev.yanzuolu.common.meter import get_running_average_meter
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.data.streaming import streaming_window_starts
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import CausalMiniMaxH3DMD
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import MiniMaxH3StreamingSFT, StreamingBatch
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import MINIMAX_H3_VIDEO_CLEAN_TIMESTEP

_Pair = tuple[list[torch.Tensor], list[torch.Tensor]]


@dataclass
class _DMDChain:
    """One pack's chain as this rank sees it between iterations.

    ``video`` and ``audio`` are the float32 working copies that windows read
    history from and write generated x0 into. ``plans[index][window]`` is
    fixed at the head, and ``gt_windows[index]`` counts the leading windows
    whose output is not written back.
    """

    chain_id: str
    chain_length: int
    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    plans: list[list[dict[str, Any]]]
    gt_windows: list[int]
    text_input_ids: list[torch.Tensor]
    text_lens: list[int]
    latent_shapes: list[Any]
    audio_shapes: list[Any]
    streaming_config: list[Any]
    next_index: int = 0


@dataclass(frozen=True)
class StreamingDMDInputs:
    """One chain window prepared once and shared by rollout, FAKE and GEN.

    ``history`` holds complete window tensors whose history rows carry the
    conditioned values. Every forward replaces the noisy rows. ``commit``
    marks the samples whose rollout becomes later history. ``score_window``
    is the window the fake and real scores read, this one when None.
    """

    chain_id: str
    window: int | tuple[int, ...]
    commit: tuple[bool, ...]
    self_history: tuple[bool, ...]
    window_inputs: StreamingInputs
    negative_window_inputs: StreamingInputs | None
    history: _Pair
    score_window: StreamingDMDInputs | None = None

    @property
    def batch_size(self) -> int:
        return self.window_inputs.batch_size

    @property
    def plans(self) -> list[dict[str, Any]]:
        return self.window_inputs.plans

    @property
    def seqlens(self) -> torch.Tensor:
        return self.window_inputs.seqlens

    @property
    def window_indices(self) -> tuple[int, ...]:
        """Each sample's window, including independently selected packed windows."""
        return (self.window,) * self.batch_size if isinstance(self.window, int) else self.window


@dataclass(frozen=True)
class StreamingDMDRollout:
    """Final noisy-row x0 with every stage's x0 and the per-sample sampling grids."""

    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    stage_video: list[list[torch.Tensor]]
    stage_audio: list[list[torch.Tensor]]
    video_grid: torch.Tensor
    audio_grid: torch.Tensor


class MiniMaxH3StreamingDMD(MiniMaxH3StreamingSFT, CausalMiniMaxH3DMD):
    """See the module docstring for the ctx contract, chains and CFG switches."""

    _chains: dict[str, _DMDChain]

    def __init__(self, config: Any) -> None:
        options = config.meta_model
        if options.get("guidance_branches") is not None or options.get("guidance_fitting") is not None:
            raise ValueError("streaming DMD does not support guidance_branches or guidance_fitting")
        if "bootstrap_probability" in options:
            raise ValueError("streaming DMD trains every window of a chain exactly once; remove bootstrap_probability")
        if float(options.get("guidance_scale", 1.0)) != 1.0 or float(options.get("negative_loss_weight", 0.0)) != 0.0:
            raise ValueError("streaming DMD uses per-network CFG fitting switches, not guidance_scale or negative_loss_weight")
        CausalMiniMaxH3DMD.__init__(self, config)
        self._configure_streaming(config)
        self._configure_chains(config)
        if self.norm_per_chunk or self.chunk_wise_weighting is not None:
            raise ValueError("streaming DMD normalizes whole windows; norm_per_chunk and chunk_wise_weighting are causal-only")
        if (self.bootstrap_loss_weight, self.first_frame_loss_weight, self.continuation_loss_weight) != (1.0, 1.0, 1.0):
            raise ValueError("streaming DMD weights every window uniformly")

        self.student_cfg_fitting = bool(options.get("student_cfg_fitting", False))
        self.fake_cfg_fitting = bool(options.get("fake_cfg_fitting", False))
        self.teacher_cfg_fitting = bool(options.get("teacher_cfg_fitting", False))
        self.cfg_fitting_scale = float(options.get("cfg_fitting_scale", 3.0))
        if not math.isfinite(self.cfg_fitting_scale) or self.cfg_fitting_scale <= 0:
            raise ValueError("cfg_fitting_scale must be finite and positive")
        lower, upper = self.teacher_guidance_scale
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper):
            raise ValueError("teacher_guidance_scale must be a finite nonnegative ordered pair")

        ratio = options.get("gen_gt_ratio", 0.0)
        if isinstance(ratio, bool) or not isinstance(ratio, Real) or not 0.0 <= ratio <= 1.0:
            raise ValueError("meta_model.gen_gt_ratio must be a number in [0, 1]")
        self.gen_gt_ratio = float(ratio)
        self.gt_prefix_window_range = (0, 0)
        if self.gen_gt_ratio > 0:
            window_range = options.get("gt_prefix_window_range")
            if (
                not isinstance(window_range, (list, tuple)) or len(window_range) != 2
                or any(type(value) is not int for value in window_range)
                or not 1 <= window_range[0] <= window_range[1]
            ):
                raise ValueError("gt_prefix_window_range must be positive integers [min, max]")
            self.gt_prefix_window_range = tuple(window_range)

        grids = (self.sampling_timesteps, self.audio_sampling_timesteps)
        if any(grid.dynamic_shift for grid in grids) or grids[0].num_sampling_steps != grids[1].num_sampling_steps:
            raise ValueError("video and audio student grids must be static and share one step count")

        self.score_path = options.get("score_path", "renoise")
        if self.score_path not in ("renoise", "ode"):
            raise ValueError("meta_model.score_path must be renoise or ode")
        if self.score_path == "ode":
            self._check_ode_path()

    def _check_ode_path(self) -> None:
        """The ODE points need deterministic DDIM steps, one time parameterization and grids ending at 0."""
        if not isinstance(self.sampler, DDIMSampler):
            raise ValueError(f"score_path ode requires the DDIMSampler, got {type(self.sampler).__name__}")
        if self.fake_grad_enabled or self.real_grad_enabled:
            raise ValueError("score_path ode scores a detached point, so fake_grad_enabled and real_grad_enabled must be false")
        horizon = self.schedule.T
        points = torch.tensor([0, horizon / 2, horizon], dtype=torch.float64 if isinstance(horizon, float) else torch.long)
        if float(self.fake_schedule.T) != float(horizon) or not all(
            torch.allclose(mine(points), reference(points), rtol=1.0e-6, atol=1.0e-6)
            for mine, reference in ((self.fake_schedule.A, self.schedule.A), (self.fake_schedule.B, self.schedule.B))
        ):
            raise ValueError("score_path ode requires fake_schedule to equal schedule in T, A and B")
        for node in (self.sampling_timesteps, self.audio_sampling_timesteps):
            grid = node.timesteps.to(torch.float32)
            if grid.ndim != 1 or not bool((grid > torch.cat((grid[1:], grid.new_zeros(1)))).all()):
                raise ValueError("score_path ode requires student grids strictly decreasing to 0")

    # ------------------------------------------------------------------
    # Chains
    # ------------------------------------------------------------------

    def _configure_chains(self, config: Any) -> None:
        """Keep rank-local chains, never checkpointed, so a resumed chain restarts at window 0."""
        if int(config.engine.get("up_size", {}).get("train", 1)) != 1:
            raise ValueError("streaming DMD keeps per-rank chain working copies and requires engine.up_size.train == 1")
        if config.data.args.get("resume_mid_chain", False):
            raise ValueError("streaming DMD does not checkpoint its chains, so data.args.resume_mid_chain must stay false")
        config.data.args.resume_mid_chain = False
        self._chains = {}

    @staticmethod
    def _chain_fields(batch: dict[str, Any]) -> tuple[str, int, int, int]:
        return str(batch["chain_id"]), int(batch["chain_index"]), int(batch["chain_length"]), int(batch["chain_windows"])

    def _start_chain(self, batch: dict[str, Any], rng: Any) -> _DMDChain:
        """Plan every trained window, draw corpus prefixes and copy the corpus latents."""
        chain_id, chain_index, chain_length, chain_windows = self._chain_fields(batch)
        assert chain_index == 0 and 0 < chain_length <= chain_windows
        for descriptor in batch["video_temporal_mapping"]:
            if VideoTemporalMapping.from_dict(descriptor) != self.video_temporal_mapping:
                raise ValueError("dataset temporal mapping disagrees with video_vae")
        device = get_device()
        video = [value.to(device=device, dtype=torch.float32, copy=True) for value in batch["video_latents"]]
        audio = [value.to(device=device, dtype=torch.float32, copy=True) for value in batch["audio_latents"]]
        for values, shapes in ((video, batch["latent_shapes"]), (audio, batch["audio_shapes"])):
            assert all(tuple(value.shape) == tuple(shape) for value, shape in zip(values, shapes, strict=True))
        text_lens = [int(length) for length in batch["text_lens"]]
        geometry = StreamingBatch(
            prompt_embeds=[torch.empty((length, 0)) for length in text_lens], reference_latents=None,
            video_shapes=[tuple(shape) for shape in batch["latent_shapes"]],
            audio_shapes=[tuple(shape) for shape in batch["audio_shapes"]],
            streaming_configs=[self._streaming_policy(policy) for policy in batch["streaming_config"]],
        )
        plans, gt_windows = [], []
        lowest, highest = self.gt_prefix_window_range
        for index in range(geometry.batch_size):
            starts = streaming_window_starts(
                geometry.video_shapes[index][1], **geometry.streaming_configs[index],
                video_temporal_mapping=self.video_temporal_mapping,
            )
            assert len(starts) == chain_windows, "chain_windows disagrees with the clip's streaming windows"
            windows = [self._plan(geometry, index, start) for start in starts[:chain_length]]
            assert all(plan["packing_rows"] <= int(batch["packing_rows"][index]) for plan in windows), (
                "a chain window exceeds the dataset packing budget"
            )
            plans.append(windows)
            generator = rng.fork("gt_prefix", index).python_generator
            gt_windows.append(generator.randint(lowest, highest) if generator.random() < self.gen_gt_ratio else 0)
        chain = _DMDChain(
            chain_id=chain_id, chain_length=chain_length, video=video, audio=audio, plans=plans,
            gt_windows=gt_windows, text_input_ids=list(batch["text_input_ids"]), text_lens=text_lens,
            latent_shapes=list(batch["latent_shapes"]), audio_shapes=list(batch["audio_shapes"]),
            streaming_config=list(batch["streaming_config"]),
        )
        self._chains[chain_id] = chain
        return chain

    def _continue_chain(self, batch: dict[str, Any]) -> _DMDChain:
        chain_id, chain_index, chain_length, _ = self._chain_fields(batch)
        chain = self._chains.get(chain_id)
        if chain is None:
            raise RuntimeError(f"chain {chain_id!r} has no pool entry: the dataloader and the DMD chain pool desynchronised")
        assert chain_index == chain.next_index and chain_length == chain.chain_length
        return chain

    def _window_history(self, latents: _Pair, plans: list[dict[str, Any]], rng: Any) -> _Pair:
        """Select history rows from the working copies with SFT's clean-condition perturbation."""
        history: _Pair = ([], [])
        anchor = MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
        for index, plan in enumerate(plans):
            video = self._select(latents[0][index], plan["video_indices"])
            video = anchor * video + (1 - anchor) * self._noise(video, rng.fork("history", index))
            video[:, plan["video_noisy_mask"].to(video.device)] = 0
            audio = self._select(latents[1][index], plan["audio_indices"], audio=True)
            audio[:, :, plan["audio_noisy_mask"].to(audio.device)] = 0
            history[0].append(video)
            history[1].append(audio)
        return history

    def _commit_window(self, inputs: StreamingDMDInputs, final: _Pair) -> None:
        """Write this window's generated x0 back as history and advance the chain."""
        chain = self._chains[inputs.chain_id]
        assert chain.next_index == inputs.window, "chain rollouts must follow their preparation order"
        for index, (plan, commit) in enumerate(zip(inputs.plans, inputs.commit, strict=True)):
            if not commit:
                continue
            for modality, working in enumerate((chain.video[index], chain.audio[index])):
                prefix = "audio" if modality else "video"
                rows = plan[f"{prefix}_indices"][plan[f"{prefix}_noisy_mask"]]
                working.index_copy_(2 if modality else 1, rows.to(working.device), final[modality][index].to(working.dtype))
        chain.next_index += 1
        if chain.next_index == chain.chain_length:
            del self._chains[chain.chain_id]

    # ------------------------------------------------------------------
    # Window forwards
    # ------------------------------------------------------------------

    @property
    def _needs_negative(self) -> bool:
        lower, upper = self.teacher_guidance_scale
        teacher_scale = self.cfg_fitting_scale if self.teacher_cfg_fitting else 1.0
        fitted = self.cfg_fitting_scale != 1.0 and (self.student_cfg_fitting or self.fake_cfg_fitting)
        return fitted or not lower == upper == teacher_scale

    @staticmethod
    def _score_inputs(inputs: StreamingDMDInputs) -> StreamingDMDInputs:
        return inputs if inputs.score_window is None else inputs.score_window

    def _real_score_inputs(self, inputs: StreamingDMDInputs) -> StreamingDMDInputs:
        """The real score shares the fake score's window unless the host supplies its own."""
        return self._score_inputs(inputs)

    @staticmethod
    def _negative(inputs: StreamingDMDInputs) -> StreamingInputs:
        assert inputs.negative_window_inputs is not None
        return inputs.negative_window_inputs

    @staticmethod
    def _window_values(inputs: StreamingDMDInputs, values: _Pair) -> _Pair:
        """Place noisy-row values into the prepared history, keeping their graphs."""
        return tuple([
            history.index_copy(
                2 if modality else 1,
                plan["audio_noisy_mask" if modality else "video_noisy_mask"].nonzero().flatten().to(history.device),
                value.to(history.dtype),
            )
            for history, value, plan in zip(inputs.history[modality], values[modality], inputs.plans, strict=True)
        ] for modality in (0, 1))

    def _window_kwargs(self, inputs: StreamingDMDInputs, values: _Pair, timesteps: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        video, audio = self._window_values(inputs, values)
        return dict(video_xts=video, audio_xts=audio, video_timesteps=timesteps[0], audio_timesteps=timesteps[1])

    def _fitted_window_prediction(
        self, model: Any, inputs: StreamingDMDInputs, values: _Pair,
        timesteps: tuple[torch.Tensor, torch.Tensor], *, fitting: bool,
    ) -> _Pair:
        return self._fitted_prediction(
            model, inputs.window_inputs, lambda: self._negative(inputs),
            scale=self.cfg_fitting_scale if fitting else 1.0, **self._window_kwargs(inputs, values, timesteps),
        )

    def _fake_window_prediction(
        self, model: Any, inputs: StreamingDMDInputs, values: _Pair, timesteps: tuple[torch.Tensor, torch.Tensor],
    ) -> _Pair:
        """The fake score's prediction on its window, shared by FAKE and GEN."""
        return self._fitted_window_prediction(model, inputs, values, timesteps, fitting=self.fake_cfg_fitting)

    def _noise_rows(self, values: _Pair, rng: Any) -> _Pair:
        return tuple([self._noise(value, rng) for value in modality] for modality in values)

    # ------------------------------------------------------------------
    # Score points
    # ------------------------------------------------------------------

    def _paired_times(self, inputs: StreamingDMDInputs, video: Any, audio: Any, rng: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(value.to(torch.float32) for value in self._sample_paired_timesteps(
            inputs.batch_size, inputs.seqlens, video, audio, rng,
        ))

    @staticmethod
    def _stage_rows(stages: list[_Pair], index: torch.Tensor) -> _Pair:
        """Each sample's rows at its own stage, from every stage's video and audio rows."""
        return tuple([stages[stage][modality][sample] for sample, stage in enumerate(index.tolist())] for modality in (0, 1))

    @staticmethod
    def _stage_times(rollout: StreamingDMDRollout, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = torch.arange(index.numel(), device=rollout.video_grid.device)
        return tuple(grid[samples, index.to(samples.device)] for grid in (rollout.video_grid, rollout.audio_grid))

    @staticmethod
    def _ode_stages(
        rollout: StreamingDMDRollout, timesteps: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Each sample's stage, whose video interval holds its video time, and both times held inside it.

        Stage k spans ``t_{k+1} <= s < t_k`` of each grid, the last stage ending at 0.
        """
        index = (rollout.video_grid > timesteps[0][:, None]).sum(1).sub(1).clamp_min(0)
        held = []
        for grid, values in zip((rollout.video_grid, rollout.audio_grid), timesteps, strict=True):
            nodes = torch.cat((grid, grid.new_zeros(grid.shape[0], 1)), dim=1)
            upper, lower = (nodes.gather(1, (index + offset)[:, None]).squeeze(1) for offset in (0, 1))
            held.append(values.clamp(lower, torch.nextafter(upper, lower)))
        return index, tuple(held)

    def _fake_point(self, ctx: dict[str, Any]) -> tuple[_Pair, _Pair, tuple[torch.Tensor, torch.Tensor], _Pair]:
        """FAKE's x0 and x_T endpoints, its paired timesteps and its noisy rows.

        ``renoise`` diffuses a rollout x0 with fresh noise. With
        ``fake_use_trajectory``, ``ode`` follows the rollout line of the
        stage holding the drawn time. Otherwise it evaluates the last
        stage's cached x_t and final x0 line at the unmodified drawn time,
        extrapolating beyond the last interval. Each line's x0 and implied
        x_T are the targets.
        """
        inputs, rng, rollout = ctx["inputs"], ctx["rng"], ctx["rollout_x0s"]
        if self.score_path == "ode":
            timesteps = self._paired_times(
                inputs, self.fake_training_timesteps, self.audio_fake_training_timesteps, rng,
            )
            if self.fake_use_trajectory:
                index, timesteps = self._ode_stages(rollout, timesteps)
                predictions = self._stage_rows(list(zip(rollout.stage_video, rollout.stage_audio)), index)
            else:
                index = torch.full_like(timesteps[0], len(rollout.stage_video) - 1, dtype=torch.long)
                predictions = (rollout.video, rollout.audio)
            xts = self._stage_rows(ctx["trajectory_xts"], index)
            nodes = self._stage_times(rollout, index)
            x0s, noises = zip(*(
                self.schedule.convert_from_pred(pred, values, t) for pred, values, t in zip(predictions, xts, nodes, strict=True)
            ))
            return x0s, noises, timesteps, tuple(
                self.sampler.step_to(pred=pred, x_t=values, t=t, s=s)
                for pred, values, t, s in zip(predictions, xts, nodes, timesteps, strict=True)
            )
        if self.fake_use_trajectory:
            stage_count = len(rollout.stage_video)
            stages = [rng.python_generator.randrange(stage_count) for _ in range(inputs.batch_size)]
            x0s = ([rollout.stage_video[stage][index] for index, stage in enumerate(stages)],
                   [rollout.stage_audio[stage][index] for index, stage in enumerate(stages)])
        else:
            x0s = (rollout.video, rollout.audio)
        timesteps = self._paired_times(inputs, self.fake_training_timesteps, self.audio_fake_training_timesteps, rng)
        noises = self._noise_rows(x0s, rng)
        return x0s, noises, timesteps, tuple(
            self.fake_schedule.forward(values, noise, t) for values, noise, t in zip(x0s, noises, timesteps, strict=True)
        )

    def _gen_stage(self, ctx: dict[str, Any]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Each sample's replayed stage and its paired score timesteps.

        ``renoise`` draws the stage uniformly, ``ode`` takes the stage holding the drawn time.
        """
        inputs, rng, rollout = ctx["inputs"], ctx["rng"], ctx["rollout_x0s"]
        if self.score_path == "ode":
            return self._ode_stages(rollout, self._paired_times(inputs, self.score_timesteps, self.audio_score_timesteps, rng))
        with local_seed(rng.seed % 2**31):
            index = torch.randint(0, rollout.video_grid.shape[1], (inputs.batch_size,))
        rng.seed = yield_seed(rng.seed)
        return index, self._paired_times(inputs, self.score_timesteps, self.audio_score_timesteps, rng)

    def _score_point(self, ctx: dict[str, Any]) -> _Pair:
        """The generator's noisy rows both scores read at the paired score timesteps.

        ``renoise`` diffuses the generator's x0 with fresh noise, keeping its
        graph attached. ``ode`` steps the replayed stage's cached x_t along
        the detached prediction.
        """
        timesteps = (ctx["score_timesteps"], ctx["audio_score_timesteps"])
        if self.score_path == "ode":
            with torch.no_grad():
                return tuple(
                    self.sampler.step_to(pred=pred, x_t=values, t=t, s=s) for pred, values, t, s in zip(
                        ctx["gen_pred"], ctx["gen_xts"], (ctx["gen_timesteps"], ctx["gen_audio_timesteps"]), timesteps,
                        strict=True,
                    )
                )
        noises = self._noise_rows(ctx["gen_pred"], ctx["rng"])
        return tuple(
            self.schedule.forward(values, noise, t) for values, noise, t in zip(ctx["gen_pred"], noises, timesteps, strict=True)
        )

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: batch, models, rng. Writes: inputs, neg_inputs.

        The text encoder runs once per iteration, never once per chain, so
        ranks at different chain phases issue the same collectives.
        """
        batch = ctx["batch"]
        chain = self._start_chain(batch, ctx["rng"]) if "video_latents" in batch else self._continue_chain(batch)
        window = chain.next_index
        source = self._inputs_from_payload({
            "prompt_embeds": self._encode_prompts(ctx["models"], chain.text_input_ids, chain.text_lens),
            "latent_shapes": chain.latent_shapes, "audio_shapes": chain.audio_shapes,
            "streaming_config": chain.streaming_config,
        })
        plans = [windows[window] for windows in chain.plans]
        window_inputs = self._window_inputs(source, plans, None)
        negative = (self._negative_window_inputs({"window_inputs": window_inputs, "inputs": source})
                    if self._needs_negative else None)
        ctx["inputs"] = StreamingDMDInputs(
            chain_id=chain.chain_id, window=window,
            commit=tuple(window >= prefix for prefix in chain.gt_windows),
            self_history=tuple(window > prefix for prefix in chain.gt_windows),
            window_inputs=window_inputs, negative_window_inputs=negative,
            history=self._window_history((chain.video, chain.audio), window_inputs.plans, ctx["rng"]),
        )
        ctx["neg_inputs"] = None
        return ctx

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, rng. Writes: rollout_x0s, trajectory_xts. (no-grad)

        Commits the final x0 to the chain before returning.
        """
        inputs, rng = ctx["inputs"], ctx["rng"]
        device = get_device()
        seqlens = [int(length) for length in inputs.seqlens]
        video_grid = self._sampling_grid(self.sampling_timesteps, seqlens, device)
        audio_grid = self._sampling_grid(self.audio_sampling_timesteps, seqlens, device)
        assert video_grid.shape == audio_grid.shape
        xts = self._noise_rows(self._noisy_rows(inputs.history, inputs.plans), rng)
        sample_rngs = [rng.fork("rollout_sampler", index) for index in range(inputs.batch_size)]
        lengths = torch.tensor(seqlens, device=device)
        trajectory, stages = [], ([], [])
        for step in range(video_grid.shape[1]):
            times = (video_grid[:, step], audio_grid[:, step])
            last = step + 1 == video_grid.shape[1]
            targets = tuple(torch.zeros_like(value) if last else grid[:, step + 1]
                            for value, grid in zip(times, (video_grid, audio_grid), strict=True))
            trajectory.append(xts)
            prediction = self._fitted_window_prediction(
                ctx["models"]["backbone"], inputs, xts, times, fitting=self.student_cfg_fitting,
            )
            for modality in (0, 1):
                stages[modality].append(prediction[modality])
            xts = tuple(self.sampler.step_to(pred=pred, x_t=values, t=t, s=s, rng=sample_rngs, seqlens=lengths)
                        for pred, values, t, s in zip(prediction, xts, times, targets, strict=True))
        self._commit_window(inputs, xts)
        ctx["rollout_x0s"] = StreamingDMDRollout(
            video=xts[0], audio=xts[1], stage_video=stages[0], stage_audio=stages[1],
            video_grid=video_grid, audio_grid=audio_grid,
        )
        ctx["trajectory_xts"] = trajectory
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: rollout_x0s, trajectory_xts, inputs, rng. Writes: fake_x0s, fake_timesteps,
        fake_audio_timesteps, fake_noises, fake_noisy_latents, fake_inputs."""
        x0s, noises, timesteps, xts = self._fake_point(ctx)
        ctx["fake_x0s"] = x0s
        ctx["fake_timesteps"], ctx["fake_audio_timesteps"] = timesteps
        ctx["fake_noises"] = noises
        ctx["fake_noisy_latents"] = xts
        ctx["fake_inputs"] = ctx["inputs"]
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, fake_inputs, fake_noisy_latents, fake timesteps. Writes: fake_pred."""
        ctx["fake_pred"] = self._fake_window_prediction(
            ctx["models"]["fake_model"], self._score_inputs(ctx["fake_inputs"]), ctx["fake_noisy_latents"],
            (ctx["fake_timesteps"], ctx["fake_audio_timesteps"]),
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: fake_pred, fake_noisy_latents, fake_noises, fake timesteps, fake_x0s. Writes: fake_loss."""
        timesteps = (ctx["fake_timesteps"], ctx["fake_audio_timesteps"])
        predictions = tuple(
            self._to_loss_pred(pred, xts, t)
            for pred, xts, t in zip(ctx["fake_pred"], ctx["fake_noisy_latents"], timesteps, strict=True)
        )
        targets = tuple(
            self.fake_schedule.convert_to_pred(x_0=x0, x_T=noise, t=t, pred_type=self.fake_loss_type)
            for x0, noise, t in zip(ctx["fake_x0s"], ctx["fake_noises"], timesteps, strict=True)
        )
        video, audio = self._reduce_window_losses({"plans": ctx["fake_inputs"].plans}, predictions, targets)
        meter = get_running_average_meter()
        meter.put_scalar("running/fake_losses/video", float(video.detach()))
        meter.put_scalar("running/fake_losses/audio", float(audio.detach()))
        ctx["fake_loss"] = video + self.audio_loss_weight * audio
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, rollout_x0s, trajectory_xts, rng. Writes: gen_timesteps,
        gen_audio_timesteps, gen_index, gen_xts, score_timesteps, audio_score_timesteps, gen_inputs."""
        index, (ctx["score_timesteps"], ctx["audio_score_timesteps"]) = self._gen_stage(ctx)
        ctx["gen_index"] = index
        ctx["gen_timesteps"], ctx["gen_audio_timesteps"] = self._stage_times(ctx["rollout_x0s"], index)
        ctx["gen_xts"] = self._stage_rows(ctx["trajectory_xts"], index)
        ctx["gen_inputs"] = ctx["inputs"]
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, gen_inputs, gen_xts, gen timesteps. Writes: gen_pred. (graph attached)"""
        ctx["gen_pred"] = self._fitted_window_prediction(
            ctx["models"]["backbone"], ctx["gen_inputs"], ctx["gen_xts"],
            (ctx["gen_timesteps"], ctx["gen_audio_timesteps"]), fitting=self.student_cfg_fitting,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def score(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, gen_inputs, gen_pred, gen_xts, gen and score timesteps, rng.
        Writes: gen_x0s, fake_score_x0s, real_score_x0s."""
        models, rng = ctx["models"], ctx["rng"]
        inputs = self._score_inputs(ctx["gen_inputs"])
        gen_x0s = ctx["gen_pred"]
        timesteps = (ctx["score_timesteps"], ctx["audio_score_timesteps"])
        score_xts = self._score_point(ctx)
        with nullcontext() if self.fake_grad_enabled else torch.no_grad():
            fake_score_x0s = self._fake_window_prediction(models["fake_model"], inputs, score_xts, timesteps)
        with nullcontext() if self.real_grad_enabled else torch.no_grad():
            inputs = self._real_score_inputs(ctx["gen_inputs"])
            real_score_x0s = self._guided_prediction(
                models["tea_model"], inputs.window_inputs, lambda: self._negative(inputs),
                guidance=self.teacher_guidance_scale,
                fitting_scale=self.cfg_fitting_scale if self.teacher_cfg_fitting else 1.0,
                rng=rng, **self._window_kwargs(inputs, score_xts, timesteps),
            )
        ctx["gen_x0s"] = gen_x0s
        ctx["fake_score_x0s"] = fake_score_x0s
        ctx["real_score_x0s"] = real_score_x0s
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: gen_inputs, gen_x0s, fake_score_x0s, real_score_x0s. Writes: gen_loss."""
        inputs = ctx["gen_inputs"]
        losses = []
        for modality, name in enumerate(("video", "audio")):
            x0s, fakes, reals = (ctx[key][modality] for key in ("gen_x0s", "fake_score_x0s", "real_score_x0s"))
            active = [index for index, value in enumerate(x0s) if value.numel()]
            terms = self._dmd_terms(
                [x0s[index] for index in active], [fakes[index] for index in active],
                [reals[index] for index in active], name, None,
            )
            total = sum((term.mean() for term in terms), x0s[0].double().sum() * 0)
            count = torch.tensor(float(inputs.batch_size), device=get_device(), dtype=torch.float64)
            all_reduce_sum(count)
            losses.append(total * get_world_size() / count)
        meter = get_running_average_meter()
        meter.put_scalar("running/dmd_losses/video", float(losses[0].detach()))
        meter.put_scalar("running/dmd_losses/audio", float(losses[1].detach()))
        meter.put_scalar("running/chain/self_history", sum(inputs.self_history) / inputs.batch_size)
        meter.put_scalar("running/chain/window", float(inputs.window) if isinstance(inputs.window, int)
                         else sum(inputs.window_indices) / inputs.batch_size)
        ctx["gen_loss"] = losses[0] + self.audio_loss_weight * losses[1]
        return ctx


EntryClass = MiniMaxH3StreamingDMD

__all__ = ["MiniMaxH3StreamingDMD", "StreamingDMDInputs", "StreamingDMDRollout", "EntryClass"]
