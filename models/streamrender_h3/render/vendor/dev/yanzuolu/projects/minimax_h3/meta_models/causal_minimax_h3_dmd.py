# SPDX-License-Identifier: Apache-2.0
"""CausalMiniMaxH3DMD: every DMD algorithm primitive for chunk-causal MiniMax-H3 T2AV.

One project == one engine, so this class has no genericity: the public surface
is the ``engines/dmd.py`` ctx primitives, and everything else is a private
recipe. The *mechanism* those primitives are written against -- layout
derivation, row packing, the cached causal rollout, the bidirectional forward,
the loss reduction, and ``validate`` -- lives in ``CausalMiniMaxH3Base``
(``causal_minimax_h3_base.py``); this file adds only the DMD algorithm on top of
it and inherits the rest unchanged.

ctx contract (see engines/dmd.py for the calling sequence)::

    prepare_inputs   reads batch/models/rng      writes inputs, neg_inputs
    sync_inputs      reads inputs/neg_inputs     yields synchronized ctx copies
    rollout          reads models/inputs/rng     writes rollout_x0s, trajectory_xts  (no-grad)
    prepare_fake     reads rollout_x0s/rng       writes fake_timesteps,
                                                 fake_audio_timesteps, fake_noises,
                                                 fake_noisy_latents, fake_inputs
    fake_forward     reads models/fake_*         writes fake_pred
    fake_loss        reads fake_pred/fake_*      writes fake_loss
    prepare_gen      reads trajectory_xts/rng    writes gen_timesteps, gen_audio_timesteps,
                                                 gen_index, gen_xts, score_timesteps,
                                                 audio_score_timesteps, gen_inputs
    gen_forward      reads models/gen_*          writes gen_pred                     (graph attached)
    score            reads models/gen_pred/...   writes gen_x0s, fake_score_x0s, real_score_x0s
    gen_loss         reads gen_x0s/*_score_x0s   writes gen_loss

Three DMD-specific facts sit on top of the base's four H3 ones.

1. **The three model nodes are all x0 models.** ``backbone`` wraps the causal
   DiT, ``fake_model`` and ``tea_model`` wrap the bidirectional one. Every
   schedule is therefore asserted ``pred_type: x_0`` -- ``schedule`` by the
   base, ``fake_schedule`` here -- which makes ``convert_from_pred`` an identity
   on x0 and makes the H3 velocity sign error unrepresentable.

2. **Scoring adds fresh Gaussian noise.** The meta model samples paired score
   timesteps and noise for each modality. The schedule applies forward diffusion
   to the generated x0, while ``diffusion.sampler`` advances generated chunks.

3. **The DMD normalizer is per sample AND per modality.** ``|x0 - real|.mean()``
   computed jointly would let video's scale rescale audio and make
   ``audio_loss_weight`` meaningless, at H3's 256:1 element ratio.

``neg_inputs`` remains ``None`` for the student rollout. Optional teacher CFG
uses an empty-text critic view while retaining the same noisy media and score
timesteps. The default ``teacher_guidance_scale: [1.0, 1.0]`` uses one teacher
forward without drawing a scale or constructing negative conditions.

Expected configuration shape::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd
      class_name: CausalMiniMaxH3DMD
      audio_loss_weight: 1.0
      fake_loss_type: x_0
      fake_grad_enabled: false
      real_grad_enabled: false
      fake_use_trajectory: false
      teacher_guidance_scale: [1.0, 1.0]  # per-sample scale shared by video and audio
      dmd_loss:
        type: dmd
        norm_clip_min: 1.0e-5
        phuber_c: 0.001
        alpha: 1.0

``config.diffusion`` provides nine components: the base's ``sampling_timesteps``,
``audio_sampling_timesteps``, ``schedule`` and ``sampler``, plus
``fake_training_timesteps`` and ``audio_fake_training_timesteps``,
``score_timesteps`` and ``audio_score_timesteps``, and ``fake_schedule``. Each
audio component differs from its video partner only in ``shift``. There is no
``tea_schedule``: the teacher returns x0, so nothing is left to convert.

``models`` must provide ``backbone``, ``fake_model``, ``tea_model`` and
``text_encoder``, plus ``video_vae``/``audio_vae`` once validation is enabled
(and ``backbone_ema`` when ``validation.validate_ema``).
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from dev.yanzuolu.common import media  # noqa: F401  (re-export; see the import block below)
from dev.yanzuolu.common.diffusion.schedule import PredictionType
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.meter import get_running_average_meter
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase

# Re-exported, not merely imported: ``ForwardInput`` and ``_RolloutX0s`` were
# defined here before the mechanism moved to the base, and DMD2 / the zelda
# variant / minimax_h3_base / the tests all still reach them through this
# module. ``media`` is re-exported for the same reason -- ``validate`` now lives
# in the base, but callers still patch it through this module's namespace.
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import (
    CausalMiniMaxH3Base,
    ForwardInput,
    _Chunk,
    _Layout,
    _PATCH_SIZE,
    _RolloutX0s,
)

__all__ = [
    "CausalMiniMaxH3Base",
    "CausalMiniMaxH3DMD",
    "ForwardInput",
    "_Chunk",
    "_Layout",
    "_PATCH_SIZE",
    "_RolloutX0s",
]

logger = get_logger()


@dataclass(frozen=True)
class _TrajectoryRolloutX0s(_RolloutX0s):
    """Final rollout for GEN plus random-stage x0s for phase-local FAKE."""

    fake_video: list[torch.Tensor]
    fake_audio: list[torch.Tensor]


class CausalMiniMaxH3DMD(CausalMiniMaxH3Base):
    """See the module docstring for the ctx contract."""

    def __init__(self, config: Any) -> None:
        # The base owns the mechanism and the shared diffusion nodes; everything
        # below is the DMD algorithm's own configuration.
        super().__init__(config)
        mm = config.meta_model

        self.audio_loss_weight = float(mm.get("audio_loss_weight", 1.0))
        self.fake_grad_enabled = bool(mm.get("fake_grad_enabled", False))
        self.real_grad_enabled = bool(mm.get("real_grad_enabled", False))
        self.fake_use_trajectory = bool(mm.get("fake_use_trajectory", False))
        guidance_min, guidance_max = mm.get("teacher_guidance_scale", [1.0, 1.0])
        self.teacher_guidance_scale = (float(guidance_min), float(guidance_max))

        assert mm.fake_loss_type in {"x_0", "x_T", "v_cos", "v_lerp"}, (
            f"unsupported fake_loss_type: {mm.fake_loss_type}"
        )
        self.fake_loss_type = mm.fake_loss_type
        assert mm.dmd_loss.type in {"dmd", "sim", "sid"}, (
            f"unsupported dmd_loss.type: {mm.dmd_loss.type}"
        )
        self.dmd_loss_type = mm.dmd_loss.type
        norm_clip_min = mm.dmd_loss.get("norm_clip_min", 1.0e-5)
        self.norm_clip_min = None if norm_clip_min is None else float(norm_clip_min)
        self.norm_per_chunk = bool(mm.dmd_loss.get("norm_per_chunk", False))
        self.chunk_wise_weighting = (
            dict(mm.chunk_wise_weighting) if mm.get("chunk_wise_weighting") else None
        )
        self.phuber_c = float(mm.dmd_loss.get("phuber_c", 0.001))
        self.sid_alpha = float(mm.dmd_loss.get("alpha", 1.0))
        assert self.phuber_c > 0, "dmd_loss.phuber_c must be positive"

        # Indexed out of the base's single build_diffusion result, not out of a
        # second build of the same config block: two builds would give the
        # sampler and this file schedule objects that only look identical.
        self.fake_training_timesteps = self._diffusion["fake_training_timesteps"]
        self.audio_fake_training_timesteps = self._diffusion[
            "audio_fake_training_timesteps"
        ]
        self.score_timesteps = self._diffusion["score_timesteps"]
        self.audio_score_timesteps = self._diffusion["audio_score_timesteps"]
        self.fake_schedule = self._diffusion["fake_schedule"]
        # No tea_schedule: with an x0 teacher there is nothing left to convert,
        # so a node here would be built and never read.

        # The base pinned ``schedule``; ``fake_schedule`` is this algorithm's own
        # and needs the same guarantee. Both the fake critic and the teacher are
        # MiniMaxH3X0Model wrappers, so with any other pred_type
        # convert_from_pred would reinterpret an x0 as a velocity and be wrong by
        # a plausible-looking amount.
        assert self.fake_schedule.pred_type == PredictionType.x_0, (
            f"diffusion.fake_schedule.pred_type must be x_0 because every "
            f"MiniMax-H3 model node returns x0, got {self.fake_schedule.pred_type}"
        )

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @contextmanager
    def gen_model_context(self, models: dict[str, Any]) -> Iterator[None]:
        """Freeze fake parameters across GEN when score gradients flow through it.

        Gradients to the fake model's *inputs* are what DMD needs; its parameter
        gradients would be accumulated into the FAKE phase's optimizer state.
        """
        frozen = [
            models[name]
            for name, enabled in (
                ("fake_model", self.fake_grad_enabled),
                # Symmetric on purpose: the teacher is frozen by trial config
                # today, but the property is enforced here for the fake model
                # and must not be held by config alone for the teacher.
                ("tea_model", self.real_grad_enabled),
            )
            if enabled
        ]
        if not frozen:
            yield
            return

        trainable = [
            {name for name, param in model.named_parameters() if param.requires_grad}
            for model in frozen
        ]
        for model in frozen:
            for param in model.parameters():
                param.requires_grad_(False)
        try:
            yield
        finally:
            for model, names in zip(frozen, trainable):
                for name, param in model.named_parameters():
                    param.requires_grad_(name in names)

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: batch, models. Writes: inputs, neg_inputs."""
        batch = ctx["batch"]
        prompt_embeds = self._encode_prompts(
            ctx["models"], batch["text_input_ids"], [int(v) for v in batch["text_lens"]]
        )
        ctx["inputs"] = self._build_inputs(batch, prompt_embeds)
        # Student rollout uses conditional inputs. Optional teacher CFG gets
        # negative conditions through its score hook.
        ctx["neg_inputs"] = None
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Reads: batch, inputs. Yields exactly one ctx per SP-group source rank.

        What crosses the wire is the raw batch plus the text embeddings, not
        ``ForwardInput``: ``SPDistForward`` recurses lists/tuples/mappings/tensors
        natively but would pickle a dataclass whole, so shipping ``ForwardInput``
        would send its ``_Layout`` tensors through ``pickle`` instead of NCCL.
        Every rank then re-derives the layout, which is pure and cheap.
        """
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return

        # The dataset hands out CPU tensors and only _build_inputs moves them, but
        # the exchange is an NCCL broadcast: a CPU leaf here raises
        # "No backend type associated with device type cpu".
        payload = (self._to_device(ctx["batch"]), ctx["inputs"].prompt_embeds)
        sync = SPDistForward(name="dmd_inputs", comm_shape=True, device=get_device())
        for batch, prompt_embeds in sync(payload):
            sub_ctx = dict(ctx)
            sub_ctx["inputs"] = self._build_inputs(batch, prompt_embeds)
            sub_ctx["neg_inputs"] = None
            yield sub_ctx

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, inputs, rng. Writes: rollout_x0s, trajectory_xts. (no-grad)"""
        trajectory_x0_chunks: list[
            tuple[list[torch.Tensor], list[torch.Tensor]]
        ] | None = [] if self.fake_use_trajectory else None
        rollout_x0s, trajectory_xts = self._rollout_latents(
            ctx["models"]["backbone"],
            ctx["inputs"],
            ctx["rng"],
            keep_trajectory=True,
            trajectory_x0_chunks=trajectory_x0_chunks,
        )
        if trajectory_x0_chunks is not None:
            inputs = ctx["inputs"]
            stage_count = int(self.sampling_timesteps.timesteps.numel())
            chunk_count = len(inputs.layouts[0].chunks)
            assert len(trajectory_x0_chunks) == chunk_count * stage_count
            trajectory_timesteps = self._sample_timesteps(
                inputs, self.sampling_timesteps, ctx["rng"]
            )
            indices = self.sampling_timesteps.index(trajectory_timesteps)
            assert bool((indices >= 0).all()), (
                "trajectory timestep is not on the video sampling grid; "
                "index() returned -1"
            )
            stage_indices = [int(value) for value in indices.tolist()]
            fake_video = [
                torch.cat(
                    [
                        trajectory_x0_chunks[
                            chunk_index * stage_count + stage_indices[index]
                        ][0][index]
                        for chunk_index in range(chunk_count)
                    ],
                    dim=1,
                )
                for index in range(inputs.batch_size)
            ]
            fake_audio = [
                torch.cat(
                    [
                        trajectory_x0_chunks[
                            chunk_index * stage_count + stage_indices[index]
                        ][1][index]
                        for chunk_index in range(chunk_count)
                    ],
                    dim=2,
                )
                for index in range(inputs.batch_size)
            ]
            fake_video = self._renorm_video_chunks(
                inputs, fake_video, rollout_x0s.video_anchors
            )
            # Keep final-stage x0s on the shared payload for GEN. prepare_fake
            # swaps only its phase-local ctx to these random-stage predictions.
            ctx["rollout_x0s"] = _TrajectoryRolloutX0s(
                video=rollout_x0s.video,
                audio=rollout_x0s.audio,
                video_eps=rollout_x0s.video_eps,
                audio_eps=rollout_x0s.audio_eps,
                fake_video=fake_video,
                fake_audio=fake_audio,
                video_anchors=rollout_x0s.video_anchors,
            )
        else:
            ctx["rollout_x0s"] = rollout_x0s
        ctx["trajectory_xts"] = trajectory_xts
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: rollout_x0s, inputs, rng. Writes: fake_timesteps,
        fake_audio_timesteps, fake_noises, fake_noisy_latents, fake_inputs."""
        inputs = ctx["inputs"]
        rng = ctx["rng"]
        rollout = ctx["rollout_x0s"]
        if self.fake_use_trajectory:
            trajectory_rollout: _TrajectoryRolloutX0s = rollout
            rollout = _RolloutX0s(
                video=trajectory_rollout.fake_video,
                audio=trajectory_rollout.fake_audio,
                video_eps=trajectory_rollout.video_eps,
                audio_eps=trajectory_rollout.audio_eps,
                video_anchors=trajectory_rollout.video_anchors,
            )
            ctx["rollout_x0s"] = rollout

        # A continuous (T: float) training-timestep node samples in float64, and
        # A(t)*x_0 would promote the whole noisy pack with it -- silently, since
        # _embed casts to fp32 anyway. Pin it where it meets the latents.
        fake_timesteps, fake_audio_timesteps = (
            value.to(torch.float32)
            for value in self._sample_paired_timesteps(
                inputs.batch_size, inputs.seqlens,
                self.fake_training_timesteps, self.audio_fake_training_timesteps, rng,
            )
        )
        fake_noises = self._sample_noises(inputs, rng)
        fake_noisy_latents = (
            self.fake_schedule.forward(rollout.video, fake_noises[0], fake_timesteps),
            self.fake_schedule.forward(rollout.audio, fake_noises[1], fake_audio_timesteps),
        )

        ctx["fake_timesteps"] = fake_timesteps
        ctx["fake_audio_timesteps"] = fake_audio_timesteps
        ctx["fake_noises"] = fake_noises
        ctx["fake_noisy_latents"] = fake_noisy_latents
        ctx["fake_inputs"] = inputs
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, fake_inputs, fake_noisy_latents, fake_timesteps. Writes: fake_pred."""
        video_xts, audio_xts = ctx["fake_noisy_latents"]
        ctx["fake_pred"] = self._bidirectional_forward(
            ctx["models"]["fake_model"],
            ctx["fake_inputs"],
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_timesteps=ctx["fake_timesteps"],
            audio_timesteps=ctx["fake_audio_timesteps"],
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: fake_pred, fake_noisy_latents, fake_noises, fake_timesteps,
        fake_audio_timesteps, rollout_x0s. Writes: fake_loss."""
        rollout = ctx["rollout_x0s"]
        video_t = ctx["fake_timesteps"]
        audio_t = ctx["fake_audio_timesteps"]
        video_xts, audio_xts = ctx["fake_noisy_latents"]
        video_noises, audio_noises = ctx["fake_noises"]
        video_pred, audio_pred = ctx["fake_pred"]

        # Each modality converts at the timestep it was noised with; crossing
        # them would build the target from a different t than the input.
        video_pred = self._to_loss_pred(video_pred, video_xts, video_t)
        audio_pred = self._to_loss_pred(audio_pred, audio_xts, audio_t)
        video_target = self.fake_schedule.convert_to_pred(
            x_0=rollout.video, x_T=video_noises, t=video_t, pred_type=self.fake_loss_type
        )
        audio_target = self.fake_schedule.convert_to_pred(
            x_0=rollout.audio, x_T=audio_noises, t=audio_t, pred_type=self.fake_loss_type
        )

        ctx["fake_loss"] = self._loss_fn(
            video_pred, audio_pred,
            inputs=ctx["fake_inputs"],
            video_targets=video_target, audio_targets=audio_target,
            key_prefix="fake_losses",
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, trajectory_xts, rng. Writes: gen_timesteps, gen_index,
        gen_xts, score_timesteps, audio_score_timesteps, gen_inputs."""
        inputs = ctx["inputs"]
        rng = ctx["rng"]
        device = get_device()

        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        gen_timesteps = self._sample_timesteps(inputs, self.sampling_timesteps, rng)
        gen_index = self.sampling_timesteps.index(gen_timesteps)
        assert bool((gen_index >= 0).all()), (
            "gen timestep is not on the video sampling grid; index() returned -1"
        )
        # Same step index, the audio grid's own shift: this is H3's native
        # "steps aligned, sigmas not" sampling.
        audio_gen_timesteps = self.audio_sampling_timesteps.timesteps.to(device)[gen_index]
        # One uniform, each modality's own shift -- the same pairing the sampler
        # produces, so the frozen teacher scores at a (sigma_v, sigma_a) it has
        # actually seen.
        score_timesteps, audio_score_timesteps = self._sample_paired_timesteps(
            inputs.batch_size, inputs.seqlens,
            self.score_timesteps, self.audio_score_timesteps, rng,
        )

        trajectory_xts = ctx["trajectory_xts"]
        gen_xts = (
            [trajectory_xts[int(gen_index[i])][0][i] for i in range(inputs.batch_size)],
            [trajectory_xts[int(gen_index[i])][1][i] for i in range(inputs.batch_size)],
        )
        ctx["gen_timesteps"] = gen_timesteps
        ctx["gen_audio_timesteps"] = audio_gen_timesteps
        ctx["gen_index"] = gen_index
        ctx["gen_xts"] = gen_xts
        ctx["score_timesteps"] = score_timesteps
        ctx["audio_score_timesteps"] = audio_score_timesteps
        ctx["gen_inputs"] = inputs
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, gen_inputs, gen_xts, gen_timesteps, rollout_x0s.
        Writes: gen_pred. (graph attached)"""
        rollout = ctx["rollout_x0s"]
        inputs = ctx["gen_inputs"]
        video_xts, audio_xts = ctx["gen_xts"]
        # DMD scores one point of the rollout trajectory, so all of a sample's
        # chunks sit at that same step -- the per-chunk table is uniform here.
        chunks = [len(layout.chunks) for layout in inputs.layouts]
        ctx["gen_pred"] = self._causal_packed_forward(
            ctx["models"]["backbone"],
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_context=rollout.video,
            audio_context=rollout.audio,
            video_eps=rollout.video_eps,
            audio_eps=rollout.audio_eps,
            video_timesteps=[
                ctx["gen_timesteps"][i].expand(n) for i, n in enumerate(chunks)
            ],
            audio_timesteps=[
                ctx["gen_audio_timesteps"][i].expand(n) for i, n in enumerate(chunks)
            ],
        )
        return ctx

    def _renorm_gen_video(
        self, ctx: dict[str, Any], video: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Use the rollout's fixed anchors for differentiable GEN predictions."""
        if not self.video_chunk_renorm:
            return video
        return self._renorm_video_chunks(
            ctx["gen_inputs"], video, ctx["rollout_x0s"].video_anchors
        )

    def _teacher_negative_inputs(
        self, ctx: dict[str, Any], inputs: ForwardInput
    ) -> ForwardInput:
        """Return the teacher's caption-free conditions for this score call."""
        return self._empty_text_inputs(inputs)

    @staticmethod
    def _teacher_cfg_combine(
        conditional: list[torch.Tensor],
        unconditional: list[torch.Tensor],
        scales: list[float] | torch.Tensor,
    ) -> list[torch.Tensor]:
        """Apply detached per-sample scales while preserving prediction gradients."""
        if isinstance(scales, torch.Tensor):
            scales = scales.detach()
        return [
            uncond + scale * (cond - uncond)
            for cond, uncond, scale in zip(
                conditional, unconditional, scales, strict=True
            )
        ]

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def score(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: models, gen_pred, gen_xts, score_timesteps, rng.
        Writes: gen_x0s, fake_score_x0s, real_score_x0s."""
        models = ctx["models"]
        rng = ctx["rng"]
        inputs = ctx["gen_inputs"]
        video_pred, audio_pred = ctx["gen_pred"]
        video_pred = self._renorm_gen_video(ctx, video_pred)
        video_xts, audio_xts = ctx["gen_xts"]
        score_t = ctx["score_timesteps"]
        audio_score_t = ctx["audio_score_timesteps"]

        # Every schedule is pinned to pred_type x_0 in __init__, so the model
        # output already IS x0 and convert_from_pred would be an identity plus a
        # discarded full-size divide by B(t).
        gen_x0s = (video_pred, audio_pred)

        # Draw score noise independently of generation stepping. The schedule
        # keeps the generator graph attached through forward diffusion.
        video_noises, audio_noises = self._sample_noises(inputs, rng)
        score_xts = (
            [
                value.to(source.dtype)
                for value, source in zip(
                    self.schedule.forward(video_pred, video_noises, score_t),
                    video_xts, strict=True,
                )
            ],
            [
                value.to(source.dtype)
                for value, source in zip(
                    self.schedule.forward(audio_pred, audio_noises, audio_score_t),
                    audio_xts, strict=True,
                )
            ],
        )

        fake_ctx = nullcontext() if self.fake_grad_enabled else torch.no_grad()
        with fake_ctx:
            fake_score_x0s = self._bidirectional_forward(
                models["fake_model"], inputs,
                video_xts=score_xts[0], audio_xts=score_xts[1],
                video_timesteps=score_t, audio_timesteps=audio_score_t,
            )

        real_ctx = nullcontext() if self.real_grad_enabled else torch.no_grad()
        with real_ctx:
            real_score_x0s = self._bidirectional_forward(
                models["tea_model"], inputs,
                video_xts=score_xts[0], audio_xts=score_xts[1],
                video_timesteps=score_t, audio_timesteps=audio_score_t,
            )
            guidance_min, guidance_max = self.teacher_guidance_scale
            if (guidance_min, guidance_max) != (1.0, 1.0):
                negative_inputs = self._teacher_negative_inputs(ctx, inputs)
                uncond_score_x0s = self._bidirectional_forward(
                    models["tea_model"], negative_inputs,
                    video_xts=score_xts[0], audio_xts=score_xts[1],
                    video_timesteps=score_t, audio_timesteps=audio_score_t,
                )
                if guidance_min == guidance_max:
                    scales: list[float] | torch.Tensor = [
                        guidance_min
                    ] * inputs.batch_size
                else:
                    scales = torch.rand(
                        (inputs.batch_size,), device=inputs.token_tags.device,
                        generator=self._generator(rng),
                    )
                    scales = (guidance_min + scales * (guidance_max - guidance_min)).detach()
                real_score_x0s = (
                    self._teacher_cfg_combine(
                        real_score_x0s[0], uncond_score_x0s[0], scales
                    ),
                    self._teacher_cfg_combine(
                        real_score_x0s[1], uncond_score_x0s[1], scales
                    ),
                )

        ctx["gen_x0s"] = gen_x0s
        ctx["fake_score_x0s"] = fake_score_x0s
        ctx["real_score_x0s"] = real_score_x0s
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: gen_x0s, fake_score_x0s, real_score_x0s. Writes: gen_loss."""
        inputs = ctx["gen_inputs"]
        video_terms = self._dmd_terms(
            ctx["gen_x0s"][0], ctx["fake_score_x0s"][0], ctx["real_score_x0s"][0],
            "video", inputs,
        )
        audio_terms = self._dmd_terms(
            ctx["gen_x0s"][1], ctx["fake_score_x0s"][1], ctx["real_score_x0s"][1],
            "audio", inputs,
        )
        ctx["gen_loss"] = self._loss_fn(
            video_terms, audio_terms, inputs=inputs, key_prefix="dmd_losses"
        )
        return ctx

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _to_loss_pred(
        self,
        pred: list[torch.Tensor],
        xts: list[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.fake_loss_type == PredictionType.x_0:
            # Identity by construction: __init__ pins the schedule to x_0, so the
            # round trip would only add a discarded full-size divide by B(t).
            return pred
        pred_x_0, pred_x_T = self.fake_schedule.convert_from_pred(pred, xts, timesteps)
        return self.fake_schedule.convert_to_pred(
            x_0=pred_x_0, x_T=pred_x_T, t=timesteps, pred_type=self.fake_loss_type
        )

    def _dmd_terms(
        self,
        x0s: list[torch.Tensor],
        fakes: list[torch.Tensor],
        reals: list[torch.Tensor],
        modality: str,
        inputs: ForwardInput,
    ) -> list[torch.Tensor]:
        """Per-sample DMD surrogate for one modality.

        The normalizer is per sample AND per modality on purpose: one joint
        normalizer would let video's scale set audio's, which would silently
        undo ``audio_loss_weight``.

        ``norm_per_chunk`` narrows it further, to per chunk as well, so a
        chunk's scale cannot be set by drift accumulated in later chunks --
        LongLive2.0's per-block normalization, and the same knob wan exposes.
        """
        terms = []
        for index, (x0, fake, real) in enumerate(zip(x0s, fakes, reals)):
            x0, fake, real = x0.double(), fake.double(), real.double()
            if self.dmd_loss_type == "sim":
                difference = real - fake
                norm = torch.sqrt((difference**2).sum() + self.phuber_c**2)
                term = difference * (fake - x0) / norm
            else:
                if self.norm_per_chunk:
                    norm = torch.empty_like(x0)
                    for chunk in inputs.layouts[index].chunks:
                        chunk_slice = self._chunk_slice(chunk, modality)
                        norm[chunk_slice] = torch.abs(
                            x0[chunk_slice] - real[chunk_slice]
                        ).mean()
                else:
                    norm = torch.abs(x0 - real).mean()
                if self.norm_clip_min is not None:
                    norm = norm.clamp_min(self.norm_clip_min)
                if self.dmd_loss_type == "dmd":
                    term = (real - fake) * (fake - x0) / norm.detach()
                else:
                    term = (
                        (real - fake)
                        * ((real - x0) - self.sid_alpha * (real - fake))
                        / norm.detach()
                    )
            get_running_average_meter().put_scalar(
                f"running/dmd_loss/{modality}_norm", norm.mean().item()
            )
            terms.append(term)
        return terms
