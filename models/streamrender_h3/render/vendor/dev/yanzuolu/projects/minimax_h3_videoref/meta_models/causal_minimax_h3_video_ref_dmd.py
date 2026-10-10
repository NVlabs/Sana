# SPDX-License-Identifier: Apache-2.0
"""Video-reference DMD conditioning with native bidirectional critics.

The student and both critics share target diffusion coordinates. Generated x0
is passed directly to the critics with its gradient intact. Real corpus video
and critic reference video use the optional score codec, or the student codec
instance when no score codec is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterator

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    CausalMiniMaxH3DMD,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_df import (
    CausalVideoRefMixin,
    VideoRefForwardInput,
    VideoRefLayout,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_ref2va import (
    MiniMaxH3Ref2VABase,
    _native_with_qwen_tags,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.video_ref_conditions import (
    encode_video_ref_targets,
    prepare_video_ref_score_conditions,
)


@dataclass(frozen=True)
class _ScoreReference:
    """Encoded reference rows needed by a critic after SP synchronization."""

    visual_rows: torch.Tensor
    audio_rows: torch.Tensor
    visual_row_anchors: torch.Tensor
    audio_row_anchors: torch.Tensor


@dataclass(frozen=True)
class _ScoreInputs:
    """Native critic conditions and corpus targets in shared H3 coordinates."""

    batch_size: int
    prompt_embeds: list[torch.Tensor]
    text_lens: list[int]
    seqlens: torch.Tensor
    layouts: list[VideoRefLayout]
    native: list[dict[str, Any]]
    reference_plans: list[_ScoreReference]
    token_tags: torch.Tensor
    clean_latents: tuple[list[torch.Tensor], list[torch.Tensor]]
    negative_inputs: _ScoreInputs | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class _VideoRefDMDInputs(VideoRefForwardInput):
    """Keep both static representations inside the engine's retained payload."""

    score_inputs: _ScoreInputs


class CausalVideoRefDMDMixin(CausalVideoRefMixin):
    """Share causal student inputs and native Ref2VA critic conditions.

    ``video_vae`` serves the student and validation. The optional
    ``score_video_vae`` encodes corpus targets and critic depth conditions.
    When absent, those calls reuse the ``video_vae`` instance. ``text_encoder`` is shared
    between a pure-text student call and a batched multimodal critic call.
    ``meta_model.score_processor_path`` selects the native Qwen tokenizer and
    visual processor assets, not separate encoder weights. Training consumes
    nonempty raw video-reference batches with synchronous target/reference
    pixels and target audio. Validation uses the configured student codec.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.score_processor_path = str(config.meta_model.score_processor_path)
        self.score_visual_anchor = float(config.meta_model.get("score_visual_anchor", 0.999))
        self._score_processor: MiniMaxH3Ref2VAPresentationProcessor | None = None

    # Native packing is shared with Ref2VA validation, not with its sampler or
    # text-only/vision encoder dispatch. Student helpers stay on the causal mixin.
    _bidirectional_kwargs = MiniMaxH3Ref2VABase._bidirectional_kwargs
    _bidirectional_forward = MiniMaxH3Ref2VABase._bidirectional_forward

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        batch, models = ctx["batch"], ctx["models"]
        assert batch["reference_video_pixels"], "VideoRef DMD requires a nonempty video-reference batch"
        score_video_vae = models.get("score_video_vae", models["video_vae"])
        score_video_temporal_mapping = score_video_vae.video_temporal_mapping
        if self._score_processor is None:
            self._score_processor = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(
                self.score_processor_path
            )
        student_reference_rng = ctx["rng"].fork("student_reference")
        student_reference = self._encode_video_media(
            models["video_vae"], batch["reference_video_pixels"],
            seeds=[
                student_reference_rng.fork(index).seed
                for index in range(len(batch["reference_video_pixels"]))
            ],
        )
        target_rng = ctx["rng"].fork("score_target")
        target_video = encode_video_ref_targets(
            score_video_vae, batch["video_pixels"],
            seeds=[target_rng.fork(index).seed for index in range(len(batch["video_pixels"]))],
            video_temporal_mapping=score_video_temporal_mapping,
        )
        target_audio = self._encode_audio_media(models["audio_vae"], batch["audio_waveform"])
        prompt_embeds = self._encode_prompts(
            models, batch["text_input_ids"], [int(value) for value in batch["text_lens"]]
        )
        reference_encode_rng = ctx["rng"].fork("score_reference_encode")
        condition_options = (
            {"include_empty_language": True}
            if self.teacher_guidance_scale != (1.0, 1.0)
            else {}
        )
        conditions = prepare_video_ref_score_conditions(
            reference_pixels=batch["reference_video_pixels"],
            prompts=batch["prompts"],
            video_vae=score_video_vae,
            text_encoder=models["text_encoder"],
            processor=self._score_processor,
            encode_seeds=[
                reference_encode_rng.fork(index).seed
                for index in range(len(batch["reference_video_pixels"]))
            ],
            noise_seed=ctx["rng"].fork("score_reference").seed,
            visual_anchor=self.score_visual_anchor,
            video_temporal_mapping=score_video_temporal_mapping,
            **condition_options,
        )
        encoded_batch = {
            key: value for key, value in batch.items()
            if key not in {"video_pixels", "reference_video_pixels", "audio_waveform"}
        }
        encoded_batch.update(
            reference_video_latents=student_reference,
            video_latents=target_video,
            audio_latents=target_audio,
        )
        student = self._build_inputs(encoded_batch, prompt_embeds)
        score_parts = {
            "prompt_embeds": conditions.prompt_embeds,
            "native": [
                _native_with_qwen_tags(
                    presentation=presentation, plan=plan, layout=layout, device=get_device()
                )
                for presentation, plan, layout in zip(
                    conditions.presentations, conditions.reference_plans, student.layouts,
                    strict=True,
                )
            ],
            "references": [
                {field.name: getattr(plan, field.name) for field in fields(_ScoreReference)}
                for plan in conditions.reference_plans
            ],
        }
        if condition_options:
            negative = conditions.negative
            assert negative is not None
            score_parts["negative"] = {
                "prompt_embeds": negative.prompt_embeds,
                "native": [
                    _native_with_qwen_tags(
                        presentation=presentation, plan=plan, layout=layout,
                        device=get_device(),
                    )
                    for presentation, plan, layout in zip(
                        negative.presentations, conditions.reference_plans,
                        student.layouts, strict=True,
                    )
                ],
            }
        # Only plain mappings/tensors cross SP. Decoder pixels, Qwen
        # presentations and complete reference-plan dataclasses stop here.
        ctx["encoded_batch"] = self._to_device(encoded_batch)
        ctx["score_parts"] = self._to_device(score_parts)
        ctx["inputs"] = self._attach_score_inputs(
            student, ctx["encoded_batch"], ctx["score_parts"]
        )
        ctx["neg_inputs"] = None
        return ctx

    @staticmethod
    def _attach_score_inputs(
        student: VideoRefForwardInput,
        encoded_batch: dict[str, Any],
        score_parts: dict[str, Any],
    ) -> _VideoRefDMDInputs:
        video, audio = encoded_batch["video_latents"], encoded_batch["audio_latents"]
        for layout, video_x0, audio_x0, reference in zip(
            student.layouts, video, audio, student.reference_latents, strict=True
        ):
            assert tuple(video_x0.shape) == layout.latent_shape
            assert tuple(audio_x0.shape) == layout.audio_shape
            assert tuple(reference.shape) == layout.reference_latent_shape
        reference_plans = [_ScoreReference(**item) for item in score_parts["references"]]
        clean_latents = (video, audio)

        def score_view(parts: dict[str, Any]) -> _ScoreInputs:
            native = parts["native"]
            text_lens = [int(embed.shape[0]) for embed in parts["prompt_embeds"]]
            assert len(native) == len(text_lens) == student.batch_size
            return _ScoreInputs(
                batch_size=student.batch_size,
                prompt_embeds=parts["prompt_embeds"],
                text_lens=text_lens,
                seqlens=torch.tensor(
                    [int(entry["seq_len"]) for entry in native],
                    dtype=torch.int32, device=student.token_tags.device,
                ),
                layouts=[replace(layout, text_len=length) for layout, length in zip(student.layouts, text_lens, strict=True)],
                native=native,
                reference_plans=reference_plans,
                token_tags=torch.cat([entry["token_tags"] for entry in native]),
                clean_latents=clean_latents,
            )

        score = score_view(score_parts)
        if "negative" in score_parts:
            score = replace(score, negative_inputs=score_view(score_parts["negative"]))
        return _VideoRefDMDInputs(
            **{field.name: getattr(student, field.name) for field in fields(VideoRefForwardInput)},
            score_inputs=score,
        )

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return
        payload = (ctx["encoded_batch"], ctx["inputs"].prompt_embeds, ctx["score_parts"])
        sync = SPDistForward(name="video_ref_dmd2_inputs", comm_shape=True, device=get_device())
        for encoded_batch, prompt_embeds, score_parts in sync(payload):
            sub_ctx = dict(ctx)
            sub_ctx["encoded_batch"] = encoded_batch
            sub_ctx["score_parts"] = score_parts
            student = self._build_inputs(encoded_batch, prompt_embeds)
            sub_ctx["inputs"] = self._attach_score_inputs(student, encoded_batch, score_parts)
            sub_ctx["neg_inputs"] = None
            yield sub_ctx

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = ctx["inputs"]
        ctx["inputs"] = replace(
            inputs, reference_eps=self._sample_reference_noises(inputs, ctx["rng"])
        )
        return super().rollout(ctx)

    def _causal_packed_forward(
        self, model: Any, inputs: VideoRefForwardInput, *,
        reference_eps: list[torch.Tensor] | None = None, **kwargs: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if reference_eps is None:
            reference_eps = inputs.reference_eps
        assert reference_eps is not None
        return super()._causal_packed_forward(
            model, inputs, reference_eps=reference_eps, **kwargs
        )

    def _critic_inputs(
        self, ctx: dict[str, Any], inputs: _VideoRefDMDInputs
    ) -> _ScoreInputs:
        """Return the critic conditions for the current training phase."""
        return inputs.score_inputs

    def _teacher_negative_inputs(
        self, ctx: dict[str, Any], inputs: _ScoreInputs
    ) -> _ScoreInputs:
        """Drop language while preserving the encoded reference conditions."""
        assert inputs.negative_inputs is not None, "teacher CFG requires prepared negative conditions"
        return inputs.negative_inputs

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_fake(self, ctx: dict[str, Any]) -> dict[str, Any]:
        # Prepare critic targets through the selected algorithm while keeping
        # the retained student payload intact.
        phase_ctx = dict(ctx, inputs=self._critic_inputs(ctx, ctx["inputs"]))
        phase_ctx = super().prepare_fake(phase_ctx)
        phase_ctx["inputs"] = ctx["inputs"]
        ctx.update(phase_ctx)
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs, rng = ctx["inputs"], ctx["rng"]
        device = get_device()
        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        gen_timesteps = self._sample_timesteps(inputs, self.sampling_timesteps, rng)
        gen_index = self.sampling_timesteps.index(gen_timesteps)
        assert bool((gen_index >= 0).all())
        score_inputs = inputs.score_inputs
        score_timesteps, audio_score_timesteps = self._sample_paired_timesteps(
            score_inputs.batch_size, score_inputs.seqlens,
            self.score_timesteps, self.audio_score_timesteps, rng,
        )
        trajectory = ctx["trajectory_xts"]
        ctx.update(
            gen_timesteps=gen_timesteps,
            gen_audio_timesteps=self.audio_sampling_timesteps.timesteps.to(device)[gen_index],
            gen_index=gen_index,
            gen_xts=(
                [trajectory[int(gen_index[i])][0][i] for i in range(inputs.batch_size)],
                [trajectory[int(gen_index[i])][1][i] for i in range(inputs.batch_size)],
            ),
            score_timesteps=score_timesteps,
            audio_score_timesteps=audio_score_timesteps,
            gen_inputs=inputs,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def score(self, ctx: dict[str, Any]) -> dict[str, Any]:
        score_ctx = dict(ctx, gen_inputs=self._critic_inputs(ctx, ctx["gen_inputs"]))
        # No codec bridge or detach. Native fake and teacher score the same
        # generated target tensors with their shared multimodal conditions.
        score_ctx = CausalMiniMaxH3DMD.score(self, score_ctx)
        for key in ("gen_x0s", "fake_score_x0s", "real_score_x0s"):
            ctx[key] = score_ctx[key]
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        score_ctx = dict(ctx, gen_inputs=self._critic_inputs(ctx, ctx["gen_inputs"]))
        score_ctx = super().gen_loss(score_ctx)
        ctx["gen_loss"] = score_ctx["gen_loss"]
        return ctx


class CausalMiniMaxH3VideoRefDMD(CausalVideoRefDMDMixin, CausalMiniMaxH3DMD):
    """Distill a causal VideoRef student with the pure DMD objective.

    Use ``MiniMaxH3CausalX0DiTSP`` for ``backbone`` and
    ``MiniMaxH3X0DiTSP`` for both ``fake_model`` and ``tea_model``.
    """


EntryClass = CausalMiniMaxH3VideoRefDMD

__all__ = ["CausalVideoRefDMDMixin", "CausalMiniMaxH3VideoRefDMD", "EntryClass"]
