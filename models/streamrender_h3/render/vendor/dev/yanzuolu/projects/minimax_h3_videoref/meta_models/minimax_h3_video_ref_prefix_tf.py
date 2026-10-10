# SPDX-License-Identifier: Apache-2.0
"""Teacher forcing with a complete reference prefix and causal target chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterator

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import combine_seed, local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import (
    _RolloutX0s,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_tf import (
    CausalMiniMaxH3TF,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
    video_temporal_mapping_from_model_config,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_latent import (
    causal_video_ref_target_shapes,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_df import (
    CausalVideoRefMixin,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_ref2va import (
    _Ref2VALayout,
    _native_with_qwen_tags,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.prefix_forward import (
    PrefixForwardMixin,
    PrefixTFInputs,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
    encode_ref2va_presentations,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import (
    EncodedReferencePlan,
    encode_ref_block_plan,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.video_ref_conditions import (
    _decoded_video,
    encode_video_ref_references,
    encode_video_ref_targets,
)

_REFERENCE_FIELDS = (
    "visual_rows",
    "audio_rows",
    "visual_row_anchors",
    "audio_row_anchors",
)


class MiniMaxH3VideoRefPrefixTF(PrefixForwardMixin, CausalMiniMaxH3TF):
    """Fit all target chunks behind one text, Qwen-vision and reference prefix.

    The prefix attends only itself. Noisy target chunks attend the prefix,
    earlier clean target chunks, and themselves. All noisy chunks of a sample
    share one video/audio timestep pair. Sampling recomputes the prefix and
    history at every denoising step because prefix text follows the video time.

    ``models.backbone`` must provide the causal H3 X0 forward interface, and
    ``models.text_encoder`` must support native multimodal Qwen presentations.
    ``data.args.processor_path`` locates the corresponding Qwen processor.
    ``data.args.chunk_size`` and ``data.args.independent_first_chunk`` select
    chunks used for both training and validation.
    Dataset samples supply the logical ``prefix_plan`` and processed
    ``ref_presentation``. This class resolves physical rows after Qwen encoding.
    Video/audio codecs remain ordinary ``models.video_vae``/``audio_vae`` nodes.
    The video codec's temporal descriptor supplies the dataset geometry and
    reference/target time coordinates used during training and validation.

    ``cfg_fitting_scale`` defaults to one. Other positive scales fit
    (conditional + (scale - 1) * detached_negative) / scale against x0.
    The negative branch removes only the language prompt and retains reference
    vision, reference latents and the same noisy targets and clean history.
    Validation guidance remains independently configured.
    """

    cfg_fitting_scale = 1.0
    validation_mode = "prefix"
    _validation_requests = staticmethod(CausalVideoRefMixin._validation_requests)
    _validation_request_prompt = staticmethod(CausalVideoRefMixin._validation_request_prompt)

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.cfg_fitting_scale = float(config.meta_model.get("cfg_fitting_scale", 1.0))
        if not math.isfinite(self.cfg_fitting_scale) or self.cfg_fitting_scale <= 0:
            raise ValueError("meta_model.cfg_fitting_scale must be finite and positive")
        self.video_temporal_mapping = video_temporal_mapping_from_model_config(config.models.video_vae)
        configured_mapping = config.data.args.get("video_temporal_mapping")
        if (
            configured_mapping is not None
            and VideoTemporalMapping.from_dict(configured_mapping) != self.video_temporal_mapping
        ):
            raise ValueError("data.args.video_temporal_mapping must match models.video_vae")
        config.data.args.video_temporal_mapping = self.video_temporal_mapping.to_dict()
        self.processor_path = str(config.data.args.processor_path)
        self.visual_anchor = float(config.meta_model.get("visual_anchor", 0.999))
        self._processor: MiniMaxH3Ref2VAPresentationProcessor | None = None
        self._validation_reference_cache: tuple[Any, ...] | None = None

    def _presentation_processor(self) -> MiniMaxH3Ref2VAPresentationProcessor:
        if self._processor is None:
            self._processor = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(
                self.processor_path
            )
        return self._processor

    @staticmethod
    def _reference_payload(plan: EncodedReferencePlan) -> dict[str, torch.Tensor]:
        return {name: getattr(plan, name) for name in _REFERENCE_FIELDS}

    def _inputs_from_payload(self, payload: dict[str, Any]) -> PrefixTFInputs:
        return self._prefix_inputs_from_payload(payload)

    def _native_conditions(
        self, presentations: Sequence[Any], references: Sequence[EncodedReferencePlan],
        video_shapes: Sequence[Sequence[int]], audio_shapes: Sequence[Sequence[int]],
    ) -> list[dict[str, Any]]:
        return [
            _native_with_qwen_tags(
                presentation=presentation, plan=plan,
                layout=_Ref2VALayout(
                    tuple(video_shape), tuple(audio_shape),
                    video_temporal_mapping=self.video_temporal_mapping,
                ),
                device=get_device(),
            )
            for presentation, plan, video_shape, audio_shape in zip(
                presentations, references, video_shapes, audio_shapes, strict=True,
            )
        ]

    def _set_training_inputs(self, ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        ctx["encoded_batch"] = payload
        ctx["inputs"] = self._inputs_from_payload(payload)
        ctx["clean_latents"] = (payload["video_latents"], payload["audio_latents"])
        if self.cfg_fitting_scale != 1.0:
            ctx["negative_inputs"] = self._inputs_from_payload(dict(
                payload, prompt_embeds=payload["negative_prompt_embeds"],
                native=payload["negative_native"],
            ))
        return ctx

    def _check_video_temporal_mapping(
        self, video_vae: Any, prefix_plans: Sequence[dict[str, Any]] = (),
    ) -> VideoTemporalMapping:
        mapping = self.video_temporal_mapping
        runtime_mapping = video_vae.video_temporal_mapping
        if runtime_mapping != mapping:
            raise ValueError("runtime video_vae temporal mapping disagrees with the configured codec")
        for plan in prefix_plans:
            declared = VideoTemporalMapping.from_dict(
                plan.get("video_temporal_mapping", H3_VIDEO_TEMPORAL_MAPPING.to_dict())
            )
            if declared != mapping:
                raise ValueError("dataset prefix plan temporal mapping disagrees with video_vae")
        return mapping

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Encode the dataset's prepared presentations and raw media for TF."""
        batch, models, rng = ctx["batch"], ctx["models"], ctx["rng"]
        mapping = self._check_video_temporal_mapping(models["video_vae"], batch["prefix_plan"])
        targets = encode_video_ref_targets(
            models["video_vae"], batch["video_pixels"],
            seeds=[rng.fork("target_encode").fork(index).seed
                   for index in range(len(batch["video_pixels"]))],
            video_temporal_mapping=mapping,
        )
        audio = CausalVideoRefMixin._encode_audio_media(
            models["audio_vae"], batch["audio_waveform"]
        )
        references = encode_video_ref_references(
            reference_pixels=batch["reference_video_pixels"],
            video_vae=models["video_vae"],
            encode_seeds=[rng.fork("reference_encode").fork(index).seed
                          for index in range(len(targets))],
            noise_seed=rng.fork("reference_noise").seed,
            visual_anchor=self.visual_anchor,
            video_temporal_mapping=mapping,
        )
        presentations = batch["ref_presentation"]
        negative_presentations = []
        if self.cfg_fitting_scale != 1.0:
            processor = self._presentation_processor()
            negative_presentations = [processor.build("", plan.qwen_media) for plan in references]
        embeddings = encode_ref2va_presentations(
            models["text_encoder"], list(presentations) + negative_presentations,
        )
        video_shapes = [tuple(value.shape) for value in targets]
        audio_shapes = [tuple(value.shape) for value in audio]
        payload = {
            "prompt_embeds": embeddings[:len(presentations)],
            "references": [self._reference_payload(plan) for plan in references],
            "prefix_plans": batch["prefix_plan"],
            "latent_shapes": video_shapes,
            "audio_shapes": audio_shapes,
            "native": self._native_conditions(presentations, references, video_shapes, audio_shapes),
            "video_latents": targets,
            "audio_latents": audio,
        }
        if self.cfg_fitting_scale != 1.0:
            payload["negative_prompt_embeds"] = embeddings[len(presentations):]
            payload["negative_native"] = self._native_conditions(
                negative_presentations, references, video_shapes, audio_shapes,
            )
        self._set_training_inputs(ctx, self._to_device(payload))
        lengths = [int(pack["sample_lens"]) for pack in ctx["inputs"].packs]
        assert lengths == [int(value) for value in batch["packing_rows"]], (
            "encoded prefix-TF rows disagree with the dataset's packing plan"
        )
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return
        sync = SPDistForward(name="video_ref_prefix_tf_inputs", comm_shape=True, device=get_device())
        for payload in sync(ctx["encoded_batch"]):
            yield self._set_training_inputs(dict(ctx), payload)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Draw one paired video/audio timestep for every sample's noisy chunks."""
        inputs, rng = ctx["inputs"], ctx["rng"]
        with local_seed(rng.seed % 2**31):
            video_t, audio_t = self.training_timesteps.sample_pair(
                (inputs.batch_size,), inputs.seqlens, get_device()
            )
        rng.seed = yield_seed(rng.seed)
        video_t, audio_t = video_t.float(), audio_t.float()
        ctx["sample_timesteps"] = (video_t, audio_t)
        ctx["train_timesteps"] = (
            [video_t[index].expand(len(layout.chunks)) for index, layout in enumerate(inputs.layouts)],
            [audio_t[index].expand(len(layout.chunks)) for index, layout in enumerate(inputs.layouts)],
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Predict only noisy target rows while reading clean corpus history."""
        video_t, audio_t = ctx["sample_timesteps"]
        arguments = dict(
            video_xts=ctx["noisy_latents"][0], audio_xts=ctx["noisy_latents"][1],
            video_timesteps=video_t, audio_timesteps=audio_t,
            video_context=ctx["clean_latents"][0], audio_context=ctx["clean_latents"][1],
            video_eps=ctx["context_eps"][0], audio_eps=ctx["context_eps"][1],
        )
        prediction = self._packed_forward(ctx["models"]["backbone"], ctx["inputs"], **arguments)
        if self.cfg_fitting_scale != 1.0:
            with torch.no_grad():
                negative = self._packed_forward(
                    ctx["models"]["backbone"], ctx["negative_inputs"], **arguments,
                )
            prediction = tuple([
                (conditional + (self.cfg_fitting_scale - 1.0) * unconditional.detach())
                / self.cfg_fitting_scale
                for conditional, unconditional in zip(positive_values, negative_values, strict=True)
            ] for positive_values, negative_values in zip(prediction, negative, strict=True))
        ctx["pred"] = prediction
        return ctx

    def _packed_forward(
        self, model: Any, inputs: PrefixTFInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        video_context: list[torch.Tensor], audio_context: list[torch.Tensor],
        video_eps: list[torch.Tensor], audio_eps: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self._prefix_forward(
            model, inputs, video_xts=video_xts, audio_xts=audio_xts,
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
            video_context=video_context, audio_context=audio_context,
            video_eps=video_eps, audio_eps=audio_eps,
        )

    def _chunk_inputs(self, inputs: PrefixTFInputs, chunk_index: int) -> PrefixTFInputs:
        return self._prefix_chunk_inputs(inputs, chunk_index)

    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        try:
            return super().validate(ctx)
        finally:
            self._validation_reference_cache = None

    def _validation_inputs_for_requests(
        self, config: Any, models: dict[str, Any], requests: Sequence[dict[str, Any]], *,
        prompts: Sequence[str] | None = None,
    ) -> PrefixTFInputs:
        validation = config.validation
        data_args = config.data.get("args", {})
        mapping = self._check_video_temporal_mapping(models["video_vae"])
        video_shape, audio_shape = causal_video_ref_target_shapes(
            height=int(validation.height), width=int(validation.width),
            num_frames=int(validation.num_frames),
            video_latent_channels=int(data_args.get("video_latent_channels", 24)),
            audio_latent_channels=int(data_args.get("audio_latent_channels", 32)),
            spatial_vae_stride=int(data_args.get("spatial_vae_stride", 16)),
            video_temporal_mapping=mapping,
        )
        preprocessing = (
            int(validation.num_frames), int(validation.get("fps", 24)),
            int(validation.get("reference_height", data_args.get("reference_height", validation.height))),
            int(validation.get("reference_width", data_args.get("reference_width", validation.width))),
            int(validation.seed),
            mapping, video_shape, audio_shape,
        )
        cached = self._validation_reference_cache
        if (
            cached is not None
            and cached[0] is requests
            and cached[1] == preprocessing
            and cached[2] is models["video_vae"]
        ):
            plans = cached[3]
        else:
            plans = []
            for request in requests:
                path = str(Path(request["reference_video_path"]).expanduser().resolve())
                seed = int(request.get("seed", validation.seed))
                pixels = CausalVideoRefMixin._load_reference_video(
                    path, num_frames=preprocessing[0], fps=preprocessing[1],
                    height=preprocessing[2], width=preprocessing[3],
                )
                media, spec = _decoded_video(pixels, video_temporal_mapping=mapping)
                plans.append(encode_ref_block_plan(
                    [spec], video_vae=models["video_vae"], audio_vae=None,
                    target_latent_t=video_shape[1],
                    encode_seed=combine_seed(seed, "prefix_reference_encode", path),
                    noise_seed=int(request.get("ref_noise_seed", combine_seed(seed, "prefix_reference_noise", path))),
                    visual_anchor=self.visual_anchor, decoded_visual_media={0: media},
                ))
            self._validation_reference_cache = (requests, preprocessing, models["video_vae"], plans)
        if prompts is None:
            prompts = [self._validation_request_prompt(request) for request in requests]
        processor = self._presentation_processor()
        presentations = [processor.build(prompt, plan.qwen_media)
                         for prompt, plan in zip(prompts, plans, strict=True)]
        embeddings = encode_ref2va_presentations(models["text_encoder"], presentations)
        return self._inputs_from_payload({
            "prompt_embeds": embeddings,
            "prefix_plans": [self._prefix_plan_factory(
                target_video_shape=video_shape, target_audio_shape=audio_shape,
                chunk_size=data_args.get("chunk_size", 5),
                independent_first_chunk=data_args.get("independent_first_chunk"),
                video_temporal_mapping=mapping,
            ) for _ in requests],
            "native": self._native_conditions(
                presentations, plans, [video_shape] * len(requests), [audio_shape] * len(requests),
            ),
            "references": [self._reference_payload(plan) for plan in plans],
            "latent_shapes": [video_shape] * len(requests),
            "audio_shapes": [audio_shape] * len(requests),
        })

    @torch.no_grad()
    def _rollout_latents(
        self, backbone: Any, inputs: PrefixTFInputs, rng: Any, *,
        keep_trajectory: bool, rngs: Sequence[Any] | None = None,
        trajectory_x0_chunks: Any = None,
    ) -> tuple[_RolloutX0s, list[Any]]:
        assert not keep_trajectory and trajectory_x0_chunks is None
        sample_rngs = list(rngs) if rngs is not None else [rng] * inputs.batch_size
        return self._guided_rollout_latents(backbone, [(1.0, inputs)], sample_rngs), []

    @torch.no_grad()
    def _guided_rollout_latents(
        self, backbone: Any, branch_inputs: Sequence[tuple[float, PrefixTFInputs]],
        rngs: Sequence[Any], *, first_branch_without_history: bool = False,
    ) -> _RolloutX0s:
        """Generate each target chunk with shared CFG noise and no persistent KV."""
        assert not first_branch_without_history
        inputs = branch_inputs[0][1]
        assert len(rngs) == inputs.batch_size
        device = inputs.token_tags.device
        video_noises = [torch.empty(layout.latent_shape, device=device).normal_(generator=self._generator(rng))
                        for layout, rng in zip(inputs.layouts, rngs, strict=True)]
        audio_noises = [torch.empty(layout.audio_shape, device=device).normal_(generator=self._generator(rng))
                        for layout, rng in zip(inputs.layouts, rngs, strict=True)]
        video_eps = [torch.empty_like(value).normal_(generator=self._generator(rng))
                     for value, rng in zip(video_noises, rngs, strict=True)]
        audio_eps = [torch.empty_like(value).normal_(generator=self._generator(rng))
                     for value, rng in zip(audio_noises, rngs, strict=True)]
        video_x0s = [torch.zeros_like(value) for value in video_noises]
        audio_x0s = [torch.zeros_like(value) for value in audio_noises]
        video_anchors = [None] * inputs.batch_size if self.video_chunk_renorm else None
        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        video_grid, audio_grid = self.sampling_timesteps.timesteps, self.audio_sampling_timesteps.timesteps
        assert video_grid.ndim == audio_grid.ndim == 1
        assert video_grid.numel() == audio_grid.numel()
        chunk_count = len(inputs.layouts[0].chunks)
        assert all(len(layout.chunks) == chunk_count for layout in inputs.layouts)
        for chunk_index in range(chunk_count):
            chunks = [layout.chunks[chunk_index] for layout in inputs.layouts]
            video_xts = [value[:, chunk.video_start:chunk.video_stop]
                         for value, chunk in zip(video_noises, chunks, strict=True)]
            audio_xts = [value[:, :, chunk.audio_start:chunk.audio_stop]
                         for value, chunk in zip(audio_noises, chunks, strict=True)]
            branches = [(weight, self._chunk_inputs(branch, chunk_index))
                        for weight, branch in branch_inputs]
            for video_step, audio_step in zip(video_grid, audio_grid, strict=True):
                video_t = video_step.expand(inputs.batch_size).to(device=device)
                audio_t = audio_step.expand(inputs.batch_size).to(device=device)
                video_s = self.sampling_timesteps.get_next_timesteps(video_t)
                audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)
                full_video = [value.clone() for value in video_x0s]
                full_audio = [value.clone() for value in audio_x0s]
                for index, chunk in enumerate(chunks):
                    full_video[index][:, chunk.video_start:chunk.video_stop] = video_xts[index]
                    full_audio[index][:, :, chunk.audio_start:chunk.audio_stop] = audio_xts[index]
                video_pred = [torch.zeros_like(value) for value in video_xts]
                audio_pred = [torch.zeros_like(value) for value in audio_xts]
                for weight, branch in branches:
                    predicted_video, predicted_audio = self._packed_forward(
                        backbone, branch, video_xts=full_video, audio_xts=full_audio,
                        video_timesteps=video_t, audio_timesteps=audio_t,
                        video_context=video_x0s, audio_context=audio_x0s,
                        video_eps=video_eps, audio_eps=audio_eps,
                    )
                    for index, chunk in enumerate(chunks):
                        video_pred[index].add_(
                            predicted_video[index][:, chunk.video_start:chunk.video_stop], alpha=weight
                        )
                        audio_pred[index].add_(
                            predicted_audio[index][:, :, chunk.audio_start:chunk.audio_stop], alpha=weight
                        )
                video_xts = self.sampler.step_to(
                    pred=video_pred, x_t=video_xts, t=video_t, s=video_s,
                    rng=rngs, seqlens=inputs.seqlens,
                )
                audio_xts = self.sampler.step_to(
                    pred=audio_pred, x_t=audio_xts, t=audio_t, s=audio_s,
                    rng=rngs, seqlens=inputs.seqlens,
                )
            video_xts = self._renorm_completed_video_chunks(
                inputs, video_xts, video_anchors, chunk_index=chunk_index
            )
            for index, chunk in enumerate(chunks):
                video_x0s[index][:, chunk.video_start:chunk.video_stop] = video_xts[index]
                audio_x0s[index][:, :, chunk.audio_start:chunk.audio_stop] = audio_xts[index]
        return _RolloutX0s(
            video=video_x0s, audio=audio_x0s, video_eps=video_eps,
            audio_eps=audio_eps, video_anchors=video_anchors,
        )


EntryClass = MiniMaxH3VideoRefPrefixTF

__all__ = ["MiniMaxH3VideoRefPrefixTF", "PrefixTFInputs", "EntryClass"]
