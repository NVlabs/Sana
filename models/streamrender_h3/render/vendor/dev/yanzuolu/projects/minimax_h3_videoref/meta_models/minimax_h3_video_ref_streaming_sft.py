# SPDX-License-Identifier: Apache-2.0
"""Paired-video inputs for the shared bidirectional streaming SFT mechanism."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.distributed.fsdp import FSDPModule

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import (
    get_unified_parallel_rank,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import combine_seed
from dev.yanzuolu.projects.minimax_h3.data.caption_timeline import (
    caption_index_for_plan,
    normalize_caption_segments,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import (
    MiniMaxH3StreamingSFT,
    StreamingBatch,
    StreamingLatentChunk,
    StreamingRollout,
    StreamingState,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.streaming_guidance import (
    GUIDANCE_CONDITIONS,
    StreamingGuidanceBranch,
)
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import (
    MINIMAX_H3_VIDEO_CLEAN_TIMESTEP,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_latent import (
    causal_video_ref_target_shapes,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_df import (
    CausalVideoRefMixin,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.video_ref_conditions import (
    encode_video_ref_targets,
    video_ref_corpus_metadata,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MiniMaxH3Ref2VAPresentationProcessor,
    Ref2VAPresentation,
    Ref2VAPresentationMedia,
    encode_ref2va_presentations,
    text_only_ref2va_presentation,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import (
    encode_reference_image,
    prepare_reference_image,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_qwen import (
    build_streaming_ref_presentation,
    streaming_reference_rgb_indices,
)


@dataclass(frozen=True)
class VideoRefStreamingBatch(StreamingBatch):
    """Streaming inputs with each sample's optional picture and fixed Qwen contexts.

    A picture is ``{"image", "clean", "latents"}``: the prepared RGB image for
    Qwen, its clean latent and the anchored latent the model reads. Without
    ``qwen_reference_video``, ``qwen_contexts`` maps each of a sample's captions
    to its encoded embeddings, tags, negative embeddings and negative tags.
    """

    pictures: list[dict[str, Any] | None] | None = None
    qwen_contexts: list[dict[str, tuple[torch.Tensor, ...]]] | None = None


class MiniMaxH3VideoRefStreamingSFT(MiniMaxH3StreamingSFT):
    """Prepare raw media or cached paired latents for shared streaming SFT.

    With Qwen visual context, every window of a clip may also carry one picture.
    Qwen reads it as Picture 1 before the reference video, and the DiT reads it
    as one near-clean image latent before the reference rows. It belongs to the
    reference group, so conditions that drop the reference drop it too.
    ``reference_drop_keeps_picture`` true instead keeps it in every
    ``guidance_branches`` state, like the keyframe of text-to-audio-video. A
    branch dropping the reference then removes only the reference rows and
    keeps the picture's rows and Qwen context, and one dropping the text
    removes only the caption. Without ``qwen_reference_video`` the states read
    the picture alone, the picture and caption, the picture alone beside the
    reference rows, and the full context. Qwen must not read the reference
    video, so no branch needs a reference-free text pass.
    A branch may also drop ``picture``, which makes the picture its own
    condition and implies ``reference_drop_keeps_picture``, so the flag may
    be omitted but not set false. Dropping it removes the picture rows and
    Picture 1 from Qwen. Qwen then reads the caption alone, the text-only
    encoding of its token ids, or nothing when the caption is dropped too.
    Along ``0 --> T --> IT --> ITS`` the states read no context, the caption
    alone, the picture and caption, and the picture and caption beside the
    reference rows. History rows are never dropped with it, and a branch
    dropping ``history`` treats them by ``history_drop`` like any host.
    ``picture_mode`` places those rows. ``rows``, the default, gives them their
    own time slot like a native image reference. ``keyframe`` places them like
    the native first-frame keyframe, at target latent 0's time on the target
    grid, while target latent 0 is still generated.
    ``qwen_reference_video`` defaults to True. False keeps the reference video
    out of Qwen, which then reads only the picture and caption, and requires
    ``data.args.picture_index_path``. Qwen context then never depends on
    reference frames, so sampling encodes each caption's context once per
    request, before its first window. Validation requests add a picture with
    ``picture``, an image path, and training samples with ``picture``, the
    prepared ``[H, W, 3]`` RGB image.

    ``reference_video_rows`` false keeps the paired reference latents out of
    the windows. The payload then carries them as ``source_latents`` for
    metas that inject them elsewhere.
    """

    _sync_inputs_name = "video_ref_streaming_sft_inputs"
    qwen_reference_video: bool = True
    picture_mode: str = "rows"
    reference_video_rows: bool = True
    reference_drop_keeps_picture: bool = False
    guidance_conditions: tuple[str, ...] = (*GUIDANCE_CONDITIONS, "picture")
    _validation_requests = staticmethod(CausalVideoRefMixin._validation_requests)
    _validation_tokenizer = CausalVideoRefMixin._validation_tokenizer

    @staticmethod
    def _validation_request_prompt(request: Any) -> str:
        if "caption_segments" not in request:
            return str(request["prompt"])
        segments = normalize_caption_segments(request["caption_segments"])
        return str(request.get("prompt") or next((item["prompt"] for item in segments if item["prompt"]), ""))

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if self.fixed_window_rope and not self.separate_reference_rope:
            raise ValueError("fixed_window_rope requires separate_reference_rope when using video references")
        self._validation_reference_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._validation_reference_pixels_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._validation_picture_cache: dict[tuple[Any, ...], tuple[Any, torch.Tensor]] = {}
        self._stream_pictures: list[dict[str, Any] | None] | None = None
        self._stream_qwen_contexts: list[dict[str, tuple[torch.Tensor, ...]]] | None = None
        self.reference_drop_keeps_picture = bool(config.meta_model.get("reference_drop_keeps_picture", False))
        if any("picture" in branch.drop for branch in self.guidance_branches or ()):
            if config.meta_model.get("reference_drop_keeps_picture", True) is False:
                raise ValueError("a guidance branch dropping the picture makes it its own condition, so "
                                 "reference_drop_keeps_picture cannot be false")
            self.reference_drop_keeps_picture = True
        if self.reference_drop_keeps_picture:
            if self.guidance_branches is None:
                raise ValueError("reference_drop_keeps_picture shapes guidance_branches, so it requires them")
            if self.qwen_visual_context and self.qwen_reference_video:
                raise ValueError("reference_drop_keeps_picture keeps the picture's Qwen context, so Qwen must not "
                                 "read the reference video: set qwen_reference_video false")

    @property
    def needs_reference_free_text_conditioning(self) -> bool:
        """A kept picture keeps the caption's Qwen context, so only a branch dropping the picture and keeping the
        caption needs a text-only pass, in training or in a distillation fit."""
        uses = zip(self.guidance_branches or (), self._guidance_possible_uses, strict=True)
        fitting = getattr(self, "cfg_fitting_guidance", None)
        if (any(needed and self._reads_caption_alone(branch) for branch, needed in uses)
                or fitting is not None and any(self._reads_caption_alone(branch) for branch in fitting[0])):
            return True
        return not self.reference_drop_keeps_picture and super().needs_reference_free_text_conditioning

    @staticmethod
    def _reads_caption_alone(branch: StreamingGuidanceBranch) -> bool:
        return "picture" in branch.drop and "text" not in branch.drop

    def _guidance_available_conditions(self, ctx: dict[str, Any]) -> tuple[frozenset[str], ...]:
        """A window's picture is a condition beside its text, reference and history."""
        return tuple(
            available | {"picture"} if plan.get("picture_shape") is not None else available
            for available, plan in zip(super()._guidance_available_conditions(ctx), ctx["window_inputs"].plans, strict=True)
        )

    def _configure_window_context(self, config: Any) -> None:
        path = config.meta_model.get("qwen_processor_path")
        if not isinstance(path, str) or not path:
            raise ValueError("qwen_visual_context requires meta_model.qwen_processor_path")
        self.qwen_processor_path = path
        self._qwen_processor_cache = None
        self.qwen_reference_video = bool(config.meta_model.get("qwen_reference_video", True))
        if not self.qwen_reference_video and config.data.args.get("picture_index_path") is None:
            raise ValueError("qwen_reference_video false requires data.args.picture_index_path")
        settings = dict(qwen_visual_context=True, qwen_processor_path=path,
                        fixed_window_rope=self.fixed_window_rope, keep_sink_reference=self.keep_sink_reference,
                        qwen_reference_video=self.qwen_reference_video)
        self.picture_mode = str(config.meta_model.get("picture_mode", "rows"))
        if self.picture_mode not in {"rows", "keyframe"}:
            raise ValueError("picture_mode must be rows or keyframe")
        if self.picture_mode != "rows" and config.data.args.get("picture_index_path") is None:
            raise ValueError("picture_mode keyframe requires data.args.picture_index_path")
        for key, value in settings.items():
            if key in config.data.args and config.data.args[key] != value:
                raise ValueError(f"data.args.{key} must match meta_model.{key}")
            config.data.args[key] = value

    def _qwen_processor(self) -> MiniMaxH3Ref2VAPresentationProcessor:
        if self._qwen_processor_cache is None:
            self._qwen_processor_cache = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(self.qwen_processor_path)
        return self._qwen_processor_cache

    def _configure_selective_video_encoding(self, config: Any) -> None:
        if "fsdp" in config.models.video_vae.get("placement", {}):
            raise ValueError("selective_video_encoding requires an unsharded video_vae because temporal block counts can differ across ranks")

    def _preselect_training_windows(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Choose each source sample's window from shape metadata before encoding."""
        batch = ctx["batch"]
        # The prepare seed is shared within an SP group, so each source needs
        # its own window stream as well as an independent stream per sample.
        source_rank = get_unified_parallel_rank() if is_unified_parallel_initialized() else 0
        geometry = StreamingBatch(
            prompt_embeds=[torch.empty((0 if self.qwen_visual_context else int(length), 0)) for length in batch["text_lens"]],
            reference_latents=([torch.empty(tuple(shape), device="meta") for shape in batch["reference_latent_shapes"]]
                               if self.reference_video_rows else None),
            video_shapes=[tuple(shape) for shape in batch["latent_shapes"]],
            audio_shapes=[tuple(shape) for shape in batch["audio_shapes"]],
            streaming_configs=[self._streaming_policy(policy) for policy in batch["streaming_config"]],
        )
        return [self._sample_streaming_plan(geometry, index, ctx["rng"].fork("streaming_window", source_rank, index))
                for index in range(geometry.batch_size)]

    @staticmethod
    def _with_picture(plan: dict[str, Any], clean: torch.Tensor, *, keyframe: bool = False) -> dict[str, Any]:
        """Attach one clean ``[C, 1, H, W]`` picture latent. ``add_noise`` anchors it for training.

        ``keyframe`` places it as the native first-frame keyframe instead of in its own time slot.
        """
        rows = (clean.shape[2] // 2) * (clean.shape[3] // 2)
        plan = dict(plan, picture_shape=tuple(clean.shape), picture_clean_latents=clean,
                    packing_rows=plan["packing_rows"] + rows)
        return dict(plan, picture_keyframe=True) if keyframe else plan

    @staticmethod
    def _plan_without_reference(plan: dict[str, Any]) -> dict[str, Any]:
        """Remove the reference rows together with the picture, which belongs to the reference group."""
        return MiniMaxH3VideoRefStreamingSFT._plan_without_picture(MiniMaxH3StreamingSFT._plan_without_reference(plan))

    @staticmethod
    def _plan_without_picture(plan: dict[str, Any]) -> dict[str, Any]:
        """Remove the picture rows, keeping the reference rows."""
        shape = plan.get("picture_shape")
        if shape is None:
            return plan
        return dict(plan, picture_shape=None, picture_clean_latents=None, picture_latents=None,
                    packing_rows=plan["packing_rows"] - (shape[2] // 2) * (shape[3] // 2))

    def _guidance_window_inputs(
        self, ctx: dict[str, Any], branch: StreamingGuidanceBranch,
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """With a kept picture, drop only the reference rows, the caption or the picture itself."""
        if not self.reference_drop_keeps_picture or not branch.drop & {"text", "reference", "picture"}:
            return super()._guidance_window_inputs(ctx, branch)
        inputs = self._picture_keeping_branch_inputs(ctx["window_inputs"], ctx["inputs"], branch.drop)
        values = ctx["noisy_latents"]
        if "history" in branch.drop:
            return self._drop_target_history(ctx, inputs, values)
        return inputs, values

    def _picture_keeping_branch_inputs(
        self, inputs: StreamingInputs, source: StreamingBatch, drop: frozenset[str],
    ) -> StreamingInputs:
        """Drop the caption, the reference rows or the picture from a window, keeping its history rows.

        The picture stays unless ``drop`` names it. Without it Qwen reads the
        caption's text-only encoding, or nothing when the caption goes too.
        """
        drop_text, drop_reference, drop_picture = "text" in drop, "reference" in drop, "picture" in drop
        embeds, tags = inputs.prompt_embeds, inputs.text_token_tags
        if drop_picture and drop_text:
            embeds, tags = [value[:0] for value in embeds], None
        elif drop_picture and self.qwen_visual_context:
            embeds = source.text_only_prompt_embeds
            assert embeds is not None, "a picture-free caption requires text-only prompt embeddings"
            tags = [torch.ones(value.shape[0], dtype=torch.long, device=value.device) for value in embeds]
        elif drop_text and self.qwen_visual_context:
            embeds, tags = source.negative_prompt_embeds, source.negative_text_token_tags
            assert embeds is not None, "visual conditioning requires a prepared media-only prefix"
        elif drop_text:
            embeds, tags = [value[:0] for value in embeds], None
        plans = ([MiniMaxH3StreamingSFT._plan_without_reference(plan) for plan in inputs.plans]
                 if drop_reference else inputs.plans)
        if drop_picture:
            plans = [self._plan_without_picture(plan) for plan in plans]
        return self._streaming_inputs_from_payload({
            "plans": [dict(plan, text_len=value.shape[0], packing_rows=plan["packing_rows"] - plan["text_len"] + value.shape[0])
                      for plan, value in zip(plans, embeds, strict=True)],
            "prompt_embeds": embeds, "text_token_tags": tags,
            "reference_latents": None if drop_reference else inputs.reference_latents,
        })

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Anchor each window's picture like its reference rows, from a fork of the sample stream."""
        plans = []
        for plan, rng in zip(ctx["plans"], ctx["sample_rngs"], strict=True):
            if plan.get("picture_shape") is not None:
                clean = plan["picture_clean_latents"].to(device=get_device(), dtype=torch.float32)
                plan = dict(plan, picture_latents=MINIMAX_H3_VIDEO_CLEAN_TIMESTEP * clean +
                            (1 - MINIMAX_H3_VIDEO_CLEAN_TIMESTEP) * self._noise(clean, rng.fork("picture")))
            plans.append(plan)
        ctx["plans"] = plans
        return super().add_noise(ctx)

    def _selected_caption_inputs(
        self, batch: dict[str, Any], plans: Sequence[dict[str, Any]] | None,
    ) -> tuple[list[str], list[torch.Tensor], list[int]]:
        """Select language from the noisy source interval before either training forward."""
        if "caption_segments" not in batch:
            return list(batch["prompts"]), list(batch["text_input_ids"]), list(batch["text_lens"])
        assert plans is not None
        indices = [caption_index_for_plan(segments, plan, self.video_temporal_mapping)
                   for segments, plan in zip(batch["caption_segments"], plans, strict=True)]
        return (
            [segments[index]["prompt"] for segments, index in zip(batch["caption_segments"], indices, strict=True)],
            [tokens[index] for tokens, index in zip(batch["caption_text_input_ids"], indices, strict=True)],
            [lengths[index] for lengths, index in zip(batch["caption_text_lens"], indices, strict=True)],
        )

    def _encode_selected_video_inputs(self, ctx: dict[str, Any], plans: list[dict[str, Any]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode selected rows into full-shape carriers for the common window path."""
        batch = ctx["batch"]
        codec, prepare_rng = ctx["models"]["video_vae"], ctx["rng"]
        video_indices = [plan["video_indices"] for plan in plans]
        reference_indices = [plan["reference_video_indices"] for plan in plans]
        encoded = (
            encode_video_ref_targets(codec, batch["video_pixels"],
                                     seeds=[prepare_rng.fork("target_encode", index).seed for index in range(len(plans))],
                                     video_temporal_mapping=self.video_temporal_mapping, latent_indices=video_indices),
            encode_video_ref_targets(codec, batch["reference_video_pixels"],
                                     seeds=[prepare_rng.fork("reference_encode", index).seed for index in range(len(plans))],
                                     video_temporal_mapping=self.video_temporal_mapping, latent_indices=reference_indices),
        )
        carriers = []
        for shapes, indices, selected in ((batch["latent_shapes"], video_indices, encoded[0]),
                                          (batch["reference_latent_shapes"], reference_indices, encoded[1])):
            modality = []
            for shape, rows, value in zip(shapes, indices, selected, strict=True):
                assert tuple(value.shape) == (shape[0], rows.numel(), *shape[2:])
                carrier = value.new_zeros(shape)
                carrier.index_copy_(1, rows.to(value.device), value)
                modality.append(carrier)
            carriers.append(modality)
        return carriers[0], carriers[1]

    @staticmethod
    def _conditioned_plan(plan: dict[str, Any], length: int) -> dict[str, Any]:
        return dict(plan, text_len=length, packing_rows=plan["packing_rows"] - plan["text_len"] + length)

    @staticmethod
    def _window_conditioning(presentations: Sequence[Ref2VAPresentation],
                             embeddings: list[torch.Tensor]) -> tuple[list[torch.Tensor], ...]:
        assert all(0 < item.media_prefix_length <= value.shape[0]
                   for item, value in zip(presentations, embeddings, strict=True))
        tags = [item.text_token_tags.to(get_device()) for item in presentations]
        # Native Qwen is causal and all media precedes the caption. Its complete
        # media prefix includes labels and timestamps as well as vision rows.
        negative = [value[:item.media_prefix_length].contiguous()
                    for value, item in zip(embeddings, presentations, strict=True)]
        negative_tags = [value[:item.media_prefix_length].contiguous()
                         for value, item in zip(tags, presentations, strict=True)]
        return embeddings, tags, negative, negative_tags

    @staticmethod
    @torch.no_grad()
    def _encode_window_presentations(models: dict[str, Any], presentations: Sequence[Ref2VAPresentation]) -> tuple[list[torch.Tensor], ...]:
        embeddings = encode_ref2va_presentations(models["text_encoder"], presentations)
        return MiniMaxH3VideoRefStreamingSFT._window_conditioning(presentations, embeddings)

    @torch.no_grad()
    def _prepare_step_context(self, states: Sequence[StreamingState], plans: list[dict[str, Any]], *,
                              models: dict[str, Any] | None, reference_pixels: Sequence[torch.Tensor] | None) -> list[dict[str, Any]]:
        assert models is not None and reference_pixels is not None
        presentations, contexts = [], []
        boundary = self.video_temporal_mapping.decode_timeline.boundary
        for state, plan, pixels in zip(states, plans, reference_pixels, strict=True):
            assert state.prompt_texts is not None
            prompts = {prompt for prompt in state.prompt_texts if prompt}
            if len(prompts) > 1:
                raise ValueError("visual CFG branches must share a caption or omit language")
            prompt = next(iter(prompts), "")
            if not self.qwen_reference_video:
                contexts.append(self._stream_qwen_contexts[state.sample_index][prompt])
                continue
            first, stop = boundary(plan["start"]), boundary(plan["stop"])
            if pixels.ndim != 4 or pixels.shape[0] != 3 or pixels.shape[1] != stop - first:
                raise ValueError("reference_pixels must contain exactly the new native interval as [3,RGB,H,W]")
            pixels = pixels.detach().cpu()
            frame_indices = torch.arange(first, stop)
            if state.reference_pixel_history is not None:
                pixels = torch.cat((state.reference_pixel_history, pixels), dim=1)
                frame_indices = torch.cat((state.reference_pixel_indices, frame_indices))
            picture = None if self._stream_pictures is None else self._stream_pictures[state.sample_index]
            presentations.append(build_streaming_ref_presentation(
                self._qwen_processor(), prompt, pixels, plan,
                reference_frame_indices=frame_indices, fixed_window_rope=self.fixed_window_rope,
                picture=None if picture is None else picture["image"],
            ))
            state.reference_pixel_history = pixels
            state.reference_pixel_indices = frame_indices
        if self.qwen_reference_video:
            embeddings, tags, negative, negative_tags = self._encode_window_presentations(models, presentations)
        else:
            embeddings, tags, negative, negative_tags = (list(values) for values in zip(*contexts, strict=True))
        if not self.keep_negative_reference:
            # The captionless branch drops the visual prefix together with its reference rows.
            negative = [value[:0] for value in embeddings]
            negative_tags = [value[:0] for value in tags]
        for index, state in enumerate(states):
            state.prompt_branches = [
                (weight, embeddings[index] if prompt else negative[index])
                for (weight, _), prompt in zip(state.prompt_branches, state.prompt_texts, strict=True)
            ]
            state.prompt_branch_tags = [tags[index] if prompt else negative_tags[index] for prompt in state.prompt_texts]
        return [self._conditioned_plan(plan, state.prompt_branches[0][1].shape[0])
                for state, plan in zip(states, plans, strict=True)]

    def _iter_stream_latents(self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]],
                             rngs: Sequence[Any], *, models: dict[str, Any] | None = None) -> Iterator[StreamingLatentChunk]:
        """Condition every window of a stream on its sample's picture and fixed contexts, which all branches share."""
        inputs = branch_inputs[0][1]
        if isinstance(inputs, VideoRefStreamingBatch):
            self._stream_pictures, self._stream_qwen_contexts = inputs.pictures, inputs.qwen_contexts
        try:
            yield from super()._iter_stream_latents(backbone, branch_inputs, rngs, models=models)
        finally:
            self._stream_pictures = self._stream_qwen_contexts = None

    def _state_plan(self, state: StreamingState, stop: int, audio_stop: int, text_len: int) -> dict[str, Any]:
        plan = super()._state_plan(state, stop, audio_stop, text_len)
        picture = None if self._stream_pictures is None else self._stream_pictures[state.sample_index]
        if picture is None:
            return plan
        return dict(self._with_picture(plan, picture["clean"], keyframe=self.picture_mode == "keyframe"),
                    picture_latents=picture["latents"])

    def _commit_reference_pixels(self, state: StreamingState, plan: dict[str, Any]) -> None:
        """Retain the reference frames the next presentation needs. Only Qwen reference video reads them."""
        if not self.qwen_reference_video:
            return
        wanted =streaming_reference_rgb_indices(dict(plan, reference_video_indices=state.reference_history_indices))
        assert state.reference_pixel_indices is not None and state.reference_pixel_history is not None
        positions = torch.searchsorted(state.reference_pixel_indices, wanted)
        assert torch.equal(state.reference_pixel_indices[positions], wanted)
        state.reference_pixel_history = state.reference_pixel_history.index_select(1, positions).contiguous()
        state.reference_pixel_indices = wanted

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        batch, models, rng = ctx["batch"], ctx["models"], ctx["rng"]
        self._check_codec(models["video_vae"])
        for descriptor in batch["video_temporal_mapping"]:
            if VideoTemporalMapping.from_dict(descriptor) != self.video_temporal_mapping:
                raise ValueError("dataset temporal mapping disagrees with video_vae")
        plans = self._preselect_training_windows(ctx) if (
            self.qwen_visual_context or self.selective_video_encoding or "caption_segments" in batch
        ) else None
        prompts, text_input_ids, text_lens = self._selected_caption_inputs(batch, plans)
        if "video_latents" in batch:
            video = batch["video_latents"]
            audio = batch["audio_latents"]
            reference = batch["reference_video_latents"]
        elif self.selective_video_encoding:
            if any(isinstance(module, FSDPModule) for module in models["video_vae"].modules()):
                raise ValueError("selective_video_encoding cannot use an FSDP video_vae with rank-dependent temporal blocks")
            video, reference = self._encode_selected_video_inputs(ctx, plans)
        else:
            video = encode_video_ref_targets(
                models["video_vae"], batch["video_pixels"],
                seeds=[rng.fork("target_encode", index).seed for index in range(len(batch["video_pixels"]))],
                video_temporal_mapping=self.video_temporal_mapping,
            )
            reference = encode_video_ref_targets(
                models["video_vae"], batch["reference_video_pixels"],
                seeds=[rng.fork("reference_encode", index).seed for index in range(len(video))],
                video_temporal_mapping=self.video_temporal_mapping,
            )
        if "video_latents" not in batch:
            audio = CausalVideoRefMixin._encode_audio_media(models["audio_vae"], batch["audio_waveform"])
        for values, shapes in ((video, batch["latent_shapes"]), (audio, batch["audio_shapes"]),
                               (reference, batch["reference_latent_shapes"])):
            assert all(tuple(value.shape) == tuple(shape) for value, shape in zip(values, shapes, strict=True))
        conditioning = {}
        text_only_embeddings = None
        images = pictures = None
        if "picture" in batch:
            if any(isinstance(module, FSDPModule) for module in models["video_vae"].modules()):
                raise ValueError("ranks encode different pictures, so picture encoding requires an unsharded video_vae")
            images = [Image.fromarray(value.numpy()) for value in batch["picture"]]
            pictures = [encode_reference_image(models["video_vae"], image, seed=rng.fork("picture_encode", index).seed).to(get_device())
                        for index, image in enumerate(images)]
        if self.qwen_visual_context:
            references = batch["reference_video_pixels"] if self.qwen_reference_video else [None] * len(prompts)
            presentations = [build_streaming_ref_presentation(
                self._qwen_processor(), prompt, pixels, plan, fixed_window_rope=self.fixed_window_rope,
                picture=None if images is None else images[index],
            ) for index, (prompt, pixels, plan) in enumerate(zip(prompts, references, plans, strict=True))]
            if self.needs_reference_free_text_conditioning:
                text_presentations = [text_only_ref2va_presentation(ids) for ids in text_input_ids]
                assert len(text_presentations) == len(presentations)
                assert all(item.input_ids.numel() == int(length)
                           for item, length in zip(text_presentations, text_lens, strict=True))
                encoded = encode_ref2va_presentations(models["text_encoder"], [*presentations, *text_presentations])
                embeddings, tags, negative, negative_tags = self._window_conditioning(presentations, encoded[:len(presentations)])
                text_only_embeddings = encoded[len(presentations):]
            else:
                embeddings, tags, negative, negative_tags = self._encode_window_presentations(models, presentations)
            plans = [self._conditioned_plan(plan, value.shape[0]) for plan, value in zip(plans, embeddings, strict=True)]
            conditioning = dict(
                prompts=prompts, text_token_tags=tags,
                negative_prompt_embeds=negative, negative_text_token_tags=negative_tags,
                media_prefix_lengths=[item.media_prefix_length for item in presentations],
            )
        else:
            embeddings = self._encode_prompts(models, text_input_ids, text_lens)
            if plans is not None:
                plans = [self._conditioned_plan(plan, value.shape[0]) for plan, value in zip(plans, embeddings, strict=True)]
            if self.needs_reference_free_text_conditioning:
                text_only_embeddings = embeddings
        if text_only_embeddings is not None:
            conditioning["text_only_prompt_embeds"] = text_only_embeddings
        if pictures is not None:
            plans = [self._with_picture(plan, picture, keyframe=self.picture_mode == "keyframe")
                     for plan, picture in zip(plans, pictures, strict=True)]
        if not self.reference_video_rows:
            conditioning["source_latents"] = reference
        payload = {
            "prompt_embeds": embeddings,
            "reference_latents": reference if self.reference_video_rows else None,
            "video_latents": video, "audio_latents": audio,
            "latent_shapes": batch["latent_shapes"], "audio_shapes": batch["audio_shapes"],
            "streaming_config": batch["streaming_config"], "packing_rows": batch["packing_rows"],
            **conditioning,
        }
        if plans is not None:
            payload["preselected_plans"] = plans
        ctx["encoded_batch"] = self._to_device(payload)
        ctx["inputs"] = self._inputs_from_payload(ctx["encoded_batch"])
        ctx["clean_latents"] = (ctx["encoded_batch"]["video_latents"], ctx["encoded_batch"]["audio_latents"])
        return ctx

    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        try:
            return super().validate(ctx)
        finally:
            self._validation_reference_cache.clear()
            self._validation_picture_cache.clear()
            self._validation_reference_pixels_cache.clear()

    def _validation_frames(self, frames: torch.Tensor, inputs: StreamingBatch, *, index: int) -> torch.Tensor:
        """The reference video on the left, the sample on the right, at the sample's height."""
        reference = inputs.reference_pixels[index]
        assert reference.shape[1] == frames.shape[2], (
            f"reference has {reference.shape[1]} frames, sample has {frames.shape[2]}"
        )
        height = int(frames.shape[3])
        if reference.shape[2] != height:
            width = round(reference.shape[3] * height / reference.shape[2])
            reference = torch.nn.functional.interpolate(
                reference.permute(1, 0, 2, 3), size=(height, width), mode="bilinear", antialias=True,
            ).permute(1, 0, 2, 3)
        return torch.cat([reference[None].to(frames.dtype), frames.cpu()], dim=4)

    def _validation_picture(
        self, models: dict[str, Any], request: Any, *, height: int, width: int, seed: int,
    ) -> dict[str, Any]:
        """Prepare and encode a request's picture, anchored with its own noise draw."""
        if not self.qwen_visual_context:
            raise ValueError("validation pictures require qwen_visual_context")
        path = str(Path(request["picture"]).expanduser().resolve())
        key = (path, height, width, seed, id(models["video_vae"]))
        if key not in self._validation_picture_cache:
            image = prepare_reference_image(path, width=width, height=height)
            self._validation_picture_cache[key] = (image, encode_reference_image(
                models["video_vae"], image, seed=combine_seed(seed, "streaming_picture_encode", path),
            ).to(get_device()))
        image, clean = self._validation_picture_cache[key]
        generator = torch.Generator().manual_seed(combine_seed(seed, "streaming_picture_noise", path))
        noise = torch.randn(clean.shape, generator=generator).to(clean)
        return {"image": image, "clean": clean,
                "latents": MINIMAX_H3_VIDEO_CLEAN_TIMESTEP * clean + (1 - MINIMAX_H3_VIDEO_CLEAN_TIMESTEP) * noise}

    def _validation_qwen_contexts(
        self, models: dict[str, Any], prompts: Sequence[str], caption_segments: list[Any] | None,
        pictures: Sequence[dict[str, Any] | None],
    ) -> list[dict[str, tuple[torch.Tensor, ...]]]:
        """Encode every caption of every sample with its picture in one encoder call."""
        captions = [sorted({str(prompt)} if caption_segments is None or caption_segments[index] is None
                           else {segment["prompt"] for segment in caption_segments[index]})
                    for index, prompt in enumerate(prompts)]
        processor = self._qwen_processor()
        presentations = [processor.build(caption, [] if picture is None else [
            Ref2VAPresentationMedia(kind="image", media=picture["image"])])
            for values, picture in zip(captions, pictures, strict=True) for caption in values]
        encoded = self._encode_window_presentations(models, presentations)
        contexts, cursor = [], 0
        for values in captions:
            contexts.append({caption: tuple(items[cursor + offset] for items in encoded)
                             for offset, caption in enumerate(values)})
            cursor += len(values)
        return contexts

    def _validation_latent_metadata(
        self, config: Any, inputs: StreamingBatch, request: Any, *, index: int,
    ) -> dict[str, Any]:
        reference = inputs.reference_latents[index]
        stride = int(config.data.args.get("spatial_vae_stride", 16))
        metadata = video_ref_corpus_metadata(
            reference, path=request["reference_video_path"],
            num_frames=int(request.get("num_frames", config.validation.num_frames)),
            height=reference.shape[2] * stride, width=reference.shape[3] * stride,
            video_temporal_mapping=self.video_temporal_mapping,
        )
        if "caption_segments" in request:
            metadata["caption_segments"] = normalize_caption_segments(request["caption_segments"])
        return metadata

    @torch.no_grad()
    def _validation_inputs_for_requests(self, config: Any, models: dict[str, Any], requests: Sequence[dict[str, Any]],
                                        *, prompts: Sequence[str] | None = None) -> StreamingBatch:
        self._check_codec(models["video_vae"])
        validation, data = config.validation, config.data.args
        references, video_shapes, audio_shapes, policies = [], [], [], []
        reference_pixels, pictures = [], []
        for request in requests:
            frames = int(request.get("num_frames", validation.num_frames))
            height, width = int(request.get("height", validation.height)), int(request.get("width", validation.width))
            fps = int(request.get("fps", validation.get("fps", 24)))
            if fps != 24:
                raise ValueError("streaming AV geometry requires 24 FPS video and 40 Hz audio latents")
            video_shape, audio_shape = causal_video_ref_target_shapes(
                height=height, width=width, num_frames=frames,
                video_latent_channels=int(data.get("video_latent_channels", 24)),
                audio_latent_channels=int(data.get("audio_latent_channels", 32)),
                spatial_vae_stride=int(data.get("spatial_vae_stride", 16)), video_temporal_mapping=self.video_temporal_mapping,
            )
            ref_height = int(request.get("reference_height", validation.get("reference_height", data.get("reference_height", height))))
            ref_width = int(request.get("reference_width", validation.get("reference_width", data.get("reference_width", width))))
            path = str(Path(request["reference_video_path"]).expanduser().resolve())
            seed = int(request.get("seed", validation.seed))
            key = (path, frames, fps, ref_height, ref_width, seed, self.video_temporal_mapping, id(models["video_vae"]))
            if key not in self._validation_reference_cache:
                pixels = CausalVideoRefMixin._load_reference_video(path, num_frames=frames, fps=fps, height=ref_height, width=ref_width)
                self._validation_reference_cache[key] = encode_video_ref_targets(
                    models["video_vae"], [pixels], seeds=[combine_seed(seed, "streaming_reference_encode", path)],
                    video_temporal_mapping=self.video_temporal_mapping,
                )[0].to(get_device())
                self._validation_reference_pixels_cache[key] = pixels
            references.append(self._validation_reference_cache[key])
            reference_pixels.append(self._validation_reference_pixels_cache[key])
            video_shapes.append(video_shape)
            audio_shapes.append(audio_shape)
            policies.append(self._validation_streaming_policy(validation, request))
            pictures.append(self._validation_picture(models, request, height=height, width=width, seed=seed)
                            if "picture" in request else None)
        explicit_prompts = prompts is not None
        prompts = [self._validation_request_prompt(request) for request in requests] if prompts is None else prompts
        caption_segments = None
        if any("caption_segments" in request for request in requests):
            caption_segments = [
                (None if "caption_segments" not in request else [
                    dict(segment, prompt=segment["prompt"] if not explicit_prompts or prompt else "")
                    for segment in normalize_caption_segments(
                        request["caption_segments"], num_frames=int(request.get("num_frames", validation.num_frames)),
                    )
                ]) for request, prompt in zip(requests, prompts, strict=True)
            ]
        if self.qwen_visual_context:
            embeds = [torch.empty((0, 0), device=get_device()) for _ in prompts]
            inputs = self._inputs_from_payload(dict(prompt_embeds=embeds, prompts=list(prompts), reference_latents=references,
                                                   latent_shapes=video_shapes, audio_shapes=audio_shapes, streaming_config=policies,
                                                   caption_segments=caption_segments))
            inputs = replace(inputs, reference_pixels=reference_pixels)
            return VideoRefStreamingBatch(
                **{field.name: getattr(inputs, field.name) for field in fields(inputs)}, pictures=pictures,
                qwen_contexts=None if self.qwen_reference_video else self._validation_qwen_contexts(
                    models, prompts, caption_segments, pictures,
                ),
            )
        captions = [[str(prompt)] if caption_segments is None or caption_segments[index] is None else
                    [segment["prompt"] for segment in caption_segments[index]] for index, prompt in enumerate(prompts)]
        tokens = [self._validation_tokenizer(config).encode(prompt) for values in captions for prompt in values]
        encoded = self._encode_prompts(models, [item[0] for item in tokens], [item[1] for item in tokens])
        embeddings_by_sample, cursor = [], 0
        for values in captions:
            embeddings_by_sample.append(encoded[cursor:cursor + len(values)])
            cursor += len(values)
        embeds = [values[0] for values in embeddings_by_sample]
        inputs = self._inputs_from_payload(dict(prompt_embeds=embeds, reference_latents=references,
                                               latent_shapes=video_shapes, audio_shapes=audio_shapes, streaming_config=policies,
                                               caption_segments=caption_segments,
                                               caption_embeds=None if caption_segments is None else embeddings_by_sample))
        return replace(inputs, reference_pixels=reference_pixels)


EntryClass = MiniMaxH3VideoRefStreamingSFT

__all__ = ["MiniMaxH3VideoRefStreamingSFT", "StreamingBatch", "VideoRefStreamingBatch", "StreamingState", "StreamingLatentChunk", "StreamingRollout", "EntryClass"]
