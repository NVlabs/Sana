# SPDX-License-Identifier: Apache-2.0
"""Causal MiniMax H3 diffusion forcing with a clean video reference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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
from dev.yanzuolu.common.seed import RandomState, combine_seed, local_seed
from dev.yanzuolu.projects.minimax_h3_videoref.data import causal_video_ref_raw
from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import (
    _stereo_chunk_indices,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.causal_video_ref_latent import (
    build_causal_video_ref_layout,
    causal_video_ref_target_shapes,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_df import (
    CausalMiniMaxH3DF,
)
from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
    video_temporal_mapping_from_model_config,
)
from dev.yanzuolu.projects.minimax_h3.modeling.tokenizer import MiniMaxH3Tokenizer
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.packing import (
    causal_video_ref_rope_offsets,
)
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask
from dev.yanzuolu.utils.naive_cache import NaiveCache

_AUDIO_CHANNELS = 2
_PATCH_SIZE = (1, 2, 2)
_ReferencePreprocessingSpec = tuple[int, int, int, int]


@dataclass(frozen=True)
class VideoRefChunk:
    """One physical ``[reference | audio | target video]`` chunk."""

    noise_start: int
    clean_start: int | None
    has_clean_copy: bool
    reference_rows: int
    audio_rows: int
    video_rows: int
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int

    @property
    def physical_rows(self) -> int:
        return self.reference_rows + self.audio_rows + self.video_rows


@dataclass(frozen=True)
class VideoRefLayout:
    """Target and reference geometry on the video codec's temporal grid."""

    text_len: int
    chunks: tuple[VideoRefChunk, ...]
    audio_row_perm: torch.Tensor
    latent_shape: tuple[int, int, int, int]
    audio_shape: tuple[int, int, int]
    reference_latent_shape: tuple[int, int, int, int]

    video_temporal_mapping: VideoTemporalMapping = field(
        default=H3_VIDEO_TEMPORAL_MAPPING, kw_only=True
    )

    @property
    def video_patch_grid(self) -> tuple[int, int, int]:
        _, latent_t, latent_h, latent_w = self.latent_shape
        return latent_t, latent_h // _PATCH_SIZE[1], latent_w // _PATCH_SIZE[2]


@dataclass(frozen=True)
class VideoRefForwardInput:
    """Static packed payload with ragged reference geometry."""

    batch_size: int
    prompt_embeds: list[torch.Tensor]
    text_lens: list[int]
    seqlens: torch.Tensor
    layouts: list[VideoRefLayout]
    sinks: list[int]
    window_sizes: list[int | None]
    sample_lens: list[int]
    split_lens: list[list[int]]
    attn_modes: list[list[str]]
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    img_pos: torch.Tensor
    reference_img_pos: torch.Tensor
    target_img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor
    attn_workloads: list[int]
    noisy_img_pos: torch.Tensor
    audio_noisy_sel: torch.Tensor
    reference_latents: list[torch.Tensor]
    reference_eps: list[torch.Tensor] | None
    teacher_forced: bool


class CausalVideoRefMixin:
    """Video-reference packing, corpus preparation and causal validation helpers.

    ``meta_model.separate_reference_rope`` places each chunk's reference before
    its target on the temporal axis. It defaults to false and applies to both
    packed training and cached inference, preserving noisy/clean copy alignment.

    ``data.args`` owns chunk size, the optional first-chunk length, sink and
    sliding window. Validation can override each setting independently. Sink
    counts include the text block, and noisy teacher-forcing copies do not count
    as history. Temporal coordinates follow the configured video codec.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._init_video_ref_context(config)

    def _init_video_ref_context(self, config: Any) -> None:
        """Initialize codec geometry and caches for TF, DF and TSCD constructors."""
        data_args = config.data.get("args", {})
        configured_mapping = data_args.get("video_temporal_mapping")
        video_vae_config = config.get("models", {}).get("video_vae")
        if video_vae_config is None:
            self.video_temporal_mapping = (
                H3_VIDEO_TEMPORAL_MAPPING
                if configured_mapping is None
                else VideoTemporalMapping.from_dict(configured_mapping)
            )
        else:
            self.video_temporal_mapping = video_temporal_mapping_from_model_config(
                video_vae_config
            )
        if (
            configured_mapping is not None
            and VideoTemporalMapping.from_dict(configured_mapping) != self.video_temporal_mapping
        ):
            raise ValueError("data.args.video_temporal_mapping must match models.video_vae")
        if configured_mapping is None and self.video_temporal_mapping != H3_VIDEO_TEMPORAL_MAPPING:
            if "args" not in config.data:
                config.data.args = {}
            config.data.args.video_temporal_mapping = self.video_temporal_mapping.to_dict()
        self._validation_tokenizer_cache: MiniMaxH3Tokenizer | None = None
        self._validation_reference_batch: (
            tuple[
                Sequence[dict[str, Any]],
                _ReferencePreprocessingSpec,
                int,
                Any,
                VideoTemporalMapping,
                list[torch.Tensor],
            ]
            | None
        ) = None

    @staticmethod
    def _batch_video_temporal_mappings(batch: dict[str, Any]) -> list[VideoTemporalMapping]:
        descriptions = batch.get("video_temporal_mapping")
        if descriptions is None:
            return [H3_VIDEO_TEMPORAL_MAPPING] * len(batch["sample_lens"])
        assert len(descriptions) == len(batch["sample_lens"])
        return [VideoTemporalMapping.from_dict(value) for value in descriptions]

    def _check_video_temporal_mapping(self, video_vae: Any) -> VideoTemporalMapping:
        mapping = self.video_temporal_mapping
        if video_vae.video_temporal_mapping != mapping:
            raise ValueError("runtime video_vae temporal mapping disagrees with the configured codec")
        return mapping

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Encode raw media when present and build the synchronized payload."""
        batch = ctx["batch"]
        raw_fields = {
            "video_pixels",
            "audio_waveform",
            "reference_video_pixels",
        }
        if raw_fields.issubset(batch) or "video_vae" in ctx["models"]:
            self._check_video_temporal_mapping(ctx["models"]["video_vae"])
        if any(
            value != self.video_temporal_mapping
            for value in self._batch_video_temporal_mappings(batch)
        ):
            raise ValueError("dataset video-reference temporal mapping disagrees with the configured geometry")
        if raw_fields.issubset(batch):
            video, audio, reference = self._encode_raw_media(ctx["models"], batch)
            encoded_batch = {
                key: value for key, value in batch.items() if key not in raw_fields
            }
            encoded_batch["video_latents"] = video
            encoded_batch["audio_latents"] = audio
            encoded_batch["reference_video_latents"] = reference
        else:
            encoded_batch = batch

        for latent_key, shape_key in (
            ("video_latents", "latent_shapes"),
            ("audio_latents", "audio_shapes"),
            ("reference_video_latents", "reference_latent_shapes"),
        ):
            for latent, shape in zip(encoded_batch[latent_key], encoded_batch[shape_key], strict=True):
                assert tuple(latent.shape) == tuple(shape), (
                    f"encoded {latent_key} shape disagrees with the dataset layout"
                )

        prompt_embeds = self._encode_prompts(
            ctx["models"],
            encoded_batch["text_input_ids"],
            [int(value) for value in encoded_batch["text_lens"]],
        )
        ctx["encoded_batch"] = encoded_batch
        ctx["inputs"] = self._build_inputs(encoded_batch, prompt_embeds)
        ctx["clean_latents"] = (
            [value.to(get_device()) for value in encoded_batch["video_latents"]],
            [value.to(get_device()) for value in encoded_batch["audio_latents"]],
        )
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Broadcast encoded batches without carrying raw media in the payload."""
        if (
            not is_unified_parallel_initialized()
            or get_unified_parallel_world_size() <= 1
        ):
            yield ctx
            return

        payload = (
            self._to_device(ctx["encoded_batch"]),
            ctx["inputs"].prompt_embeds,
        )
        sync = SPDistForward(
            name="causal_video_ref_inputs", comm_shape=True, device=get_device()
        )
        for encoded_batch, prompt_embeds in sync(payload):
            sub_ctx = dict(ctx)
            sub_ctx["encoded_batch"] = encoded_batch
            sub_ctx["inputs"] = self._build_inputs(encoded_batch, prompt_embeds)
            sub_ctx["clean_latents"] = (
                encoded_batch["video_latents"],
                encoded_batch["audio_latents"],
            )
            yield sub_ctx

    @staticmethod
    @torch.no_grad()
    def _encode_raw_media(
        models: dict[str, Any], batch: dict[str, Any]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Encode target, audio, and ragged reference media with H3 codecs."""
        video_vae = models["video_vae"]
        audio_vae = models["audio_vae"]
        video_out: list[torch.Tensor] = []
        audio_out: list[torch.Tensor] = []
        reference_out: list[torch.Tensor] = []
        for pixels, waveform, reference_pixels in zip(
            batch["video_pixels"],
            batch["audio_waveform"],
            batch["reference_video_pixels"],
            strict=True,
        ):
            video_out.extend(CausalVideoRefMixin._encode_video_media(video_vae, [pixels]))
            reference_out.extend(
                CausalVideoRefMixin._encode_video_media(video_vae, [reference_pixels])
            )

            audio_out.extend(CausalVideoRefMixin._encode_audio_media(audio_vae, [waveform]))
        return video_out, audio_out, reference_out

    @staticmethod
    @torch.no_grad()
    def _encode_video_media(
        video_vae: Any,
        pixels: Sequence[torch.Tensor],
        *,
        seeds: Sequence[int] | None = None,
    ) -> list[torch.Tensor]:
        """Encode normalized video latents with optional per-sample posterior seeds."""
        device = get_device()
        outputs = []
        sample_seeds = [None] * len(pixels) if seeds is None else seeds
        for value, seed in zip(pixels, sample_seeds, strict=True):
            normalized = video_vae.processor.transform_tensor(value.to(device)).transpose(0, 1)
            with local_seed(None if seed is None else int(seed) % 2**31):
                latent = video_vae.encode_videos([normalized], transform_input=False)[0]
            mean = video_vae.latents_mean.to(latent).view(-1, 1, 1, 1)
            std = video_vae.latents_std.to(latent).view(-1, 1, 1, 1)
            outputs.append((latent - mean) / std)
        return outputs

    @staticmethod
    @torch.no_grad()
    def _encode_audio_media(
        audio_vae: Any, waveforms: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Encode shared normalized stereo targets independently of the video codec."""
        outputs = []
        for waveform in waveforms:
            audio_input = waveform.to(
                device=get_device(), dtype=audio_vae.param_dtype
            ).unsqueeze(1)
            encoded = audio_vae.encoder(audio_vae.preprocess(audio_input, 32000))
            if audio_vae.attn_proj:
                encoded = audio_vae.pre_block(encoded.transpose(1, 2)).transpose(1, 2)
            latent = audio_vae.mean_proj(encoded)
            mean = audio_vae.latents_mean.to(latent).view(1, -1, 1)
            std = audio_vae.latents_std.to(latent).view(1, -1, 1)
            outputs.append((latent - mean) / std)
        return outputs

    @staticmethod
    def _build_video_ref_layout(
        *,
        text_len: int,
        latent_shape: tuple[int, int, int, int],
        audio_shape: tuple[int, int, int],
        reference_latent_shape: tuple[int, int, int, int],
        video_chunk_ranges: Sequence[Sequence[int]],
        audio_chunk_ranges: Sequence[Sequence[int]],
        chunk_starts: Sequence[int],
        clean_chunk_starts: Sequence[int | None],
        video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
    ) -> VideoRefLayout:
        reference_frame_rows = (reference_latent_shape[2] // 2) * (
            reference_latent_shape[3] // 2
        )
        target_frame_rows = (latent_shape[2] // 2) * (latent_shape[3] // 2)
        chunks = []
        for video_range, audio_range, start, clean_start_value in zip(
            video_chunk_ranges,
            audio_chunk_ranges,
            chunk_starts,
            clean_chunk_starts,
            strict=True,
        ):
            video_start, video_stop = (int(value) for value in video_range)
            audio_start, audio_stop = (int(value) for value in audio_range)
            clean_start = None if clean_start_value is None else int(clean_start_value)
            noise_start = int(start)
            chunks.append(
                VideoRefChunk(
                    noise_start=noise_start,
                    clean_start=clean_start,
                    has_clean_copy=(
                        clean_start is not None and clean_start != noise_start
                    ),
                    reference_rows=(video_stop - video_start) * reference_frame_rows,
                    audio_rows=(audio_stop - audio_start) * _AUDIO_CHANNELS,
                    video_rows=(video_stop - video_start) * target_frame_rows,
                    video_start=video_start,
                    video_stop=video_stop,
                    audio_start=audio_start,
                    audio_stop=audio_stop,
                )
            )
        audio_row_perm = torch.cat(
            [
                _stereo_chunk_indices(
                    audio_shape[2], chunk.audio_start, chunk.audio_stop
                )
                for chunk in chunks
            ]
        )
        return VideoRefLayout(
            text_len=int(text_len),
            chunks=tuple(chunks),
            audio_row_perm=audio_row_perm,
            latent_shape=latent_shape,
            audio_shape=audio_shape,
            reference_latent_shape=reference_latent_shape,
            video_temporal_mapping=video_temporal_mapping,
        )

    def _build_inputs(
        self, batch: dict[str, Any], prompt_embeds: list[torch.Tensor]
    ) -> VideoRefForwardInput:
        device = get_device()
        batch_size = len(batch["sample_lens"])
        text_lens = [int(value) for value in batch["text_lens"]]
        sample_lens = [int(value) for value in batch["sample_lens"]]
        latent_shapes = [
            tuple(int(value) for value in shape) for shape in batch["latent_shapes"]
        ]
        audio_shapes = [
            tuple(int(value) for value in shape) for shape in batch["audio_shapes"]
        ]
        reference_shapes = [
            tuple(int(value) for value in shape)
            for shape in batch["reference_latent_shapes"]
        ]
        temporal_mappings = self._batch_video_temporal_mappings(batch)
        layouts = [
            self._build_video_ref_layout(
                text_len=text_lens[index],
                latent_shape=latent_shapes[index],
                audio_shape=audio_shapes[index],
                reference_latent_shape=reference_shapes[index],
                video_chunk_ranges=batch["video_chunk_ranges"][index],
                audio_chunk_ranges=batch["audio_chunk_ranges"][index],
                chunk_starts=batch["chunk_starts"][index],
                clean_chunk_starts=batch["clean_chunk_starts"][index],
                video_temporal_mapping=temporal_mappings[index],
            )
            for index in range(batch_size)
        ]
        offsets = self._offsets(sample_lens)
        position_ids = torch.cat(batch["position_ids"])
        if self.config.meta_model.get("separate_reference_rope", False):
            for layout, sample_offset in zip(layouts, offsets, strict=True):
                reference_spans = [
                    layout.video_temporal_mapping.position_start(chunk.video_stop, 0.0)
                    - layout.video_temporal_mapping.position_start(chunk.video_start, 0.0)
                    for chunk in layout.chunks
                ]
                for chunk, (reference_offset, target_offset) in zip(
                    layout.chunks,
                    causal_video_ref_rope_offsets(reference_spans),
                    strict=True,
                ):
                    starts = [chunk.noise_start]
                    if chunk.has_clean_copy:
                        assert chunk.clean_start is not None
                        starts.append(chunk.clean_start)
                    for chunk_start in starts:
                        start = sample_offset + chunk_start
                        target_start = start + chunk.reference_rows
                        stop = start + chunk.physical_rows
                        position_ids[start:target_start, 0] += reference_offset
                        position_ids[target_start:stop, 0] += target_offset

        def packed_positions(key: str) -> torch.Tensor:
            return torch.cat(
                [
                    batch[key][index].to(torch.long) + offsets[index]
                    for index in range(batch_size)
                ]
            ).to(device)

        img_pos = packed_positions("img_pos")
        audio_pos = packed_positions("audio_pos")
        clean_rows = torch.cat(batch["inverse_indices"]).to(device) == 0
        noisy_img_pos = img_pos[~clean_rows[img_pos]]
        audio_noisy_sel = torch.nonzero(~clean_rows[audio_pos], as_tuple=False).view(-1)
        q_ranges = torch.cat(
            [batch["q_ranges"][index] + offsets[index] for index in range(batch_size)]
        ).to(device)
        k_ranges = torch.cat(
            [batch["k_ranges"][index] + offsets[index] for index in range(batch_size)]
        ).to(device)
        return VideoRefForwardInput(
            batch_size=batch_size,
            prompt_embeds=[embed.to(device) for embed in prompt_embeds],
            text_lens=text_lens,
            seqlens=torch.tensor(
                [int(value) for value in batch["seqlens"]],
                dtype=torch.int32,
                device=device,
            ),
            layouts=layouts,
            sinks=[int(value) for value in batch["sinks"]],
            window_sizes=[
                None if value is None else int(value) for value in batch["window_sizes"]
            ],
            sample_lens=sample_lens,
            split_lens=[[int(value) for value in row] for row in batch["split_lens"]],
            attn_modes=[list(row) for row in batch["attn_modes"]],
            position_ids=position_ids.to(device),
            token_tags=torch.cat(batch["token_tags"]).to(device),
            img_pos=img_pos,
            reference_img_pos=packed_positions("reference_img_pos"),
            target_img_pos=packed_positions("target_img_pos"),
            audio_pos=audio_pos,
            text_pos=packed_positions("text_pos"),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=torch.cat(batch["attn_type_map"]).to(device),
            attn_workloads=[int(value) for value in batch["attn_workloads"]],
            noisy_img_pos=noisy_img_pos,
            audio_noisy_sel=audio_noisy_sel,
            reference_latents=[
                value.to(device=device, dtype=torch.float32)
                for value in batch["reference_video_latents"]
            ],
            reference_eps=(
                [value.to(device) for value in batch["reference_eps"]]
                if "reference_eps" in batch
                else None
            ),
            teacher_forced=any(
                chunk.has_clean_copy for layout in layouts for chunk in layout.chunks
            ),
        )

    def _empty_text_inputs(
        self, inputs: VideoRefForwardInput
    ) -> VideoRefForwardInput:
        """Rebuild a video-reference input with structural zero-text rows."""
        forcing = "teacher" if inputs.teacher_forced else "diffusion"
        samples = []
        for index, layout in enumerate(inputs.layouts):
            sample = {
                "text_lens": 0,
                "reference_video_latents": inputs.reference_latents[index],
                **build_causal_video_ref_layout(
                    text_len=0,
                    target_video_shape=layout.latent_shape,
                    target_audio_shape=layout.audio_shape,
                    reference_video_shape=layout.reference_latent_shape,
                    forcing=forcing,
                    sink=inputs.sinks[index],
                    window_size=inputs.window_sizes[index],
                    video_chunk_ranges=[
                        (chunk.video_start, chunk.video_stop) for chunk in layout.chunks
                    ],
                    audio_chunk_ranges=[
                        (chunk.audio_start, chunk.audio_stop) for chunk in layout.chunks
                    ],
                    video_temporal_mapping=layout.video_temporal_mapping,
                ),
            }
            if inputs.reference_eps is not None:
                sample["reference_eps"] = inputs.reference_eps[index]
            samples.append(sample)
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}
        return self._build_inputs(
            batch,
            [embedding[:0] for embedding in inputs.prompt_embeds],
        )

    def _sample_reference_noises(
        self, inputs: VideoRefForwardInput, rng: Any
    ) -> list[torch.Tensor]:
        device = get_device()
        return [
            torch.empty(layout.reference_latent_shape, device=device).normal_(
                generator=self._generator(rng)
            )
            for layout in inputs.layouts
        ]

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Noise targets and draw the reference clean-timestep noise once."""
        inputs = ctx["inputs"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_timesteps, audio_timesteps = ctx["train_timesteps"]
        video_noises, audio_noises = self._sample_noises(inputs, ctx["rng"])
        ctx["noisy_latents"] = (
            [
                self._noise_chunks(clean, noise, timesteps, layout, "video")
                for clean, noise, timesteps, layout in zip(
                    clean_video,
                    video_noises,
                    video_timesteps,
                    inputs.layouts,
                    strict=True,
                )
            ],
            [
                self._noise_chunks(clean, noise, timesteps, layout, "audio")
                for clean, noise, timesteps, layout in zip(
                    clean_audio,
                    audio_noises,
                    audio_timesteps,
                    inputs.layouts,
                    strict=True,
                )
            ],
        )
        if inputs.teacher_forced:
            ctx["context_eps"] = self._sample_noises(inputs, ctx["rng"])
        reference_eps = self._sample_reference_noises(inputs, ctx["rng"])
        ctx["reference_eps"] = reference_eps
        ctx["inputs"] = replace(inputs, reference_eps=reference_eps)
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Run one target-only packed denoising forward."""
        inputs = ctx["inputs"]
        video_xts, audio_xts = ctx["noisy_latents"]
        video_t, audio_t = ctx["train_timesteps"]
        forward_kwargs: dict[str, Any] = {}
        if inputs.teacher_forced:
            clean_video, clean_audio = ctx["clean_latents"]
            video_eps, audio_eps = ctx["context_eps"]
            forward_kwargs.update(
                video_context=clean_video,
                audio_context=clean_audio,
                video_eps=video_eps,
                audio_eps=audio_eps,
            )
        ctx["pred"] = self._causal_packed_forward(
            ctx["models"]["backbone"],
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
            reference_eps=ctx["reference_eps"],
            **forward_kwargs,
        )
        return ctx

    def _causal_packed_forward(
        self,
        model: Any,
        inputs: VideoRefForwardInput,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: list[torch.Tensor],
        audio_timesteps: list[torch.Tensor],
        reference_eps: list[torch.Tensor],
        video_context: list[torch.Tensor] | None = None,
        audio_context: list[torch.Tensor] | None = None,
        video_eps: list[torch.Tensor] | None = None,
        audio_eps: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Materialize reference-clean and target-noisy physical rows."""
        device = inputs.token_tags.device
        video_dim = inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        audio_dim = inputs.layouts[0].audio_shape[1]
        video_timesteps = [
            value.to(device=device, dtype=torch.float32) for value in video_timesteps
        ]
        audio_timesteps = [
            value.to(device=device, dtype=torch.float32) for value in audio_timesteps
        ]
        video_blocks: list[torch.Tensor] = []
        audio_blocks: list[torch.Tensor] = []
        video_eps_blocks: list[torch.Tensor] = []
        audio_eps_blocks: list[torch.Tensor] = []
        row_timesteps: list[torch.Tensor] = []
        zero = torch.zeros((), device=device, dtype=torch.float32)

        def push(
            reference_rows: torch.Tensor,
            reference_eps_rows: torch.Tensor,
            target_audio_rows: torch.Tensor,
            target_video_rows: torch.Tensor,
            target_audio_eps: torch.Tensor,
            target_video_eps: torch.Tensor,
            audio_timestep: torch.Tensor,
            video_timestep: torch.Tensor,
        ) -> None:
            reference_count = reference_rows.shape[0]
            audio_count = target_audio_rows.shape[0]
            video_count = target_video_rows.shape[0]
            video_blocks.extend(
                (
                    reference_rows,
                    target_video_rows.new_zeros((audio_count, video_dim)),
                    target_video_rows,
                )
            )
            audio_blocks.extend(
                (
                    target_audio_rows.new_zeros((reference_count, audio_dim)),
                    target_audio_rows,
                    target_audio_rows.new_zeros((video_count, audio_dim)),
                )
            )
            video_eps_blocks.extend(
                (
                    reference_eps_rows,
                    target_video_eps.new_zeros((audio_count, video_dim)),
                    target_video_eps,
                )
            )
            audio_eps_blocks.extend(
                (
                    target_audio_eps.new_zeros((reference_count, audio_dim)),
                    target_audio_eps,
                    target_audio_eps.new_zeros((video_count, audio_dim)),
                )
            )
            row_timesteps.extend(
                (
                    zero.expand(reference_count),
                    audio_timestep.expand(audio_count),
                    video_timestep.expand(video_count),
                )
            )

        for index, layout in enumerate(inputs.layouts):
            text_rows = layout.text_len
            video_blocks.append(video_xts[index].new_zeros((text_rows, video_dim)))
            audio_blocks.append(audio_xts[index].new_zeros((text_rows, audio_dim)))
            video_eps_blocks.append(video_xts[index].new_zeros((text_rows, video_dim)))
            audio_eps_blocks.append(audio_xts[index].new_zeros((text_rows, audio_dim)))
            row_timesteps.append(zero.expand(text_rows))
            for chunk_index, chunk in enumerate(layout.chunks):
                reference_rows = self._video_rows(
                    inputs.reference_latents[index],
                    chunk.video_start,
                    chunk.video_stop,
                )
                reference_eps_rows = self._video_rows(
                    reference_eps[index], chunk.video_start, chunk.video_stop
                )
                target_video_rows = self._video_rows(
                    video_xts[index], chunk.video_start, chunk.video_stop
                )
                target_audio_rows = self._audio_rows(
                    audio_xts[index], chunk.audio_start, chunk.audio_stop
                )
                push(
                    reference_rows,
                    reference_eps_rows,
                    target_audio_rows,
                    target_video_rows,
                    target_audio_rows.new_zeros(target_audio_rows.shape),
                    target_video_rows.new_zeros(target_video_rows.shape),
                    audio_timesteps[index][chunk_index],
                    video_timesteps[index][chunk_index],
                )
                if not chunk.has_clean_copy:
                    continue
                assert (
                    video_context is not None
                    and audio_context is not None
                    and video_eps is not None
                    and audio_eps is not None
                )
                push(
                    reference_rows,
                    reference_eps_rows,
                    self._audio_rows(
                        audio_context[index], chunk.audio_start, chunk.audio_stop
                    ),
                    self._video_rows(
                        video_context[index], chunk.video_start, chunk.video_stop
                    ),
                    self._audio_rows(
                        audio_eps[index], chunk.audio_start, chunk.audio_stop
                    ),
                    self._video_rows(
                        video_eps[index], chunk.video_start, chunk.video_stop
                    ),
                    zero,
                    zero,
                )

        rows = sum(inputs.sample_lens)
        sample_lens = list(inputs.sample_lens)
        split_lens = [value for sample in inputs.split_lens for value in sample]
        attn_modes = [value for sample in inputs.attn_modes for value in sample]
        q_ranges, k_ranges = inputs.q_ranges, inputs.k_ranges
        attn_type_map, attn_workloads = inputs.attn_type_map, inputs.attn_workloads
        position_ids, token_tags, pad = self._pad_for_sp(
            rows,
            video_blocks=video_blocks,
            audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks,
            audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps,
            position_ids=inputs.position_ids,
            token_tags=inputs.token_tags,
            sample_lens=sample_lens,
            video_dim=video_dim,
            audio_dim=audio_dim,
        )
        if pad:
            split_lens.append(pad)
            attn_modes.append("causal")
            pad_q, pad_k, pad_type, pad_work = _prepare_flex_attention_mask(
                [pad],
                ["causal"],
                sink=inputs.sinks[0],
                window_size=inputs.window_sizes[0],
            )
            q_ranges = torch.cat((q_ranges, pad_q.to(device) + rows))
            k_ranges = torch.cat((k_ranges, pad_k.to(device) + rows))
            attn_type_map = torch.cat((attn_type_map, pad_type.to(device)))
            attn_workloads = list(attn_workloads) + [int(pad_work)]

        kwargs = self._common_kwargs(
            inputs,
            x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=position_ids,
            token_tags=token_tags,
            img_pos=inputs.img_pos,
            audio_pos=inputs.audio_pos,
            text_pos=inputs.text_pos,
            infer_out_pos=inputs.noisy_img_pos,
        )
        kwargs.update(
            sample_lens=sample_lens,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=attn_workloads,
            attention_mask=self._block_mask(
                inputs,
                device,
                sample_lens=sample_lens,
                split_lens=split_lens,
                attn_modes=attn_modes,
            ),
        )
        video_logits, audio_logits = model(**kwargs)
        return self._split_causal_output(
            inputs,
            video_logits,
            audio_logits.index_select(0, inputs.audio_noisy_sel),
        )

    def _chunk_forward(
        self,
        model: Any,
        inputs: VideoRefForwardInput,
        *,
        chunk_index: int,
        role: str,
        video_rows_source: list[torch.Tensor],
        audio_rows_source: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        cache: NaiveCache,
        update_cache: bool,
        video_eps_source: list[torch.Tensor] | None = None,
        audio_eps_source: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run one reference and target chunk against the inherited KV loop."""
        assert inputs.reference_eps is not None
        device = inputs.token_tags.device
        video_dim = inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        audio_dim = inputs.layouts[0].audio_shape[1]
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)
        video_blocks: list[torch.Tensor] = []
        audio_blocks: list[torch.Tensor] = []
        video_eps_blocks: list[torch.Tensor] = []
        audio_eps_blocks: list[torch.Tensor] = []
        row_timesteps: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        token_tags: list[torch.Tensor] = []
        img_pos: list[torch.Tensor] = []
        target_img_pos: list[torch.Tensor] = []
        audio_pos: list[torch.Tensor] = []
        sample_lens: list[int] = []
        source_offsets = self._offsets(inputs.sample_lens)
        cursor = 0
        zero = torch.zeros((), device=device, dtype=torch.float32)

        for index, layout in enumerate(inputs.layouts):
            chunk = layout.chunks[chunk_index]
            start = chunk.noise_start if role == "noise" else chunk.clean_start
            assert start is not None
            reference_rows = self._video_rows(
                inputs.reference_latents[index],
                chunk.video_start,
                chunk.video_stop,
            )
            reference_eps_rows = self._video_rows(
                inputs.reference_eps[index], chunk.video_start, chunk.video_stop
            )
            target_video_rows = self._video_rows(
                video_rows_source[index], 0, video_rows_source[index].shape[1]
            )
            target_audio_rows = self._audio_rows(
                audio_rows_source[index], 0, audio_rows_source[index].shape[2]
            )
            if video_eps_source is None:
                target_video_eps = target_video_rows.new_zeros(target_video_rows.shape)
                target_audio_eps = target_audio_rows.new_zeros(target_audio_rows.shape)
            else:
                target_video_eps = self._video_rows(
                    video_eps_source[index], 0, video_eps_source[index].shape[1]
                )
                target_audio_eps = self._audio_rows(
                    audio_eps_source[index], 0, audio_eps_source[index].shape[2]
                )
            video_blocks.extend(
                (
                    reference_rows,
                    target_video_rows.new_zeros((chunk.audio_rows, video_dim)),
                    target_video_rows,
                )
            )
            audio_blocks.extend(
                (
                    target_audio_rows.new_zeros((chunk.reference_rows, audio_dim)),
                    target_audio_rows,
                    target_audio_rows.new_zeros((chunk.video_rows, audio_dim)),
                )
            )
            video_eps_blocks.extend(
                (
                    reference_eps_rows,
                    target_video_rows.new_zeros((chunk.audio_rows, video_dim)),
                    target_video_eps,
                )
            )
            audio_eps_blocks.extend(
                (
                    target_audio_rows.new_zeros((chunk.reference_rows, audio_dim)),
                    target_audio_eps,
                    target_audio_rows.new_zeros((chunk.video_rows, audio_dim)),
                )
            )
            noisy = role == "noise"
            row_timesteps.extend(
                (
                    zero.expand(chunk.reference_rows),
                    (audio_timesteps[index] if noisy else zero).expand(
                        chunk.audio_rows
                    ),
                    (video_timesteps[index] if noisy else zero).expand(
                        chunk.video_rows
                    ),
                )
            )
            source_start = source_offsets[index] + start
            position_ids.append(
                inputs.position_ids[source_start : source_start + chunk.physical_rows]
            )
            token_tags.append(
                inputs.token_tags[source_start : source_start + chunk.physical_rows]
            )
            reference_pos = torch.arange(
                cursor,
                cursor + chunk.reference_rows,
                device=device,
                dtype=torch.long,
            )
            audio_pos.append(
                torch.arange(
                    cursor + chunk.reference_rows,
                    cursor + chunk.reference_rows + chunk.audio_rows,
                    device=device,
                    dtype=torch.long,
                )
            )
            target_pos = torch.arange(
                cursor + chunk.reference_rows + chunk.audio_rows,
                cursor + chunk.physical_rows,
                device=device,
                dtype=torch.long,
            )
            img_pos.append(torch.cat((reference_pos, target_pos)))
            target_img_pos.append(target_pos)
            sample_lens.append(chunk.physical_rows)
            cursor += chunk.physical_rows

        packed_position_ids, packed_token_tags, _ = self._pad_for_sp(
            cursor,
            video_blocks=video_blocks,
            audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks,
            audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps,
            position_ids=torch.cat(position_ids),
            token_tags=torch.cat(token_tags),
            sample_lens=sample_lens,
            video_dim=video_dim,
            audio_dim=audio_dim,
        )
        target_img_pos_tensor = torch.cat(target_img_pos)
        kwargs = self._common_kwargs(
            inputs,
            x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=packed_position_ids,
            token_tags=packed_token_tags,
            img_pos=torch.cat(img_pos),
            audio_pos=torch.cat(audio_pos),
            text_pos=torch.empty(0, device=device, dtype=torch.long),
            infer_out_pos=target_img_pos_tensor,
        )
        kwargs.update(
            sample_lens=sample_lens,
            past_key_values=cache,
            update_past_key_values=update_cache,
        )
        video_logits, audio_logits = model(**kwargs)
        if role == "clean":
            return [], []

        video_out: list[torch.Tensor] = []
        audio_out: list[torch.Tensor] = []
        video_cursor = 0
        audio_cursor = 0
        for layout in inputs.layouts:
            chunk = layout.chunks[chunk_index]
            video_out.append(
                minimax_h3_unpatchify_video_tokens(
                    video_logits[video_cursor : video_cursor + chunk.video_rows],
                    latent_shape=(
                        chunk.video_stop - chunk.video_start,
                        layout.video_patch_grid[1],
                        layout.video_patch_grid[2],
                        layout.latent_shape[0],
                    ),
                    patch_size=_PATCH_SIZE,
                )[0]
            )
            if chunk.audio_rows:
                audio_out.append(
                    minimax_h3_unpack_audio_tokens(
                        audio_logits[audio_cursor : audio_cursor + chunk.audio_rows],
                        audio_t=chunk.audio_rows,
                        audio_channel=_AUDIO_CHANNELS,
                    )
                )
            else:
                audio_out.append(
                    audio_logits.new_empty((_AUDIO_CHANNELS, layout.audio_shape[1], 0))
                )
            video_cursor += chunk.video_rows
            audio_cursor += chunk.audio_rows
        return video_out, audio_out

    @staticmethod
    def _validation_requests(validation: Any) -> list[Any]:
        requests = list(validation.requests)
        limit = int(validation.get("num_prompts", 0))
        return requests[:limit] if limit > 0 else requests

    @staticmethod
    def _validation_request_prompt(request: Any) -> str:
        return str(request["prompt"])

    def _validation_tokenizer(self, config: Any) -> MiniMaxH3Tokenizer:
        if self._validation_tokenizer_cache is None:
            self._validation_tokenizer_cache = MiniMaxH3Tokenizer(
                str(config.data.args.tokenizer_path)
            )
        return self._validation_tokenizer_cache

    @staticmethod
    def _load_reference_video(
        path_text: str,
        *,
        num_frames: int,
        fps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return causal_video_ref_raw.decode_reference_video(
            Path(path_text).expanduser(),
            start_frame=0,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
        )

    @torch.no_grad()
    def _validation_reference_latent(
        self,
        video_vae: Any,
        request: dict[str, Any],
        *,
        seed: int,
        num_frames: int,
        fps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        pixels = self._load_reference_video(
            str(request["reference_video_path"]),
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
        )
        resolved_path = Path(request["reference_video_path"]).expanduser().resolve()
        posterior_seed = combine_seed(seed, "video_ref_encode", str(resolved_path))
        return self._encode_video_media(
            video_vae, [pixels], seeds=[posterior_seed]
        )[0].to(torch.float32)

    def _validation_reference_latents(
        self,
        video_vae: Any,
        requests: Sequence[dict[str, Any]],
        *,
        seed: int,
        num_frames: int,
        fps: int,
        height: int,
        width: int,
    ) -> list[torch.Tensor]:
        preprocessing = (num_frames, fps, height, width)
        mapping = video_vae.video_temporal_mapping
        previous = self._validation_reference_batch
        if (
            previous is not None
            and previous[0] is requests
            and previous[1] == preprocessing
            and previous[2] == seed
            and previous[3] is video_vae
            and previous[4] == mapping
        ):
            return previous[5]
        references = [
            self._validation_reference_latent(
                video_vae,
                request,
                seed=seed,
                num_frames=num_frames,
                fps=fps,
                height=height,
                width=width,
            )
            for request in requests
        ]
        self._validation_reference_batch = (
            requests, preprocessing, seed, video_vae, mapping, references
        )
        return references

    def _validation_reference_eps(
        self,
        validation: Any,
        request: dict[str, Any],
        shape: Sequence[int],
    ) -> torch.Tensor:
        resolved_path = Path(request["reference_video_path"]).expanduser().resolve()
        rng = RandomState(
            combine_seed(int(validation.seed), "video_ref", str(resolved_path))
        )
        return torch.empty(tuple(shape), device=get_device()).normal_(
            generator=self._generator(rng)
        )

    def _validation_inputs_for_requests(
        self,
        config: Any,
        models: dict[str, Any],
        requests: Sequence[dict[str, Any]],
        *,
        prompts: Sequence[str] | None = None,
    ) -> VideoRefForwardInput:
        if prompts is None:
            prompts = [self._validation_request_prompt(request) for request in requests]
        validation = config.validation
        data_args = dict(config.data.get("args", {}))
        mapping = self._check_video_temporal_mapping(models["video_vae"])
        target_shape, audio_shape = causal_video_ref_target_shapes(
            height=int(validation.height),
            width=int(validation.width),
            num_frames=int(validation.num_frames),
            video_latent_channels=int(data_args.get("video_latent_channels", 24)),
            audio_latent_channels=int(data_args.get("audio_latent_channels", 32)),
            spatial_vae_stride=int(data_args.get("spatial_vae_stride", 16)),
            video_temporal_mapping=mapping,
        )
        layout_options = {
            key: validation.get(key, data_args.get(key, default))
            for key, default in (
                ("chunk_size", 5),
                ("independent_first_chunk", None),
                ("sink", 1),
                ("window_size", None),
            )
        }
        tokenizer = self._validation_tokenizer(config)
        references = self._validation_reference_latents(
            models["video_vae"],
            requests,
            seed=int(validation.seed),
            num_frames=int(validation.num_frames),
            fps=int(validation.fps),
            height=int(validation.height),
            width=int(validation.width),
        )
        samples = []
        for prompt, request, reference in zip(
            prompts, requests, references, strict=True
        ):
            text_input_ids, text_len = tokenizer.encode(str(prompt))
            samples.append(
                {
                    "prompts": str(prompt),
                    "text_input_ids": text_input_ids,
                    "text_lens": text_len,
                    "reference_video_latents": reference,
                    "reference_eps": self._validation_reference_eps(
                        validation, request, reference.shape
                    ),
                    **build_causal_video_ref_layout(
                        text_len=text_len,
                        target_video_shape=target_shape,
                        target_audio_shape=audio_shape,
                        reference_video_shape=reference.shape,
                        video_temporal_mapping=mapping,
                        **layout_options,
                    ),
                }
            )
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}
        prompt_embeds = self._encode_prompts(
            models,
            batch["text_input_ids"],
            [int(value) for value in batch["text_lens"]],
        )
        return self._build_inputs(batch, prompt_embeds)


class CausalMiniMaxH3VideoRefDF(CausalVideoRefMixin, CausalMiniMaxH3DF):
    """Single-slot causal DF conditioned on one clean video latent reference."""


EntryClass = CausalMiniMaxH3VideoRefDF

__all__ = [
    "CausalMiniMaxH3VideoRefDF",
    "EntryClass",
    "VideoRefChunk",
    "VideoRefForwardInput",
    "VideoRefLayout",
]
