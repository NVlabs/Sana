"""Latent corpus and packed layout for causal video-reference diffusion forcing."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import get_worker_info

from dev.yanzuolu.common.data import WorkerResumeContext, WorkerStateEnvelope
from dev.yanzuolu.common.seed import yield_seed
from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import (
    CausalTextOnlyT2AVDataset,
    _audio_chunk_ranges,
    _chunk_position_ids,
    _positive_chunk_size,
    _video_chunk_ranges,
    _video_chunk_t_grid,
)
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS
from dev.yanzuolu.projects.minimax_h3.modeling.packing import _axis_from_sqrt_area
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
    minimax_h3_audio_latent_t,
)
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask

_VIDEO_TAG = 0
_TEXT_TAG = 1
_AUDIO_TAG = 2
_AUDIO_CHANNELS = 2
_PATCH_SIZE = (1, 2, 2)


def causal_video_ref_target_shapes(
    *,
    height: int,
    width: int,
    num_frames: int,
    video_latent_channels: int = 24,
    audio_latent_channels: int = 32,
    spatial_vae_stride: int = 16,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    """Return normalized target video and audio latent shapes."""
    height = int(height)
    width = int(width)
    num_frames = int(num_frames)
    spatial_vae_stride = int(spatial_vae_stride)
    if height % spatial_vae_stride or width % spatial_vae_stride:
        raise ValueError("height and width must be divisible by spatial_vae_stride")
    latent_h = height // spatial_vae_stride
    latent_w = width // spatial_vae_stride
    if latent_h % _PATCH_SIZE[1] or latent_w % _PATCH_SIZE[2]:
        raise ValueError(
            "target latent height and width must be divisible by patch (2, 2)"
        )
    latent_t = video_temporal_mapping.target_latent_t(num_frames)
    audio_t = minimax_h3_audio_latent_t(num_frames / float(MINIMAX_H3_SUPPORTED_FPS))
    return (
        (int(video_latent_channels), latent_t, latent_h, latent_w),
        (_AUDIO_CHANNELS, int(audio_latent_channels), audio_t),
    )


def _reference_chunk_position_ids(
    *,
    video_start: int,
    video_stop: int,
    latent_h: int,
    latent_w: int,
    origin: int,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> torch.Tensor:
    """Build reference RoPE rows on the target temporal grid."""
    sqrt_area = float((latent_h * latent_w) ** 0.5)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_SIZE[1], sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_SIZE[2], sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)
    times = _video_chunk_t_grid(
        video_start, video_stop, float(origin), video_temporal_mapping=video_temporal_mapping
    )
    positions = torch.zeros((times.numel(), frame.shape[0], 3), dtype=torch.float64)
    positions[:, :, 0] = times[:, None]
    positions[:, :, 1:] = frame[None]
    return positions.reshape(-1, 3)


def _latent_shapes(
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    reference_video_shape: Sequence[int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int], tuple[int, int, int, int]]:
    target = tuple(int(value) for value in target_video_shape)
    audio = tuple(int(value) for value in target_audio_shape)
    reference = tuple(int(value) for value in reference_video_shape)
    if len(target) != 4 or len(reference) != 4 or len(audio) != 3:
        raise ValueError("video latents must be [C,T,H,W] and audio must be [2,C,T]")
    if target[:2] != reference[:2]:
        raise ValueError("reference and target video latent C/T must match")
    if audio[0] != _AUDIO_CHANNELS:
        raise ValueError("target audio latent must be stereo")
    if any(shape[2] % 2 or shape[3] % 2 for shape in (target, reference)):
        raise ValueError("video latent H/W must be divisible by patch (2, 2)")
    return target, audio, reference


def _nonnegative_chunk_count(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _partition_ranges(
    ranges: Sequence[Sequence[int]], total: int, name: str, *, allow_empty: bool,
) -> list[tuple[int, int]]:
    result = [(int(start), int(stop)) for start, stop in ranges]
    cursor = 0
    for start, stop in result:
        if start != cursor or stop < start or stop > total or (stop == start and not allow_empty):
            raise ValueError(f"{name} must partition [0, {total}) in order")
        cursor = stop
    if cursor != total:
        raise ValueError(f"{name} must cover all {total} latent steps")
    return result


def build_causal_video_ref_layout(
    *,
    text_len: int,
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    reference_video_shape: Sequence[int],
    forcing: str = "diffusion",
    chunk_size: int = 5,
    independent_first_chunk: int | None = None,
    sink: int = 1,
    window_size: int | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
    video_chunk_ranges: Sequence[Sequence[int]] | None = None,
    audio_chunk_ranges: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Build ``[reference video | target audio | target video]`` media chunks.

    Chunk lengths count video latents, with an optional independent first
    length and a shorter final chunk. Sink and window count non-noise blocks,
    including the leading text block. Reference rows use the target timeline.
    Explicit ranges preserve an already resolved chunk partition.
    """
    if forcing not in {"teacher", "diffusion"}:
        raise ValueError(f"forcing must be 'teacher' or 'diffusion', got {forcing!r}")
    chunk_size = _positive_chunk_size(chunk_size, "chunk_size")
    if independent_first_chunk is not None:
        _positive_chunk_size(independent_first_chunk, "independent_first_chunk")
    sink = _nonnegative_chunk_count(sink, "sink")
    if window_size is not None:
        window_size = _nonnegative_chunk_count(window_size, "window_size")
    target_shape, audio_shape, reference_shape = _latent_shapes(
        target_video_shape, target_audio_shape, reference_video_shape
    )
    text_len = int(text_len)
    _, latent_t, target_h, target_w = target_shape
    _, _, reference_h, reference_w = reference_shape
    audio_t = audio_shape[2]
    if min(target_shape) <= 0 or min(reference_shape) <= 0 or audio_shape[1] <= 0 or audio_t < 0:
        raise ValueError("video dimensions must be positive and audio T must be nonnegative")
    video_ranges = _partition_ranges(
        _video_chunk_ranges(
            latent_t, chunk_size=chunk_size, independent_first_chunk=independent_first_chunk
        ) if video_chunk_ranges is None else video_chunk_ranges,
        latent_t,
        "video_chunk_ranges",
        allow_empty=False,
    )
    audio_ranges = _partition_ranges(
        _audio_chunk_ranges(
            latent_t, audio_t, video_ranges=video_ranges,
            video_temporal_mapping=video_temporal_mapping,
        ) if audio_chunk_ranges is None else audio_chunk_ranges,
        audio_t,
        "audio_chunk_ranges",
        allow_empty=True,
    )
    if len(video_ranges) != len(audio_ranges):
        raise ValueError("video and audio must have the same number of chunks")
    target_frame_rows = (target_h // 2) * (target_w // 2)
    reference_frame_rows = (reference_h // 2) * (reference_w // 2)

    positions = [
        torch.stack(
            (
                torch.arange(text_len, dtype=torch.float64),
                torch.zeros(text_len, dtype=torch.float64),
                torch.zeros(text_len, dtype=torch.float64),
            ),
            dim=-1,
        )
    ]
    tags = [torch.full((text_len,), _TEXT_TAG, dtype=torch.long)]
    role_rows = [torch.zeros(text_len, dtype=torch.long)]
    split_lens = [text_len]
    attn_modes = ["full"]
    text_pos = torch.arange(text_len, dtype=torch.long)

    reference_img_pos_parts: list[torch.Tensor] = []
    all_target_img_pos_parts: list[torch.Tensor] = []
    noisy_target_img_pos_parts: list[torch.Tensor] = []
    audio_pos_parts: list[torch.Tensor] = []
    chunk_starts: list[int] = []
    clean_chunk_starts: list[int | None] = []
    cursor = text_len
    last_chunk = len(video_ranges) - 1

    for chunk_index, (
        (video_start, video_stop),
        (audio_start, audio_stop),
    ) in enumerate(zip(video_ranges, audio_ranges, strict=True)):
        reference_rows = (video_stop - video_start) * reference_frame_rows
        target_video_rows = (video_stop - video_start) * target_frame_rows
        audio_rows = (audio_stop - audio_start) * _AUDIO_CHANNELS
        chunk_rows = reference_rows + audio_rows + target_video_rows
        chunk_positions = torch.cat(
            (
                _reference_chunk_position_ids(
                    video_start=video_start,
                    video_stop=video_stop,
                    latent_h=reference_h,
                    latent_w=reference_w,
                    origin=text_len,
                    video_temporal_mapping=video_temporal_mapping,
                ),
                _chunk_position_ids(
                    video_start=video_start,
                    video_stop=video_stop,
                    audio_start=audio_start,
                    audio_stop=audio_stop,
                    latent_h=target_h,
                    latent_w=target_w,
                    origin=text_len,
                    video_temporal_mapping=video_temporal_mapping,
                ),
            )
        )
        chunk_tags = torch.cat(
            (
                torch.full((reference_rows,), _VIDEO_TAG, dtype=torch.long),
                torch.full((audio_rows,), _AUDIO_TAG, dtype=torch.long),
                torch.full((target_video_rows,), _VIDEO_TAG, dtype=torch.long),
            )
        )
        roles = (
            (("noise",) if chunk_index == last_chunk else ("noise", "full"))
            if forcing == "teacher"
            else ("full",)
        )
        noise_start = cursor
        chunk_starts.append(noise_start)

        for role in roles:
            positions.append(chunk_positions)
            tags.append(chunk_tags)
            split_lens.append(chunk_rows)
            attn_modes.append(role)
            is_noisy = forcing == "diffusion" or role == "noise"
            if is_noisy:
                role_rows.append(
                    torch.cat(
                        (
                            torch.zeros(reference_rows, dtype=torch.long),
                            torch.ones(
                                audio_rows + target_video_rows, dtype=torch.long
                            ),
                        )
                    )
                )
            else:
                role_rows.append(torch.zeros(chunk_rows, dtype=torch.long))

            reference_img_pos_parts.append(
                torch.arange(cursor, cursor + reference_rows, dtype=torch.long)
            )
            audio_pos_parts.append(
                torch.arange(
                    cursor + reference_rows,
                    cursor + reference_rows + audio_rows,
                    dtype=torch.long,
                )
            )
            target_img_pos = torch.arange(
                cursor + reference_rows + audio_rows,
                cursor + chunk_rows,
                dtype=torch.long,
            )
            all_target_img_pos_parts.append(target_img_pos)
            if is_noisy:
                noisy_target_img_pos_parts.append(target_img_pos)
            cursor += chunk_rows

        clean_chunk_starts.append(
            None
            if chunk_index == last_chunk
            else noise_start + chunk_rows
            if forcing == "teacher"
            else noise_start
        )

    q_ranges, k_ranges, attn_type_map, attn_workloads = _prepare_flex_attention_mask(
        split_lens,
        attn_modes,
        sink=int(sink),
        window_size=None if window_size is None else int(window_size),
    )
    reference_img_pos = torch.cat(reference_img_pos_parts)
    target_img_pos = torch.cat(noisy_target_img_pos_parts)
    audio_pos = torch.cat(audio_pos_parts)
    img_pos = torch.cat(
        [
            torch.cat((reference, target))
            for reference, target in zip(
                reference_img_pos_parts, all_target_img_pos_parts, strict=True
            )
        ]
    )
    packed = {
        "latent_shapes": target_shape,
        "audio_shapes": audio_shape,
        "reference_latent_shapes": reference_shape,
        "video_chunk_ranges": video_ranges,
        "audio_chunk_ranges": audio_ranges,
        "chunk_starts": chunk_starts,
        "clean_chunk_starts": clean_chunk_starts,
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "sample_lens": cursor,
        "packing_rows": cursor - text_len,
        "seqlens": latent_t * target_frame_rows + audio_t * _AUDIO_CHANNELS,
        "position_ids": torch.cat(positions),
        "token_tags": torch.cat(tags),
        "inverse_indices": torch.cat(role_rows),
        "text_pos": text_pos,
        "img_pos": img_pos,
        "reference_img_pos": reference_img_pos,
        "target_img_pos": target_img_pos,
        "img_pos_for_infer_output": target_img_pos,
        "audio_pos": audio_pos,
        "q_ranges": q_ranges,
        "k_ranges": k_ranges,
        "attn_type_map": attn_type_map,
        "attn_workloads": attn_workloads,
        "sinks": int(sink),
        "window_sizes": None if window_size is None else int(window_size),
    }
    if video_temporal_mapping != H3_VIDEO_TEMPORAL_MAPPING:
        packed["video_temporal_mapping"] = video_temporal_mapping.to_dict()
    return packed


class _CausalVideoRefDataset(CausalTextOnlyT2AVDataset):
    """Shared causal VideoRef chunk options and codec-derived target geometry."""

    _MIN_SINK = 0

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        chunk_size: int = 5,
        independent_first_chunk: int | None = None,
        sink: int = 1,
        window_size: int | None = None,
        video_temporal_mapping: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.chunk_size = _positive_chunk_size(chunk_size, "chunk_size")
        self.independent_first_chunk = (
            None if independent_first_chunk is None
            else _positive_chunk_size(independent_first_chunk, "independent_first_chunk")
        )
        self.video_temporal_mapping = (
            H3_VIDEO_TEMPORAL_MAPPING if video_temporal_mapping is None
            else VideoTemporalMapping.from_dict(video_temporal_mapping)
        )
        sink = _nonnegative_chunk_count(sink, "sink")
        if window_size is not None:
            window_size = _nonnegative_chunk_count(window_size, "window_size")
        super().__init__(
            seed, resume_context, sink=sink, window_size=window_size, **kwargs
        )

    def _target_video_latent_t(self, frame_count: int) -> int:
        return self.video_temporal_mapping.target_latent_t(frame_count)


class CausalVideoRefLatentT2AVDataset(_CausalVideoRefDataset):
    """Stateful latent stream with one normalized video reference."""

    _STATE_SCHEMA = "minimax_h3_videoref_causal_video_ref_latent_df_worker"
    _STATE_VERSION = 2
    forcing = "diffusion"

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        latent_dir: str,
        **kwargs: Any,
    ) -> None:
        self.latent_paths = sorted(Path(latent_dir).glob("**/prompt*.pt"))
        if not self.latent_paths:
            raise FileNotFoundError(f"No latent files found in: {latent_dir}")
        super().__init__(seed, resume_context, **kwargs)

    def _pack_video_ref_sample(
        self, *, prompt: str, reference_video_shape: Sequence[int]
    ) -> dict[str, Any]:
        text_input_ids, text_len = self.tokenizer.encode(prompt)
        return {
            "prompts": prompt,
            "text_input_ids": text_input_ids,
            "text_lens": text_len,
            **build_causal_video_ref_layout(
                text_len=text_len,
                target_video_shape=self.latent_shape,
                target_audio_shape=self.audio_shape,
                reference_video_shape=reference_video_shape,
                forcing=self.forcing,
                chunk_size=self.chunk_size,
                independent_first_chunk=self.independent_first_chunk,
                sink=self.sink,
                window_size=self.window_size,
                video_temporal_mapping=self.video_temporal_mapping,
            ),
        }

    def __iter__(self):
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                "worker topology mismatch: context expects "
                f"{effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        if logical_worker_id in self._decoded_worker_states:
            offset, avg_seqlen, cnt = self._decoded_worker_states[logical_worker_id]
        else:
            offset, avg_seqlen, cnt = self._initial_worker_state(logical_worker_id)

        while True:
            rng = random.Random(offset)
            samples: list[dict[str, Any]] = []
            cur_rows = 0
            num_retries = 0
            while len(samples) == 0 or cur_rows + avg_seqlen <= self.max_seqlen:
                path = self.latent_paths[rng.randrange(len(self.latent_paths))]
                entry = torch.load(path, map_location="cpu", weights_only=True)
                prompt = entry["prompt"]
                video = entry["video"]
                audio = entry["audio"]
                reference = entry["reference_video"]
                _, _, reference_shape = _latent_shapes(
                    video.shape, audio.shape, reference.shape
                )
                if (
                    tuple(video.shape) != self.latent_shape
                    or tuple(audio.shape) != self.audio_shape
                ):
                    raise ValueError(
                        f"{path}: corpus target latents disagree with the configured layout"
                    )
                if self.text_dropout > 0.0 and rng.random() < self.text_dropout:
                    prompt = ""
                candidate = self._pack_video_ref_sample(
                    prompt=prompt, reference_video_shape=reference_shape
                )
                candidate["video_latents"] = video
                candidate["audio_latents"] = audio
                candidate["reference_video_latents"] = reference
                packing_rows = int(candidate["packing_rows"])
                if (
                    self.max_seqlen_per_sample is not None
                    and packing_rows > self.max_seqlen_per_sample
                ) or cur_rows + packing_rows > self.max_seqlen:
                    if cur_rows + packing_rows > self.max_seqlen:
                        num_retries += 1
                        if num_retries >= self.max_retries:
                            break
                    continue
                avg_seqlen = avg_seqlen * cnt / (cnt + 1) + packing_rows / (cnt + 1)
                cnt += 1
                cur_rows += packing_rows
                num_retries = 0
                samples.append(candidate)

            if not samples:
                raise ValueError("no video-reference latent atom fits max_seqlen")
            offset = yield_seed(offset)
            batch = {key: [sample[key] for sample in samples] for key in samples[0]}
            state_after = self._encode_worker_state(
                logical_worker_id, offset, avg_seqlen, cnt
            )
            yield WorkerStateEnvelope(batch, logical_worker_id, state_after)


class CausalVideoRefLatentDiffusionForcingT2AVDataset(CausalVideoRefLatentT2AVDataset):
    """Explicit diffusion-forcing dataset entry class."""


EntryClass = CausalVideoRefLatentDiffusionForcingT2AVDataset

__all__ = [
    "CausalVideoRefLatentDiffusionForcingT2AVDataset",
    "CausalVideoRefLatentT2AVDataset",
    "EntryClass",
    "build_causal_video_ref_layout",
    "causal_video_ref_target_shapes",
]
