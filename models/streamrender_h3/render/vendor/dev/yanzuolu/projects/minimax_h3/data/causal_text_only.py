"""Stateful synthetic causal text-only dataset for MiniMax H3 DMD training."""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dev.yanzuolu.common.data import WorkerResumeContext, WorkerStateEnvelope, WorkerStateLoadError
from dev.yanzuolu.common.seed import yield_seed
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS
from dev.yanzuolu.projects.minimax_h3.modeling.packing import _axis_from_sqrt_area
from dev.yanzuolu.projects.minimax_h3.modeling.tokenizer import MiniMaxH3Tokenizer
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
    minimax_h3_audio_latent_t,
    minimax_h3_video_latent_t,
)
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask


_VIDEO_LATENTS_PER_CHUNK = 5
_AUDIO_CHANNELS = 2


def _video_position_start(
    index: int,
    origin: float,
    *,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> float:
    """Return one video latent's shared-clock start without building prior rows."""
    return video_temporal_mapping.position_start(index, origin)


def _positive_chunk_size(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _video_chunk_ranges(
    latent_t: int,
    *,
    chunk_size: int = _VIDEO_LATENTS_PER_CHUNK,
    independent_first_chunk: int | None = None,
) -> list[tuple[int, int]]:
    """Split a latent sequence with an optional first length and a short tail."""
    chunk_size = _positive_chunk_size(chunk_size, "chunk_size")
    first_chunk = (
        chunk_size
        if independent_first_chunk is None
        else _positive_chunk_size(independent_first_chunk, "independent_first_chunk")
    )
    if latent_t <= 0:
        return []
    first_stop = min(first_chunk, latent_t)
    return [(0, first_stop)] + [
        (start, min(start + chunk_size, latent_t))
        for start in range(first_stop, latent_t, chunk_size)
    ]


def _video_chunk_t_grid(
    start: int,
    stop: int,
    origin: float,
    *,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> torch.Tensor:
    """Build a chunk's temporal positions directly from the closed form."""
    return torch.tensor(
        [
            _video_position_start(index, origin, video_temporal_mapping=video_temporal_mapping)
            for index in range(start, stop)
        ],
        dtype=torch.float64,
    )


def _audio_chunk_ranges(
    latent_t: int,
    audio_t: int,
    *,
    video_ranges: Sequence[Sequence[int]] | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> list[tuple[int, int]]:
    """Assign audio starts to video chunks on the shared 40 Hz clock.

    Audio latent ``j`` belongs to chunk ``c`` exactly when its start lies in
    ``[T_c, T_{c+1})``. Integer ceiling on the codec's decoded frame boundaries
    keeps an audio latent exactly on a boundary with the following video chunk.
    """
    if video_ranges is None:
        video_ranges = _video_chunk_ranges(latent_t)
    boundary = video_temporal_mapping.decode_timeline.clock_boundary_ceil
    return [
        (min(boundary(int(start)), audio_t), min(boundary(int(stop)), audio_t))
        for start, stop in video_ranges
    ]


def _stereo_chunk_indices(audio_t: int, start: int, stop: int) -> torch.Tensor:
    """Map one chunk into native channel-major ``[L(all) | R(all)]`` rows."""
    left = torch.arange(start, stop, dtype=torch.long)
    right = torch.arange(audio_t + start, audio_t + stop, dtype=torch.long)
    return torch.cat((left, right))


def _gather_stereo_chunk(native_rows: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Gather ``[L(start:stop) | R(start:stop)]`` from native global order."""
    audio_t = int(native_rows.shape[0]) // _AUDIO_CHANNELS
    return native_rows.index_select(0, _stereo_chunk_indices(audio_t, start, stop))


def _scatter_stereo_chunks(
    chunks: Sequence[torch.Tensor],
    chunk_ranges: Sequence[tuple[int, int]],
    audio_t: int,
) -> torch.Tensor:
    """Invert chunk-local stereo gathers back to native global order."""
    native = chunks[0].new_empty((audio_t * _AUDIO_CHANNELS, *chunks[0].shape[1:]))
    for chunk, (start, stop) in zip(chunks, chunk_ranges, strict=True):
        native.index_copy_(0, _stereo_chunk_indices(audio_t, start, stop), chunk)
    return native


def _chunk_position_ids(
    *,
    video_start: int,
    video_stop: int,
    audio_start: int,
    audio_stop: int,
    latent_h: int,
    latent_w: int,
    origin: int,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
) -> torch.Tensor:
    """Build one chunk's float64 positions in audio-first, video-second order.

    Video time is evaluated from the closed form for only this chunk. This is
    intentionally not a slice of a clip-wide materialization because later
    position rebasing needs chunks to remain independently constructible.
    Spatial coordinates reuse H3's private helper so training stays bit-for-bit
    aligned with the model's validated packed-sequence geometry.
    """
    sqrt_area = float((latent_h * latent_w) ** 0.5)
    h_grid = _axis_from_sqrt_area(latent_h, 2, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, 2, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)
    frame_rows = int(frame.shape[0])

    video_t = _video_chunk_t_grid(
        video_start, video_stop, float(origin), video_temporal_mapping=video_temporal_mapping
    )
    video = torch.zeros(video_t.numel(), frame_rows, 3, dtype=torch.float64)
    video[:, :, 0] = video_t[:, None]
    video[:, :, 1:] = frame[None]

    audio_t = torch.arange(audio_start, audio_stop, dtype=torch.float64)
    audio = torch.zeros(audio_t.numel() * _AUDIO_CHANNELS, 3, dtype=torch.float64)
    audio[:, 0] = (float(origin) + audio_t).repeat(_AUDIO_CHANNELS)
    audio[: audio_t.numel(), 2] = float(w_grid[0])
    audio[audio_t.numel() :, 2] = float(w_grid[-1])

    return torch.cat((audio, video.reshape(-1, 3)), dim=0)


class CausalTextOnlyT2AVDataset(IterableDataset):
    """Infinite worker-local stream of chunk-packed MiniMax H3 shape batches."""

    _STATE_SCHEMA = "jarvis_minimax_h3_causal_text_only_worker"
    _STATE_VERSION = 1
    _MIN_SINK = 1
    forcing = "teacher"

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        tokenizer_path: str,
        paths: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
        default_prompt: str = "A synthetic training video with synchronized audio.",
        height: int = 768,
        width: int = 1280,
        num_frames: int = 192,
        video_latent_channels: int = 24,
        audio_latent_channels: int = 32,
        spatial_vae_stride: int = 16,
        sink: int = 1,
        window_size: int | None = None,
        total_seqlen: int | None = None,
        max_seqlen: int | None = None,
        max_seqlen_per_sample: int | None = None,
        max_retries: int = 5,
        sync_group_size: int = 1,
        text_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.resume_context = resume_context
        self.text_dropout = float(text_dropout)
        self.sync_group_size = int(sync_group_size)
        if self.sync_group_size <= 0:
            raise ValueError(f"sync_group_size must be positive, got {sync_group_size}")
        self.tokenizer = MiniMaxH3Tokenizer(tokenizer_path)

        self._decoded_worker_states: dict[int, tuple[int, float, int]] = {}
        for logical_id, snapshot in resume_context.committed_states.items():
            try:
                state = pickle.loads(snapshot)
                restored = (state["offset"], float(state["avg_seqlen"]), state["cnt"])
            except (KeyError, TypeError, ValueError, pickle.UnpicklingError, EOFError) as exc:
                raise WorkerStateLoadError(
                    f"invalid snapshot for logical worker {logical_id}"
                ) from exc
            if (
                (state.get("schema"), state.get("version"), state.get("logical_worker_id"))
                != (self._STATE_SCHEMA, self._STATE_VERSION, logical_id)
                or not isinstance(restored[0], int)
                or not math.isfinite(restored[1])
                or not isinstance(restored[2], int)
                or restored[2] < 0
            ):
                raise WorkerStateLoadError(
                    f"invalid snapshot for logical worker {logical_id}"
                )
            self._decoded_worker_states[logical_id] = restored

        loaded: list[str] = []
        seen: set[str] = set()

        def add_prompt(prompt: str) -> None:
            prompt = prompt.strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                loaded.append(prompt)

        for prompt in prompts or []:
            add_prompt(str(prompt))
        text_files: list[Path] = []
        for raw_path in paths or []:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {raw_path}")
            if path.is_dir():
                text_files.extend(sorted(item for item in path.rglob("*.txt") if item.is_file()))
            elif path.suffix == ".txt":
                text_files.append(path)
            else:
                raise ValueError(f"Only .txt files are supported, got: {raw_path}")
        for file_path in sorted(set(text_files)):
            with file_path.open("r", encoding="utf8") as handle:
                for line in handle:
                    add_prompt(line)
        if not loaded:
            add_prompt(default_prompt)
        if not loaded:
            raise ValueError("No valid prompts found; pass prompts, paths, or default_prompt")
        self.prompts = loaded

        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.video_latent_channels = int(video_latent_channels)
        self.audio_latent_channels = int(audio_latent_channels)
        self.spatial_vae_stride = int(spatial_vae_stride)
        if min(
            self.height,
            self.width,
            self.video_latent_channels,
            self.audio_latent_channels,
            self.spatial_vae_stride,
        ) <= 0:
            raise ValueError("spatial and latent dimensions must be positive")
        if self.height % self.spatial_vae_stride or self.width % self.spatial_vae_stride:
            raise ValueError("height and width must be divisible by spatial_vae_stride")

        self.latent_t = self._target_video_latent_t(self.num_frames)
        self.latent_h = self.height // self.spatial_vae_stride
        self.latent_w = self.width // self.spatial_vae_stride
        if self.latent_h % 2 or self.latent_w % 2:
            raise ValueError("MiniMax H3 latent height and width must be divisible by patch (2, 2)")
        self.audio_t = minimax_h3_audio_latent_t(
            self.num_frames / float(MINIMAX_H3_SUPPORTED_FPS)
        )
        self.latent_shape = (
            self.video_latent_channels,
            self.latent_t,
            self.latent_h,
            self.latent_w,
        )
        self.audio_shape = (
            _AUDIO_CHANNELS,
            self.audio_latent_channels,
            self.audio_t,
        )

        self.sink = int(sink)
        if self.sink < self._MIN_SINK:
            raise ValueError(f"sink must be at least {self._MIN_SINK}")
        self.window_size = None if window_size is None else int(window_size)

        resolved_max_seqlen = (
            max_seqlen
            if total_seqlen is None
            else int(total_seqlen) // resume_context.world_size
        )
        if resolved_max_seqlen is None or int(resolved_max_seqlen) <= 0:
            raise ValueError(
                f"max_seqlen must be provided and positive, got {resolved_max_seqlen}"
            )
        self.max_seqlen = int(resolved_max_seqlen)
        self.max_seqlen_per_sample = (
            None if max_seqlen_per_sample is None else int(max_seqlen_per_sample)
        )
        self.max_retries = int(max_retries)
        if self.max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {max_retries}")

    def _target_video_latent_t(self, frame_count: int) -> int:
        return minimax_h3_video_latent_t(frame_count)

    def _initial_worker_state(self, logical_worker_id: int) -> tuple[int, float, int]:
        offset = (
            self.seed
            + (self.resume_context.rank // self.sync_group_size)
            * (self.resume_context.num_workers or 1)
            + logical_worker_id
        )
        return offset, 0.0, 0

    def _encode_worker_state(
        self,
        logical_worker_id: int,
        offset: int,
        avg_seqlen: float,
        cnt: int,
    ) -> bytes:
        return pickle.dumps(
            {
                "schema": self._STATE_SCHEMA,
                "version": self._STATE_VERSION,
                "logical_worker_id": logical_worker_id,
                "offset": offset,
                "avg_seqlen": avg_seqlen,
                "cnt": cnt,
            }
        )

    def initial_worker_snapshot(self, logical_worker_id: int) -> bytes:
        return self._encode_worker_state(
            logical_worker_id, *self._initial_worker_state(logical_worker_id)
        )

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
        worker_prompts = list(self.prompts)
        random.Random(self.seed).shuffle(worker_prompts)

        while True:
            rng = random.Random(offset)
            samples: list[dict[str, Any]] = []
            cur_seqlen = 0
            num_retries = 0
            while len(samples) == 0 or cur_seqlen + avg_seqlen <= self.max_seqlen:
                prompt = worker_prompts[rng.randrange(len(worker_prompts))]
                if self.text_dropout > 0.0 and rng.random() < self.text_dropout:
                    prompt = ""
                candidate = self._pack_sample(prompt=prompt)
                candidate_seqlen = candidate["seqlens"]
                if (
                    (
                        self.max_seqlen_per_sample is not None
                        and candidate_seqlen > self.max_seqlen_per_sample
                    )
                    or cur_seqlen + candidate_seqlen > self.max_seqlen
                ):
                    if cur_seqlen + candidate_seqlen > self.max_seqlen:
                        num_retries += 1
                        if num_retries >= self.max_retries:
                            break
                    continue
                avg_seqlen = avg_seqlen * cnt / (cnt + 1) + candidate_seqlen / (cnt + 1)
                cnt += 1
                cur_seqlen += candidate_seqlen
                num_retries = 0
                samples.append(candidate)

            offset = yield_seed(offset)
            batch = {key: [sample[key] for sample in samples] for key in samples[0]}
            state_after = self._encode_worker_state(
                logical_worker_id, offset, avg_seqlen, cnt
            )
            yield WorkerStateEnvelope(batch, logical_worker_id, state_after)

    def _pack_sample(self, *, prompt: str) -> dict[str, Any]:
        text_input_ids, num_text_tokens = self.tokenizer.encode(prompt)
        frame_rows = (self.latent_h // 2) * (self.latent_w // 2)
        video_ranges = _video_chunk_ranges(self.latent_t)
        audio_ranges = _audio_chunk_ranges(self.latent_t, self.audio_t)

        positions = [
            torch.stack(
                (
                    torch.arange(num_text_tokens, dtype=torch.float64),
                    torch.zeros(num_text_tokens, dtype=torch.float64),
                    torch.zeros(num_text_tokens, dtype=torch.float64),
                ),
                dim=-1,
            )
        ]
        tags = [torch.full((num_text_tokens,), 1, dtype=torch.long)]
        inverse = [torch.zeros(num_text_tokens, dtype=torch.long)]
        text_pos = torch.arange(num_text_tokens, dtype=torch.long)
        img_pos_parts: list[torch.Tensor] = []
        audio_pos_parts: list[torch.Tensor] = []
        split_lens = [num_text_tokens]
        attn_modes = ["full"]
        cursor = num_text_tokens

        for chunk_index, ((video_start, video_stop), (audio_start, audio_stop)) in enumerate(
            zip(video_ranges, audio_ranges, strict=True)
        ):
            video_rows = (video_stop - video_start) * frame_rows
            audio_rows = (audio_stop - audio_start) * _AUDIO_CHANNELS
            chunk_rows = audio_rows + video_rows
            chunk_positions = _chunk_position_ids(
                video_start=video_start,
                video_stop=video_stop,
                audio_start=audio_start,
                audio_stop=audio_stop,
                latent_h=self.latent_h,
                latent_w=self.latent_w,
                origin=num_text_tokens,
            )
            chunk_tags = torch.cat(
                (
                    torch.full((audio_rows,), 2, dtype=torch.long),
                    torch.zeros(video_rows, dtype=torch.long),
                )
            )
            if self.forcing == "teacher":
                roles = (
                    ("noise",)
                    if chunk_index == len(video_ranges) - 1
                    else ("noise", "full")
                )
            else:
                roles = ("full",)
            for role in roles:
                split_lens.append(chunk_rows)
                attn_modes.append(role)
                positions.append(chunk_positions)
                tags.append(chunk_tags)
                inverse.append(
                    torch.full(
                        (chunk_rows,),
                        1 if self.forcing == "diffusion" or role == "noise" else 0,
                        dtype=torch.long,
                    )
                )
                audio_pos_parts.append(torch.arange(cursor, cursor + audio_rows, dtype=torch.long))
                img_pos_parts.append(
                    torch.arange(
                        cursor + audio_rows,
                        cursor + audio_rows + video_rows,
                        dtype=torch.long,
                    )
                )
                cursor += chunk_rows

        q_ranges, k_ranges, attn_type_map, attn_workloads = _prepare_flex_attention_mask(
            split_lens,
            attn_modes,
            sink=self.sink,
            window_size=self.window_size,
        )
        img_pos = torch.cat(img_pos_parts)
        audio_pos = torch.cat(audio_pos_parts)
        unique_media_rows = self.latent_t * frame_rows + self.audio_t * _AUDIO_CHANNELS

        return {
            "prompts": prompt,
            "text_input_ids": text_input_ids,
            "text_lens": num_text_tokens,
            "latent_shapes": self.latent_shape,
            "audio_shapes": self.audio_shape,
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            "sample_lens": cursor,
            "seqlens": unique_media_rows,
            "position_ids": torch.cat(positions, dim=0),
            "token_tags": torch.cat(tags),
            "inverse_indices": torch.cat(inverse),
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "img_pos_for_infer_output": img_pos,
            "q_ranges": q_ranges,
            "k_ranges": k_ranges,
            "attn_type_map": attn_type_map,
            "attn_workloads": attn_workloads,
            "sinks": self.sink,
            "window_sizes": self.window_size,
        }


__all__ = ["CausalTextOnlyT2AVDataset"]
