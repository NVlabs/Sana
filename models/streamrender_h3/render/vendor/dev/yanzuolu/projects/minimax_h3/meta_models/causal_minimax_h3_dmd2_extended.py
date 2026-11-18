# SPDX-License-Identifier: Apache-2.0
"""Extended-horizon DMD2 with supervision restricted to the original H3 window.

The rollout owns a variable number of complete five-latent chunks followed by
whatever ragged tail the input layout carries.  Critics keep seeing the original
window and its original coordinates.  Only the causal generator forward sees
the detached clean prefix at its absolute extended-horizon coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterator, Sequence

import torch

from dev.yanzuolu.common.distributed import ops
from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.distributed.unified_parallel import (
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import (
    _AUDIO_CHANNELS,
    _VIDEO_LATENTS_PER_CHUNK,
    _audio_chunk_ranges,
    _chunk_position_ids,
    _video_chunk_ranges,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import (
    _AUDIO_TAG,
    _PATCH_SIZE,
    _TEXT_TAG,
    _VIDEO_TAG,
    ForwardInput,
    _Chunk,
    _Layout,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    _TrajectoryRolloutX0s,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd2 import (
    CausalMiniMaxH3DMD2,
)
from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from dev.yanzuolu.projects.minimax_h3.modeling.packing import minimax_h3_packed_sequence
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    minimax_h3_audio_latent_t,
    minimax_h3_frame_count_from_video_latent_t,
)
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask


@dataclass(frozen=True)
class _ExtendedForwardInput(ForwardInput):
    """A synchronized input carrying its own rollout horizon."""

    rollout_full_chunks: int = 0


@dataclass(frozen=True)
class _GeneratorPack:
    """Physical clean-prefix/noisy-suffix sequence metadata for one GEN forward."""

    sample_lens: list[int]
    split_lens: list[list[int]]
    attn_modes: list[list[str]]
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos: torch.Tensor
    noisy_img_pos: torch.Tensor
    audio_noisy_sel: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor
    attn_workloads: list[int]
    output_layouts: list[_Layout]
    prefix_chunks: int
    prefix_video_stops: list[int]
    prefix_audio_stops: list[int]


@dataclass(frozen=True)
class _ExtendedRolloutX0s(_TrajectoryRolloutX0s):
    """Supervised suffix plus detached prefix state needed by the GEN phase."""

    prefix_video: list[torch.Tensor]
    prefix_audio: list[torch.Tensor]
    prefix_video_eps: list[torch.Tensor]
    prefix_audio_eps: list[torch.Tensor]
    extended_inputs: ForwardInput
    prefix_chunks: int


def _copy_forward_input(inputs: ForwardInput, **extra: Any) -> _ExtendedForwardInput:
    values = {field.name: getattr(inputs, field.name) for field in fields(ForwardInput)}
    values.update(extra)
    return _ExtendedForwardInput(**values)


class CausalMiniMaxH3DMD2Extended(CausalMiniMaxH3DMD2):
    """Roll beyond 192f while supervising only the input-derived suffix window."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        raw_range = config.meta_model.rollout_chunk_range
        if (
            not isinstance(raw_range, (list, tuple))
            or len(raw_range) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw_range
            )
        ):
            raise ValueError("meta_model.rollout_chunk_range must contain two integers")
        minimum, maximum = (int(value) for value in raw_range)
        if minimum <= 0 or minimum > maximum:
            raise ValueError(
                "meta_model.rollout_chunk_range must be a positive inclusive range"
            )
        self.rollout_chunk_range = (minimum, maximum)

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Assign one first-SP-group horizon to every UP local source slot."""
        ctx = super().prepare_inputs(ctx)
        inputs = ctx["inputs"]
        candidates = self._valid_rollout_full_chunks(inputs)

        up_size = (
            get_unified_parallel_world_size()
            if is_unified_parallel_initialized()
            else 1
        )
        rank = ops.get_rank()
        candidate = (
            self._draw_rollout_full_chunks(candidates, ctx["rng"])
            if rank < up_size
            else None
        )
        gathered = ops.all_gather_object(candidate)
        slot_chunks = gathered[:up_size]
        if any(value is None for value in slot_chunks):
            raise RuntimeError(
                "the first SP group did not provide every rollout horizon"
            )
        slot_chunks = [int(value) for value in slot_chunks]
        local_slot = (
            get_unified_parallel_rank() if is_unified_parallel_initialized() else 0
        )
        ctx["inputs"] = _copy_forward_input(
            inputs, rollout_full_chunks=slot_chunks[local_slot]
        )
        ctx["rollout_chunk_slots"] = slot_chunks
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Attach the first group's slot horizon after each SP source broadcast."""
        slot_chunks = [int(value) for value in ctx["rollout_chunk_slots"]]
        for local_slot, synced_ctx in enumerate(super().sync_inputs(ctx)):
            if local_slot >= len(slot_chunks):
                raise RuntimeError(
                    "sync_inputs yielded more UP slots than were sampled"
                )
            synced_ctx["inputs"] = _copy_forward_input(
                synced_ctx["inputs"],
                rollout_full_chunks=slot_chunks[local_slot],
            )
            yield synced_ctx

    @staticmethod
    def _supervised_full_chunks(inputs: ForwardInput) -> int:
        counts = []
        tails = []
        for layout in inputs.layouts:
            latent_t = int(layout.latent_shape[1])
            full_chunks, tail = divmod(latent_t, _VIDEO_LATENTS_PER_CHUNK)
            counts.append(full_chunks)
            tails.append(tail)
        if len(set(counts)) != 1 or len(set(tails)) != 1:
            raise ValueError(
                "all packed samples must share one supervised chunk layout"
            )
        if tails[0] <= 0:
            raise ValueError(
                "extended DMD2 requires the input layout to end in a ragged tail"
            )
        if len(inputs.layouts[0].chunks) != counts[0] + 1:
            raise ValueError(
                "input logical chunks disagree with its video latent shape"
            )
        return counts[0]

    @staticmethod
    def _suffix_audio_latents(layout: _Layout, full_chunks: int) -> int:
        source_t = int(layout.latent_shape[1])
        supervised_full_chunks, tail = divmod(source_t, _VIDEO_LATENTS_PER_CHUNK)
        if full_chunks < supervised_full_chunks:
            return -1
        latent_t = full_chunks * _VIDEO_LATENTS_PER_CHUNK + tail
        frame_count = minimax_h3_frame_count_from_video_latent_t(latent_t)
        audio_t = minimax_h3_audio_latent_t(
            frame_count / float(MINIMAX_H3_SUPPORTED_FPS)
        )
        prefix_chunks = full_chunks - supervised_full_chunks
        audio_ranges = _audio_chunk_ranges(latent_t, audio_t)
        return audio_t - audio_ranges[prefix_chunks][0]

    def _valid_rollout_full_chunks(self, inputs: ForwardInput) -> list[int]:
        supervised_full_chunks = self._supervised_full_chunks(inputs)
        minimum, maximum = self.rollout_chunk_range
        if minimum < supervised_full_chunks:
            raise ValueError(
                f"rollout_chunk_range minimum {minimum} is shorter than the "
                f"input-derived supervised window of {supervised_full_chunks} full chunks"
            )
        candidates = [
            full_chunks
            for full_chunks in range(minimum, maximum + 1)
            if all(
                self._suffix_audio_latents(layout, full_chunks) == layout.audio_shape[2]
                for layout in inputs.layouts
            )
        ]
        if not candidates:
            audio_lengths = [layout.audio_shape[2] for layout in inputs.layouts]
            raise ValueError(
                f"rollout_chunk_range [{minimum}, {maximum}] has no horizon whose "
                f"extended suffix audio length matches input lengths {audio_lengths}"
            )
        return candidates

    @staticmethod
    def _draw_rollout_full_chunks(candidates: Sequence[int], rng: Any) -> int:
        return int(rng.python_generator.choice(candidates))

    @staticmethod
    def _text_positions(text_len: int, device: torch.device) -> torch.Tensor:
        return torch.stack(
            (
                torch.arange(text_len, dtype=torch.float64, device=device),
                torch.zeros(text_len, dtype=torch.float64, device=device),
                torch.zeros(text_len, dtype=torch.float64, device=device),
            ),
            dim=-1,
        )

    def _build_extended_rollout_inputs(
        self, inputs: ForwardInput, full_chunks: int
    ) -> ForwardInput:
        """Build the ordinary noisy/clean causal ABI at an extended horizon."""
        device = inputs.token_tags.device
        sample_parts: list[dict[str, Any]] = []
        for index, source_layout in enumerate(inputs.layouts):
            source_t = int(source_layout.latent_shape[1])
            supervised_full, tail = divmod(source_t, _VIDEO_LATENTS_PER_CHUNK)
            if full_chunks < supervised_full:
                raise ValueError(
                    "rollout horizon is shorter than the supervised suffix"
                )
            latent_t = full_chunks * _VIDEO_LATENTS_PER_CHUNK + tail
            frame_count = minimax_h3_frame_count_from_video_latent_t(latent_t)
            audio_t = minimax_h3_audio_latent_t(
                frame_count / float(MINIMAX_H3_SUPPORTED_FPS)
            )
            latent_shape = (
                source_layout.latent_shape[0],
                latent_t,
                source_layout.latent_shape[2],
                source_layout.latent_shape[3],
            )
            audio_shape = (
                source_layout.audio_shape[0],
                source_layout.audio_shape[1],
                audio_t,
            )
            text_len = source_layout.text_len
            frame_rows = (latent_shape[2] // _PATCH_SIZE[1]) * (
                latent_shape[3] // _PATCH_SIZE[2]
            )
            video_ranges = _video_chunk_ranges(latent_t)
            audio_ranges = _audio_chunk_ranges(latent_t, audio_t)

            positions = [self._text_positions(text_len, device)]
            tags = [torch.full((text_len,), _TEXT_TAG, dtype=torch.long, device=device)]
            inverse = [torch.zeros(text_len, dtype=torch.long, device=device)]
            split_lens = [text_len]
            attn_modes = ["full"]
            img_pos_parts: list[torch.Tensor] = []
            audio_pos_parts: list[torch.Tensor] = []
            cursor = text_len
            for chunk_index, (
                (video_start, video_stop),
                (audio_start, audio_stop),
            ) in enumerate(zip(video_ranges, audio_ranges, strict=True)):
                video_rows = (video_stop - video_start) * frame_rows
                audio_rows = (audio_stop - audio_start) * _AUDIO_CHANNELS
                chunk_rows = video_rows + audio_rows
                chunk_positions = _chunk_position_ids(
                    video_start=video_start,
                    video_stop=video_stop,
                    audio_start=audio_start,
                    audio_stop=audio_stop,
                    latent_h=latent_shape[2],
                    latent_w=latent_shape[3],
                    origin=text_len,
                ).to(device)
                chunk_tags = torch.cat(
                    (
                        torch.full(
                            (audio_rows,), _AUDIO_TAG, dtype=torch.long, device=device
                        ),
                        torch.full(
                            (video_rows,), _VIDEO_TAG, dtype=torch.long, device=device
                        ),
                    )
                )
                roles = (
                    ("noise",)
                    if chunk_index == len(video_ranges) - 1
                    else ("noise", "full")
                )
                for role in roles:
                    split_lens.append(chunk_rows)
                    attn_modes.append(role)
                    positions.append(chunk_positions)
                    tags.append(chunk_tags)
                    inverse.append(
                        torch.full(
                            (chunk_rows,),
                            1 if role == "noise" else 0,
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    audio_pos_parts.append(
                        torch.arange(cursor, cursor + audio_rows, device=device)
                    )
                    img_pos_parts.append(
                        torch.arange(
                            cursor + audio_rows,
                            cursor + chunk_rows,
                            device=device,
                        )
                    )
                    cursor += chunk_rows

            token_tags = torch.cat(tags)
            layout = self._build_layout(
                split_lens=split_lens,
                attn_modes=attn_modes,
                token_tags=token_tags,
                text_len=text_len,
                sample_len=cursor,
                latent_shape=latent_shape,
                audio_shape=audio_shape,
            )
            q_ranges, k_ranges, type_map, workload = _prepare_flex_attention_mask(
                split_lens,
                attn_modes,
                sink=inputs.sinks[index],
                window_size=inputs.window_sizes[index],
                device=device,
            )
            sample_parts.append(
                {
                    "layout": layout,
                    "split_lens": split_lens,
                    "attn_modes": attn_modes,
                    "sample_len": cursor,
                    "seqlen": latent_t * frame_rows + audio_t * _AUDIO_CHANNELS,
                    "position_ids": torch.cat(positions),
                    "token_tags": token_tags,
                    "inverse": torch.cat(inverse),
                    "img_pos": torch.cat(img_pos_parts),
                    "audio_pos": torch.cat(audio_pos_parts),
                    "text_pos": torch.arange(text_len, device=device),
                    "q_ranges": q_ranges,
                    "k_ranges": k_ranges,
                    "type_map": type_map,
                    "workload": workload,
                }
            )

        sample_lens = [part["sample_len"] for part in sample_parts]
        offsets = self._offsets(sample_lens)
        img_pos = torch.cat(
            [part["img_pos"] + offset for part, offset in zip(sample_parts, offsets)]
        )
        audio_pos = torch.cat(
            [part["audio_pos"] + offset for part, offset in zip(sample_parts, offsets)]
        )
        text_pos = torch.cat(
            [part["text_pos"] + offset for part, offset in zip(sample_parts, offsets)]
        )
        q_ranges = torch.cat(
            [part["q_ranges"] + offset for part, offset in zip(sample_parts, offsets)]
        )
        k_ranges = torch.cat(
            [part["k_ranges"] + offset for part, offset in zip(sample_parts, offsets)]
        )
        token_tags = torch.cat([part["token_tags"] for part in sample_parts])
        clean_rows = torch.cat([part["inverse"] for part in sample_parts]) == 0
        layouts = [part["layout"] for part in sample_parts]
        self._check_layout_agrees(layouts, offsets, img_pos, audio_pos, token_tags)

        native = [
            self._native_to_device(
                minimax_h3_packed_sequence(
                    text_len=layout.text_len,
                    latent_t=layout.latent_shape[1],
                    latent_h=layout.latent_shape[2],
                    latent_w=layout.latent_shape[3],
                    audio_t=layout.audio_shape[2],
                    audio_channel=_AUDIO_CHANNELS,
                    include_keyframe_cond=False,
                ),
                device,
            )
            for layout in layouts
        ]
        return ForwardInput(
            batch_size=inputs.batch_size,
            prompt_embeds=list(inputs.prompt_embeds),
            text_lens=list(inputs.text_lens),
            seqlens=torch.tensor(
                [part["seqlen"] for part in sample_parts],
                dtype=torch.int32,
                device=device,
            ),
            layouts=layouts,
            sinks=list(inputs.sinks),
            window_sizes=list(inputs.window_sizes),
            sample_lens=sample_lens,
            split_lens=[part["split_lens"] for part in sample_parts],
            attn_modes=[part["attn_modes"] for part in sample_parts],
            position_ids=torch.cat([part["position_ids"] for part in sample_parts]),
            token_tags=token_tags,
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=torch.cat([part["type_map"] for part in sample_parts]),
            attn_workloads=[part["workload"] for part in sample_parts],
            noisy_img_pos=img_pos[~clean_rows[img_pos]],
            audio_noisy_sel=torch.nonzero(~clean_rows[audio_pos], as_tuple=False).view(
                -1
            ),
            native=native,
            clean_latents=None,
        )

    @staticmethod
    def _suffix_bounds(
        extended_layout: _Layout, suffix_layout: _Layout
    ) -> tuple[int, int, int]:
        prefix_chunks = len(extended_layout.chunks) - len(suffix_layout.chunks)
        if prefix_chunks < 0:
            raise ValueError("extended layout is shorter than the supervised layout")
        first_suffix = extended_layout.chunks[prefix_chunks]
        return prefix_chunks, first_suffix.video_start, first_suffix.audio_start

    @staticmethod
    def _slice_suffix(
        values: Sequence[torch.Tensor],
        video_starts: Sequence[int],
        audio_starts: Sequence[int],
        modality: str,
    ) -> list[torch.Tensor]:
        if modality == "video":
            return [
                value[:, start:].detach().clone()
                for value, start in zip(values, video_starts, strict=True)
            ]
        return [
            value[:, :, start:].detach().clone()
            for value, start in zip(values, audio_starts, strict=True)
        ]

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Roll the full horizon, then retain only the original suffix for critics."""
        tail_inputs = ctx["inputs"]
        if not isinstance(tail_inputs, _ExtendedForwardInput):
            raise TypeError(
                "prepare_inputs must attach rollout_full_chunks to ForwardInput"
            )
        extended_inputs = self._build_extended_rollout_inputs(
            tail_inputs, tail_inputs.rollout_full_chunks
        )
        extended_ctx = dict(ctx)
        extended_ctx["inputs"] = extended_inputs
        extended_ctx = super().rollout(extended_ctx)
        full_rollout = extended_ctx["rollout_x0s"]

        bounds = [
            self._suffix_bounds(extended, suffix)
            for extended, suffix in zip(
                extended_inputs.layouts, tail_inputs.layouts, strict=True
            )
        ]
        prefix_counts = {value[0] for value in bounds}
        if len(prefix_counts) != 1:
            raise ValueError("all samples must have the same prefix chunk count")
        prefix_chunks = prefix_counts.pop()
        video_starts = [value[1] for value in bounds]
        audio_starts = [value[2] for value in bounds]

        suffix_video = self._slice_suffix(
            full_rollout.video, video_starts, audio_starts, "video"
        )
        suffix_audio = self._slice_suffix(
            full_rollout.audio, video_starts, audio_starts, "audio"
        )
        suffix_video_eps = self._slice_suffix(
            full_rollout.video_eps, video_starts, audio_starts, "video"
        )
        suffix_audio_eps = self._slice_suffix(
            full_rollout.audio_eps, video_starts, audio_starts, "audio"
        )
        for index, layout in enumerate(tail_inputs.layouts):
            if suffix_video[index].shape != layout.latent_shape:
                raise ValueError(
                    "extended rollout video suffix disagrees with input layout"
                )
            if suffix_audio[index].shape != layout.audio_shape:
                raise ValueError(
                    "extended rollout audio suffix disagrees with input layout"
                )

        if self.fake_use_trajectory:
            fake_video = self._slice_suffix(
                full_rollout.fake_video, video_starts, audio_starts, "video"
            )
            fake_audio = self._slice_suffix(
                full_rollout.fake_audio, video_starts, audio_starts, "audio"
            )
        else:
            fake_video = [value.clone() for value in suffix_video]
            fake_audio = [value.clone() for value in suffix_audio]

        prefix_video = [
            value[:, :stop].detach().clone()
            for value, stop in zip(full_rollout.video, video_starts, strict=True)
        ]
        prefix_audio = [
            value[:, :, :stop].detach().clone()
            for value, stop in zip(full_rollout.audio, audio_starts, strict=True)
        ]
        prefix_video_eps = [
            value[:, :stop].detach().clone()
            for value, stop in zip(full_rollout.video_eps, video_starts, strict=True)
        ]
        prefix_audio_eps = [
            value[:, :, :stop].detach().clone()
            for value, stop in zip(full_rollout.audio_eps, audio_starts, strict=True)
        ]

        ctx["rollout_x0s"] = _ExtendedRolloutX0s(
            video=suffix_video,
            audio=suffix_audio,
            video_eps=suffix_video_eps,
            audio_eps=suffix_audio_eps,
            fake_video=fake_video,
            fake_audio=fake_audio,
            prefix_video=prefix_video,
            prefix_audio=prefix_audio,
            prefix_video_eps=prefix_video_eps,
            prefix_audio_eps=prefix_audio_eps,
            extended_inputs=extended_inputs,
            prefix_chunks=prefix_chunks,
            video_anchors=full_rollout.video_anchors,
        )
        ctx["trajectory_xts"] = [
            (
                self._slice_suffix(video, video_starts, audio_starts, "video"),
                self._slice_suffix(audio, video_starts, audio_starts, "audio"),
            )
            for video, audio in extended_ctx["trajectory_xts"]
        ]
        return ctx

    @staticmethod
    def _output_layout(
        source: _Layout,
        extended_chunks: Sequence[_Chunk],
    ) -> _Layout:
        chunks: list[_Chunk] = []
        video_cursor = 0
        audio_cursor = 0
        for index, chunk in enumerate(extended_chunks):
            video_len = chunk.video_stop - chunk.video_start
            audio_len = chunk.audio_stop - chunk.audio_start
            chunks.append(
                _Chunk(
                    noise_start=0,
                    clean_start=None,
                    has_clean_copy=False,
                    audio_rows=chunk.audio_rows,
                    video_rows=chunk.video_rows,
                    video_start=video_cursor,
                    video_stop=video_cursor + video_len,
                    audio_start=audio_cursor,
                    audio_stop=audio_cursor + audio_len,
                )
            )
            video_cursor += video_len
            audio_cursor += audio_len
        if (
            video_cursor != source.latent_shape[1]
            or audio_cursor != source.audio_shape[2]
        ):
            raise ValueError(
                "extended suffix geometry disagrees with the source window"
            )
        audio_t = source.audio_shape[2]
        audio_row_perm = torch.cat(
            [
                torch.cat(
                    (
                        torch.arange(chunk.audio_start, chunk.audio_stop),
                        torch.arange(
                            audio_t + chunk.audio_start,
                            audio_t + chunk.audio_stop,
                        ),
                    )
                )
                for chunk in chunks
            ]
        )
        return _Layout(
            text_len=source.text_len,
            chunks=tuple(chunks),
            audio_row_perm=audio_row_perm,
            latent_shape=source.latent_shape,
            audio_shape=source.audio_shape,
        )

    def _build_generator_pack(
        self,
        tail_inputs: ForwardInput,
        rollout: _ExtendedRolloutX0s,
    ) -> _GeneratorPack:
        """Pack clean-only prefix rows and ordinary supervised suffix rows."""
        device = tail_inputs.token_tags.device
        extended_inputs = rollout.extended_inputs
        sample_parts: list[dict[str, Any]] = []
        output_layouts: list[_Layout] = []
        prefix_video_stops: list[int] = []
        prefix_audio_stops: list[int] = []
        audio_occurrence_cursor = 0

        for index, (extended, source) in enumerate(
            zip(extended_inputs.layouts, tail_inputs.layouts, strict=True)
        ):
            prefix_chunks, prefix_video_stop, prefix_audio_stop = self._suffix_bounds(
                extended, source
            )
            if prefix_chunks != rollout.prefix_chunks:
                raise ValueError(
                    "rollout prefix metadata disagrees with extended layout"
                )
            prefix_video_stops.append(prefix_video_stop)
            prefix_audio_stops.append(prefix_audio_stop)
            positions = [self._text_positions(extended.text_len, device)]
            tags = [
                torch.full(
                    (extended.text_len,), _TEXT_TAG, dtype=torch.long, device=device
                )
            ]
            split_lens = [extended.text_len]
            attn_modes = ["full"]
            img_pos_parts: list[torch.Tensor] = []
            audio_pos_parts: list[torch.Tensor] = []
            noisy_img_parts: list[torch.Tensor] = []
            audio_noisy_parts: list[torch.Tensor] = []
            cursor = extended.text_len

            for chunk_index, chunk in enumerate(extended.chunks):
                chunk_positions = _chunk_position_ids(
                    video_start=chunk.video_start,
                    video_stop=chunk.video_stop,
                    audio_start=chunk.audio_start,
                    audio_stop=chunk.audio_stop,
                    latent_h=extended.latent_shape[2],
                    latent_w=extended.latent_shape[3],
                    origin=extended.text_len,
                ).to(device)
                chunk_tags = torch.cat(
                    (
                        torch.full(
                            (chunk.audio_rows,),
                            _AUDIO_TAG,
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.full(
                            (chunk.video_rows,),
                            _VIDEO_TAG,
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )
                if chunk_index < prefix_chunks:
                    roles = ("full",)
                else:
                    roles = (
                        ("noise",)
                        if chunk_index == len(extended.chunks) - 1
                        else ("noise", "full")
                    )
                for role in roles:
                    split_lens.append(chunk.rows)
                    attn_modes.append(role)
                    positions.append(chunk_positions)
                    tags.append(chunk_tags)
                    audio_pos_parts.append(
                        torch.arange(cursor, cursor + chunk.audio_rows, device=device)
                    )
                    img_rows = torch.arange(
                        cursor + chunk.audio_rows,
                        cursor + chunk.rows,
                        device=device,
                    )
                    img_pos_parts.append(img_rows)
                    if role == "noise":
                        noisy_img_parts.append(img_rows)
                        audio_noisy_parts.append(
                            torch.arange(
                                audio_occurrence_cursor,
                                audio_occurrence_cursor + chunk.audio_rows,
                                device=device,
                            )
                        )
                    audio_occurrence_cursor += chunk.audio_rows
                    cursor += chunk.rows

            q_ranges, k_ranges, type_map, workload = _prepare_flex_attention_mask(
                split_lens,
                attn_modes,
                sink=tail_inputs.sinks[index],
                window_size=tail_inputs.window_sizes[index],
                device=device,
            )
            sample_parts.append(
                {
                    "sample_len": cursor,
                    "split_lens": split_lens,
                    "attn_modes": attn_modes,
                    "position_ids": torch.cat(positions),
                    "token_tags": torch.cat(tags),
                    "img_pos": torch.cat(img_pos_parts),
                    "audio_pos": torch.cat(audio_pos_parts),
                    "text_pos": torch.arange(extended.text_len, device=device),
                    "noisy_img_pos": torch.cat(noisy_img_parts),
                    "audio_noisy_sel": torch.cat(audio_noisy_parts),
                    "q_ranges": q_ranges,
                    "k_ranges": k_ranges,
                    "type_map": type_map,
                    "workload": workload,
                }
            )
            output_layouts.append(
                self._output_layout(source, extended.chunks[prefix_chunks:])
            )

        sample_lens = [part["sample_len"] for part in sample_parts]
        offsets = self._offsets(sample_lens)
        return _GeneratorPack(
            sample_lens=sample_lens,
            split_lens=[part["split_lens"] for part in sample_parts],
            attn_modes=[part["attn_modes"] for part in sample_parts],
            position_ids=torch.cat([part["position_ids"] for part in sample_parts]),
            token_tags=torch.cat([part["token_tags"] for part in sample_parts]),
            img_pos=torch.cat(
                [
                    part["img_pos"] + offset
                    for part, offset in zip(sample_parts, offsets)
                ]
            ),
            audio_pos=torch.cat(
                [
                    part["audio_pos"] + offset
                    for part, offset in zip(sample_parts, offsets)
                ]
            ),
            text_pos=torch.cat(
                [
                    part["text_pos"] + offset
                    for part, offset in zip(sample_parts, offsets)
                ]
            ),
            noisy_img_pos=torch.cat(
                [
                    part["noisy_img_pos"] + offset
                    for part, offset in zip(sample_parts, offsets)
                ]
            ),
            audio_noisy_sel=torch.cat(
                [part["audio_noisy_sel"] for part in sample_parts]
            ),
            q_ranges=torch.cat(
                [
                    part["q_ranges"] + offset
                    for part, offset in zip(sample_parts, offsets)
                ]
            ),
            k_ranges=torch.cat(
                [
                    part["k_ranges"] + offset
                    for part, offset in zip(sample_parts, offsets)
                ]
            ),
            attn_type_map=torch.cat([part["type_map"] for part in sample_parts]),
            attn_workloads=[part["workload"] for part in sample_parts],
            output_layouts=output_layouts,
            prefix_chunks=rollout.prefix_chunks,
            prefix_video_stops=prefix_video_stops,
            prefix_audio_stops=prefix_audio_stops,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Select an extended-grid stage while keeping score inputs tail-local."""
        inputs = ctx["inputs"]
        rollout = ctx["rollout_x0s"]
        if not isinstance(rollout, _ExtendedRolloutX0s):
            raise TypeError("extended rollout payload is required for GEN")
        extended_inputs = rollout.extended_inputs
        rng = ctx["rng"]
        device = get_device()

        self.sampling_timesteps.set_timesteps(
            seqlen=extended_inputs.seqlens, device=device
        )
        self.audio_sampling_timesteps.set_timesteps(
            seqlen=extended_inputs.seqlens, device=device
        )
        gen_timesteps = self._sample_timesteps(
            extended_inputs, self.sampling_timesteps, rng
        )
        gen_index = self.sampling_timesteps.index(gen_timesteps)
        if not bool((gen_index >= 0).all()):
            raise ValueError("gen timestep is not on the extended video sampling grid")
        audio_gen_timesteps = self.audio_sampling_timesteps.timesteps.to(device)[
            gen_index
        ]

        score_timesteps, audio_score_timesteps = self._sample_paired_timesteps(
            inputs.batch_size,
            inputs.seqlens,
            self.score_timesteps,
            self.audio_score_timesteps,
            rng,
        )
        trajectory_xts = ctx["trajectory_xts"]
        gen_xts = (
            [
                trajectory_xts[int(gen_index[index])][0][index]
                for index in range(inputs.batch_size)
            ],
            [
                trajectory_xts[int(gen_index[index])][1][index]
                for index in range(inputs.batch_size)
            ],
        )

        ctx["gen_timesteps"] = gen_timesteps
        ctx["gen_audio_timesteps"] = audio_gen_timesteps
        ctx["gen_index"] = gen_index
        ctx["gen_xts"] = gen_xts
        ctx["score_timesteps"] = score_timesteps
        ctx["audio_score_timesteps"] = audio_score_timesteps
        ctx["gen_inputs"] = inputs
        ctx["extended_generator_pack"] = self._build_generator_pack(inputs, rollout)
        return ctx

    def _extended_generator_forward(
        self,
        model: Any,
        tail_inputs: ForwardInput,
        pack: _GeneratorPack,
        rollout: _ExtendedRolloutX0s,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward one absolute-coordinate sequence without prefix noisy rows."""
        device = tail_inputs.token_tags.device
        extended_inputs = rollout.extended_inputs
        video_dim = (
            extended_inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        )
        audio_dim = extended_inputs.layouts[0].audio_shape[1]
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)
        zero = torch.zeros((), dtype=torch.float32, device=device)

        video_blocks: list[torch.Tensor] = []
        audio_blocks: list[torch.Tensor] = []
        video_eps_blocks: list[torch.Tensor] = []
        audio_eps_blocks: list[torch.Tensor] = []
        row_timesteps: list[torch.Tensor] = []

        def push(
            audio_rows: torch.Tensor,
            video_rows: torch.Tensor,
            audio_eps_rows: torch.Tensor,
            video_eps_rows: torch.Tensor,
            audio_t: torch.Tensor,
            video_t: torch.Tensor,
        ) -> None:
            n_audio = audio_rows.shape[0]
            n_video = video_rows.shape[0]
            video_blocks.extend(
                (video_rows.new_zeros((n_audio, video_dim)), video_rows)
            )
            audio_blocks.extend(
                (audio_rows, audio_rows.new_zeros((n_video, audio_dim)))
            )
            video_eps_blocks.extend(
                (video_eps_rows.new_zeros((n_audio, video_dim)), video_eps_rows)
            )
            audio_eps_blocks.extend(
                (audio_eps_rows, audio_eps_rows.new_zeros((n_video, audio_dim)))
            )
            row_timesteps.extend((audio_t.expand(n_audio), video_t.expand(n_video)))

        for sample_index, extended in enumerate(extended_inputs.layouts):
            text_rows = extended.text_len
            video_blocks.append(
                video_xts[sample_index].new_zeros((text_rows, video_dim))
            )
            audio_blocks.append(
                audio_xts[sample_index].new_zeros((text_rows, audio_dim))
            )
            video_eps_blocks.append(
                video_xts[sample_index].new_zeros((text_rows, video_dim))
            )
            audio_eps_blocks.append(
                audio_xts[sample_index].new_zeros((text_rows, audio_dim))
            )
            row_timesteps.append(zero.expand(text_rows))

            prefix_video_stop = pack.prefix_video_stops[sample_index]
            prefix_audio_stop = pack.prefix_audio_stops[sample_index]
            for chunk_index, chunk in enumerate(extended.chunks):
                if chunk_index < pack.prefix_chunks:
                    push(
                        self._audio_rows(
                            rollout.prefix_audio[sample_index],
                            chunk.audio_start,
                            chunk.audio_stop,
                        ).detach(),
                        self._video_rows(
                            rollout.prefix_video[sample_index],
                            chunk.video_start,
                            chunk.video_stop,
                        ).detach(),
                        self._audio_rows(
                            rollout.prefix_audio_eps[sample_index],
                            chunk.audio_start,
                            chunk.audio_stop,
                        ).detach(),
                        self._video_rows(
                            rollout.prefix_video_eps[sample_index],
                            chunk.video_start,
                            chunk.video_stop,
                        ).detach(),
                        zero,
                        zero,
                    )
                    continue

                video_start = chunk.video_start - prefix_video_stop
                video_stop = chunk.video_stop - prefix_video_stop
                audio_start = chunk.audio_start - prefix_audio_stop
                audio_stop = chunk.audio_stop - prefix_audio_stop
                push(
                    self._audio_rows(audio_xts[sample_index], audio_start, audio_stop),
                    self._video_rows(video_xts[sample_index], video_start, video_stop),
                    audio_xts[sample_index].new_zeros((chunk.audio_rows, audio_dim)),
                    video_xts[sample_index].new_zeros((chunk.video_rows, video_dim)),
                    audio_timesteps[sample_index],
                    video_timesteps[sample_index],
                )
                if chunk_index == len(extended.chunks) - 1:
                    continue
                push(
                    self._audio_rows(
                        rollout.audio[sample_index], audio_start, audio_stop
                    ),
                    self._video_rows(
                        rollout.video[sample_index], video_start, video_stop
                    ),
                    self._audio_rows(
                        rollout.audio_eps[sample_index], audio_start, audio_stop
                    ),
                    self._video_rows(
                        rollout.video_eps[sample_index], video_start, video_stop
                    ),
                    zero,
                    zero,
                )

        rows = sum(pack.sample_lens)
        sample_lens = list(pack.sample_lens)
        split_lens = [length for sample in pack.split_lens for length in sample]
        attn_modes = [mode for sample in pack.attn_modes for mode in sample]
        q_ranges = pack.q_ranges
        k_ranges = pack.k_ranges
        attn_type_map = pack.attn_type_map
        attn_workloads = list(pack.attn_workloads)
        position_ids, token_tags, pad = self._pad_for_sp(
            rows,
            video_blocks=video_blocks,
            audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks,
            audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps,
            position_ids=pack.position_ids,
            token_tags=pack.token_tags,
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
                sink=tail_inputs.sinks[0],
                window_size=tail_inputs.window_sizes[0],
                device=device,
            )
            q_ranges = torch.cat((q_ranges, pad_q + rows))
            k_ranges = torch.cat((k_ranges, pad_k + rows))
            attn_type_map = torch.cat((attn_type_map, pad_type))
            attn_workloads.append(int(pad_work))

        kwargs = self._common_kwargs(
            tail_inputs,
            x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=position_ids,
            token_tags=token_tags,
            img_pos=pack.img_pos,
            audio_pos=pack.audio_pos,
            text_pos=pack.text_pos,
            infer_out_pos=pack.noisy_img_pos,
        )
        kwargs.update(
            sample_lens=sample_lens,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=attn_workloads,
            attention_mask=self._block_mask(
                tail_inputs,
                device,
                sample_lens=sample_lens,
                split_lens=split_lens,
                attn_modes=attn_modes,
            ),
        )
        video_logits, audio_logits = model(**kwargs)
        audio_logits = audio_logits.index_select(0, pack.audio_noisy_sel)

        video_out: list[torch.Tensor] = []
        audio_out: list[torch.Tensor] = []
        video_cursor = 0
        audio_cursor = 0
        for layout in pack.output_layouts:
            video_rows = sum(chunk.video_rows for chunk in layout.chunks)
            audio_rows = sum(chunk.audio_rows for chunk in layout.chunks)
            video_out.append(
                minimax_h3_unpatchify_video_tokens(
                    video_logits[video_cursor : video_cursor + video_rows],
                    latent_shape=(
                        layout.latent_shape[1],
                        layout.latent_shape[2] // _PATCH_SIZE[1],
                        layout.latent_shape[3] // _PATCH_SIZE[2],
                        layout.latent_shape[0],
                    ),
                    patch_size=_PATCH_SIZE,
                )[0]
            )
            chunk_audio = audio_logits[audio_cursor : audio_cursor + audio_rows]
            native_audio = chunk_audio.new_empty(chunk_audio.shape).index_copy(
                0, layout.audio_row_perm.to(chunk_audio.device), chunk_audio
            )
            audio_out.append(
                minimax_h3_unpack_audio_tokens(
                    native_audio,
                    audio_t=layout.audio_shape[2] * _AUDIO_CHANNELS,
                    audio_channel=_AUDIO_CHANNELS,
                )
            )
            video_cursor += video_rows
            audio_cursor += audio_rows
        if (
            video_cursor != video_logits.shape[0]
            or audio_cursor != audio_logits.shape[0]
        ):
            raise ValueError(
                "generator suffix output rows disagree with packed selectors"
            )
        return video_out, audio_out

    def _renorm_gen_video(
        self, ctx: dict[str, Any], video: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Keep the full rollout's anchor and global chunk indices on the suffix."""
        if not self.video_chunk_renorm:
            return video
        rollout = ctx["rollout_x0s"]
        assert isinstance(rollout, _ExtendedRolloutX0s)
        return self._renorm_video_chunks(
            ctx["gen_inputs"], video, rollout.video_anchors,
            chunk_offset=rollout.prefix_chunks,
        )

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def gen_forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Predict only the suffix while conditioning on detached clean prefix rows."""
        rollout = ctx["rollout_x0s"]
        if not isinstance(rollout, _ExtendedRolloutX0s):
            raise TypeError("extended rollout payload is required for GEN")
        ctx["gen_pred"] = self._extended_generator_forward(
            ctx["models"]["backbone"],
            ctx["gen_inputs"],
            ctx["extended_generator_pack"],
            rollout,
            video_xts=ctx["gen_xts"][0],
            audio_xts=ctx["gen_xts"][1],
            video_timesteps=ctx["gen_timesteps"],
            audio_timesteps=ctx["gen_audio_timesteps"],
        )
        return ctx


__all__ = ["CausalMiniMaxH3DMD2Extended"]
