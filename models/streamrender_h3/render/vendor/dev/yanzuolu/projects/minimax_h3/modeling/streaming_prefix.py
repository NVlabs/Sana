# SPDX-License-Identifier: Apache-2.0
"""One-way static prefixes for packed streaming AV windows.

Training evaluates the complete window with two attention blocks. The prefix
reads itself, and the dynamic block reads both blocks. Inference evaluates the
prefix once, retaining its per-layer keys and values, then evaluates only the
dynamic rows against that read-only cache. The caller owns cache lifetime and
keeps one cache per model, condition branch and diffusion timestep.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import minimax_h3_unpatchify_video_tokens
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.causal_model import CausalMiniMaxH3DiTModel
from dev.yanzuolu.utils import flex_attn
from dev.yanzuolu.utils.naive_cache import NaiveCache


def streaming_prefix_media_masks(plan: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    """Select the generated sink and its aligned references without duplicating W.

    Masks index each modality's selected latent rows, before spatial patching
    and stereo packing. A bootstrap has no static prefix.
    """
    stop = 0 if plan["is_bootstrap"] else int(plan["sink_history_stop"])
    mapping = VideoTemporalMapping.from_dict(plan["video_temporal_mapping"])
    audio_stop = mapping.decode_timeline.clock_boundary_ceil(stop)
    reference = plan["reference_video_indices"]
    return {
        "video": (plan["video_indices"] < stop) & ~plan["video_noisy_mask"],
        "audio": (plan["audio_indices"] < audio_stop) & ~plan["audio_noisy_mask"],
        "reference": None if reference is None else reference < stop,
    }


def streaming_prefix_rows(inputs: StreamingInputs, index: int) -> torch.Tensor:
    """Mark Qwen, picture, sink reference and generated sink rows in native order."""
    plan, pack = inputs.plans[index], inputs.packs[index]
    selected = torch.zeros(int(pack["sample_lens"]), dtype=torch.bool, device=inputs.token_tags.device)
    if plan["is_bootstrap"]:
        return selected
    masks = streaming_prefix_media_masks(plan)
    text_len = inputs.text_lens[index]
    picture = plan.get("picture_shape")
    picture_rows = 0 if picture is None else (picture[2] // 2) * (picture[3] // 2)
    selected[:text_len + picture_rows] = True
    cursor = text_len + picture_rows
    if masks["reference"] is not None:
        shape = plan["reference_video_shape"]
        reference_rows = masks["reference"].repeat_interleave((shape[2] // 2) * (shape[3] // 2))
        selected[cursor:cursor + reference_rows.numel()] = reference_rows.to(selected.device)
        cursor += reference_rows.numel()
    selected[cursor:cursor + int(pack["audio_rows"])] = masks["audio"].repeat(2).to(selected.device)
    cursor += int(pack["audio_rows"])
    shape = plan["target_video_shape"]
    selected[cursor:] = masks["video"].repeat_interleave((shape[2] // 2) * (shape[3] // 2)).to(selected.device)
    return selected


@dataclass(frozen=True)
class StreamingPrefixForward:
    """Packed model arguments and selectors restoring native streaming outputs."""

    kwargs: dict[str, Any]
    prefix_lens: tuple[int, ...]
    audio_output_indices: torch.Tensor
    audio_output_rows: int

    def restore_outputs(self, prediction: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        video, audio = prediction
        restored_audio = audio.new_zeros((self.audio_output_rows, audio.shape[-1]))
        return video, restored_audio.index_copy(0, self.audio_output_indices, audio)


def build_streaming_prefix_forward(
    inputs: StreamingInputs, kwargs: dict[str, Any], *, part: str = "full", sp_size: int = 1,
    active: Sequence[bool] | None = None, attention_head_dim: int | None = None,
    prefix_masks: Sequence[torch.Tensor] | None = None,
) -> StreamingPrefixForward:
    """Repack native streaming kwargs while preserving row times and positions.

    ``full`` supplies the training mask. ``prefix`` and ``suffix`` supply one
    dense document per sample for cache fill and cache reads. Empty prefix
    documents execute a constant dummy row, which is removed when caches split.
    Alignment padding is a separate document and never persists in a cache.
    """
    if part not in {"full", "prefix", "suffix"}:
        raise ValueError("part must be full, prefix or suffix")
    if active is not None and (part != "prefix" or len(active) != inputs.batch_size):
        raise ValueError("active must describe each sample of a prefix fill")
    if prefix_masks is not None and len(prefix_masks) != inputs.batch_size:
        raise ValueError("prefix_masks must describe each streaming sample")
    device = kwargs["x"].device
    sources, sample_lens, prefix_lens, splits, rectangles = [], [], [], [], []
    native_cursor = cursor = 0
    for index, pack in enumerate(inputs.packs):
        prefix_mask = (
            streaming_prefix_rows(inputs, index)
            if prefix_masks is None
            else prefix_masks[index].to(device=device, dtype=torch.bool)
        )
        if prefix_mask.numel() != int(pack["sample_lens"]):
            raise ValueError("a custom prefix mask must cover every packed row")
        if active is not None and not active[index]:
            prefix_mask = torch.zeros_like(prefix_mask)
        prefix = prefix_mask.nonzero().flatten() + native_cursor
        dynamic = (~prefix_mask).nonzero().flatten() + native_cursor
        prefix_lens.append(prefix.numel())
        rows = torch.cat((prefix, dynamic)) if part == "full" else prefix if part == "prefix" else dynamic
        if rows.numel() == 0:
            rows = torch.tensor([-1], device=device)
        sources.append(rows)
        length = rows.numel()
        sample_lens.append(length)
        if part == "full" and prefix.numel():
            splits.extend((prefix.numel(), dynamic.numel()))
            rectangles.append(((cursor, cursor + prefix.numel()), (cursor, cursor + prefix.numel())))
            if dynamic.numel():
                rectangles.append(((cursor + prefix.numel(), cursor + length), (cursor, cursor + length)))
        else:
            splits.append(length)
            rectangles.append(((cursor, cursor + length), (cursor, cursor + length)))
        native_cursor += int(pack["sample_lens"])
        cursor += length
    pad = sp_size - cursor % sp_size if sp_size > 1 else 0
    if pad:
        sources.append(torch.full((pad,), -1, dtype=torch.long, device=device))
        sample_lens.append(pad)
        splits.append(pad)
        rectangles.append(((cursor, cursor + pad), (cursor, cursor + pad)))
    source = torch.cat(sources)
    valid = source >= 0
    safe_source = source.clamp(min=0)
    destination = source.new_full((kwargs["x"].shape[1],), -1)
    destination[source[valid]] = valid.nonzero().flatten()

    def rows(value: torch.Tensor, dim: int, fill: float = 0) -> torch.Tensor:
        selected = value.index_select(dim, safe_source)
        shape = [1] * selected.ndim
        shape[dim] = valid.numel()
        return torch.where(valid.reshape(shape), selected, selected.new_full((), fill))

    result = dict(kwargs)
    result.pop("packed_seq_params", None)
    for key in ("x", "audio_x", "eps", "audio_eps", "img_position_ids"):
        result[key] = rows(kwargs[key], 1)
    result["token_tags"] = rows(kwargs["token_tags"], 0, -1)
    if "timestep_conditioning_mask" in kwargs:
        result["timestep_conditioning_mask"] = rows(kwargs["timestep_conditioning_mask"], 0)
    timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
    result["unique_timesteps"], result["inverse_indices"] = torch.unique(rows(timesteps, 0), return_inverse=True)
    kept_positions = {}
    for key in ("img_pos_info", "audio_pos_info", "text_pos_info", "img_pos_for_infer_output_info"):
        original = kwargs[key]["position_ids"]
        mapped = destination[original]
        kept_positions[key] = mapped >= 0
        result[key] = {"position_ids": mapped[mapped >= 0]}
    output_mask = kept_positions["img_pos_for_infer_output_info"]
    result["update_mask"] = kwargs["update_mask"][output_mask]
    audio_mask = kept_positions["audio_pos_info"]
    if "update_audio_mask" in kwargs:
        result["update_audio_mask"] = kwargs["update_audio_mask"][audio_mask]
    text_mask = kept_positions["text_pos_info"]
    result["prompt_embeds"] = kwargs["prompt_embeds"][text_mask]
    text_cursor = 0
    text_lens = []
    for length in inputs.text_lens:
        kept = int(text_mask[text_cursor:text_cursor + length].sum())
        if kept:
            text_lens.append(kept)
        text_cursor += length
    text_cu = torch.tensor([0, *text_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    result["refiner_packed_seq_params"] = {"cu_seqlens_q": text_cu, "max_seqlen_q": max(text_lens, default=0)}
    result["sample_lens"] = sample_lens
    q_ranges = torch.tensor([query for query, _ in rectangles], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([key for _, key in rectangles], dtype=torch.int32, device=device)
    result.update(q_ranges=q_ranges, k_ranges=k_ranges,
                  attn_type_map=torch.zeros(len(rectangles), dtype=torch.int32, device=device))
    result["attn_workloads"] = [sum((q1-q0) * (k1-k0) for (q0,q1),(k0,k1) in rectangles)]
    if part == "full":
        flash = flex_attn.FLEX_FLASH_ATTN_AVAILABLE and attention_head_dim is not None and attention_head_dim <= 128
        result["attention_mask"] = None if flash else flex_attn.create_sparse_mask(
            sample_lens, splits, ["full"] * len(splits), device=device,
            sink=[0], window_size=[None], block_size=128,
        )
    return StreamingPrefixForward(
        kwargs=result, prefix_lens=tuple(prefix_lens),
        audio_output_indices=audio_mask.nonzero().flatten(), audio_output_rows=audio_mask.numel(),
    )


def split_streaming_prefix_cache(cache: NaiveCache, lengths: Sequence[int]) -> list[NaiveCache]:
    """Return independent single-sample caches, excluding fill and alignment dummies."""
    output, offset = [], 0
    for index, length in enumerate(lengths):
        sample = NaiveCache(cache.num_layers, 1, sink=[1], window_size=[0])
        sample.kvlens = [length]
        sample.chunk_lens = [[length]]
        for layer in range(cache.num_layers):
            sample.key_cache[layer] = cache.key_cache[layer][offset:offset + length].detach().clone()
            sample.value_cache[layer] = cache.value_cache[layer][offset:offset + length].detach().clone()
        output.append(sample)
        offset += cache.kvlens[index]
    return output


def merge_streaming_prefix_caches(caches: Sequence[NaiveCache], *, padding: bool = False) -> NaiveCache:
    """Pack selected streams for one read-only forward, with optional empty padding."""
    # Batch-one serving is the hot path. Reuse the request-owned prefix bank
    # rather than concatenating all 50 layers on every NFE.
    if len(caches) == 1:
        merged = caches[0]
        if padding and merged.batch_size == 1:
            merged.batch_size += 1
            merged.kvlens.append(0)
            merged.sink.append(0)
            merged.window_size.append(0)
            merged.chunk_lens[0].append(0)
            merged.curr_rope = torch.cat(
                (merged.curr_rope, merged.curr_rope.new_zeros(1))
            )
        elif padding and not (
            merged.batch_size == 2
            and len(merged.kvlens) == 2
            and int(merged.kvlens[1]) == 0
        ):
            raise ValueError("unexpected reused prefix-cache padding layout")
        return merged
    merged = NaiveCache.merge(list(caches))
    if padding:
        merged.batch_size += 1
        merged.kvlens.append(0)
        merged.sink.append(0)
        merged.window_size.append(0)
        merged.chunk_lens[0].append(0)
        merged.curr_rope = torch.cat((merged.curr_rope, merged.curr_rope.new_zeros(1)))
    return merged


def _empty_prefix_cache(like: NaiveCache) -> NaiveCache:
    cache = NaiveCache(like.num_layers, 1, sink=[1], window_size=[0])
    cache.chunk_lens = [[0]]
    for layer in range(like.num_layers):
        cache.key_cache[layer] = like.key_cache[layer][:0]
        cache.value_cache[layer] = like.value_cache[layer][:0]
    return cache


class StreamingPrefixForwardMixin:
    """Evaluate prefix-isolated windows or the dynamic rows of cached streams."""

    @staticmethod
    def _require_streaming_prefix_model(model: Any) -> None:
        if not isinstance(getattr(model, "dit", None), CausalMiniMaxH3DiTModel):
            raise TypeError("streaming prefixes require a MiniMaxH3CausalX0DiT or MiniMaxH3CausalX0DiTSP model")

    def _streaming_prefix_forward(
        self, model: Any, inputs: StreamingInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
        caches: Sequence[NaiveCache | None] | None = None,
        prefix_masks: Sequence[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self._require_streaming_prefix_model(model)
        if caches is not None:
            if len(caches) != inputs.batch_size:
                raise ValueError("one prefix cache is required for each streaming sample")
            for index, cache in enumerate(caches):
                if cache is None and bool(streaming_prefix_rows(inputs, index).any()):
                    raise ValueError("a continuation prefix must be filled before reading its cache")
            prototype = next((cache for cache in caches if cache is not None), None)
            caches = None if prototype is None else [
                cache if cache is not None else _empty_prefix_cache(prototype) for cache in caches
            ]
        packed = build_streaming_prefix_forward(inputs, self._streaming_kwargs(
            model, inputs, video_xts=video_xts, audio_xts=audio_xts,
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
        ), part="full" if caches is None else "suffix", sp_size=self._sp_size(),
            attention_head_dim=model.dit.arch.attention_head_dim,
            prefix_masks=prefix_masks)
        if caches is not None:
            merged = merge_streaming_prefix_caches(
                caches, padding=self._sp_size() > 1
            )
            if merged._persistent_key_arenas is None:
                merged.prepare_persistent_attention_arenas(
                    int(os.environ.get("H3_CACHE_QUERY_CAPACITY", "12288"))
                )
            packed.kwargs.update(
                past_key_values=merged, update_past_key_values=False
            )
        video_logits, audio_logits = packed.restore_outputs(model(**packed.kwargs))
        videos, audios = [], []
        video_cursor = audio_cursor = 0
        for plan, pack in zip(inputs.plans, inputs.packs, strict=True):
            channels, _, height, width = plan["target_video_shape"]
            frames = int(pack["noisy_video_frames"])
            count = frames * (height // 2) * (width // 2)
            videos.append(minimax_h3_unpatchify_video_tokens(
                video_logits[video_cursor:video_cursor + count],
                latent_shape=(frames, height // 2, width // 2, channels), patch_size=(1, 2, 2),
            )[0] if frames else video_logits.new_empty((channels, 0, height, width)))
            audio = audio_logits[audio_cursor:audio_cursor + int(pack["audio_rows"])].index_select(0, pack["audio_noisy_sel"])
            audios.append(audio.reshape(2, int(pack["noisy_audio_frames"]), plan["target_audio_shape"][1]).permute(0, 2, 1))
            video_cursor += count
            audio_cursor += int(pack["audio_rows"])
        return videos, audios

    @torch.no_grad()
    def _streaming_prefix_capture_forward(
        self, model: Any, inputs: StreamingInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
        prefix_masks: Sequence[torch.Tensor],
    ) -> tuple[tuple[list[torch.Tensor], list[torch.Tensor]], list[NaiveCache]]:
        """Run one full masked forward and tap its prefix K/V for a later NFE."""
        self._require_streaming_prefix_model(model)
        packed = build_streaming_prefix_forward(
            inputs,
            self._streaming_kwargs(
                model, inputs, video_xts=video_xts, audio_xts=audio_xts,
                video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
            ),
            part="full",
            sp_size=self._sp_size(),
            attention_head_dim=model.dit.arch.attention_head_dim,
            prefix_masks=prefix_masks,
        )
        sample_lens = [int(value) for value in packed.kwargs["sample_lens"]]
        capture_lens = [int(value) for value in packed.prefix_lens]
        capture_lens.extend([0] * (len(sample_lens) - len(capture_lens)))
        cache = NaiveCache(
            self._num_layers(model), len(sample_lens),
            sink=[1] * len(sample_lens), window_size=[0] * len(sample_lens),
        )
        cache.prepare_selected_capture(
            capture_lens, sample_lens, packed.kwargs["x"].device
        )
        packed.kwargs["capture_key_values"] = cache
        video_logits, audio_logits = packed.restore_outputs(model(**packed.kwargs))

        videos, audios = [], []
        video_cursor = audio_cursor = 0
        for plan, pack in zip(inputs.plans, inputs.packs, strict=True):
            channels, _, height, width = plan["target_video_shape"]
            frames = int(pack["noisy_video_frames"])
            count = frames * (height // 2) * (width // 2)
            videos.append(minimax_h3_unpatchify_video_tokens(
                video_logits[video_cursor:video_cursor + count],
                latent_shape=(frames, height // 2, width // 2, channels),
                patch_size=(1, 2, 2),
            )[0] if frames else video_logits.new_empty((channels, 0, height, width)))
            audio = audio_logits[
                audio_cursor:audio_cursor + int(pack["audio_rows"])
            ].index_select(0, pack["audio_noisy_sel"])
            audios.append(audio.reshape(
                2, int(pack["noisy_audio_frames"]), plan["target_audio_shape"][1]
            ).permute(0, 2, 1))
            video_cursor += count
            audio_cursor += int(pack["audio_rows"])

        # Keep the zero-length SP padding document and avoid cloning 50 layers.
        caches = [cache] if inputs.batch_size == 1 else split_streaming_prefix_cache(
            cache, packed.prefix_lens
        )
        return (videos, audios), caches

    @torch.no_grad()
    def _streaming_prefix_prefill(
        self, model: Any, inputs: StreamingInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
        active: Sequence[bool] | None = None,
        prefix_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[NaiveCache]:
        self._require_streaming_prefix_model(model)
        packed = build_streaming_prefix_forward(inputs, self._streaming_kwargs(
            model, inputs, video_xts=video_xts, audio_xts=audio_xts,
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
        ), part="prefix", sp_size=self._sp_size(), active=active,
            prefix_masks=prefix_masks)
        samples = len(packed.kwargs["sample_lens"])
        cache = NaiveCache(self._num_layers(model), samples, sink=[1] * samples, window_size=[0] * samples)
        packed.kwargs.update(past_key_values=cache, update_past_key_values=True)
        model(**packed.kwargs)
        return split_streaming_prefix_cache(cache, packed.prefix_lens)


__all__ = [
    "StreamingPrefixForwardMixin", "StreamingPrefixForward", "build_streaming_prefix_forward",
    "streaming_prefix_media_masks", "streaming_prefix_rows", "split_streaming_prefix_cache", "merge_streaming_prefix_caches",
]
