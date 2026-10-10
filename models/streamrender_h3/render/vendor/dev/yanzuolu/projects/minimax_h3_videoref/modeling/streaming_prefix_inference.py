# SPDX-License-Identifier: Apache-2.0
"""Request-local prefix banks for bidirectional streaming continuations.

The bootstrap uses the ordinary full window. Its first continuation freezes
the prepared Qwen context, picture, sink AV and matching reference rows.
Each student, guidance branch and paired denoising time has its own prefix
KV bank. Only the window and current chunk are recomputed afterwards.

Banks belong to individual streaming states, so shrinking active batches
cannot exchange samples' histories. A collective cache-miss flag keeps
prefill calls aligned across data-parallel groups, including groups running
the sampling loop's bootstrap-shaped padding rounds. Samples that already
have a bank participate with empty prefix rows. Finished streams release
their snapshots and banks.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import all_reduce_max
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch, StreamingLatentChunk, StreamingState
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3.modeling.streaming_prefix import (
    StreamingPrefixForwardMixin,
    streaming_prefix_media_masks,
)
from dev.yanzuolu.utils.naive_cache import NaiveCache


def validate_static_prefix_captions(timelines: Sequence[Sequence[dict[str, Any]] | None] | None) -> None:
    """Allow caption timelines whose text remains constant throughout each clip."""
    if any(len({segment["prompt"] for segment in timeline or ()}) > 1 for timeline in timelines or ()):
        raise ValueError("a fixed student prefix requires a single caption per clip")


@dataclass(frozen=True)
class _PrefixCondition:
    prompt: torch.Tensor
    tags: torch.Tensor | None
    picture: torch.Tensor | None
    reference_indices: torch.Tensor | None
    reference: torch.Tensor | None


@dataclass(frozen=True)
class _PrefixSnapshot:
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor
    conditions: tuple[_PrefixCondition, ...]


@dataclass
class _PrefixStreamingState(StreamingState):
    prefix_prompt_texts: tuple[str, ...] | None = None
    prefix_snapshot: _PrefixSnapshot | None = None
    prefix_banks: dict[tuple[int, int, int, float, float], NaiveCache] = field(default_factory=dict)


class VideoRefStreamingPrefixInferenceMixin(StreamingPrefixForwardMixin):
    """Cache a fixed first-continuation prefix separately for each sampling stage."""

    def start_stream(self, **kwargs: Any) -> list[_PrefixStreamingState]:
        states = super().start_stream(**kwargs)
        for state in states:
            self._validate_prefix_policy(state.streaming_config)
        return [_PrefixStreamingState(
            **{item.name: getattr(state, item.name) for item in fields(StreamingState)},
            prefix_prompt_texts=None if state.prompt_texts is None else tuple(state.prompt_texts),
        ) for state in states]

    def _iter_stream_latents(
        self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]], rngs: Sequence[Any], *,
        models: dict[str, Any] | None = None,
    ) -> Iterator[StreamingLatentChunk]:
        for weight, branch in branch_inputs:
            if weight:
                validate_static_prefix_captions(branch.caption_segments)
        yield from super()._iter_stream_latents(backbone, branch_inputs, rngs, models=models)

    def step(self, backbone: Any, states: Sequence[_PrefixStreamingState], *args: Any, **kwargs: Any) -> Any:
        for state in states:
            texts = None if state.prompt_texts is None else tuple(state.prompt_texts)
            if state.next_video == 0:
                state.prefix_prompt_texts = texts
            elif texts != state.prefix_prompt_texts:
                raise ValueError("a cached stream cannot change its conditioning prompts")
        previous = getattr(self, "_prefix_active_states", None)
        self._prefix_active_states = states
        try:
            return super().step(backbone, states, *args, **kwargs)
        finally:
            self._prefix_active_states = previous

    def _freeze_stream_prefix(
        self, states: Sequence[_PrefixStreamingState], branches: Sequence[tuple[float, StreamingInputs]],
        plans: list[dict[str, Any]], histories: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> tuple[list[tuple[float, StreamingInputs]], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Keep the first continuation's conditioned prefix while rebuilding current windows."""
        fixed_history = (list(histories[0]), list(histories[1]))
        for index, (state, plan) in enumerate(zip(states, plans, strict=True)):
            if plan["is_bootstrap"]:
                continue
            masks = streaming_prefix_media_masks(plan)
            if state.prefix_snapshot is None:
                conditions = []
                for _, branch in branches:
                    selected = branch.plans[index]
                    reference_mask = streaming_prefix_media_masks(selected)["reference"]
                    conditions.append(_PrefixCondition(
                        prompt=branch.prompt_embeds[index].detach().clone(),
                        tags=None if branch.text_token_tags is None else branch.text_token_tags[index].detach().clone(),
                        picture=(None if selected.get("picture_shape") is None
                                 else selected["picture_latents"].detach().clone()),
                        reference_indices=(None if reference_mask is None
                                           else selected["reference_video_indices"][reference_mask].clone()),
                        reference=(None if branch.reference_latents is None else self._select(
                            branch.reference_latents[index], reference_mask.nonzero().flatten(),
                        ).detach().clone()),
                    ))
                state.prefix_snapshot = _PrefixSnapshot(
                    video_indices=plan["video_indices"][masks["video"]].clone(),
                    audio_indices=plan["audio_indices"][masks["audio"]].clone(),
                    video=self._select(histories[0][index], masks["video"].nonzero().flatten()).detach().clone(),
                    audio=self._select(histories[1][index], masks["audio"].nonzero().flatten(), audio=True).detach().clone(),
                    conditions=tuple(conditions),
                )
            snapshot = state.prefix_snapshot
            for modality, name in enumerate(("video", "audio")):
                assert torch.equal(plan[f"{name}_indices"][masks[name]], getattr(snapshot, f"{name}_indices")), (
                    "a cached stream must keep its first continuation's sink indices"
                )
                current = histories[modality][index]
                fixed_history[modality][index] = current.index_copy(
                    2 if modality else 1, masks[name].nonzero().flatten().to(current.device),
                    getattr(snapshot, name).to(current),
                )

        fixed_branches = []
        for branch_index, (weight, branch) in enumerate(branches):
            embeddings, selected_plans = list(branch.prompt_embeds), list(branch.plans)
            tags = None if branch.text_token_tags is None else list(branch.text_token_tags)
            references = None if branch.reference_latents is None else list(branch.reference_latents)
            for index, state in enumerate(states):
                if plans[index]["is_bootstrap"]:
                    continue
                condition = state.prefix_snapshot.conditions[branch_index]
                plan = selected_plans[index]
                if not torch.equal(embeddings[index], condition.prompt):
                    raise ValueError("a cached stream cannot change its conditioning prefix")
                embeddings[index] = condition.prompt
                if tags is not None:
                    tags[index] = condition.tags
                if condition.picture is not None:
                    selected_plans[index] = dict(plan, picture_latents=condition.picture)
                if references is not None:
                    mask = streaming_prefix_media_masks(plan)["reference"]
                    assert torch.equal(plan["reference_video_indices"][mask], condition.reference_indices), (
                        "a cached stream must keep its first continuation's sink reference indices"
                    )
                    references[index] = references[index].index_copy(
                        1, mask.nonzero().flatten().to(references[index].device), condition.reference.to(references[index]),
                    )
            fixed_branches.append((weight, self._streaming_inputs_from_payload(dict(
                plans=selected_plans, prompt_embeds=embeddings, reference_latents=references, text_token_tags=tags,
            ))))
        return fixed_branches, fixed_history

    def _denoise_window(
        self, backbone: Any, branches: Sequence[tuple[float, StreamingInputs]], plans: list[dict[str, Any]],
        histories: tuple[list[torch.Tensor], list[torch.Tensor]], video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor], *, rngs: list[Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        states = self._prefix_active_states
        assert states is not None and len(states) == len(plans)
        branches, histories = self._freeze_stream_prefix(states, branches, plans, histories)
        device = video_xts[0].device
        seqlens = [plan["packing_rows"] for plan in plans]
        video_grid = self._sampling_grid(self.sampling_timesteps, seqlens, device)
        audio_grid = self._sampling_grid(self.audio_sampling_timesteps, seqlens, device)
        if video_grid.shape != audio_grid.shape:
            raise ValueError("video and audio sampling grids must have equal step counts")
        for stage in range(video_grid.shape[1]):
            vt, at = video_grid[:, stage], audio_grid[:, stage]
            vs = video_grid[:, stage + 1] if stage + 1 < video_grid.shape[1] else torch.zeros_like(vt)
            audio_s = audio_grid[:, stage + 1] if stage + 1 < audio_grid.shape[1] else torch.zeros_like(at)
            video, audio = [], []
            for index, plan in enumerate(plans):
                v, a = histories[0][index].clone(), histories[1][index].clone()
                v[:, plan["video_noisy_mask"].to(device)] = video_xts[index]
                a[:, :, plan["audio_noisy_mask"].to(device)] = audio_xts[index]
                video.append(v)
                audio.append(a)
            keys = [[(id(backbone), self._sp_size(), branch, float(vt[index]), float(at[index]))
                     for index in range(len(states))] for branch in range(len(branches))]
            missing = [[not plan["is_bootstrap"] and key not in state.prefix_banks
                        for state, plan, key in zip(states, plans, branch_keys, strict=True)] for branch_keys in keys]
            needs_prefill = torch.tensor(any(any(values) for values in missing), dtype=torch.int32, device=device)
            all_reduce_max(needs_prefill)
            vp, ap = [torch.zeros_like(value) for value in video_xts], [torch.zeros_like(value) for value in audio_xts]
            for branch_index, (weight, branch) in enumerate(branches):
                kwargs = dict(video_xts=video, audio_xts=audio, video_timesteps=vt, audio_timesteps=at)
                if bool(needs_prefill):
                    fresh = self._streaming_prefix_prefill(backbone, branch, active=missing[branch_index], **kwargs)
                    for state, key, cache, keep in zip(states, keys[branch_index], fresh, missing[branch_index], strict=True):
                        if keep:
                            state.prefix_banks[key] = cache
                caches = [state.prefix_banks.get(key) for state, key in zip(states, keys[branch_index], strict=True)]
                prediction = self._streaming_prefix_forward(
                    backbone, branch, caches=caches if any(cache is not None for cache in caches) else None, **kwargs,
                )
                for accumulators, values in zip((vp, ap), prediction, strict=True):
                    for accumulator, value in zip(accumulators, values, strict=True):
                        accumulator.add_(value, alpha=weight)
            lengths = torch.tensor(seqlens, device=device)
            video_xts = self.sampler.step_to(pred=vp, x_t=video_xts, t=vt, s=vs, rng=rngs, seqlens=lengths)
            audio_xts = self.sampler.step_to(pred=ap, x_t=audio_xts, t=at, s=audio_s, rng=rngs, seqlens=lengths)
        return video_xts, audio_xts

    def _commit_stream_state(
        self, state: _PrefixStreamingState, plan: dict[str, Any], video: torch.Tensor,
        selected_audio: torch.Tensor, reference: torch.Tensor | None, is_last: bool,
    ) -> None:
        super()._commit_stream_state(state, plan, video, selected_audio, reference, is_last)
        if state.finished:
            state.prefix_banks.clear()
            state.prefix_snapshot = None


__all__ = ["VideoRefStreamingPrefixInferenceMixin", "validate_static_prefix_captions"]
