# SPDX-License-Identifier: Apache-2.0
"""DMD with corpus GEN repacking or corpus-prefix rollout conditioning."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from numbers import Real
from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    CausalMiniMaxH3DMD,
    ForwardInput,
    _RolloutX0s,
)


@dataclass(frozen=True)
class _PrefixForwardInput(ForwardInput):
    """Per-payload prefix decisions shared by rollout, FAKE and GEN."""

    corpus_prefix_chunks: tuple[int, ...] = ()
    suffix_gan_only: bool = False


def _prefix_inputs(
    inputs: ForwardInput, counts: tuple[int, ...]
) -> _PrefixForwardInput:
    values = {field.name: getattr(inputs, field.name) for field in fields(ForwardInput)}
    return _PrefixForwardInput(**values, corpus_prefix_chunks=counts)


class CausalMiniMaxH3DMDGT(CausalMiniMaxH3DMD):
    """Select one of two mutually exclusive corpus conditioning modes.

    ``gen_repack`` selects complete GEN samples without altering rollout or
    FAKE. ``rollout_prefix`` selects leading full chunks to commit as corpus
    cache while preserving every denoising forward. FAKE score trains on the
    mixed corpus/generated clip. GEN keeps the DMD objective on the complete
    clip in either mode.
    """

    _carry_clean_latents = True
    _gen_query_keys = ("gen_xts", "gen_timesteps", "gen_audio_timesteps")

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        ratio = config.meta_model.gen_gt_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, Real):
            raise ValueError("meta_model.gen_gt_ratio must be a number in [0, 1]")
        self.gen_gt_ratio = float(ratio)
        if not 0.0 <= self.gen_gt_ratio <= 1.0:
            raise ValueError("meta_model.gen_gt_ratio must be a number in [0, 1]")
        self.gt_mode = config.meta_model.get("gt_mode", "gen_repack")
        if self.gt_mode not in ("gen_repack", "rollout_prefix"):
            raise ValueError("meta_model.gt_mode must be gen_repack or rollout_prefix")
        if self.gt_mode == "rollout_prefix":
            chunk_range = config.meta_model.gt_prefix_chunk_range
            if (
                not isinstance(chunk_range, (list, tuple))
                or len(chunk_range) != 2
                or any(type(value) is not int for value in chunk_range)
                or not 1 <= chunk_range[0] <= chunk_range[1]
            ):
                raise ValueError(
                    "gt_prefix_chunk_range must be positive integers [min, max]"
                )
            self.gt_prefix_chunk_range = tuple(chunk_range)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def prepare_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Replace selected samples' rollout repack with noised corpus media."""
        if self.gt_mode == "rollout_prefix":
            return self._prepare_prefix_gen(ctx)
        # Fork before the parent advances its GEN stream. Constructing child
        # streams is order-free and does not mutate the parent, so ratio zero is
        # exactly the inherited DMD path, including every subsequent RNG draw.
        gt_rng = ctx["rng"].fork("dmd2_gt_repack")
        ctx = super().prepare_gen(ctx)

        inputs = ctx["gen_inputs"]
        decision_rng = gt_rng.fork("decision")
        gt_mask = [
            decision_rng.python_generator.random() < self.gen_gt_ratio
            for _ in range(inputs.batch_size)
        ]
        ctx["gen_gt_mask"] = gt_mask
        if not any(gt_mask):
            return ctx

        if inputs.clean_latents is None:
            raise ValueError(
                "MiniMax-H3 GT repack requires corpus video/audio latents"
            )
        clean_video, clean_audio = inputs.clean_latents
        query_video_eps, query_audio_eps = self._sample_noises(
            inputs, gt_rng.fork("query_noise")
        )
        context_video_eps, context_audio_eps = self._sample_noises(
            inputs, gt_rng.fork("context_noise")
        )

        rollout = ctx["rollout_x0s"]
        query_key, video_t_key, audio_t_key = self._gen_query_keys
        rollout_video_xts, rollout_audio_xts = ctx[query_key]
        video_timesteps = ctx[video_t_key]
        audio_timesteps = ctx[audio_t_key]

        ctx[query_key] = (
            [
                self.schedule.forward(
                    x_0=clean_video[index],
                    x_T=query_video_eps[index],
                    t=video_timesteps[index : index + 1].to(torch.float32),
                )
                if use_gt
                else rollout_video_xts[index]
                for index, use_gt in enumerate(gt_mask)
            ],
            [
                self.schedule.forward(
                    x_0=clean_audio[index],
                    x_T=query_audio_eps[index],
                    t=audio_timesteps[index : index + 1].to(torch.float32),
                )
                if use_gt
                else rollout_audio_xts[index]
                for index, use_gt in enumerate(gt_mask)
            ],
        )
        # GEN owns a phase-local shallow copy of the rollout payload. Replacing
        # this key only changes the clean occurrences consumed by inherited
        # gen_forward and cannot affect FAKE or the offline rollout pool.
        ctx["rollout_x0s"] = _RolloutX0s(
            video=[
                clean_video[index] if use_gt else rollout.video[index]
                for index, use_gt in enumerate(gt_mask)
            ],
            audio=[
                clean_audio[index] if use_gt else rollout.audio[index]
                for index, use_gt in enumerate(gt_mask)
            ],
            video_eps=[
                context_video_eps[index] if use_gt else rollout.video_eps[index]
                for index, use_gt in enumerate(gt_mask)
            ],
            audio_eps=[
                context_audio_eps[index] if use_gt else rollout.audio_eps[index]
                for index, use_gt in enumerate(gt_mask)
            ],
            video_anchors=rollout.video_anchors,
        )
        return ctx

    def _video_renorm_start_chunks(self, inputs: ForwardInput) -> tuple[int, ...]:
        """Anchor on the first generated chunk after each sample's corpus prefix."""
        if isinstance(inputs, _PrefixForwardInput):
            return inputs.corpus_prefix_chunks
        return super()._video_renorm_start_chunks(inputs)

    def _check_prefix_geometry(self, inputs: ForwardInput) -> None:
        if inputs.clean_latents is None:
            raise ValueError("rollout_prefix requires corpus video/audio latents")
        maximum = self.gt_prefix_chunk_range[1]
        for layout in inputs.layouts:
            eligible = 0
            for chunk in layout.chunks:
                if (
                    not chunk.has_clean_copy
                    or chunk.video_stop - chunk.video_start != 5
                ):
                    break
                eligible += 1
            if maximum > eligible:
                raise ValueError(
                    f"gt_prefix_chunk_range maximum {maximum} exceeds {eligible} "
                    "leading cacheable full chunks; the tail cannot be selected"
                )

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().prepare_inputs(ctx)
        if self.gt_mode == "rollout_prefix":
            self._check_prefix_geometry(ctx["inputs"])
        return ctx

    @staticmethod
    def _replace_prefix(
        values: list[torch.Tensor],
        corpus: list[torch.Tensor],
        inputs: _PrefixForwardInput,
        modality: str,
    ) -> list[torch.Tensor]:
        """Build new mixed tensors without modifying corpus or trajectory storage."""
        dim = 1 if modality == "video" else 2
        result = []
        for value, clean, layout, count in zip(
            values, corpus, inputs.layouts, inputs.corpus_prefix_chunks, strict=True
        ):
            if not count:
                result.append(value)
                continue
            chunk = layout.chunks[count - 1]
            stop = chunk.video_stop if modality == "video" else chunk.audio_stop
            result.append(
                torch.cat(
                    (
                        clean.narrow(dim, 0, stop).detach(),
                        value.narrow(dim, stop, value.shape[dim] - stop),
                    ),
                    dim=dim,
                )
            )
        return result

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        if self.gt_mode != "rollout_prefix":
            return super().rollout(ctx)
        inputs = ctx["inputs"]
        self._check_prefix_geometry(inputs)
        rng = ctx["rng"].fork("dmd2_gt_prefix")
        decision = rng.fork("decision").python_generator
        lengths = rng.fork("length").python_generator
        minimum, maximum = self.gt_prefix_chunk_range
        counts = tuple(
            lengths.randint(minimum, maximum)
            if decision.random() < self.gen_gt_ratio
            else 0
            for _ in range(inputs.batch_size)
        )
        # No tag is necessary for all-rollout payloads. This also keeps ratio=0
        # on the exact inherited payload and RNG path.
        if not any(counts):
            return super().rollout(ctx)
        inputs = _prefix_inputs(inputs, counts)
        ctx["inputs"] = inputs
        ctx = super().rollout(ctx)
        rollout = ctx["rollout_x0s"]
        video, audio = inputs.clean_latents
        replacements = {
            "video": self._replace_prefix(rollout.video, video, inputs, "video"),
            "audio": self._replace_prefix(rollout.audio, audio, inputs, "audio"),
        }
        if self.fake_use_trajectory:
            replacements.update(
                fake_video=self._replace_prefix(
                    rollout.fake_video, video, inputs, "video"
                ),
                fake_audio=self._replace_prefix(
                    rollout.fake_audio, audio, inputs, "audio"
                ),
            )
        # Preserve cache-fill eps and the concrete trajectory payload type.
        ctx["rollout_x0s"] = replace(rollout, **replacements)
        return ctx

    @staticmethod
    def _prefix_clean_chunk_sources(
        inputs: ForwardInput,
        *,
        chunk_index: int,
        video_rows_source: list[torch.Tensor],
        audio_rows_source: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Select corpus rows for a clean cache commit without changing its eps."""
        if isinstance(inputs, _PrefixForwardInput):
            video, audio = inputs.clean_latents
            video_rows_source = list(video_rows_source)
            audio_rows_source = list(audio_rows_source)
            for index, count in enumerate(inputs.corpus_prefix_chunks):
                if chunk_index < count:
                    chunk = inputs.layouts[index].chunks[chunk_index]
                    video_rows_source[index] = video[index][
                        :, chunk.video_start : chunk.video_stop
                    ].detach()
                    audio_rows_source[index] = audio[index][
                        :, :, chunk.audio_start : chunk.audio_stop
                    ].detach()
        return video_rows_source, audio_rows_source

    def _chunk_forward(
        self,
        model: Any,
        inputs: ForwardInput,
        *,
        chunk_index: int,
        role: str,
        video_rows_source: list[torch.Tensor],
        audio_rows_source: list[torch.Tensor],
        update_cache: bool,
        **kwargs: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if role == "clean" and update_cache:
            video_rows_source, audio_rows_source = self._prefix_clean_chunk_sources(
                inputs,
                chunk_index=chunk_index,
                video_rows_source=video_rows_source,
                audio_rows_source=audio_rows_source,
            )
        return super()._chunk_forward(
            model,
            inputs,
            chunk_index=chunk_index,
            role=role,
            video_rows_source=video_rows_source,
            audio_rows_source=audio_rows_source,
            update_cache=update_cache,
            **kwargs,
        )

    def _prepare_prefix_gen(self, ctx: dict[str, Any]) -> dict[str, Any]:
        rng = ctx["rng"].fork("dmd2_gt_prefix_query")
        ctx = super().prepare_gen(ctx)
        inputs = ctx["gen_inputs"]
        if not isinstance(inputs, _PrefixForwardInput):
            ctx["gen_gt_mask"] = [False] * inputs.batch_size
            return ctx
        ctx["gen_gt_mask"] = [count > 0 for count in inputs.corpus_prefix_chunks]
        query_noise = self._sample_noises(inputs, rng)
        query_key, video_t_key, audio_t_key = self._gen_query_keys
        mixed_xts = []
        for modality, corpus, noises, xts, timesteps in zip(
            ("video", "audio"),
            inputs.clean_latents,
            query_noise,
            ctx[query_key],
            (ctx[video_t_key], ctx[audio_t_key]),
            strict=True,
        ):
            dim = 1 if modality == "video" else 2
            mixed = []
            for index, (clean, noise, xt, layout, count) in enumerate(
                zip(
                    corpus,
                    noises,
                    xts,
                    inputs.layouts,
                    inputs.corpus_prefix_chunks,
                    strict=True,
                )
            ):
                if not count:
                    mixed.append(xt)
                    continue
                chunk = layout.chunks[count - 1]
                stop = chunk.video_stop if modality == "video" else chunk.audio_stop
                prefix_xt = self.schedule.forward(
                    x_0=clean.narrow(dim, 0, stop),
                    x_T=noise.narrow(dim, 0, stop),
                    t=timesteps[index : index + 1].to(torch.float32),
                )
                mixed.append(
                    torch.cat(
                        (prefix_xt, xt.narrow(dim, stop, xt.shape[dim] - stop)), dim=dim
                    )
                )
            mixed_xts.append(mixed)
        ctx[query_key] = tuple(mixed_xts)
        # rollout_x0s already carries corpus prefix x0 and the exact eps used to
        # commit its cache. Never resample clean eps in this mode.
        return ctx


__all__ = ["CausalMiniMaxH3DMDGT"]
