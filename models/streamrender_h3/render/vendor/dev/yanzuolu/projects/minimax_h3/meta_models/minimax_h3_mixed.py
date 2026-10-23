# SPDX-License-Identifier: Apache-2.0
"""Mixed full and chunk-causal diffusion forcing for MiniMax-H3.

Each sequence-parallel group draws one attention mode for the entire physical
pack on every microstep.  Causal steps retain the parent diffusion-forcing
routing and per-chunk paired timesteps.  Full steps keep the same rows while
rewriting routing to one dense rectangle per sample and expanding one paired
timestep over all chunks of that sample.

Validation is independent of the training draw.  Causal validation uses the
parent KV-cached rollout.  Bidirectional validation samples the full physical
sequence without a cache through the same hybrid backbone.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Sequence

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import combine_seed, local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import (
    ForwardInput,
    _RolloutX0s,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_df import CausalMiniMaxH3DF
from dev.yanzuolu.projects.minimax_h3.modeling.hybrid_model import MINIMAX_H3_DENSE_ATTENTION

_FULL = "full"
_CAUSAL = "causal"
_BIDIRECTIONAL = "bidirectional"
_VALIDATION_MODES = (_BIDIRECTIONAL, _CAUSAL)


class MiniMaxH3Mixed(CausalMiniMaxH3DF):
    """Train one MiniMax-H3 parameter tree with full and causal attention."""

    def __init__(self, config: Any) -> None:
        validation = config.validation
        validations = validation if isinstance(validation, list) else [validation]
        modes = [entry.get("mode") for entry in validations]
        if not modes or any(mode not in _VALIDATION_MODES for mode in modes):
            raise ValueError(
                "each validation.mode must be exactly 'bidirectional' or "
                f"'causal', got {modes!r}"
            )
        if len(validations) > 1:
            names = [entry.get("name") for entry in validations]
            if any(name is None for name in names) or len(names) != len(set(names)):
                raise ValueError(
                    "multiple validation configs must each have a unique name, "
                    f"got {names!r}"
                )
        self.validation_mode = modes[0]

        super().__init__(config)
        probability = config.meta_model.get("full_attention_probability")
        if probability is None:
            raise ValueError("meta_model.full_attention_probability is required")
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "meta_model.full_attention_probability must lie in [0, 1], got "
                f"{probability}"
            )
        self.full_attention_probability = probability

    def _draw_attention_mode(self, ctx: dict[str, Any]) -> str:
        stream = random.Random(
            combine_seed(ctx["rng"].seed, "minimax_h3_attention_mode")
        )
        return _FULL if stream.random() < self.full_attention_probability else _CAUSAL

    @staticmethod
    def _require_hybrid_backbone(backbone: Any) -> None:
        if not bool(getattr(backbone, "_minimax_h3_hybrid_abi", False)):
            raise TypeError(
                "MiniMaxH3Mixed requires a backbone carrying "
                "_minimax_h3_hybrid_abi"
            )

    @staticmethod
    def _require_diffusion_forcing_layouts(inputs: ForwardInput) -> None:
        if any(
            chunk.has_clean_copy
            for layout in inputs.layouts
            for chunk in layout.chunks
        ):
            raise ValueError(
                "full-attention MiniMax-H3 steps require diffusion-forcing "
                "layouts without physical clean copies"
            )

    @staticmethod
    def _is_full_routing(inputs: ForwardInput) -> bool:
        if not inputs.sample_lens:
            return False
        return all(
            len(modes) == 1
            and modes[0] == _FULL
            and len(splits) == 1
            and int(splits[0]) == int(length)
            for modes, splits, length in zip(
                inputs.attn_modes,
                inputs.split_lens,
                inputs.sample_lens,
                strict=True,
            )
        )

    def _full_routing_inputs(self, inputs: ForwardInput) -> ForwardInput:
        """Replace only attention routing with one full rectangle per sample."""
        self._require_diffusion_forcing_layouts(inputs)
        lengths = [int(value) for value in inputs.sample_lens]
        offsets = self._offsets(lengths)
        rectangles = torch.tensor(
            [
                [offset, offset + length]
                for offset, length in zip(offsets, lengths, strict=True)
            ],
            dtype=inputs.q_ranges.dtype,
            device=inputs.q_ranges.device,
        )
        return replace(
            inputs,
            split_lens=[[length] for length in lengths],
            attn_modes=[[_FULL] for _ in lengths],
            q_ranges=rectangles,
            k_ranges=rectangles.clone(),
            attn_type_map=torch.zeros(
                len(lengths),
                dtype=inputs.attn_type_map.dtype,
                device=inputs.attn_type_map.device,
            ),
            attn_workloads=[length * length for length in lengths],
        )

    @staticmethod
    def _block_mask(
        inputs: ForwardInput,
        device: torch.device,
        *,
        sample_lens: list[int],
        split_lens: list[int],
        attn_modes: list[str],
    ) -> Any:
        if MiniMaxH3Mixed._is_full_routing(inputs):
            return MINIMAX_H3_DENSE_ATTENTION.bind(list(sample_lens))
        return CausalMiniMaxH3DF._block_mask(
            inputs,
            device,
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
        )

    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        config = ctx["config"]
        validation = config.validation
        if not isinstance(validation, list):
            self.validation_mode = validation.mode
            return super().validate(ctx)

        original_cache = self._validation_packer_cache
        try:
            for entry in validation:
                config["validation"] = entry
                self.validation_mode = entry.mode
                self._validation_packer_cache = None
                super().validate(ctx)
            return ctx
        finally:
            config["validation"] = validation
            self._validation_packer_cache = original_cache

    @torch.no_grad()
    def _bidirectional_rollout_latents(
        self,
        backbone: Any,
        inputs: ForwardInput,
        rngs: Sequence[Any],
    ) -> _RolloutX0s:
        """Cache-free full-sequence sampling through the hybrid backbone."""
        device = get_device()
        video_xts = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(rngs[index])
            )
            for index, layout in enumerate(inputs.layouts)
        ]
        audio_xts = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(rngs[index])
            )
            for index, layout in enumerate(inputs.layouts)
        ]

        self.sampling_timesteps.set_timesteps(
            seqlen=inputs.seqlens, device=device
        )
        self.audio_sampling_timesteps.set_timesteps(
            seqlen=inputs.seqlens, device=device
        )
        video_grid = self.sampling_timesteps.timesteps
        audio_grid = self.audio_sampling_timesteps.timesteps
        if video_grid.dim() != 1 or audio_grid.dim() != 1:
            raise NotImplementedError(
                "dynamic-shift sampling grids are not supported by full-sequence "
                "MiniMax-H3 sampling"
            )
        if video_grid.numel() != audio_grid.numel():
            raise ValueError(
                "sampling_timesteps and audio_sampling_timesteps must have the "
                f"same number of steps, got {video_grid.numel()} and "
                f"{audio_grid.numel()}"
            )

        chunk_counts = [len(layout.chunks) for layout in inputs.layouts]
        for step in range(int(video_grid.numel())):
            video_t = video_grid[step].expand(inputs.batch_size).to(device)
            audio_t = audio_grid[step].expand(inputs.batch_size).to(device)
            video_s = self.sampling_timesteps.get_next_timesteps(video_t)
            audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)
            video_pred, audio_pred = self._causal_packed_forward(
                backbone,
                inputs,
                video_xts=video_xts,
                audio_xts=audio_xts,
                video_context=video_xts,
                audio_context=audio_xts,
                video_eps=video_xts,
                audio_eps=audio_xts,
                video_timesteps=[
                    video_t[index].expand(count)
                    for index, count in enumerate(chunk_counts)
                ],
                audio_timesteps=[
                    audio_t[index].expand(count)
                    for index, count in enumerate(chunk_counts)
                ],
                text_timesteps=video_t,
            )
            video_xts = self.sampler.step_to(
                pred=video_pred,
                x_t=video_xts,
                t=video_t,
                s=video_s,
                rng=list(rngs),
                seqlens=inputs.seqlens,
            )
            audio_xts = self.sampler.step_to(
                pred=audio_pred,
                x_t=audio_xts,
                t=audio_t,
                s=audio_s,
                rng=list(rngs),
                seqlens=inputs.seqlens,
            )

        return _RolloutX0s(
            video=video_xts,
            audio=audio_xts,
            video_eps=None,
            audio_eps=None,
        )

    @torch.no_grad()
    def _bidirectional_guided_rollout_latents(
        self,
        backbone: Any,
        branch_inputs: Sequence[tuple[float, ForwardInput]],
        rngs: Sequence[Any],
    ) -> _RolloutX0s:
        """Sample dense caption branches on one shared trajectory."""
        inputs = branch_inputs[0][1]
        device = get_device()
        video_xts = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(rngs[index])
            )
            for index, layout in enumerate(inputs.layouts)
        ]
        audio_xts = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(rngs[index])
            )
            for index, layout in enumerate(inputs.layouts)
        ]

        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(
            seqlen=inputs.seqlens, device=device
        )
        video_grid = self.sampling_timesteps.timesteps
        audio_grid = self.audio_sampling_timesteps.timesteps
        if video_grid.dim() != 1 or audio_grid.dim() != 1:
            raise NotImplementedError(
                "dynamic-shift sampling grids are not supported by full-sequence "
                "MiniMax-H3 sampling"
            )
        if video_grid.numel() != audio_grid.numel():
            raise ValueError(
                "sampling_timesteps and audio_sampling_timesteps must have the "
                f"same number of steps, got {video_grid.numel()} and "
                f"{audio_grid.numel()}"
            )

        for step in range(int(video_grid.numel())):
            video_t = video_grid[step].expand(inputs.batch_size).to(device)
            audio_t = audio_grid[step].expand(inputs.batch_size).to(device)
            video_s = self.sampling_timesteps.get_next_timesteps(video_t)
            audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)

            video_pred: list[torch.Tensor] | None = None
            audio_pred: list[torch.Tensor] | None = None
            for weight, branch in branch_inputs:
                chunk_counts = [len(layout.chunks) for layout in branch.layouts]
                branch_video, branch_audio = self._causal_packed_forward(
                    backbone,
                    branch,
                    video_xts=video_xts,
                    audio_xts=audio_xts,
                    video_context=video_xts,
                    audio_context=audio_xts,
                    video_eps=video_xts,
                    audio_eps=audio_xts,
                    video_timesteps=[
                        video_t[index].expand(count)
                        for index, count in enumerate(chunk_counts)
                    ],
                    audio_timesteps=[
                        audio_t[index].expand(count)
                        for index, count in enumerate(chunk_counts)
                    ],
                    text_timesteps=video_t,
                )
                if video_pred is None:
                    video_pred = [float(weight) * value for value in branch_video]
                    audio_pred = [float(weight) * value for value in branch_audio]
                else:
                    video_pred = [
                        pred + float(weight) * value
                        for pred, value in zip(
                            video_pred, branch_video, strict=True
                        )
                    ]
                    assert audio_pred is not None
                    audio_pred = [
                        pred + float(weight) * value
                        for pred, value in zip(
                            audio_pred, branch_audio, strict=True
                        )
                    ]

            assert video_pred is not None and audio_pred is not None
            video_xts = self.sampler.step_to(
                pred=video_pred,
                x_t=video_xts,
                t=video_t,
                s=video_s,
                rng=list(rngs),
                seqlens=inputs.seqlens,
            )
            audio_xts = self.sampler.step_to(
                pred=audio_pred,
                x_t=audio_xts,
                t=audio_t,
                s=audio_s,
                rng=list(rngs),
                seqlens=inputs.seqlens,
            )

        return _RolloutX0s(
            video=video_xts,
            audio=audio_xts,
            video_eps=None,
            audio_eps=None,
        )

    @torch.no_grad()
    def _guided_rollout_latents(
        self,
        backbone: Any,
        branch_inputs: Sequence[tuple[float, ForwardInput]],
        rngs: Sequence[Any],
        *,
        first_branch_without_history: bool = False,
    ) -> _RolloutX0s:
        """Run caption guidance in the configured validation attention mode."""
        self._require_hybrid_backbone(backbone)
        if self.validation_mode == _CAUSAL:
            return super()._guided_rollout_latents(
                backbone,
                branch_inputs,
                rngs,
                first_branch_without_history=first_branch_without_history,
            )

        dense_branch_inputs = [
            (weight, self._full_routing_inputs(inputs))
            for weight, inputs in branch_inputs
        ]
        return self._bidirectional_guided_rollout_latents(
            backbone, dense_branch_inputs, rngs
        )

    @torch.no_grad()
    def _rollout_latents(
        self,
        backbone: Any,
        inputs: ForwardInput,
        rng: Any,
        *,
        keep_trajectory: bool,
        rngs: Sequence[Any] | None = None,
        trajectory_x0_chunks: list[
            tuple[list[torch.Tensor], list[torch.Tensor]]
        ]
        | None = None,
    ) -> tuple[
        _RolloutX0s,
        list[tuple[list[torch.Tensor], list[torch.Tensor]]],
    ]:
        self._require_hybrid_backbone(backbone)
        if self.validation_mode == _CAUSAL:
            return super()._rollout_latents(
                backbone,
                inputs,
                rng,
                keep_trajectory=keep_trajectory,
                rngs=rngs,
                trajectory_x0_chunks=trajectory_x0_chunks,
            )

        sample_rngs = list(rngs) if rngs is not None else [rng] * inputs.batch_size
        full_inputs = self._full_routing_inputs(inputs)
        return self._bidirectional_rollout_latents(
            backbone, full_inputs, sample_rngs
        ), []

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Draw the pack mode and its corresponding paired timesteps."""
        mode = self._draw_attention_mode(ctx)
        ctx["minimax_h3_attention_mode"] = mode
        if mode == _CAUSAL:
            return super().sample_timesteps(ctx)

        inputs = ctx["inputs"]
        chunk_counts = [len(layout.chunks) for layout in inputs.layouts]
        if any(count <= 0 for count in chunk_counts):
            raise ValueError("every full-attention sample must contain a chunk")
        self._require_diffusion_forcing_layouts(inputs)

        rng = ctx["rng"]
        with local_seed(rng.seed % 2**31):
            drawn = self.training_timesteps.sample_pair(
                (inputs.batch_size,), inputs.seqlens, get_device()
            )
        rng.seed = yield_seed(rng.seed)

        video_t, audio_t = (value.to(torch.float32) for value in drawn)
        ctx["train_timesteps"] = (
            [
                video_t[index].expand(count)
                for index, count in enumerate(chunk_counts)
            ],
            [
                audio_t[index].expand(count)
                for index, count in enumerate(chunk_counts)
            ],
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self._require_hybrid_backbone(ctx["models"]["backbone"])
        if ctx["minimax_h3_attention_mode"] == _CAUSAL:
            return super().forward(ctx)

        video_xts, audio_xts = ctx["noisy_latents"]
        clean_video, clean_audio = ctx["clean_latents"]
        video_eps, audio_eps = ctx["context_eps"]
        video_t, audio_t = ctx["train_timesteps"]
        ctx["pred"] = self._causal_packed_forward(
            ctx["models"]["backbone"],
            self._full_routing_inputs(ctx["inputs"]),
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_context=clean_video,
            audio_context=clean_audio,
            video_eps=video_eps,
            audio_eps=audio_eps,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
            text_timesteps=torch.stack([timesteps[0] for timesteps in video_t]),
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().compute_loss(ctx)
        mode = ctx["minimax_h3_attention_mode"]
        metrics = ctx.setdefault("metrics", {})
        metrics["train/mode_full_ratio"] = 1.0 if mode == _FULL else 0.0
        metrics[f"train/loss_{mode}"] = float(ctx["loss"].detach().item())
        return ctx


EntryClass = MiniMaxH3Mixed

__all__ = ["EntryClass", "MiniMaxH3Mixed"]
