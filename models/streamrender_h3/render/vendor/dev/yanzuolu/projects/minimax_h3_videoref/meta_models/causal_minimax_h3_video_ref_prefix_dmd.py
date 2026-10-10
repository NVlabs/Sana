# SPDX-License-Identifier: Apache-2.0
"""Reference-prefix DMD critics paired with a chunk-causal student."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_df import (
    VideoRefForwardInput,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_dmd import (
    CausalMiniMaxH3VideoRefDMD,
    _ScoreInputs,
    _VideoRefDMDInputs,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.prefix_forward import (
    PrefixForwardMixin,
    PrefixTFInputs,
)


@dataclass(frozen=True)
class _ScoreHistory:
    """Detached target history and its clean-endpoint noise."""

    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    video_eps: list[torch.Tensor]
    audio_eps: list[torch.Tensor]


@dataclass(frozen=True)
class _PrefixScoreInputs(_ScoreInputs):
    """Static critic packing plus the history selected for one training phase."""

    prefix_inputs: PrefixTFInputs
    history: _ScoreHistory | None = None


class CausalVideoRefPrefixDMDMixin(PrefixForwardMixin):
    """Bind prefix critics to the causal student's final detached history.

    Fake denoising and both DMD scores use the same final rollout history and
    clean-endpoint noise. Only noisy target rows contribute predictions.
    """

    def _attach_score_inputs(
        self,
        student: VideoRefForwardInput,
        encoded_batch: dict[str, Any],
        score_parts: dict[str, Any],
    ) -> _VideoRefDMDInputs:
        inputs = super()._attach_score_inputs(student, encoded_batch, score_parts)
        native_score = inputs.score_inputs
        plans = [
            self._prefix_plan_factory(
                target_video_shape=layout.latent_shape,
                target_audio_shape=layout.audio_shape,
                video_temporal_mapping=layout.video_temporal_mapping,
                video_chunk_ranges=[
                    (chunk.video_start, chunk.video_stop) for chunk in layout.chunks
                ],
                audio_chunk_ranges=[
                    (chunk.audio_start, chunk.audio_stop) for chunk in layout.chunks
                ],
            )
            for layout in student.layouts
        ]

        def prefix_view(native_inputs: _ScoreInputs) -> _PrefixScoreInputs:
            prefix = self._prefix_inputs_from_payload({
                "native": native_inputs.native,
                "prompt_embeds": native_inputs.prompt_embeds,
                "references": score_parts["references"],
                "latent_shapes": [layout.latent_shape for layout in student.layouts],
                "audio_shapes": [layout.audio_shape for layout in student.layouts],
                "prefix_plans": plans,
            })
            values = {
                field.name: getattr(native_inputs, field.name)
                for field in fields(_ScoreInputs)
            }
            values.update(
                layouts=prefix.layouts, seqlens=prefix.seqlens,
                token_tags=prefix.token_tags,
            )
            return _PrefixScoreInputs(**values, prefix_inputs=prefix)

        score = prefix_view(native_score)
        if native_score.negative_inputs is not None:
            score = replace(score, negative_inputs=prefix_view(native_score.negative_inputs))
        return replace(inputs, score_inputs=score)

    def _critic_inputs(
        self, ctx: dict[str, Any], inputs: _VideoRefDMDInputs
    ) -> _PrefixScoreInputs:
        score = inputs.score_inputs
        assert isinstance(score, _PrefixScoreInputs)
        rollout = ctx["rollout_x0s"]
        history = _ScoreHistory(
            video=[value.detach() for value in rollout.video],
            audio=[value.detach() for value in rollout.audio],
            video_eps=[value.detach() for value in rollout.video_eps],
            audio_eps=[value.detach() for value in rollout.audio_eps],
        )
        return replace(score, history=history)

    @staticmethod
    def _history_kwargs(inputs: _PrefixScoreInputs) -> dict[str, Any]:
        history = inputs.history
        assert history is not None
        return {
            "video_context": history.video,
            "audio_context": history.audio,
            "video_eps": history.video_eps,
            "audio_eps": history.audio_eps,
        }

    def _teacher_negative_inputs(
        self, ctx: dict[str, Any], inputs: _PrefixScoreInputs
    ) -> _PrefixScoreInputs:
        """Use empty-language prefix rows with the same phase history and noise."""
        negative = super()._teacher_negative_inputs(ctx, inputs)
        assert isinstance(negative, _PrefixScoreInputs)
        assert inputs.history is not None
        return replace(negative, history=inputs.history)

    def _bidirectional_forward(
        self,
        model: Any,
        inputs: _PrefixScoreInputs,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self._prefix_forward(
            model, inputs.prefix_inputs,
            video_xts=video_xts, audio_xts=audio_xts,
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
            **self._history_kwargs(inputs),
        )


class CausalMiniMaxH3VideoRefPrefixDMD(
    CausalVideoRefPrefixDMDMixin, CausalMiniMaxH3VideoRefDMD
):
    """Train a causal student and prefix fake score against a frozen teacher.

    Both critic nodes use ordinary causal H3 X0 wrappers such as
    ``MiniMaxH3CausalX0DiTSP``.
    """


EntryClass = CausalMiniMaxH3VideoRefPrefixDMD

__all__ = [
    "CausalVideoRefPrefixDMDMixin",
    "CausalMiniMaxH3VideoRefPrefixDMD",
    "EntryClass",
]
