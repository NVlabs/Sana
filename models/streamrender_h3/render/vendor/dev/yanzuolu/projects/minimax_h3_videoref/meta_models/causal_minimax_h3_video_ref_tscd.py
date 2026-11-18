# SPDX-License-Identifier: Apache-2.0
"""Trajectory-segmented consistency distillation with a clean video reference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_tscd import (
    CausalMiniMaxH3TSCD,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_df import (
    VideoRefForwardInput,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_tf import (
    CausalMiniMaxH3VideoRefTF,
)


class CausalMiniMaxH3VideoRefTSCD(
    CausalMiniMaxH3TSCD,
    CausalMiniMaxH3VideoRefTF,
):
    """TSCD over corpus targets conditioned on one clean video reference."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # CausalMiniMaxH3TSCD deliberately terminates cooperative __init__, so
        # initialize codec geometry and the video-reference caches explicitly.
        self._init_video_ref_context(config)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Add TSCD target noise and one shared clean-reference noise draw."""
        ctx = super().add_noise(ctx)
        inputs = ctx["inputs"]
        reference_eps = self._sample_reference_noises(inputs, ctx["rng"])
        ctx["reference_eps"] = reference_eps
        ctx["inputs"] = replace(inputs, reference_eps=reference_eps)
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
        reference_eps: list[torch.Tensor] | None = None,
        video_context: list[torch.Tensor] | None = None,
        audio_context: list[torch.Tensor] | None = None,
        video_eps: list[torch.Tensor] | None = None,
        audio_eps: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Supply the same reference noise to student, teacher, and EMA."""
        if reference_eps is None:
            reference_eps = inputs.reference_eps
        assert reference_eps is not None, "video-reference forward requires reference_eps"
        return super()._causal_packed_forward(
            model,
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            reference_eps=reference_eps,
            video_context=video_context,
            audio_context=audio_context,
            video_eps=video_eps,
            audio_eps=audio_eps,
        )


EntryClass = CausalMiniMaxH3VideoRefTSCD

__all__ = ["CausalMiniMaxH3VideoRefTSCD", "EntryClass"]
