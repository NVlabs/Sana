# SPDX-License-Identifier: Apache-2.0
"""DMD2 with reference-prefix critics and a chunk-causal student."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.discriminator_output import (
    MiniMaxH3DiscriminatorOutput,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_dmd import (
    _ScoreInputs,
    _VideoRefDMDInputs as _VideoRefDMD2Inputs,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_dmd2 import (
    CausalMiniMaxH3VideoRefDMD2,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_prefix_dmd import (
    CausalVideoRefPrefixDMDMixin,
    _PrefixScoreInputs,
    _ScoreHistory,
)


class CausalMiniMaxH3VideoRefPrefixDMD2(
    CausalVideoRefPrefixDMDMixin, CausalMiniMaxH3VideoRefDMD2
):
    """Add GAN objectives to prefix critics conditioned on causal outputs.

    Generated GAN samples use the student's detached final rollout history.
    Real samples and their R1 perturbations use detached corpus history with
    the same clean-endpoint noise. GAN heads select only noisy target rows.
    """

    def _gan_inputs(self, ctx: dict[str, Any], *, role: str) -> _PrefixScoreInputs:
        inputs = super()._gan_inputs(ctx, role=role)
        assert isinstance(inputs, _PrefixScoreInputs)
        if role != "real":
            return inputs
        assert inputs.history is not None
        video, audio = inputs.clean_latents
        history = replace(
            inputs.history,
            video=[value.detach() for value in video],
            audio=[value.detach() for value in audio],
        )
        return replace(inputs, history=history)

    def _pred_gan_logits(
        self,
        fake_model: Any,
        inputs: _PrefixScoreInputs,
        *,
        noisy_latents: tuple[list[torch.Tensor], list[torch.Tensor]],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        gan_chunk_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | MiniMaxH3DiscriminatorOutput:
        prefix = inputs.prefix_inputs
        kwargs = self._prefix_kwargs(
            fake_model, prefix,
            video_xts=noisy_latents[0], audio_xts=noisy_latents[1],
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
            **self._history_kwargs(inputs),
        )
        kwargs.update(self._prefix_gan_kwargs(prefix))
        k_lens = torch.as_tensor(
            kwargs["sample_lens"], dtype=torch.int32, device=prefix.token_tags.device
        )
        live_documents = torch.arange(
            k_lens.numel(), device=k_lens.device
        ) < inputs.batch_size
        head_options = (
            {"gan_chunk_keep_mask": gan_chunk_keep_mask}
            if gan_chunk_keep_mask is not None
            else {}
        )
        return fake_model(
            **kwargs,
            classify_mode=True,
            gan_video_timesteps=video_timesteps,
            gan_k_lens=k_lens,
            gan_live_documents=live_documents,
            **head_options,
        )


EntryClass = CausalMiniMaxH3VideoRefPrefixDMD2

__all__ = ["CausalMiniMaxH3VideoRefPrefixDMD2", "EntryClass"]
