# SPDX-License-Identifier: Apache-2.0
"""DMD2 objectives with corpus GEN repacking or corpus-prefix conditioning."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd import ForwardInput
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd2 import (
    CausalMiniMaxH3DMD2,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd_gt import (
    CausalMiniMaxH3DMDGT,
    _PrefixForwardInput,
    _prefix_inputs as _prefix_inputs,
)
from dev.yanzuolu.projects.minimax_h3.modeling.discriminator_output import (
    MiniMaxH3DiscriminatorOutput,
)


class CausalMiniMaxH3DMD2GT(CausalMiniMaxH3DMDGT, CausalMiniMaxH3DMD2):
    """Compose corpus conditioning with DMD2 objectives.

    With ``rollout_prefix``, FAKE losses exclude corpus-prefix chunks. V3 also
    excludes them before its global feature aggregation. GEN keeps DMD and GAN
    losses on the complete clip.
    """

    @staticmethod
    def _require_chunk_discriminator(model: Any) -> None:
        # PEFT ModulesToSaveWrapper and FSDP retain the real module in the tree.
        # Inspect that tree rather than relying on the outer wrapper's type.
        from dev.yanzuolu.projects.minimax_h3.modeling.discriminator import (
            MiniMaxH3DMD2DiscriminatorV2,
        )

        heads = [
            module
            for module in model.modules()
            if isinstance(module, MiniMaxH3DMD2DiscriminatorV2)
        ]
        if not heads or any(head.video_chunk_size != 5 for head in heads):
            raise ValueError(
                "rollout_prefix requires a V2 or V3 discriminator with video_chunk_size=5"
            )

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        if self.gt_mode == "rollout_prefix":
            self._require_chunk_discriminator(ctx["models"]["fake_model"])
        return super().prepare_inputs(ctx)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def fake_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = ctx["fake_inputs"]
        if not isinstance(inputs, _PrefixForwardInput):
            return super().fake_loss(ctx)
        # The tag is confined to FAKE loss. Parent fake_loss routes fake, real
        # and optional aR1 through _pred_gan_logits and preserves three separate
        # backward terms. GEN's untagged discriminator call scores every chunk.
        fake_ctx = dict(ctx)
        fake_ctx["fake_inputs"] = replace(inputs, suffix_gan_only=True)
        fake_ctx = super().fake_loss(fake_ctx)
        fake_ctx["fake_inputs"] = inputs
        return fake_ctx

    def _pred_gan_logits(
        self, fake_model: Any, inputs: ForwardInput, **kwargs: Any
    ) -> torch.Tensor | MiniMaxH3DiscriminatorOutput:
        if not isinstance(inputs, _PrefixForwardInput) or not inputs.suffix_gan_only:
            return super()._pred_gan_logits(fake_model, inputs, **kwargs)
        keep = torch.cat(
            [
                torch.arange(len(layout.chunks), device=inputs.token_tags.device) >= count
                for layout, count in zip(
                    inputs.layouts, inputs.corpus_prefix_chunks, strict=True
                )
            ]
        )
        from dev.yanzuolu.projects.minimax_h3.modeling.discriminator import (
            MiniMaxH3DMD2DiscriminatorV3,
        )

        if any(isinstance(module, MiniMaxH3DMD2DiscriminatorV3) for module in fake_model.modules()):
            return super()._pred_gan_logits(
                fake_model, inputs, gan_chunk_keep_mask=keep, **kwargs
            )
        logits = super()._pred_gan_logits(fake_model, inputs, **kwargs)
        assert isinstance(logits, torch.Tensor)
        assert (
            logits.shape[0] == keep.numel()
        ), "V2 logits must follow sample-major chunk order"
        return logits[keep]


__all__ = ["CausalMiniMaxH3DMD2GT"]
