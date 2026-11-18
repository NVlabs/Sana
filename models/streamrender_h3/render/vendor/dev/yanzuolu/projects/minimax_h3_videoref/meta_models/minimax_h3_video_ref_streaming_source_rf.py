# SPDX-License-Identifier: Apache-2.0
"""Resampling forcing whose current chunk starts from the semantic video.

``MiniMaxH3VideoRefStreamingRF`` reads the paired semantic video as near-clean
reference rows in every window. This meta keeps its data, pictures, Qwen
context, chains, resample pass and validation requests, but its windows carry
no reference rows. The semantic latent instead enters through the noisy input
of the current chunk, the bridge of ``SemanticSourceMixin`` in
``streaming_semantic``. The resample pass noises the chunk with the same bridge
at ``sigma u_s``, so the written-back history follows the training
distribution.

Each chain keeps every clip's semantic latents with its conditioning, and each
window's payload carries them as ``source_latents``. With a training UP size
above 1 the SP input broadcast therefore gives every group rank the source
rank's sources along with its working copies.

With ``picture_mode: keyframe`` every window carries the picture as the
native first-frame keyframe block at target latent 0's time. Target latent 0
itself is an ordinary chunk row, bridged, supervised and written back.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_source_rf
      class_name: MiniMaxH3VideoRefStreamingSourceRF
      source_sigma: 0.95
      qwen_reference_video: false
      picture_mode: keyframe
      resample_timesteps: {...}
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import (
    MiniMaxH3VideoRefStreamingRF,
    _ChainState,
    _VideoRefChainConditioning,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_semantic import SemanticSourceMixin, SourceStreamingBatch


@dataclass
class _SourceChainConditioning(_VideoRefChainConditioning):
    """The parent's chain conditioning plus every clip's complete semantic latents."""

    sources: list[torch.Tensor] | None = None


class SemanticChainMixin:
    """Plan chain windows without reference rows and carry each clip's semantic latents in every window's payload."""

    def _chain_head_inputs(self, batch: dict[str, Any]) -> tuple[StreamingBatch, None]:
        """Plan every window without reference rows. The semantic latents stay with the conditioning."""
        geometry, _ = super()._chain_head_inputs(batch)
        return replace(geometry, reference_latents=None), None

    def _chain_head_conditioning(self, ctx: dict[str, Any], entry: _ChainState) -> _SourceChainConditioning:
        conditioning = super()._chain_head_conditioning(ctx, entry)
        sources = [value.to(get_device()) for value in ctx["batch"]["reference_video_latents"]]
        assert all(tuple(value.shape) == tuple(shape) for value, shape in zip(sources, entry.latent_shapes, strict=True)), (
            "the semantic source must share the target's latent grid"
        )
        values = {field.name: getattr(conditioning, field.name) for field in fields(conditioning)}
        return _SourceChainConditioning(**values, sources=sources)

    def _chain_window_conditioning(
        self, ctx: dict[str, Any], entry: _ChainState, window: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        plans, conditioning = super()._chain_window_conditioning(ctx, entry, window)
        return plans, dict(conditioning, source_latents=entry.conditioning.sources)


class MiniMaxH3VideoRefStreamingSourceRF(SemanticSourceMixin, SemanticChainMixin, MiniMaxH3VideoRefStreamingRF):
    """Resampling forcing that bridges each chunk from its semantic latent instead of reading reference rows."""

    def _start_chain(self, ctx: dict[str, Any]) -> _ChainState:
        """Resample each clip's chunk at the video time ``sigma u_s``, audio at its own ``t_s``."""
        entry = super()._start_chain(ctx)
        video, audio = entry.resample_timesteps
        entry.resample_timesteps = (video * self.source_sigma, audio)
        return entry


EntryClass = MiniMaxH3VideoRefStreamingSourceRF

__all__ = ["MiniMaxH3VideoRefStreamingSourceRF", "SemanticChainMixin", "SourceStreamingBatch", "EntryClass"]
