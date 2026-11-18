# SPDX-License-Identifier: Apache-2.0
"""Resampling forcing whose target video rows also read their backward optical flow.

This meta keeps everything of ``MiniMaxH3VideoRefStreamingRF``, the reference
rows, pictures, Qwen context, chains, resample pass, captionless branch and
validation requests, and adds the flow of every target latent as per-token
AdaLN conditioning, as ``StreamingGeometryMixin`` in ``streaming_geometry``
describes. Each chain keeps every clip's complete flow maps on the device with
its conditioning, and each window's payload carries them, so with a training
UP size above 1 the SP input broadcast gives every group rank the source
rank's maps along with its working copies.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_geometry_rf
      class_name: MiniMaxH3VideoRefStreamingGeometryRF
    data:
      args:
        geometry_root: /path/to/torcs_clips
    validation:
      requests:
        - geometry_path: /path/to/torcs_clips/<window>/geometry.safetensors
    models:
      backbone:
        args:
          adaln_condition_dim: 256
          adaln_condition_channels: 20
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import (
    MiniMaxH3VideoRefStreamingRF,
    _ChainState,
    _VideoRefChainConditioning,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.streaming_geometry import StreamingGeometryMixin


@dataclass
class _GeometryChainConditioning(_VideoRefChainConditioning):
    """The parent's chain conditioning plus every clip's complete flow condition maps."""

    adaln_condition_maps: list[torch.Tensor] | None = None


class GeometryChainMixin:
    """Keep each clip's flow maps with the chain and carry them in every window's payload."""

    def _chain_head_conditioning(self, ctx: dict[str, Any], entry: _ChainState) -> _GeometryChainConditioning:
        conditioning = super()._chain_head_conditioning(ctx, entry)
        maps = [value.to(get_device()) for value in ctx["batch"]["adaln_condition_maps"]]
        assert all(value.shape[0] == shape[1] for value, shape in zip(maps, entry.latent_shapes, strict=True)), (
            "every target latent needs one condition map"
        )
        values = {field.name: getattr(conditioning, field.name) for field in fields(conditioning)}
        return _GeometryChainConditioning(**values, adaln_condition_maps=maps)

    def _chain_window_conditioning(
        self, ctx: dict[str, Any], entry: _ChainState, window: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        plans, conditioning = super()._chain_window_conditioning(ctx, entry, window)
        return plans, dict(conditioning, adaln_condition_maps=entry.conditioning.adaln_condition_maps)


class MiniMaxH3VideoRefStreamingGeometryRF(StreamingGeometryMixin, GeometryChainMixin, MiniMaxH3VideoRefStreamingRF):
    """Resampling forcing with backward optical flow on every target video row."""


EntryClass = MiniMaxH3VideoRefStreamingGeometryRF

__all__ = ["GeometryChainMixin", "MiniMaxH3VideoRefStreamingGeometryRF", "EntryClass"]
