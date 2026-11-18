# SPDX-License-Identifier: Apache-2.0
"""Per-token AdaLN condition maps for streaming H3 windows.

A plan may carry ``adaln_condition_maps`` ``[T, C, 4h, 4w]``: one map per
latent of its sample's complete target video, at four times the h x w video
token grid. Every target video row, history and current chunk alike, reads the
map of its own latent. The maps are selected at the plan's ``video_indices``
only when the kwargs are built, so a branch that edits a plan's rows, by
dropping the reference or the target history, stays aligned. The DiT receives
the selected maps as ``adaln_condition_maps``, one tensor per sample since the
samples of a pack may differ in resolution, and, as ``adaln_condition_rows``,
the packed rows they describe in (latent, h, w) order, which is the order of
the patchified target video rows at the tail of every sample. Text, picture or
keyframe, reference, audio and padding rows are never listed.
"""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs


class StreamingAdalnConditionMixin:
    """Route each plan's ``adaln_condition_maps`` to its target video rows."""

    def _streaming_kwargs(self, model: Any, inputs: StreamingInputs, **kwargs: Any) -> dict[str, Any]:
        result = super()._streaming_kwargs(model, inputs, **kwargs)
        device = result["x"].device
        maps, rows = [], []
        stop = 0
        for plan, pack in zip(inputs.plans, inputs.packs, strict=True):
            complete = plan["adaln_condition_maps"]
            height, width = plan["target_video_shape"][2:]
            assert tuple(complete.shape[2:]) == (2 * height, 2 * width), (
                f"condition maps {tuple(complete.shape[2:])} are not four times the token grid of latents {(height, width)}"
            )
            maps.append(complete.index_select(0, plan["video_indices"].to(complete.device)).to(device))
            stop += int(pack["sample_lens"])
            rows.append(torch.arange(stop - int(pack["video_rows"]), stop, device=device))
        result["adaln_condition_maps"] = maps
        result["adaln_condition_rows"] = torch.cat(rows)
        return result


__all__ = ["StreamingAdalnConditionMixin"]
