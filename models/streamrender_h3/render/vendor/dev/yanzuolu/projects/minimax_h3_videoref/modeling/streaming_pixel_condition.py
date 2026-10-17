# SPDX-License-Identifier: Apache-2.0
"""Pixel condition videos as the DiT's per-token input condition on streaming H3 windows.

A plan may carry ``condition_frames`` uint8 [F, H, W, 3], every frame of its
sample's complete target-aligned condition video, or None where a branch drops
the condition. Every target video row, history and current chunk alike, then
reads the frames of its own latent. The frames are selected at the plan's
``video_indices`` and packed by ``pixel_condition_maps`` only when the kwargs
are built, so a branch that edits a plan's rows, by dropping the reference or
the target history, stays aligned, and only the window's frames are converted.
The DiT receives the maps as ``input_condition_maps``, one tensor per sample,
and, as ``input_condition_rows``, the packed rows they describe in
(latent, h, w) order, which is the order of the patchified target video rows
at the tail of every sample. Text, picture or keyframe, reference, audio and
padding rows are never listed. A sample without frames contributes an empty
map and no rows, so its rows keep exactly the plain embedding. Plans without
the key belong to a network that reads no pixel condition, such as a
distillation teacher, and their forward receives no input condition.
"""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3_videoref.data.pixel_condition import pixel_condition_channels, pixel_condition_maps


class StreamingPixelConditionMixin:
    """Route each plan's ``condition_frames`` to its target video rows.

    ``pixel_condition_unshuffle`` is the pixel-unshuffle factor and
    ``pixel_condition_scale`` the map cells per token side, which the DiT's
    input condition encoder expects.
    """

    pixel_condition_unshuffle: int
    pixel_condition_scale: int

    def _streaming_kwargs(self, model: Any, inputs: StreamingInputs, **kwargs: Any) -> dict[str, Any]:
        result = super()._streaming_kwargs(model, inputs, **kwargs)
        if not any("condition_frames" in plan for plan in inputs.plans):
            return result
        device = result["x"].device
        unshuffle, scale = self.pixel_condition_unshuffle, self.pixel_condition_scale
        maps, rows = [], []
        stop = 0
        for plan, pack in zip(inputs.plans, inputs.packs, strict=True):
            stop += int(pack["sample_lens"])
            height, width = plan["target_video_shape"][2:]
            grid = (height // 2 * scale, width // 2 * scale)
            frames = plan["condition_frames"]
            if frames is None:
                maps.append(torch.zeros((0, pixel_condition_channels(unshuffle), *grid), dtype=torch.bfloat16, device=device))
                continue
            value = pixel_condition_maps(frames, plan["video_indices"], video_temporal_mapping=self.video_temporal_mapping,
                                         unshuffle=unshuffle).to(device)
            assert tuple(value.shape[2:]) == grid, (
                f"condition maps {tuple(value.shape[2:])} are not {scale} times the token grid of latents {(height, width)}"
            )
            maps.append(value)
            rows.append(torch.arange(stop - int(pack["video_rows"]), stop, device=device))
        result["input_condition_maps"] = maps
        result["input_condition_rows"] = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long, device=device)
        return result


__all__ = ["StreamingPixelConditionMixin"]
