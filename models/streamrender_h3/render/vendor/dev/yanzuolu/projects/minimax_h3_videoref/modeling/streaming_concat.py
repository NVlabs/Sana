# SPDX-License-Identifier: Apache-2.0
"""Channel-concatenated video conditions for streaming H3 windows.

A plan may carry ``video_condition_latents`` ``[C, n, H, W]``, a clean latent
aligned with its ``video_indices``. Every target video row, history and
current chunk alike, then reads the patchified condition after its own
patchified latent, so the video patch embedding sees ``[x | condition]``.
Text, picture or keyframe, reference, audio and padding rows read zeros, and
the condition columns carry zero noise. Patchify is channel-major, so this
equals concatenating the latents along channels before patchifying.
"""

from __future__ import annotations

from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs


class StreamingVideoConditionMixin:
    """Append each plan's ``video_condition_latents`` to its target video rows as extra input channels."""

    @staticmethod
    def _with_video_condition(plan: dict[str, Any], condition: torch.Tensor) -> dict[str, Any]:
        """Attach the condition at the plan's video indices from a complete ``[C, T, H, W]`` latent."""
        indices = plan["video_indices"].to(condition.device)
        return dict(plan, video_condition_latents=condition.index_select(1, indices))

    def _streaming_kwargs(self, model: Any, inputs: StreamingInputs, **kwargs: Any) -> dict[str, Any]:
        result = super()._streaming_kwargs(model, inputs, **kwargs)
        x = result["x"]
        conditions = []
        offset = 0
        for plan, pack in zip(inputs.plans, inputs.packs, strict=True):
            condition = plan["video_condition_latents"]
            count = plan["video_indices"].numel()
            assert tuple(condition.shape[1:]) == (count, *plan["target_video_shape"][2:])
            rows = self._video_rows(condition.to(x), 0, count)
            assert rows.shape[0] == pack["video_rows"]
            conditions.append(rows.new_zeros((int(pack["sample_lens"]) - rows.shape[0], rows.shape[1])))
            conditions.append(rows)
            offset += int(pack["sample_lens"])
        conditions.append(x.new_zeros((x.shape[1] - offset, conditions[-1].shape[1])))
        condition = torch.cat(conditions).unsqueeze(0)
        result["x"] = torch.cat((x, condition), dim=-1)
        result["eps"] = torch.cat((result["eps"], torch.zeros_like(condition)), dim=-1)
        return result


__all__ = ["StreamingVideoConditionMixin"]
