# SPDX-License-Identifier: Apache-2.0
"""Diffusion forcing behind a complete text, Qwen-vision and reference prefix."""

from __future__ import annotations

from typing import Any

from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_prefix_df import (
    build_video_ref_prefix_df_plan,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_prefix_tf import (
    MiniMaxH3VideoRefPrefixTF,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.prefix_forward import (
    PrefixTFInputs,
)


class MiniMaxH3VideoRefPrefixDF(MiniMaxH3VideoRefPrefixTF):
    """Fit noisy target chunks in a single causal sequence.

    Pair this entry with ``VideoRefPrefixRawDiffusionForcingT2AVDataset``.
    Training packs ``P, N0, N1, ...`` with one video/audio timestep pair per
    sample. Each target reads the complete prefix, earlier noisy target chunks
    and itself. Text and Qwen-vision rows follow the video timestep, and
    reference rows follow the native anchor convention.

    Validation reuses prefix-TF sampling and CFG with ``P, C0, ..., Nk``.
    Earlier generated chunks use the existing clean-endpoint convention,
    while only the current chunk follows the sampling schedule. The causal
    backbone and checkpoint interface are identical to prefix TF.
    """

    _prefix_plan_factory = staticmethod(build_video_ref_prefix_df_plan)

    def _inputs_from_payload(self, payload: dict[str, Any]) -> PrefixTFInputs:
        assert all(plan.get("target_mode") == "df" for plan in payload["prefix_plans"]), (
            "prefix DF requires plans from VideoRefPrefixRawDiffusionForcingT2AVDataset"
        )
        return super()._inputs_from_payload(payload)


EntryClass = MiniMaxH3VideoRefPrefixDF

__all__ = ["MiniMaxH3VideoRefPrefixDF", "PrefixTFInputs", "EntryClass"]
