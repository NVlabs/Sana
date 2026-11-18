# SPDX-License-Identifier: Apache-2.0
"""DMD2 with reference-prefix diffusion-forcing critics and a causal student."""

from __future__ import annotations

from typing import Any

from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_prefix_df import (
    build_video_ref_prefix_df_plan,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_dmd2 import (
    CausalMiniMaxH3VideoRefDMD2,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_prefix_dmd import (
    _PrefixScoreInputs,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_prefix_dmd2 import (
    CausalMiniMaxH3VideoRefPrefixDMD2,
)


class CausalMiniMaxH3VideoRefPrefixDFDMD2(CausalMiniMaxH3VideoRefPrefixDMD2):
    """Score one causal target sequence behind the reference prefix.

    Fake and teacher use one video/audio timestep pair per sample. Generator
    GAN gradients propagate through the target sequence, including prior chunks.
    Critic conditions contain no separate clean target history.
    """

    _prefix_plan_factory = staticmethod(build_video_ref_prefix_df_plan)
    _critic_inputs = CausalMiniMaxH3VideoRefDMD2._critic_inputs
    _teacher_negative_inputs = CausalMiniMaxH3VideoRefDMD2._teacher_negative_inputs
    _gan_inputs = CausalMiniMaxH3VideoRefDMD2._gan_inputs

    @staticmethod
    def _history_kwargs(inputs: _PrefixScoreInputs) -> dict[str, Any]:
        return {}


EntryClass = CausalMiniMaxH3VideoRefPrefixDFDMD2

__all__ = ["CausalMiniMaxH3VideoRefPrefixDFDMD2", "EntryClass"]
