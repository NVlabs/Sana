# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing marker for causal MiniMax H3 video-reference training."""

from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_df import (
    CausalMiniMaxH3VideoRefDF,
)


class CausalMiniMaxH3VideoRefTF(CausalMiniMaxH3VideoRefDF):
    """Select the teacher-forcing video-reference dataset layout."""


EntryClass = CausalMiniMaxH3VideoRefTF

__all__ = ["CausalMiniMaxH3VideoRefTF", "EntryClass"]
