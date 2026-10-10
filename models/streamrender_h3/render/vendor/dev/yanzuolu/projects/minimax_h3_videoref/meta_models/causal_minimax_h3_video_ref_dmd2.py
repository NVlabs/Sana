# SPDX-License-Identifier: Apache-2.0
"""Video-reference DMD2 with native bidirectional critics."""

from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_dmd2 import (
    CausalMiniMaxH3DMD2,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.causal_minimax_h3_video_ref_dmd import (
    CausalVideoRefDMDMixin,
    _ScoreInputs as _ScoreInputs,
    _ScoreReference as _ScoreReference,
    _VideoRefDMDInputs,
)


_VideoRefDMD2Inputs = _VideoRefDMDInputs


class CausalMiniMaxH3VideoRefDMD2(CausalVideoRefDMDMixin, CausalMiniMaxH3DMD2):
    """Train a causal VideoRef student with native Ref2VA DMD2 critics."""


EntryClass = CausalMiniMaxH3VideoRefDMD2

__all__ = ["CausalMiniMaxH3VideoRefDMD2", "EntryClass"]
