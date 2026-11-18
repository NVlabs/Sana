# SPDX-License-Identifier: Apache-2.0
"""Diffusion forcing for a chunk-causal MiniMax-H3 T2AV corpus fit."""

from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_tf import CausalMiniMaxH3TF


class CausalMiniMaxH3DF(CausalMiniMaxH3TF):
    """Reuse the TF fit with the single-slot DF layout supplied by the dataset."""


__all__ = ["CausalMiniMaxH3DF"]
