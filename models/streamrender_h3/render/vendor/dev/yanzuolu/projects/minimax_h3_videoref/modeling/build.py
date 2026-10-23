# SPDX-License-Identifier: Apache-2.0
"""Flat constructor entries for videoref streaming networks."""

from __future__ import annotations

from dev.yanzuolu.projects.minimax_h3.modeling.build import MiniMaxH3GANX0DiTSPV2
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.config import MiniMaxH3DiTArchConfig
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.discriminator import MiniMaxH3SampleChunkDiscriminatorV2


class MiniMaxH3StreamingGANX0DiTSPV2(MiniMaxH3GANX0DiTSPV2):
    """Bidirectional SP x0 fake score whose V2 head scores one explicit row set per window sample.

    Parameters and checkpoint layout match ``MiniMaxH3GANX0DiTSPV2``. Streaming
    DMD2 passes each window target as explicit chunk ranges, so
    ``discriminator_video_chunk_size`` never shapes its chunks.
    """

    def _build_discriminator(
        self, *, arch: MiniMaxH3DiTArchConfig, num_queries: int, num_taps: int,
    ) -> MiniMaxH3SampleChunkDiscriminatorV2:
        return MiniMaxH3SampleChunkDiscriminatorV2(
            arch=arch, num_queries=num_queries, num_taps=num_taps,
            video_chunk_size=self._discriminator_video_chunk_size,
        )


__all__ = ["MiniMaxH3StreamingGANX0DiTSPV2"]
