# SPDX-License-Identifier: Apache-2.0
"""Raw reference-prefix diffusion forcing with exact Qwen packing budgets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_prefix_tf import (
    VideoRefPrefixRawTeacherForcingT2AVDataset,
    build_video_ref_prefix_tf_plan,
)


def build_video_ref_prefix_df_plan(
    *,
    target_video_shape: Sequence[int],
    target_audio_shape: Sequence[int],
    chunk_size: int = 5,
    independent_first_chunk: int | None = None,
    video_temporal_mapping: VideoTemporalMapping = H3_VIDEO_TEMPORAL_MAPPING,
    video_chunk_ranges: Sequence[Sequence[int]] | None = None,
    audio_chunk_ranges: Sequence[Sequence[int]] | None = None,
    target_chunk_index: int | None = None,
) -> dict[str, Any]:
    """Define one causal target sequence behind a bidirectional reference prefix.

    Training uses ``P, N0, ..., Nlast``. Each target chunk reads the prefix,
    itself and every earlier noisy target chunk. Validation of chunk ``k``
    uses ``P, C0, ..., C(k-1), Nk`` with the same visibility. Block roles
    remain 0 for prefix, 1 for clean and 2 for noisy target outputs.
    """
    plan = build_video_ref_prefix_tf_plan(
        target_video_shape=target_video_shape,
        target_audio_shape=target_audio_shape,
        chunk_size=chunk_size,
        independent_first_chunk=independent_first_chunk,
        video_temporal_mapping=video_temporal_mapping,
        video_chunk_ranges=video_chunk_ranges,
        audio_chunk_ranges=audio_chunk_ranges,
        target_chunk_index=target_chunk_index,
    )
    if target_chunk_index is None:
        num_chunks = len(plan["video_chunk_ranges"])
        plan["block_roles"] = [0] + [2] * num_chunks
        plan["block_chunk_indices"] = [None, *range(num_chunks)]
        _, latent_t, latent_h, latent_w = plan["target_video_shape"]
        plan["target_packing_rows"] = (
            latent_t * (latent_h // 2) * (latent_w // 2)
            + 2 * plan["target_audio_shape"][2]
        )
    plan["target_mode"] = "df"
    plan["attn_modes"] = ["full"] * len(plan["block_roles"])
    return plan


class VideoRefPrefixRawDiffusionForcingT2AVDataset(VideoRefPrefixRawTeacherForcingT2AVDataset):
    """Own prefix-DF plans while sharing media decoding and Qwen presentation budgets."""

    _STATE_SCHEMA = "minimax_h3_videoref_prefix_df_raw_worker"
    _prefix_plan_factory = staticmethod(build_video_ref_prefix_df_plan)
    forcing = "diffusion"


EntryClass = VideoRefPrefixRawDiffusionForcingT2AVDataset

__all__ = [
    "VideoRefPrefixRawDiffusionForcingT2AVDataset",
    "build_video_ref_prefix_df_plan",
    "EntryClass",
]
