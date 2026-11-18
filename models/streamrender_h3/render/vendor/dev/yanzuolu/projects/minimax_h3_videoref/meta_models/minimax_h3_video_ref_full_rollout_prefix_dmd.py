# SPDX-License-Identifier: Apache-2.0
"""Complete fresh student rollouts with a fixed conditioning prefix.

The full-rollout host generates every window with fixed student weights
before either optimizer runs. It retains the detached canonical clip and
each window's noisy-row trajectory, then samples FAKE and GEN windows via
``inner_batch``. Every window payload carries the same student-prefix seed
for its clip, so reconstruction preserves its original prefix inputs.

The student and its EMA use dense bootstrap attention and an isolated
prefix on continuations. Selected student windows keep their prefix input
type, including negative and guided branches. Fake and real score windows
keep their original dense conditions. GEN recomputes the complete prefix
with gradients on each forward. Training retains no prefix KV or graph
between forwards; streaming inference caches KV separately for each paired
diffusion timestep and guidance branch.

Configuration additions::

    data:
      module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_long_latent
      class_name: VideoRefStreamingLongLatentT2AVDataset
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_full_rollout_prefix_dmd
      class_name: MiniMaxH3VideoRefFullRolloutPrefixDMD
      inner_batch:
        fake: {size: 5, include_first: true}
        gen: {size: 5, include_first: true}
      score_path: renoise
      fixed_window_rope: true
      separate_reference_rope: true
      qwen_reference_video: false
    models:
      backbone:
        module: dev.yanzuolu.projects.minimax_h3.modeling.build
        class_name: MiniMaxH3CausalX0DiTSP
        placement:
          fsdp:
            wrap_modules: [CausalMiniMaxH3DiTBlock]
    engine:
      offline: 1

The EMA inherits the causal student wrapper. Both scores retain their
bidirectional wrappers. Static-prefix and full-rollout constraints apply,
including fixed captions, a complete bootstrap sink and no history_refresh.
"""

from __future__ import annotations

from collections.abc import Sequence

from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_full_rollout_dmd import (
    MiniMaxH3VideoRefFullRolloutDMD,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_prefix_dmd import (
    FixedStudentPrefixDMDMixin,
    _PrefixStreamingInputs,
)


class MiniMaxH3VideoRefFullRolloutPrefixDMD(FixedStudentPrefixDMDMixin, MiniMaxH3VideoRefFullRolloutDMD):
    """Train selected windows of fresh complete rollouts with the student's fixed prefix."""

    def _pack_selected_windows(self, windows: Sequence[StreamingInputs]) -> StreamingInputs:
        """Preserve student prefix routing while packing each role's selected windows."""
        prefix = isinstance(windows[0], _PrefixStreamingInputs)
        assert all(isinstance(window, _PrefixStreamingInputs) == prefix for window in windows), (
            "selected windows must belong to the same student or score role"
        )
        packed = super()._pack_selected_windows(windows)
        return _PrefixStreamingInputs(**vars(packed)) if prefix else packed


EntryClass = MiniMaxH3VideoRefFullRolloutPrefixDMD

__all__ = ["MiniMaxH3VideoRefFullRolloutPrefixDMD", "EntryClass"]
