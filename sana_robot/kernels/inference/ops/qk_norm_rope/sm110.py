# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sana_robot.kernels.inference.ops.qk_norm_rope import sm120
from sana_robot.kernels.inference.ops.qk_norm_rope.portable import QKNormRopeResult


def serves_shape(channels: int, head_dim: int, rows: int) -> bool:
    """Return whether the kernel serves ``rows`` token rows of ``channels`` channels on Jetson Thor: the shapes of the sm120 kernel."""
    return sm120.serves_shape(channels, head_dim, rows)


def fused_qk_norm_rope(q, k, q_weight, k_weight, freqs, layout, head_dim, eps) -> QKNormRopeResult:
    """The Jetson Thor form of ``portable.fused_qk_norm_rope``: the sm120 kernel, with its first-call check of the inverse RMS rows against torch's."""
    return sm120.fused_qk_norm_rope(q, k, q_weight, k_weight, freqs, layout, head_dim, eps)
