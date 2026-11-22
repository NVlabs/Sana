# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sana_robot.kernels.inference.ops.cross_qk_norm import sm120


def supports(channels: int) -> bool:
    """Return whether rows of ``channels`` channels are served on Jetson Thor: the widths of the sm120 kernel."""
    return sm120.supports(channels)


def cross_qk_norm(q_rows, k_rows, q_weight, k_weight, eps: float) -> None:
    """The Jetson Thor form of ``portable.cross_qk_norm``: the sm120 kernel, with its first-call check of the inverse RMS rows against torch's."""
    sm120.cross_qk_norm(q_rows, k_rows, q_weight, k_weight, eps)
