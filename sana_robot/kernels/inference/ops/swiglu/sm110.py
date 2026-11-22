# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.kernels.inference.ops.swiglu import sm120


def supports(in_features: int, hidden_features: int) -> bool:
    """Return whether the kernel serves a SwiGLU of these sizes on Jetson Thor: the sizes of the sm120 kernel."""
    return sm120.supports(in_features, hidden_features)


def swiglu_gate_up(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The Jetson Thor form of ``sm120.swiglu_gate_up``: the sm120 kernel with its default launch constants."""
    return sm120.swiglu_gate_up(x, weight, bias)
