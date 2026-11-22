# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch


def cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build the learning rate schedule: a linear warmup from 0, a cosine down to ``min_ratio`` of the peak at ``total_steps``, and that floor afterwards.

    Args:
        optimizer: the optimizer whose learning rate is the peak.
        warmup_steps: the number of training steps of the linear warmup.
        total_steps: the training step at which the cosine reaches the floor.
        min_ratio: the floor as a fraction of the peak learning rate.

    Returns:
        The schedule; step it once after every optimizer step.
    """

    def ratio(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if step > total_steps:
            return min_ratio
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) * (1 - min_ratio) + min_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)
