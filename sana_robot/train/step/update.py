# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sana_robot.train.optim.clipping import (
    GradientGroups,
    clip_gradient_groups,
    gradient_group_norms,
    total_gradient_norm,
)
from sana_robot.train.optim.ema import update_average


def update_parameters(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule,
    groups: GradientGroups,
    average: dict[str, torch.Tensor],
    clip_norm,
    max_norm: float,
    process_group=None,
) -> dict:
    """Finish a training step after the backward pass: clip the gradients by group and then as a whole, step the optimizer, move the averaged weights and step the learning rate.

    Args:
        model: The policy that holds the gradients.
        optimizer: Its optimizer.
        schedule: The learning rate schedule.
        groups: The gradient groups from ``gradient_groups``.
        average: The averaged weights from ``start_average``.
        clip_norm: The function that clips the whole gradient, called as ``clip_norm(parameters, max_norm)``:
            ``Accelerator.clip_grad_norm_``, or ``torch.nn.utils.clip_grad_norm_`` without Accelerate.
        max_norm: The maximum norm of the whole gradient.
        process_group: The FSDP group that holds the shards of one gradient, or None on a single process.

    Returns:
        A dict with ``group_norms``, the norm of every norm group before clipping; ``gradient_norm``, their total;
        ``clip_norms`` and ``clip_scales`` of the clip groups; and ``clipped_norm``, the total after the group clip.
    """
    group_norms = gradient_group_norms(groups, process_group)
    clip_norms, clip_scales = clip_gradient_groups(groups, group_norms)
    clipped_norm = clip_norm(model.parameters(), max_norm)
    optimizer.step()
    update_average(average, model)
    schedule.step()
    return {
        "group_norms": group_norms,
        "gradient_norm": total_gradient_norm(group_norms),
        "clip_norms": clip_norms,
        "clip_scales": clip_scales,
        "clipped_norm": clipped_norm,
    }
