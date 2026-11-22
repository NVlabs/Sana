# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

FSDP_WRAPPER = "_fsdp_wrapped_module"
NORM_GROUPS = (
    ("attn_res_final", ("attn_res.final_proj.",)),
    ("attn_res_attn", ("attn_res.attn_proj.",)),
    ("attn_res_mlp", ("attn_res.mlp_proj.",)),
    ("blocks", ("blocks.",)),
    ("t_embed", ("t_embedder.", "t_block.", "t_block_robot.")),
    ("robot_io", ("action_embed.", "state_embed.", "action_head.")),
    ("x_embedder", ("x_embedder.",)),
    ("final_layer", ("final_layer.",)),
)
OTHER_GROUP = "other"
REST_GROUP = "rest"
CLIP_EPSILON = 1.0e-6


def plain_name(name: str) -> str:
    """Return a parameter name without the path segments that FSDP wrapping adds."""
    return ".".join(part for part in name.split(".") if part != FSDP_WRAPPER)


def norm_group(name: str) -> str:
    """Return the gradient norm group of a parameter name: the first group with a matching prefix."""
    for group, prefixes in NORM_GROUPS:
        if name.startswith(prefixes):
            return group
    return OTHER_GROUP


@dataclass(frozen=True)
class GradientGroups:
    """The trainable parameters by gradient norm group, and the clip group and maximum norm of each."""

    names: list[str]
    parameters: list[list[nn.Parameter]]
    clip_names: list[str]
    clip_index: list[int]
    clip_rows: list[torch.Tensor]
    max_norms: torch.Tensor


def gradient_groups(model: nn.Module, max_norms: dict[str, float], device) -> GradientGroups:
    """Group the trainable parameters for gradient norms and clipping.

    Args:
        model: the policy, before or after FSDP wrapping.
        max_norms: the maximum gradient norm of each clip group; a clip group is a norm group name, or ``rest`` for all norm groups that are not named. A norm group without a clip group is not clipped by group.
        device: the device of the gradients.

    Returns:
        The groups for ``gradient_group_norms`` and ``clip_gradient_groups``.
    """
    members = {name: [] for name in (*(group for group, _ in NORM_GROUPS), OTHER_GROUP)}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            members[norm_group(plain_name(name))].append(parameter)
    names = [name for name, parameters in members.items() if parameters]
    assignment = [name if name in max_norms else REST_GROUP for name in names]
    clip_names = [name for name in (*members, REST_GROUP) if name in assignment and name in max_norms]
    clip_index = [clip_names.index(name) if name in clip_names else -1 for name in assignment]
    clip_rows = [
        torch.tensor([row for row, index in enumerate(clip_index) if index == clip], device=device)
        for clip in range(len(clip_names))
    ]
    limits = torch.tensor([float(max_norms[name]) for name in clip_names], dtype=torch.float32, device=device)
    return GradientGroups(names, [members[name] for name in names], clip_names, clip_index, clip_rows, limits)


@torch.no_grad()
def gradient_group_norms(groups: GradientGroups, process_group=None) -> torch.Tensor:
    """Return the gradient L2 norm of every norm group as float32; ``process_group`` is the FSDP group that holds the shards of one gradient."""
    squares = torch.zeros(len(groups.names), dtype=torch.float32, device=groups.max_norms.device)
    for index, parameters in enumerate(groups.parameters):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if gradients:
            squares[index] = torch.stack(torch._foreach_norm(gradients, 2.0, dtype=torch.float32)).square().sum()
    if process_group is not None:
        dist.all_reduce(squares, group=process_group)
    return squares.sqrt()


@torch.no_grad()
def clip_gradient_groups(groups: GradientGroups, norms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the gradients of each clip group in place by ``min(1, max_norm / (norm + 1e-6))``; return the clip group norms before scaling and the scales."""
    if not groups.clip_names:
        return groups.max_norms, groups.max_norms
    squares = norms.float().square()
    clip_norms = torch.stack([squares[rows].sum() for rows in groups.clip_rows]).sqrt()
    scales = torch.clamp(groups.max_norms / (clip_norms + CLIP_EPSILON), max=1.0)
    for parameters, index in zip(groups.parameters, groups.clip_index):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if gradients and index >= 0:
            torch._foreach_mul_(gradients, scales[index])
    return clip_norms, scales


def total_gradient_norm(norms: torch.Tensor) -> torch.Tensor:
    """Return the gradient L2 norm of the whole model from the norms of its groups."""
    return norms.square().sum().sqrt()
