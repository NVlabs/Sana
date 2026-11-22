# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

NORMALIZATION_LAYERS = (nn.LayerNorm, nn.GroupNorm, _BatchNorm, _InstanceNorm)


def parameter_groups(model: nn.Module, weight_decay: float, decay_biases_and_norms: bool = False) -> list[dict]:
    """Split the trainable parameters into a group with weight decay and a group without it that holds the biases and the layers of ``NORMALIZATION_LAYERS``; the RMSNorm gains are decayed."""
    exempt_ids = set()
    for module in model.modules():
        if isinstance(module, NORMALIZATION_LAYERS) and not decay_biases_and_norms:
            exempt_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    decayed, exempt = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_bias = name.rsplit(".", 1)[-1] == "bias" and not decay_biases_and_norms
        (exempt if is_bias or id(parameter) in exempt_ids else decayed).append(parameter)
    return [{"params": decayed, "weight_decay": weight_decay}, {"params": exempt, "weight_decay": 0.0}]


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    betas,
    eps: float,
    weight_decay: float,
    decay_biases_and_norms: bool = False,
) -> torch.optim.AdamW:
    """Build AdamW over the trainable parameters of the model.

    Args:
        model: the policy, before or after FSDP wrapping.
        learning_rate: the peak learning rate that the schedule scales.
        betas: the two AdamW moment decays.
        eps: the AdamW denominator term.
        weight_decay: the decay of every parameter except the biases and the layers of ``NORMALIZATION_LAYERS``;
            the RMSNorm gains are decayed.
        decay_biases_and_norms: decay the biases and the normalization layers too.

    Returns:
        The optimizer with two parameter groups.
    """
    groups = parameter_groups(model, weight_decay, decay_biases_and_norms)
    return torch.optim.AdamW(groups, lr=learning_rate, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
