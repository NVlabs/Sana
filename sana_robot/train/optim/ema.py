# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

EMA_DECAY = 0.999


def start_average(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy the model's parameters as the float32 averaged weights, one tensor per parameter name; under FSDP each rank holds its own shards."""
    return {name: parameter.detach().float().clone() for name, parameter in model.named_parameters()}


@torch.no_grad()
def update_average(average: dict[str, torch.Tensor], model: nn.Module) -> None:
    """Move the averaged weights toward the model's parameters by ``1 - EMA_DECAY``."""
    current = [parameter.detach().float() for parameter in model.parameters()]
    torch._foreach_lerp_(list(average.values()), current, 1 - EMA_DECAY)


@contextmanager
def averaged_weights(average: dict[str, torch.Tensor], model: nn.Module):
    """Hold the averaged weights in the model inside the block and put the training weights back after it."""
    parameters = dict(model.named_parameters())
    training = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(average[name])
    try:
        yield model
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(training[name])
