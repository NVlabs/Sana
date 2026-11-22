# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import torch


def batch_to_device(batch: dict, device) -> dict:
    """Return a batch with its tensors on the device; the frames stay where they are until they are encoded."""
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for name, value in batch.items()
    }


def training_prompts(batch: dict, instruction_dropout: float) -> list:
    """Return the prompts of every sample of a batch, a sample losing its instruction with the dropout probability."""
    return [
        unconditional if random.random() < instruction_dropout else conditional
        for conditional, unconditional in zip(batch["prompts"], batch["unconditional_prompts"])
    ]
