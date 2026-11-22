# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

AVERAGE_KEY = "state_dict_ema"
WEIGHT_KEYS = (AVERAGE_KEY, "state_dict")


def load_weights(model: torch.nn.Module, path) -> None:
    """Load policy weights from a file that holds a state dict, bare or under ``state_dict_ema`` or ``state_dict``."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    for key in WEIGHT_KEYS:
        state = state.get(key, state)
    model.load_state_dict(state, strict=True)
