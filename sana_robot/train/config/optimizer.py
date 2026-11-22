# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OptimizerConfig:
    """AdamW, its learning rate schedule and the gradient clipping."""

    learning_rate: float = 1.0e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1.0e-10
    weight_decay: float = 1.0e-4
    decay_biases_and_norms: bool = False
    warmup_steps: int = 1000
    min_learning_rate_ratio: float = 0.01
    gradient_clip: float = 1.0
    gradient_clip_groups: Dict[str, float] = field(default_factory=dict)
