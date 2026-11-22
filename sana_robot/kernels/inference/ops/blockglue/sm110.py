# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial

from sana_robot.kernels.inference.ops.blockglue import sm120
from sana_robot.kernels.inference.ops.dispatch import TunedTable

THOR_BOUNDARY_CONFIGS: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int]] = {
    (160, 1, 1, 1): (1, 2, 0, 0, 0, 0),
    (160, 1, 2, 1): (1, 2, 0, 0, 0, 0),
    (160, 2, 1, 1): (2, 2, 0, 0, 0, 0),
    (160, 2, 2, 1): (2, 2, 0, 0, 0, 0),
    (160, 3, 1, 1): (2, 2, 0, 0, 0, 0),
    (160, 3, 2, 1): (2, 2, 0, 0, 0, 0),
    (160, 4, 1, 1): (2, 2, 0, 0, 0, 0),
    (160, 4, 2, 1): (2, 2, 0, 0, 0, 0),
    (193, 1, 1, 1): (2, 2, 0, 0, 0, 0),
    (193, 1, 2, 1): (2, 2, 0, 0, 0, 0),
    (193, 2, 1, 1): (2, 2, 0, 0, 0, 0),
    (193, 2, 2, 1): (2, 2, 0, 0, 1, 0),
    (193, 3, 1, 1): (2, 2, 0, 0, 1, 0),
    (193, 3, 2, 1): (2, 2, 0, 0, 0, 0),
    (193, 4, 1, 1): (2, 2, 0, 0, 0, 0),
    (193, 4, 2, 1): (2, 2, 0, 0, 0, 0),
    (240, 1, 1, 1): (2, 2, 0, 0, 0, 0),
    (240, 1, 2, 1): (2, 2, 0, 0, 0, 0),
    (240, 2, 1, 1): (2, 2, 0, 0, 0, 0),
    (240, 2, 2, 1): (2, 2, 0, 0, 0, 0),
    (240, 3, 1, 1): (2, 2, 0, 0, 0, 0),
    (240, 3, 2, 1): (2, 2, 0, 0, 0, 0),
    (240, 4, 1, 1): (2, 2, 0, 0, 0, 0),
    (240, 4, 2, 1): (2, 2, 0, 0, 0, 0),
    (273, 1, 1, 1): (2, 2, 0, 0, 1, 0),
    (273, 1, 2, 1): (2, 2, 0, 0, 0, 0),
    (273, 2, 1, 1): (2, 2, 0, 0, 1, 0),
    (273, 2, 2, 1): (2, 2, 0, 0, 1, 0),
    (273, 3, 1, 1): (2, 2, 0, 0, 1, 0),
    (273, 3, 2, 1): (2, 2, 0, 0, 1, 0),
    (273, 4, 1, 0): (2, 2, 0, 0, 1, 0),
    (273, 4, 1, 1): (2, 2, 1, 0, 0, 0),
    (353, 1, 1, 1): (1, 2, 0, 0, 0, 0),
    (353, 1, 2, 1): (1, 2, 0, 0, 0, 0),
    (353, 2, 1, 1): (2, 2, 0, 0, 0, 0),
    (353, 2, 2, 1): (2, 2, 0, 0, 1, 0),
    (353, 3, 1, 1): (2, 2, 0, 0, 1, 0),
    (353, 3, 2, 1): (2, 2, 0, 0, 0, 0),
    (353, 4, 1, 1): (2, 2, 0, 0, 0, 0),
    (353, 4, 2, 1): (2, 2, 0, 0, 0, 0),
    (513, 1, 1, 1): (1, 2, 0, 0, 0, 0),
    (513, 1, 2, 1): (1, 2, 0, 0, 0, 0),
    (513, 2, 1, 1): (2, 2, 0, 0, 1, 0),
    (513, 2, 2, 1): (2, 2, 0, 0, 0, 0),
    (513, 3, 1, 1): (2, 2, 1, 0, 1, 0),
    (513, 3, 2, 1): (2, 2, 0, 0, 0, 0),
    (513, 4, 1, 1): (2, 2, 1, 0, 0, 1),
    (513, 4, 2, 1): (2, 2, 0, 0, 0, 0),
}
THOR_LAYER_NORM_CONFIGS: dict[int, tuple[int, int, int]] = {
    160: (1, 2, 0),
    193: (2, 2, 1),
    240: (2, 2, 1),
    273: (1, 2, 0),
    320: (2, 2, 1),
    353: (2, 2, 1),
    480: (2, 2, 1),
    513: (2, 2, 1),
}
THOR_GATE_RESIDUAL_CONFIGS: dict[int, tuple[int, int]] = {
    193: (0, 2),
    240: (1, 8),
    273: (1, 8),
    353: (1, 8),
    513: (1, 8),
}
BOUNDARY_CONFIGS = TunedTable(
    {**sm120.BOUNDARY_CONFIGS, **THOR_BOUNDARY_CONFIGS}, tuned=(key[0] for key in THOR_BOUNDARY_CONFIGS)
)
LAYER_NORM_CONFIGS = TunedTable({**sm120.LAYER_NORM_CONFIGS, **THOR_LAYER_NORM_CONFIGS}, tuned=THOR_LAYER_NORM_CONFIGS)
GATE_RESIDUAL_CONFIGS = TunedTable(
    {**sm120.GATE_RESIDUAL_CONFIGS, **THOR_GATE_RESIDUAL_CONFIGS}, tuned=THOR_GATE_RESIDUAL_CONFIGS
)

serves_width = sm120.serves_width
gate_residual = partial(sm120.gate_residual, configs=GATE_RESIDUAL_CONFIGS)
sublayer_boundary = partial(sm120.sublayer_boundary, configs=BOUNDARY_CONFIGS)
layer_norm_modulate = partial(sm120.layer_norm_modulate, configs=LAYER_NORM_CONFIGS)
