# SPDX-License-Identifier: Apache-2.0
"""Placement-time initialization of the per-token AdaLN condition.

Not vendored code -- this file is ours.

A checkpoint written before a DiT gained ``adaln_condition_dim`` holds neither
the condition encoder nor the block heads, so under ``meta_init`` they reach
placement as meta tensors. This plugin materializes each such module on the
CPU before sharding. Heads become zero, so the DiT starts out computing its
checkpoint's function, and encoders take PyTorch's default initialization
under ``seed``. Every encoder copy, the EMA's and PEFT's frozen
``original_module`` beside its trainable copy alike, draws from the same seed,
so all copies start equal. Modules that already hold values are left alone.

Seeding from such a checkpoint lists the plugin before the DCP load, which
must allow exactly these tensors to be missing, and trains both modules
beside a LoRA through ``modules_to_save``::

    placement:
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.adaln_condition_init
          class_name: InitMiniMaxH3AdalnCondition
          seed: 1019
        - module: dev.yanzuolu.common.plugin.dcp_weights
          class_name: ShardedDCPWeights
          path: /path/to/checkpoint
          key: models.backbone
          allow_missing: ['\\.adaln_condition_(encoder|head)\\.']
    adapter:
      modules_to_save: [adaln_condition_encoder, adaln_condition_head]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dev.yanzuolu.common.model.placement import PlacementPlugin
from dev.yanzuolu.common.seed import local_seed

from .transformer.adaln_condition import (
    MiniMaxH3AdalnConditionEncoder,
    MiniMaxH3AdalnConditionHead,
)


class InitMiniMaxH3AdalnCondition(PlacementPlugin):
    """Materialize meta condition encoders and heads: default encoders, zero heads."""

    module_types: tuple[type[nn.Module], ...] = (MiniMaxH3AdalnConditionEncoder, MiniMaxH3AdalnConditionHead)

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, nn.Module):
            return state
        seed = int(self.config["seed"])
        for module in model.modules():
            if not isinstance(module, self.module_types):
                continue
            meta = [parameter.is_meta for parameter in module.parameters()]
            if not any(meta):
                continue
            assert all(meta), f"{type(module).__name__} is partially meta; refusing to guess its values"
            module.to_empty(device=torch.device("cpu"))
            with local_seed(seed):
                module.reset_parameters()
        return state


__all__ = ["InitMiniMaxH3AdalnCondition"]
