# SPDX-License-Identifier: Apache-2.0
"""Placement-time initialization of the per-token input condition.

Not vendored code -- this file is ours.

A checkpoint written before a DiT gained ``input_condition_channels`` lacks
the input condition encoder, so under ``meta_init`` it reaches placement as
meta tensors. This plugin materializes it on the CPU before sharding exactly
as ``InitMiniMaxH3AdalnCondition`` does for the AdaLN condition: PyTorch's
default initialization under ``seed`` for every layer and a zero output
projection, so the DiT starts out computing its checkpoint's function and
every copy, the EMA's and PEFT's frozen ``original_module`` alike, starts
equal. Modules that already hold values are left alone.

Seeding from such a checkpoint lists the plugin before the weight or DCP
load, which must tolerate exactly these tensors missing, and trains the
encoder beside a LoRA through ``modules_to_save``::

    placement:
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.input_condition_init
          class_name: InitMiniMaxH3InputCondition
          seed: 1019
        - module: dev.yanzuolu.common.plugin.dcp_weights
          class_name: ShardedDCPWeights
          path: /path/to/checkpoint
          key: models.backbone
          allow_missing: ['\\.input_condition_encoder\\.']
    adapter:
      modules_to_save: [input_condition_encoder]

The sharded released-weight load leaves missing keys meta by construction,
so a trial seeding from the released weights needs no ``allow_missing``.
"""

from __future__ import annotations

from .adaln_condition_init import InitMiniMaxH3AdalnCondition
from .transformer.input_condition import MiniMaxH3InputConditionEncoder


class InitMiniMaxH3InputCondition(InitMiniMaxH3AdalnCondition):
    """Materialize meta input condition encoders: default layers, zero output projection."""

    module_types = (MiniMaxH3InputConditionEncoder,)


__all__ = ["InitMiniMaxH3InputCondition"]
