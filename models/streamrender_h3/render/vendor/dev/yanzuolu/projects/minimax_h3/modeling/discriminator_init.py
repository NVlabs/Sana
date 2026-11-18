# SPDX-License-Identifier: Apache-2.0
"""Placement-time initialization for the MiniMax H3 DMD2 discriminator.

New discriminator weights are initialized before placement while tensors are
whole, so Xavier initialization uses the full fan-in and fan-out. A loaded V2
head keeps its parameters when only the V3 global branch needs initialization.

Example::

    placement:
      fsdp: { wrap_modules: [MiniMaxH3DiscriminatorBlock] }
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.discriminator_init
          class_name: InitMiniMaxH3Discriminator
          seed: 1019
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dev.yanzuolu.common.model.placement import PlacementPlugin

from .discriminator import MiniMaxH3DMD2Discriminator, MiniMaxH3DMD2DiscriminatorV3


class InitMiniMaxH3Discriminator(PlacementPlugin):
    """Materialize and deterministically initialize meta discriminators."""

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if isinstance(model, nn.Module) and hasattr(model, "discriminator"):
            model = model.discriminator
        if not isinstance(model, nn.Module):
            return state

        generator = torch.Generator(device="cpu").manual_seed(int(self.config["seed"]))

        for module in model.modules():
            if not isinstance(module, MiniMaxH3DMD2Discriminator):
                continue
            parameters = list(module.parameters())
            if (
                isinstance(module, MiniMaxH3DMD2DiscriminatorV3)
                and any(parameter.is_meta for parameter in parameters)
                and not all(parameter.is_meta for parameter in parameters)
            ):
                module.materialize_missing_global_parameters(generator=generator)
            elif any(parameter.is_meta for parameter in parameters):
                module.to_empty(device=torch.device("cpu"))
                module.reset_parameters(generator=generator)

        return state


__all__ = ["InitMiniMaxH3Discriminator"]
