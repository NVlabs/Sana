# SPDX-License-Identifier: Apache-2.0
"""A LoRA adapter whose initial weight file may cover only part of the adapter.

peft requires every ``modules_to_save`` copy in the state dict it loads, so a
LoRA-only export cannot initialise a model that adds one, for example a
widened video patch embedding trained as a full copy. This adapter attaches
the LoRA like ``PeftLoraAdapter`` and then loads ``weight`` with every adapter
tensor absent from the file kept at its current value::

    adapter:
      module: dev.yanzuolu.projects.minimax_h3_videoref.modeling.adapters
      class_name: PartialWeightPeftLoraAdapter
      weight: /path/to/backbone_ema_lora.safetensors
      modules_to_save: [video_patch_proj]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dev.yanzuolu.common.config import CfgNode
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.model.adapters import PeftLoraAdapter
from dev.yanzuolu.common.model.weights import load_state_dict_file

logger = get_logger()


class PartialWeightPeftLoraAdapter(PeftLoraAdapter):
    """Attach the LoRA, then load a weight file that may omit adapter tensors."""

    def __init__(self, config: Any):
        self.weight = config.get("weight", None)
        super().__init__(CfgNode({key: value for key, value in config.items() if key != "weight"}))

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        state = super().__call__(state)
        if self.weight is None:
            return state
        loaded = load_state_dict_file(Path(self.weight))
        current = get_peft_model_state_dict(state["model"])
        kept = current.keys() - loaded.keys()
        result = set_peft_model_state_dict(state["model"], {**{key: current[key] for key in kept}, **loaded})
        logger.info(
            "[%s] adapter weights loaded from %s: %d unexpected keys, %d kept at their current values",
            state["name"], self.weight, len(result.unexpected_keys), len(kept),
        )
        return state


__all__ = ["PartialWeightPeftLoraAdapter"]
