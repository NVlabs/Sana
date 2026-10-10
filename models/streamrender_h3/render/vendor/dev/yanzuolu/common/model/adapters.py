"""Adapter application stage.

Attaches at most one trainable LoRA to the model (config key ``adapter``, a
single dict; absent means no adapter). Composing several LoRAs and merging them
into the base is an offline export concern, handled by ``common.plugin.export``
(``content: merged``), not something done during the build.

``custom_module_mapping`` teaches peft a base-layer class it does not know::

    adapter:
      r: 256
      lora_alpha: 256
      target_modules: [qkv_proj, out_proj]
      custom_module_mapping:
        - source: {module: projects.x.modeling.model, class_name: _Linear}
          target: {module: projects.x.modeling.lora, class_name: XLoraLinear}
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ..logging import get_logger
from ..seed import local_seed

logger = get_logger()


def apply_adapters(state: dict[str, Any]) -> dict[str, Any]:
    adapter_config = state["config"].get("adapter", None)
    if adapter_config is None:
        return state
    module = importlib.import_module(adapter_config.get("module", __name__))
    adapter_cls = getattr(module, adapter_config.get("class_name", "PeftLoraAdapter"))

    state = adapter_cls(adapter_config)(state)
    if "ema_model" in state:
        # The adapter rebinds "model" (e.g. to a PeftModel), so take the shadow's.
        ema_state = {"name": state["name"] + "_ema", "config": state["config"], "model": state["ema_model"]}
        state["ema_model"] = adapter_cls(adapter_config)(ema_state)["model"]

    return state


def _load_class(spec: Any) -> type:
    """``{module, class_name}`` -> the class. No registry: the config names it."""
    return getattr(importlib.import_module(spec.module), spec.class_name)


class PeftLoraAdapter:
    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            modules_to_save=self.config.get("modules_to_save", None),
            rank_pattern=self.config.get("rank_pattern", {}),
            alpha_pattern=self.config.get("alpha_pattern", {}),
        )
        # peft dispatches on the base layer's TYPE, and only knows torch's own
        # (Linear, Embedding, ConvNd, ...). A project whose projections are a
        # custom class -- a tensor-parallel shim, a fused MoE linear -- gets
        # "Target module ... is not supported" instead of a LoRA. Naming the
        # source class and the peft layer that handles it keeps everything else
        # peft's: parameter creation, init, scaling, merging, state-dict keys.
        mapping = self.config.get("custom_module_mapping", None)
        if mapping:
            resolved = {_load_class(pair.source): _load_class(pair.target) for pair in mapping}
            peft_config._register_custom_module(resolved)
            logger.info(
                "[%s] LoRA custom modules: %s",
                state["name"],
                ", ".join(f"{s.__name__} -> {t.__name__}" for s, t in resolved.items()),
            )
        with local_seed(self.config.get("seed", 1019)):
            state["model"] = get_peft_model(state["model"], peft_config)
        # Count modules that actually received a LoRA injection (duck-typed on the
        # lora_A attribute peft adds to every wrapped layer), not the configured
        # patterns -- one pattern can hit many modules.
        hit_modules = sum(1 for module in state["model"].modules() if hasattr(module, "lora_A"))
        logger.info(
            "[%s] LoRA attached: r=%s alpha=%s, %d target modules, %s trainable params",
            state["name"], self.config.r, self.config.lora_alpha, hit_modules,
            f"{sum(p.numel() for p in state['model'].parameters() if p.requires_grad):,}",
        )

        weight = self.config.get("weight", None)
        if weight is not None:
            from peft import set_peft_model_state_dict

            from .weights import load_state_dict_file

            result = set_peft_model_state_dict(state["model"], load_state_dict_file(Path(weight)))
            logger.info(
                "[%s] adapter weights loaded from %s: %d unexpected keys",
                state["name"],
                weight,
                len(result.unexpected_keys),
            )

        return state
