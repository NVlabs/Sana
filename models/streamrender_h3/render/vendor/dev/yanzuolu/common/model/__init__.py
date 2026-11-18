"""Fixed model build pipeline."""

from __future__ import annotations

import time
from typing import Any

import torch

from ..logging import get_logger
from .adapters import apply_adapters
from .instantiate import instantiate_model
from .placement import place_model
from .runtime import apply_runtime
from .weights import load_initial_weight

logger = get_logger()


def build_model(name: str, model_config: Any) -> dict[str, Any]:
    state = {"name": name, "config": model_config}

    state = instantiate_model(state)
    state = load_initial_weight(state)
    state = apply_adapters(state)
    state = apply_runtime(state)
    state = place_model(state)

    return state


def build_models(configs: Any) -> dict[str, "torch.nn.Module"]:
    """Build every configured model.

    Returns a flat ``{name: module}`` dict (EMA models appear as ``{name}_ema``)
    shaped to drop directly into the checkpointer state tree::

        state = {"models": build_models(config.models), "optimizers": ...}
    """
    models: dict[str, Any] = {}
    for name, model_config in configs.items():
        logger.info("Building model '%s'...", name)
        start = time.perf_counter()
        state = build_model(name, model_config)
        models[name] = state["model"]
        has_ema = "ema_model" in state
        if has_ema:
            models[f"{name}_ema"] = state["ema_model"]
        logger.info(
            "Building model '%s' done%s in %.1fs", name, " (+ema)" if has_ema else "", time.perf_counter() - start
        )
    return models
