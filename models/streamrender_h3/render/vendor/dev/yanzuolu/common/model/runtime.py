"""Runtime configuration stage.

The storage dtype of the trained weights is not set here: it belongs to
``placement.weight_dtype``, which casts on the GPU during placement/shard so no
CPU-side private copy is ever made.

``apply_runtime`` is a generic orchestrator, mirroring ``place_model``: it routes
to a runtime class (``module`` / ``class_name``, default ``DefaultRuntime``) and
runs each plugin's before/after hooks around that core, for the main model and
the EMA shadow. Nothing framework-specific lives in the orchestrator -- the
nn.Module assumptions (train mode, requires_grad, dtype annotation, per-submodule
tagging) all live in ``DefaultRuntime``, so an alternative runtime class is free
to target something that is not an nn.Module.

``DefaultRuntime`` finalizes how a built nn.Module runs:
    - ``training`` / ``requires_grad`` set train mode and gradient tracking.
    - ``param_dtype`` is the compute dtype, annotated on the module for
      forward-time consumers (same vocabulary as FSDP2's
      ``MixedPrecisionPolicy.param_dtype``).
    - every submodule is tagged with ``layer_name`` (its dotted path under the
      model name, ``{name}_ema`` for the EMA) for error localization.
A non-Module build product is a no-op.

The EMA shadow, when present, is always eval + frozen (updated by the engine's
averaging step, never by gradients): the orchestrator runs the same core class
against it with ``training`` / ``requires_grad`` forced off, ``param_dtype`` and
everything else carried over.

Plugins subclass ``RuntimePlugin`` and override ``before_runtime`` and/or
``after_runtime`` (both default to a no-op, both return state). Each ``plugins``
entry routes on ``module`` / ``class_name`` (the remaining keys are the plugin's
own config). A plugin runs on the EMA too by default with the identical config;
an ``ema`` node adjusts that -- ``enabled: false`` skips the EMA pass, any other
keys are deep-merged over the plugin config as EMA-specific overrides. Plugin
classes always face a single model; all EMA orchestration lives here.

Example::

    runtime:
      training: true
      param_dtype: bfloat16       # compute in bf16
      plugins:
        - module: common.plugin.gradient_checkpointing
          class_name: TransformersGradientCheckpointing
          use_reentrant: false
        - module: common.plugin.autocast
          class_name: AutocastNoGradWrapper
          autocast: { enabled: true, dtype: bfloat16 }
          ema:
            grad_enabled: false   # override for the EMA pass only
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

import torch

from ..config import CfgNode
from ..logging import get_logger

logger = get_logger()


class RuntimePlugin:
    """Base class for runtime plugins.

    Override ``before_runtime`` and/or ``after_runtime`` -- they run,
    respectively, before and after the core runtime class for the single model
    the plugin faces. Both default to a no-op and must return the (possibly
    mutated) state dict. ``self.config`` is the plugin's config node (its own
    fields plus ``module`` / ``class_name``; on the EMA pass the ema overrides
    are already merged in).
    """

    def __init__(self, config: Any):
        self.config = config

    def before_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


def apply_runtime(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"].get("runtime", {})
    module = importlib.import_module(config.get("module", __name__))
    runtime_cls = getattr(module, config.get("class_name", "DefaultRuntime"))
    plugin_configs = config.get("plugins", [])

    # Main model: each plugin instantiated with its own config (the ema node
    # stripped off); the core runs on the config as written. Deep-copied so a
    # plugin can never mutate the config tree in place.
    main_plugins = [
        (pc.class_name, getattr(importlib.import_module(pc.module), pc.class_name)(
            CfgNode(copy.deepcopy({k: v for k, v in pc.items() if k != "ema"}))))
        for pc in plugin_configs
    ]
    for _, plugin in main_plugins:
        state = plugin.before_runtime(state)
    state = runtime_cls(config)(state)
    for class_name, plugin in main_plugins:
        state = plugin.after_runtime(state)
        logger.info("[%s] runtime plugin applied: %s", state["name"], class_name)

    # EMA shadow: always eval + frozen (updated by the engine's averaging step,
    # never by gradients), so its core config forces training/requires_grad off
    # while carrying param_dtype etc. over. Each plugin runs too, with its ema
    # node's keys merged on as overrides; ema.enabled=false drops it from the EMA.
    # Same sequence as the main pass -- before hooks, core, after hooks.
    if "ema_model" in state:
        ema_plugins = []
        for pc in plugin_configs:
            ema_node = pc.get("ema", {})
            if not ema_node.get("enabled", True):
                continue
            effective = CfgNode(copy.deepcopy({k: v for k, v in pc.items() if k != "ema"}))
            CfgNode._merge_dict(effective, {k: v for k, v in ema_node.items() if k != "enabled"})
            ema_plugins.append((pc.class_name, getattr(importlib.import_module(pc.module), pc.class_name)(effective)))
        ema_config = CfgNode({
            **copy.deepcopy({k: v for k, v in config.items() if k != "plugins"}),
            "training": False, "requires_grad": False,
        })
        ema_state = {"name": state["name"] + "_ema", "config": state["config"], "model": state["ema_model"]}
        for _, plugin in ema_plugins:
            ema_state = plugin.before_runtime(ema_state)
        ema_state = runtime_cls(ema_config)(ema_state)
        for class_name, plugin in ema_plugins:
            ema_state = plugin.after_runtime(ema_state)
            logger.info("[%s] runtime plugin applied: %s", ema_state["name"], class_name)
        state["ema_model"] = ema_state["model"]
    return state


class DefaultRuntime:
    """Finalize how an nn.Module runs: train mode, gradient tracking, compute
    dtype annotation, and per-submodule ``layer_name`` tagging.

    A build product that is not an nn.Module has nothing to configure, so this is
    a no-op for it (the device move / sharding / dtype vocabulary are all Module
    concepts).
    """

    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, torch.nn.Module):
            return state

        name = state["name"]
        config = self.config

        if "training" in config:
            model.train(config.training)
        if "requires_grad" in config:
            model.requires_grad_(config.requires_grad)

        for path, module in model.named_modules():
            module.layer_name = f"{name}.{path}" if path else name

        if "param_dtype" in config:
            # Compute dtype, annotated on the module for forward-time consumers
            # (same vocabulary as FSDP2's MixedPrecisionPolicy.param_dtype).
            model.param_dtype = getattr(torch, config.param_dtype)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        extras = ""
        if "training" in config:
            extras += f", train={config.training}"
        if "param_dtype" in config:
            extras += f", param_dtype={config.param_dtype}"
        logger.info(
            "[%s] runtime: trainable %s / total %s params%s", name, f"{trainable:,}", f"{total:,}", extras
        )
        return state
