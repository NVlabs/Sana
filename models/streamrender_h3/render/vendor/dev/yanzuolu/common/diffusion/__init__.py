"""Portable diffusion primitives and their builder.

``build_diffusion`` instantiates every component of a ``{name: node}`` mapping
at once, where each node is the usual ``{module, class_name, args}`` triple
(``None`` nodes are skipped)::

    components = build_diffusion({
        "training_timesteps": {module: common.diffusion.timestep.training.uniform, ...},
        "schedule":           {module: common.diffusion.schedule.lerp, ...},
        "sampler":            {module: common.diffusion.sampler.consistency, ...},
        ...
    })

A name containing ``sampler`` is a sampler: it is constructed last, with the
already-built schedule component named ``name.replace("sampler", "schedule")``
injected as its ``schedule=`` argument (``sampler``/``schedule``,
``fake_sampler``/``fake_schedule``, ...).

The subpackage ``__init__``s (``sampler``/``schedule``/``timestep``/
``timestep.sampling``/``timestep.training``) implement only the base classes;
concrete classes live one file per class and are loaded by ``module`` path.
"""

from __future__ import annotations

import importlib
from typing import Any


def _instantiate(node: Any, **extra: Any) -> Any:
    cls = getattr(importlib.import_module(node["module"]), node["class_name"])
    return cls(**extra, **dict(node.get("args", {}) or {}))


def build_diffusion(configs: dict[str, Any]) -> dict[str, Any]:
    """Instantiate all diffusion components; samplers get their schedule injected."""
    components: dict[str, Any] = {}
    samplers = {}
    for name, node in configs.items():
        if node is None:
            continue
        if "sampler" in name:
            samplers[name] = node
        else:
            components[name] = _instantiate(node)
    for name, node in samplers.items():
        schedule_name = name.replace("sampler", "schedule")
        assert schedule_name in components, f"[{name}] requires a schedule component named '{schedule_name}'"
        components[name] = _instantiate(node, schedule=components[schedule_name])
    return components
