"""Build LR schedulers from per-optimizer config nodes.

An ``lr_scheduler`` node sits beside an optimizer item's ``args`` (scheduling
every param_group of that optimizer), or beside a group's ``args`` (scheduling
that single param_group; the item-level scheduler then covers the remaining
groups, including the default group)::

    models:
      backbone:
        optimizer:
          - class_name: AdamW
            args:
              lr: 1.0e-5
            lr_scheduler:                    # item level: whole optimizer
              module: common.lr_scheduler.warmup
              class_name: WarmupLRScheduler
              args:
                warmup_t: 1000
            groups:
              - match: ["*embed*"]
                args:
                  lr: 1.0e-6
                lr_scheduler:                # group level: this group only
                  module: common.lr_scheduler.multi_step
                  class_name: MultiStepLRScheduler
                  args:
                    milestones: [10000]

``module`` is imported with :mod:`importlib`, ``class_name`` is looked up on
the module, and ``args`` is passed as constructor keyword arguments on top of
the scheduled param_groups. Group order maps to ``optimizer.param_groups``
order (config groups first, then the default group when present).

A scheduler is a pure ``step(iter)`` object owned by the engine: it is stepped
every iteration and deliberately not checkpointed (see ``common/optim.py``).
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from ..logging import get_logger

logger = get_logger()


class BaseLRScheduler(ABC):
    """ Base class for learning rate scheduler. """

    @abstractmethod
    def step(self, t):
        """ Update lr at each step. """
        pass


def _build_one(node: Any, targets: list) -> Any:
    """``targets`` is a list of ``(optimizer, group_index)`` pairs.

    Indices rather than the param_group dicts: both ``Optimizer.load_state_dict``
    and DCP's ``set_optimizer_state_dict`` replace ``optimizer.param_groups`` with
    the dicts stored in the checkpoint, so any dict captured before a resume is
    orphaned afterwards and every lr written to it is silently discarded.
    """
    scheduler_cls = getattr(importlib.import_module(node["module"]), node["class_name"])
    return scheduler_cls(targets, **dict(node.get("args", {}) or {}))


def build_lr_schedulers(optimizers: dict[str, Any], model_configs: Any) -> dict[str, Any]:
    """Build the lr scheduler(s) configured inside each model's ``optimizer`` items.

    ``optimizers`` is the result of ``build_optimizers`` (same config order);
    keys mirror its labels: ``name`` / ``name.<class_name>`` for item-level
    schedulers, with a ``.g<i>`` suffix for group-level ones.
    """
    lr_schedulers: dict[str, Any] = {}
    for name, model_config in model_configs.items():
        if "optimizer" not in model_config:
            continue
        multi = len(model_config.optimizer) > 1
        for item in model_config.optimizer:
            item_name = item.get("class_name", "AdamW")
            optimizer = optimizers[name][item_name] if multi else optimizers[name]
            label = f"{name}.{item_name}" if multi else name

            remaining = list(range(len(optimizer.param_groups)))
            for gi, group in enumerate(item.get("groups", None) or []):
                node = group.get("lr_scheduler", None)
                if not node:
                    continue
                lr_schedulers[f"{label}.g{gi}"] = _build_one(node, [(optimizer, gi)])
                remaining.remove(gi)
                logger.info("[%s.g%s] lr_scheduler %s", label, gi, node["class_name"])

            node = item.get("lr_scheduler", None)
            if node:
                assert remaining, f"[{label}] item-level lr_scheduler configured but every param_group already has its own"
                lr_schedulers[label] = _build_one(node, [(optimizer, gi) for gi in remaining])
                logger.info(
                    "[%s] lr_scheduler %s: %s param_groups", label, node["class_name"], len(remaining)
                )
    return lr_schedulers
