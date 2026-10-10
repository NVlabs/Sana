"""Optimizer construction and checkpoint-tree membership.

An ``optimizer`` node hangs under ``models.<name>`` as a *list* (a single
optimizer is a one-element list); its presence is what marks a model as
*trained* -- and therefore checkpointed. A frozen model simply omits the node
(do not configure an empty one).

Single optimizer -- constructor kwargs live under ``args``::

    optimizer:
      - class_name: AdamW          # module defaults to torch.optim
        args:
          lr: 1.0e-5
          weight_decay: 0.01

Parameter groups -- a group's ``args`` override this optimizer's ``args``
defaults; params matching no group fall to a default group carrying those
defaults::

    optimizer:
      - class_name: AdamW
        args:
          lr: 1.0e-5
          weight_decay: 0.01
        groups:
          - match: ["*.bias", "*norm*"]
            args:
              weight_decay: 0.0
          - match: ["*embed*"]
            args:
              lr: 1.0e-6

Multiple optimizers over one model -- identity is the ``class_name`` itself
(duplicate classes are rejected: same-class hyperparameter differences belong in
``groups``). ``match`` (a list of fnmatch patterns) partitions the trainable
parameters, first match by list order. The single item without a ``match`` is
the catch-all for everything unclaimed::

    optimizer:
      - module: projects.x.optim
        class_name: Muon
        match: ["blocks.*.weight"]
        args:
          lr: 2.0e-2
      - class_name: AdamW          # no match => catch-all
        args:
          lr: 1.0e-5

An optional ``lr_scheduler`` node may sit beside ``args`` (item level) or
beside a group's ``args`` (group level); it is consumed by
``common.lr_scheduler.build_lr_schedulers``, not here.

Return shape: a single-optimizer model yields a bare optimizer leaf
(``optimizers[name] = optimizer``); a multi-optimizer model yields a nested
``optimizers[name] = {class_name: optimizer}``. Either way the persistence
``OptimizerHandler`` pairs each optimizer with ``models.<name>`` by path
convention -- the nested FQN ``optimizers.backbone.Muon`` resolves to
``models.backbone`` by stripping trailing segments.

Timing contract: ``build_optimizers`` must run *after* ``build_models``.
Placement has already sharded the parameters by then, so each optimizer takes
DTensor shards directly and its optimizer states are naturally 1/N sharded. An
optimizer must never be created mid-pipeline: the weights stage loads with
``assign=True``, which swaps parameter objects, so an optimizer built earlier
would hold orphaned tensors.

Assembling the checkpoint state tree::

    models = build_models(config.models)
    optimizers = build_optimizers(models, config.models)
    state = {
        "models": checkpoint_models(models, optimizers),
        "optimizers": optimizers,
        "dataloader": ...,
        "runtime": {"iter": 0},
    }

Deliberately out of scope: LR schedulers -- a scheduler is a pure ``step(iter)``
function owned by the engine, not persisted here.

The ``_ema`` suffix is a reserved framework name (the EMA-derived model of a
trained model); ``checkpoint_models`` relies on it to decide EMA membership.
"""

from __future__ import annotations

import fnmatch
import importlib
from typing import Any

from ..logging import get_logger

logger = get_logger()


def build_optimizers(models: dict[str, Any], model_configs: Any) -> dict[str, Any]:
    """Build the optimizer(s) for every model that configures an ``optimizer`` list.

    See the module docstring for the config schema, the match/group semantics,
    and the returned leaf shapes. Models without an ``optimizer`` node are
    skipped (they are frozen).
    """
    optimizers: dict[str, Any] = {}
    for name, model_config in model_configs.items():
        if "optimizer" not in model_config:
            continue
        optimizer_configs = model_config.optimizer
        assert isinstance(optimizer_configs, list), (
            f"[{name}] optimizer must be a list; a single optimizer is a one-element list "
            "(see the module docstring for the schema)"
        )

        named = [(param_name, param) for param_name, param in models[name].named_parameters() if param.requires_grad]
        assert named, f"[{name}] configures an optimizer but has no trainable parameters (fully frozen); remove the optimizer node or unfreeze"

        multi = len(optimizer_configs) > 1
        if multi:
            # Identity is the class itself: two items of the same class are
            # rejected because same-class hyperparameter differences belong in
            # groups, not in a second instance.
            classes = [item.get("class_name", "AdamW") for item in optimizer_configs]
            assert len(set(classes)) == len(classes), (
                f"[{name}] duplicate optimizer classes {classes}; use groups for per-parameter hyperparameters"
            )

        # Optimizer-level partition: each trainable param goes to the first item
        # whose match patterns hit it; the lone item without match is catch-all.
        catch_all_indices = [i for i, item in enumerate(optimizer_configs) if "match" not in item]
        assert len(catch_all_indices) <= 1, (
            f"[{name}] more than one optimizer without a match (catch-all): items {catch_all_indices}; at most one may omit match"
        )
        catch_all = catch_all_indices[0] if catch_all_indices else None

        assigned: list[list] = [[] for _ in optimizer_configs]
        unowned: list[str] = []
        for param_name, param in named:
            owner = None
            for i, item in enumerate(optimizer_configs):
                if "match" in item and any(fnmatch.fnmatch(param_name, pattern) for pattern in item.match):
                    owner = i
                    break
            if owner is None:
                owner = catch_all
            if owner is None:
                unowned.append(param_name)
            else:
                assigned[owner].append((param_name, param))
        assert not unowned, f"[{name}] {len(unowned)} trainable params matched no optimizer and there is no catch-all: {unowned[:5]}"

        nested: dict[str, Any] = {}
        for i, item in enumerate(optimizer_configs):
            item_params = assigned[i]
            item_name = item.get("class_name", "AdamW")
            assert item_params, f"[{name}] optimizer '{item_name}' (#{i}) matched no parameters; check its match patterns"

            unknown = set(item) - {"module", "class_name", "match", "groups", "args", "lr_scheduler"}
            assert not unknown, (
                f"[{name}] optimizer '{item_name}' (#{i}) has unknown keys {sorted(unknown)}; "
                "constructor kwargs (lr, weight_decay, ...) belong under 'args'"
            )
            module = importlib.import_module(item.get("module", "torch.optim"))
            optimizer_cls = getattr(module, item.get("class_name", "AdamW"))
            kwargs = dict(item.get("args", {}) or {})

            groups = item.get("groups", None)
            if groups:
                # Within-optimizer grouping: first matching group by list order,
                # unmatched params fall to a default group. A group's keys become
                # param_group overrides on top of the optimizer's kwargs defaults.
                group_params: list[list] = [[] for _ in groups]
                default_params: list = []
                for param_name, param in item_params:
                    g = None
                    for gi, group in enumerate(groups):
                        if any(fnmatch.fnmatch(param_name, pattern) for pattern in group.match):
                            g = gi
                            break
                    if g is None:
                        default_params.append(param)
                    else:
                        group_params[g].append(param)
                param_groups: list[dict] = []
                for gi, group in enumerate(groups):
                    # An empty group means its match patterns hit nothing -- a
                    # typo would silently hand these params the default-group
                    # hyperparameters, so fail loud like every other match guard.
                    assert group_params[gi], f"[{name}] optimizer '{item_name}' group #{gi} {list(group.match)} matched no parameters"
                    unknown = set(group) - {"match", "args", "lr_scheduler"}
                    assert not unknown, (
                        f"[{name}] optimizer '{item_name}' group #{gi} has unknown keys {sorted(unknown)}; "
                        "param_group overrides (lr, weight_decay, ...) belong under 'args'"
                    )
                    overrides = dict(group.get("args", {}) or {})
                    param_groups.append({"params": group_params[gi], **overrides})
                if default_params:
                    param_groups.append({"params": default_params})
                optimizer = optimizer_cls(param_groups, **kwargs)
                n_groups = len(param_groups)
            else:
                optimizer = optimizer_cls([param for _, param in item_params], **kwargs)
                n_groups = 1

            if multi:
                nested[item_name] = optimizer

            label = f"{name}.{item_name}" if multi else name
            lr_note = f", lr={kwargs['lr']}" if "lr" in kwargs else ""
            group_note = f", {n_groups} groups" if n_groups > 1 else ""
            logger.info(
                "[%s] optimizer %s: %s params%s%s",
                label, optimizer_cls.__name__, f"{sum(p.numel() for _, p in item_params):,}", lr_note, group_note,
            )

        optimizers[name] = nested if multi else optimizer

    return optimizers


def checkpoint_models(models: dict[str, Any], optimizers: dict[str, Any]) -> dict[str, Any]:
    """Filter ``models`` down to the checkpoint tree members.

    A model belongs in the checkpoint iff it has an optimizer, or it is the EMA
    of a trained model (``<name>_ema`` where ``<name>`` has an optimizer).
    References are passed through, not copied.
    """
    included = {
        name: module
        for name, module in models.items()
        if name in optimizers or name.removesuffix("_ema") in optimizers
    }
    excluded = [name for name in models if name not in included]
    logger.info(
        "Checkpoint models: included %s, excluded (frozen) %s", sorted(included), sorted(excluded)
    )
    return included
