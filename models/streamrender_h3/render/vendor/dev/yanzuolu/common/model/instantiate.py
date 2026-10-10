"""Model architecture instantiation stage.

The router selects one of three explicit construction contracts:

* :class:`DefaultModelInstantiate` calls the configured class constructor.
* :class:`DiffusersModelInstantiate` builds from a diffusers snapshot config via
  ``load_config``/``from_config``.
* :class:`TransformersModelInstantiate` builds a transformers model from its
  ``config_class``.

Snapshot families are never auto-detected. A model carrying
``pretrained_model_name_or_path`` must explicitly select the matching snapshot
class; a wrong selection fails on that class's fixed API contract. Model
``from_pretrained`` remains deliberately banned because it combines construction
with an uncontrolled loading pipeline and can privately materialize another
copy of the weights. Snapshot weights instead flow only through the weights
stage's mmap+assign path.

``meta_init`` defaults to true. ``accelerate.init_empty_weights`` then constructs
a parameter-only meta shell (buffers remain materialized), and the weights stage
assigns file-backed tensors into that shell. ``meta_init: false`` preserves
constructor initialization for from-scratch or deliberately partial-coverage
workflows.

When an ``ema`` node is present, the first three stages (instantiate, weights,
adapters) build a structurally identical EMA alongside the main model -- same
config, name suffixed ``_ema`` -- so no copy is ever taken. Matching the two sets
of initial values is the engine's step-0 sync contract, not the build's.
"""

from __future__ import annotations

import importlib
import inspect
from contextlib import nullcontext
from typing import Any

from ..logging import get_logger

logger = get_logger()


def instantiate_model(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    stage_config = config.get("instantiate", {})
    module = importlib.import_module(stage_config.get("module", __name__))
    instantiate_cls = getattr(module, stage_config.get("class_name", "DefaultModelInstantiate"))
    state = instantiate_cls(stage_config)(state)

    if "ema" in config:
        ema_state = {"name": state["name"] + "_ema", "config": config}
        ema_state = instantiate_cls(stage_config)(ema_state)
        state["ema_model"] = ema_state["model"]
        logger.info("[%s_ema] EMA shell built alongside (same config)", state["name"])

    return state


def _init_context(config: Any):
    if config.get("meta_init", True):
        from accelerate import init_empty_weights

        return init_empty_weights
    return nullcontext


def _log_instantiation(state: dict[str, Any], config: Any, model: Any, source: str) -> None:
    shell = "meta shell" if config.get("meta_init", True) else "materialized"
    parameters = getattr(model, "parameters", None)
    parameter_count = (
        f"{sum(p.numel() for p in parameters()):,}"
        if callable(parameters)
        else "n/a"
    )
    logger.info(
        "[%s] instantiated %s (%s) %s: %s params",
        state["name"], type(model).__name__, shell, source,
        parameter_count,
    )


class DefaultModelInstantiate:
    """Build a model only by calling its constructor with ``args``."""

    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        config = state["config"]
        pretrained = config.get("pretrained_model_name_or_path", None)
        assert pretrained is None, (
            "DefaultModelInstantiate does not handle pretrained_model_name_or_path; "
            "select DiffusersModelInstantiate or TransformersModelInstantiate explicitly"
        )

        module = importlib.import_module(config.module)
        model_cls = getattr(module, config.class_name)
        with _init_context(config)():
            model = model_cls(**config.get("args", {}))

        _log_instantiation(state, config, model, "from constructor args")
        state["model"] = model
        return state


class DiffusersModelInstantiate:
    """Build a diffusers model from its snapshot's bundled config."""

    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        config = state["config"]
        module = importlib.import_module(config.module)
        model_cls = getattr(module, config.class_name)
        pretrained = config.pretrained_model_name_or_path

        # Field overrides must go to from_config, whose kwargs take precedence
        # over config values -- load_config silently drops unknown kwargs instead
        # of merging them.
        args = dict(config.get("args", {}))
        if args:
            # from_config swallows keys that are not __init__ parameters without
            # raising, so validate typos against the constructor signature.
            params = inspect.signature(model_cls.__init__).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                unknown = sorted(set(args) - set(params))
                assert not unknown, f"args {unknown} are not parameters of {model_cls.__name__}.__init__"

        load_config_kwargs = {}
        if "subfolder" in config:
            load_config_kwargs["subfolder"] = config.subfolder
        model_config = model_cls.load_config(pretrained, **load_config_kwargs)
        with _init_context(config)():
            model = model_cls.from_config(model_config, **args)

        _log_instantiation(state, config, model, f"from bundled config of {pretrained}")
        state["model"] = model
        return state


class TransformersModelInstantiate:
    """Build a transformers backbone from its snapshot config.

    ``init_empty_weights`` places only Parameters on meta; constructor-created
    buffers such as rotary-embedding state remain real.
    """

    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        config = state["config"]
        module = importlib.import_module(config.module)
        model_cls = getattr(module, config.class_name)
        pretrained = config.pretrained_model_name_or_path
        args = dict(config.get("args", {}))

        if "subfolder" in config:
            model_config, unused = model_cls.config_class.from_pretrained(
                pretrained,
                subfolder=config.subfolder,
                **args,
                return_unused_kwargs=True,
            )
        else:
            model_config, unused = model_cls.config_class.from_pretrained(
                pretrained,
                **args,
                return_unused_kwargs=True,
            )
        assert not unused, f"unused transformers config args: {sorted(unused)}"

        with _init_context(config)():
            model = model_cls(model_config)

        _log_instantiation(state, config, model, f"from bundled config of {pretrained}")
        state["model"] = model
        return state


class FromPretrainedInstantiate:
    """Build weightless assets (tokenizers, processors) via ``cls.from_pretrained``.

    ``from_pretrained`` is banned for models on the 1x-memory route (it fuses
    construction with uncontrolled materialization); this class exists for
    assets where that rationale does not apply -- a tokenizer has no weight
    tensors, loading it IS pure file reading. Everything, including
    ``pretrained_model_name_or_path``, is passed through ``args``: the node
    deliberately carries no node-level ``pretrained_model_name_or_path``, so
    the weights stage sees a completed construction and naturally no-ops.
    The result is fully materialized; ``meta_init`` must not be set.
    """

    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        config = state["config"]
        assert not config.get("meta_init", False), (
            "FromPretrainedInstantiate materializes directly; meta_init is not applicable"
        )
        assert "pretrained_model_name_or_path" not in config, (
            "FromPretrainedInstantiate takes pretrained_model_name_or_path inside args; "
            "a node-level field would wrongly engage the weights stage"
        )
        module = importlib.import_module(config.module)
        model_cls = getattr(module, config.class_name)
        args = dict(config.get("args", {}))
        loaded = model_cls.from_pretrained(**args)

        logger.info(
            "[%s] instantiated %s from_pretrained %s",
            state["name"], type(loaded).__name__, args.get("pretrained_model_name_or_path", "?"),
        )
        state["model"] = loaded
        return state
