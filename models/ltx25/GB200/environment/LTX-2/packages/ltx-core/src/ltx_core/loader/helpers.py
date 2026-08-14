"""Shared model-construction helpers used by both SingleGPUModelBuilder and StreamingModelBuilder."""

from __future__ import annotations

from os import PathLike, fspath
from typing import TypeVar

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator

_M = TypeVar("_M", bound=nn.Module)
_PathArg = str | PathLike[str]


def as_path_list(paths: _PathArg | tuple[_PathArg, ...] | list[_PathArg]) -> list[str]:
    """Normalize checkpoint path args to the ``list[str]`` shape registries key on."""
    if isinstance(paths, (str, PathLike)):
        return [fspath(paths)]
    return [fspath(p) for p in paths]


def load_state_dict(
    paths: _PathArg | tuple[_PathArg, ...] | list[_PathArg],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device | None,
    sd_ops: SDOps | None = None,
) -> StateDict:
    """Load a state dict from disk, using registry caching."""
    path_list = as_path_list(paths)
    cached = registry.get(path_list, sd_ops)
    if cached is not None:
        return cached
    result = loader.load(path_list, sd_ops=sd_ops, device=device)
    # ``add`` returns the retained copy when the registry rewrites storage (e.g. pin to CPU).
    return registry.add(path_list, sd_ops=sd_ops, state_dict=result)


def read_model_metadata(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read the full metadata dict from the first shard of a checkpoint."""
    first = model_path[0] if isinstance(model_path, tuple) else model_path
    return loader.metadata(first)


def read_model_config(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read the ``config`` sub-dict (transformer/vae/scheduler/...) from a checkpoint's metadata."""
    return read_model_metadata(model_path, loader).get("config", {})


def create_meta_model(
    configurator: type[ModelConfigurator[_M]],
    metadata: dict,
    module_ops: tuple[ModuleOps, ...] = (),
) -> _M:
    """Create a model on the meta device and apply module operations."""
    with torch.device("meta"):
        model = configurator.from_metadata(metadata)
    for op in module_ops:
        if op.matcher(model):
            model = op.mutator(model)
    return model
