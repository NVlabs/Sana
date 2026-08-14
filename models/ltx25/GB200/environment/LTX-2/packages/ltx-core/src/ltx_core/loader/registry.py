from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Protocol

from torch import nn

from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import SDOps


def module_registry_key(
    *,
    configurator: type,
    config: dict,
    module_ops_names: tuple[str, ...],
) -> str:
    """Hex digest identity of a model shell: configurator + init config + module ops."""
    payload = "\0".join(
        (
            f"{configurator.__module__}.{configurator.__qualname__}",
            json.dumps(config, sort_keys=True, separators=(",", ":")),
            ",".join(module_ops_names),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Registry(Protocol):
    """Cache for state dictionaries and/or structurally keyed model shells."""

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> StateDict: ...

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None: ...

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None: ...

    def clear(self) -> None: ...

    def get_model(self, key: str) -> nn.Module | None: ...

    def add_model(self, key: str, model: nn.Module) -> nn.Module: ...

    def pop_model(self, key: str) -> nn.Module | None: ...


class DummyRegistry(Registry):
    """Registry that stores neither state dictionaries nor model instances."""

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> StateDict:
        _ = paths, sd_ops
        return state_dict

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        _ = paths, sd_ops
        return None

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        _ = paths, sd_ops
        return None

    def clear(self) -> None:
        pass

    def get_model(self, key: str) -> nn.Module | None:
        _ = key
        return None

    def add_model(self, key: str, model: nn.Module) -> nn.Module:
        _ = key
        return model

    def pop_model(self, key: str) -> nn.Module | None:
        _ = key
        return None


class ModelRegistry(Registry):
    """In-process cache of state dictionaries and/or model shells.
    Toggle independently via ``cache_weights`` / ``cache_models``:
    - both ``True`` (default): cache checkpoint tensors and structural shells;
    - ``cache_models=True, cache_weights=False``: reuse shells, reload weights each build;
    - ``cache_models=False, cache_weights=True``: cache tensors only (rebuild structure each time);
    - both ``False``: same effective behavior as :class:`DummyRegistry`.
    Model shells are keyed by :func:`module_registry_key` hex digests. A structural hit
    reuses the shell; callers always reassign weights onto it. Non-persistent buffers
    installed by module ops are kept on the shell across ``dispose`` and are not stored
    in the weight map.
    """

    def __init__(self, *, cache_weights: bool = True, cache_models: bool = True) -> None:
        self._cache_weights = cache_weights
        self._cache_models = cache_models
        self._state_dicts: dict[str, StateDict] = {}
        self._models: dict[str, nn.Module] = {}
        self._lock = threading.Lock()

    def _generate_id(self, paths: list[str], sd_ops: SDOps | None) -> str:
        m = hashlib.sha256()
        parts = [str(Path(p).resolve()) for p in paths]
        if sd_ops is not None:
            parts.append(sd_ops.name)
        m.update("\0".join(parts).encode("utf-8"))
        return m.hexdigest()

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> StateDict:
        if not self._cache_weights:
            return state_dict
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            if sd_id in self._state_dicts:
                raise ValueError(f"State dict retrieved from {paths} with {sd_ops} already added, check with get first")
            self._state_dicts[sd_id] = state_dict
        return state_dict

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.pop(self._generate_id(paths, sd_ops), None)

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.get(self._generate_id(paths, sd_ops), None)

    def clear(self) -> None:
        with self._lock:
            self._state_dicts.clear()
            self._models.clear()

    def get_model(self, key: str) -> nn.Module | None:
        with self._lock:
            return self._models.get(key)

    def add_model(self, key: str, model: nn.Module) -> nn.Module:
        if not self._cache_models:
            return model
        with self._lock:
            if key in self._models:
                raise ValueError(f"Model for {key} already added, check with get_model first")
            self._models[key] = model
        return model

    def pop_model(self, key: str) -> nn.Module | None:
        with self._lock:
            return self._models.pop(key, None)
