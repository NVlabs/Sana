import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from diffusion.refiner.vendor.ltx_core.loader.primitives import StateDict
from diffusion.refiner.vendor.ltx_core.loader.sd_ops import SDOps


class Registry(Protocol):
    """
    Protocol for managing state dictionaries in a registry.
    It is used to store state dictionaries and reuse them later without loading them again.
    Implementations must provide:
    - add: Add a state dictionary to the registry
    - pop: Remove a state dictionary from the registry
    - get: Retrieve a state dictionary from the registry
    - clear: Clear all state dictionaries from the registry
    """

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> None: ...

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None: ...

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None: ...

    def clear(self) -> None: ...


class DummyRegistry(Registry):
    """
    Dummy registry that does not store state dictionaries.
    """

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> None:
        pass

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        pass

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        pass

    def clear(self) -> None:
        pass


@dataclass
class StateDictRegistry(Registry):
    """
    Registry that stores state dictionaries in a dictionary.
    """

    _state_dicts: dict[str, StateDict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _generate_id(self, paths: list[str], sd_ops: SDOps) -> str:
        m = hashlib.sha256()
        parts = [str(Path(p).resolve()) for p in paths]
        if sd_ops is not None:
            parts.append(sd_ops.name)
        m.update("\0".join(parts).encode("utf-8"))
        return m.hexdigest()

    def add(self, paths: list[str], sd_ops: SDOps | None, state_dict: StateDict) -> str:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            if sd_id in self._state_dicts:
                raise ValueError(f"State dict retrieved from {paths} with {sd_ops} already added, check with get first")
            self._state_dicts[sd_id] = state_dict
        return sd_id

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.pop(self._generate_id(paths, sd_ops), None)

    def get(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.get(self._generate_id(paths, sd_ops), None)

    def clear(self) -> None:
        with self._lock:
            self._state_dicts.clear()
