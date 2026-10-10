"""Small metric meters shared by engines and deep model code.

``AverageMeter`` is for count-weighted means over a logging window. Each
``put_scalar`` contributes one observation for that key, and ``sync`` merges
``(sum, count)`` pairs across distributed ranks before the engine logs.

``RunningAccumulator`` is for counters and other sum-style values accumulated
across logging windows.

Model internals and low-level primitives may report scalars without threading
extra return values through every call by using the global singletons::

    get_running_average_meter().put_scalar("model/foo", value)
    get_running_accumulator().put_scalar("data/tokens", tokens)

The training engine owns the log boundary: it syncs these singletons, emits the
merged metrics, then resets their local window state.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from .distributed import ops


_RUNNING_AVERAGE_METER: "AverageMeter | None" = None
_RUNNING_ACCUMULATOR: "RunningAccumulator | None" = None


def _to_number(value: Any) -> float | int:
    if torch.is_tensor(value):
        value = value.detach().item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


class AverageMeter:
    """Count-weighted scalar averages for one logging window."""

    def __init__(self) -> None:
        self.reset()

    def put_scalar(self, key: str, value: Any) -> None:
        """Add one scalar observation for ``key``."""
        if value is None:
            return
        if key not in self._sum:
            self._keys.append(key)
            self._sum[key] = 0.0
            self._count[key] = 0
        self._sum[key] += float(_to_number(value))
        self._count[key] += 1

    def update(self, values: dict[str, Any]) -> None:
        """Add every scalar in ``values`` as one observation."""
        for key, value in values.items():
            self.put_scalar(key, value)

    def sync(self) -> None:
        """Merge sums and counts across ranks; no-op without distributed."""
        gathered = ops.all_gather_object((self._sum, self._count))

        keys: set[str] = set()
        for sums, _counts in gathered:
            keys.update(sums)

        self._keys = sorted(keys)
        self._sum = {}
        self._count = {}
        for key in self._keys:
            total_sum = 0.0
            total_count = 0
            for sums, counts in gathered:
                if key in sums:
                    total_sum += float(sums[key])
                    total_count += int(counts[key])
            if total_count > 0:
                self._sum[key] = total_sum
                self._count[key] = total_count

    @property
    def avg(self) -> dict[str, float]:
        return {key: self._sum[key] / self._count[key] for key in self.keys}

    @property
    def keys(self) -> list[str]:
        return sorted(self._keys)

    def items(self) -> Iterator[tuple[str, float]]:
        for key in self.keys:
            yield key, self._sum[key] / self._count[key]

    def reset(self) -> None:
        self._keys: list[str] = []
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}


class RunningAccumulator:
    """Integer counter deltas promoted into a synchronized cumulative sum."""

    def __init__(self) -> None:
        self.sum: dict[str, int] = {}
        self.reset()

    def put_scalar(self, key: str, value: Any) -> None:
        """Add ``value`` to the running sum for ``key``."""
        if value is None:
            return
        if key not in self._sum:
            self._sum[key] = 0
        self._sum[key] += int(_to_number(value))

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self.put_scalar(key, value)

    def sync(self) -> None:
        """Add every rank's local delta to the synchronized cumulative sum."""
        gathered = ops.all_gather_object(self._sum)
        for sums in gathered:
            for key, value in sums.items():
                self.sum[key] = self.sum.get(key, 0) + int(value)

    @property
    def keys(self) -> list[str]:
        return sorted(self.sum)

    def items(self) -> Iterator[tuple[str, int]]:
        for key in self.keys:
            yield key, self.sum[key]

    def reset(self) -> None:
        self._sum: dict[str, int] = {}

    def state_dict(self) -> dict[str, tuple[tuple[str, int], ...]]:
        # Keep dynamic metric keys in one DCP leaf so an empty fresh instance can restore them.
        return {"sum": tuple(self.sum.items())}

    def load_state_dict(self, state_dict: dict[str, tuple[tuple[str, int], ...]]) -> None:
        self.sum = {key: int(value) for key, value in state_dict["sum"]}
        self.reset()


def get_running_average_meter() -> AverageMeter:
    """Return the process-wide average meter, creating it on first use."""
    global _RUNNING_AVERAGE_METER
    if _RUNNING_AVERAGE_METER is None:
        _RUNNING_AVERAGE_METER = AverageMeter()
    return _RUNNING_AVERAGE_METER


def get_running_accumulator() -> RunningAccumulator:
    """Return the process-wide accumulator, creating it on first use."""
    global _RUNNING_ACCUMULATOR
    if _RUNNING_ACCUMULATOR is None:
        _RUNNING_ACCUMULATOR = RunningAccumulator()
    return _RUNNING_ACCUMULATOR
