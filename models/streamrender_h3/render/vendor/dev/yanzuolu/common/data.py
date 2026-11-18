"""Stateful worker-local dataset protocol and checkpointable dataloader."""

from __future__ import annotations

import importlib
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, NamedTuple, cast

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import IterableDataset

from .distributed import ops
from .logging import get_logger

logger = get_logger()


class WorkerStateLoadError(ValueError):
    """A dataset-specific worker snapshot cannot be restored."""


@dataclass(frozen=True)
class WorkerResumeContext:
    rank: int
    world_size: int
    num_workers: int
    next_logical_worker_id: int
    committed_states: Mapping[int, bytes]


class WorkerStateEnvelope(NamedTuple):
    batch: Any
    logical_worker_id: int
    state_after: bytes


DatasetFactory = Callable[[WorkerResumeContext], IterableDataset]


class StatefulDataLoader:
    """Commit worker snapshots only when their corresponding batches are consumed."""

    _SCHEMA = "jarvis_stateful_dataloader"
    _VERSION = 2

    def __init__(
        self,
        factory: DatasetFactory,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        pin_memory: bool = True,
        strict: bool = True,
    ):
        self.factory = factory
        self.num_workers = int(num_workers)
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.strict = bool(strict)
        self.rank = ops.get_rank()
        self.world_size = ops.get_world_size()
        self.consumed = 0
        self.committed_states: dict[int, bytes] = {}
        self._prepared_dataset: IterableDataset | None = None
        self._loader: TorchDataLoader | None = None
        self._iterator: Any = None

    def _reset_for_partial_resume(self, consumed: int) -> None:
        dataset = self.factory(self._context(consumed, {}))
        states = {
            worker: dataset.initial_worker_snapshot(worker)
            for worker in range(min(consumed, self.effective_workers))
        }
        self.consumed, self.committed_states, self._prepared_dataset = consumed, states, dataset

    @property
    def effective_workers(self) -> int:
        return self.num_workers or 1

    def _context(self, consumed: int, states: Mapping[int, bytes]) -> WorkerResumeContext:
        return WorkerResumeContext(
            rank=self.rank,
            world_size=self.world_size,
            num_workers=self.num_workers,
            next_logical_worker_id=consumed % self.effective_workers,
            committed_states=dict(states),
        )

    def _start(self) -> None:
        dataset = self._prepared_dataset
        if dataset is None:
            dataset = self.factory(self._context(self.consumed, self.committed_states))
        kwargs: dict[str, Any] = {}
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        try:
            loader = TorchDataLoader(
                dataset,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                **kwargs,
            )
            self._loader = loader
            self._iterator = iter(loader)
        finally:
            self._prepared_dataset = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._start()
        envelope = next(cast(Iterator[WorkerStateEnvelope], self._iterator))
        expected_worker = self.consumed % self.effective_workers
        if (
            not isinstance(envelope, WorkerStateEnvelope)
            or envelope.logical_worker_id != expected_worker
            or not isinstance(envelope.state_after, bytes)
        ):
            raise RuntimeError(f"invalid worker envelope for logical worker {expected_worker}")
        self.committed_states[expected_worker] = envelope.state_after
        self.consumed += 1
        return envelope.batch

    def state_dict(self) -> dict[str, bytes]:
        payload = {
            "schema": self._SCHEMA,
            "version": self._VERSION,
            "world_size": self.world_size,
            "num_workers": self.num_workers,
            "consumed": self.consumed,
            "worker_states": dict(self.committed_states),
        }
        return {f"rank_{self.rank}": pickle.dumps(payload)}

    def load_state_dict(self, state_dict: Any, strict: bool | None = None) -> None:
        if self._iterator is not None or self._loader is not None:
            raise RuntimeError("dataloader state must be loaded before iteration starts")
        key = f"rank_{self.rank}"
        try:
            payload = pickle.loads(state_dict[key])
        except (KeyError, TypeError, pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(f"invalid dataloader checkpoint for {key}") from exc
        if (payload.get("schema"), payload.get("version")) != (self._SCHEMA, self._VERSION):
            raise ValueError("unsupported dataloader checkpoint")
        use_strict = self.strict if strict is None else strict
        saved_topology = (payload["world_size"], payload["num_workers"])
        current_topology = (self.world_size, self.num_workers)
        topology_mismatch = saved_topology != current_topology
        if topology_mismatch and use_strict:
            raise ValueError("dataloader resume topology mismatch; world_size and num_workers must be unchanged")
        consumed = payload["consumed"]
        worker_states = payload["worker_states"]
        saved_effective_workers = payload["num_workers"] or 1
        expected_keys = set(range(min(consumed, saved_effective_workers)))
        if (
            consumed < 0
            or set(worker_states) != expected_keys
            or any(not isinstance(value, bytes) for value in worker_states.values())
        ):
            raise ValueError("invalid dataloader worker snapshots")
        context = self._context(consumed, worker_states)
        try:
            if topology_mismatch:
                raise WorkerStateLoadError("dataloader topology changed")
            dataset = self.factory(context)
        except WorkerStateLoadError:
            if use_strict:
                raise
            self._reset_for_partial_resume(consumed)
            return
        self.consumed, self.committed_states, self._prepared_dataset = consumed, worker_states, dataset


def build_dataloader(config: Any, seed: int) -> StatefulDataLoader:
    module = importlib.import_module(config.module)
    dataset_cls = getattr(module, config.class_name)
    num_workers = config.get("num_workers", 0)
    kwargs = dict(config.get("args", {}))
    loader = StatefulDataLoader(
        factory=lambda context: dataset_cls(seed=seed, resume_context=context, **kwargs),
        num_workers=num_workers,
        prefetch_factor=config.get("prefetch_factor", None),
        pin_memory=config.get("pin_memory", True),
        strict=config.get("strict", True),
    )
    logger.info("dataloader built: %s, num_workers=%d, seed=%d", dataset_cls.__name__, num_workers, seed)
    return loader
