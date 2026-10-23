"""Process-group-safe distributed helpers.

Thin wrappers over ``torch.distributed`` that degrade gracefully when no
process group is initialized -- single-process runs, tests, or any construction
path that must work before (or without) a launcher. Rank/size fall back to the
launcher's environment variables; collectives such as ``barrier`` become
no-ops. Callers can therefore use one code path for both single- and
multi-process execution.

This is for *consumers* of the process group. Bootstrap -- creating the group,
binding the device, seeding -- lives in ``common.distributed.init`` and reads
the environment directly, since it runs before the group exists.

Only the helpers with real callers live here; collectives (all_reduce,
all_gather, gather/broadcast_object) are added when metrics/validation need
them, not speculatively.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch
import torch.distributed as dist


def _pg_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank: the process group if initialized, else ``RANK`` env (default 0)."""
    if _pg_ready():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    """Global world size: the process group if initialized, else ``WORLD_SIZE`` env (default 1)."""
    if _pg_ready():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_world_size() -> int:
    """Local world size from ``LOCAL_WORLD_SIZE`` env (default 1)."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_local_rank() -> int:
    """Local rank from ``LOCAL_RANK`` env (default 0)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """True on global rank 0."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks. No-op when no process group is initialized (single process)."""
    if _pg_ready():
        dist.barrier()


def all_gather_object(value):
    """Gather one object from every WORLD rank, or return the local value alone."""
    if not _pg_ready() or dist.get_world_size() == 1:
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def all_reduce_max(tensor: torch.Tensor) -> None:
    """In-place WORLD maximum reduction; no-op without an initialized process group."""
    if _pg_ready() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)


def all_reduce_sum(tensor: torch.Tensor, async_op: bool = False):
    """In-place WORLD sum reduction; returns the async handle (or None).

    No-op without an initialized process group (single process)."""
    if _pg_ready() and dist.get_world_size() > 1:
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)
    return None


def all_gather(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> list[torch.Tensor]:
    """Gather equal-shaped tensors from every rank in ``group`` (default WORLD)."""
    if not _pg_ready() or dist.get_world_size(group) == 1:
        return [tensor]
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def get_device() -> torch.device:
    """The current CUDA device when available, else CPU (single-process/tests)."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def to_device(data, device: torch.device):
    """Recursively move tensor leaves in a nested payload to ``device``."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return data.__class__((key, to_device(value, device)) for key, value in data.items())
    if isinstance(data, list):
        return [to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(to_device(value, device) for value in data)
    return data
