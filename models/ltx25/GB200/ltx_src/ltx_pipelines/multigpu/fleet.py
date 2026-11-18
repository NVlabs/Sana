"""Worker fleet, wire protocol, and SPMD job execution driven by the MGPU controller.
Everything the controller (``MGPUController`` in ``controller.py``) drives lives here: the on-the-wire
payloads, the per-job NCCL input broadcast (``_RankLink`` / ``_run_job``), the worker entrypoint +
loops, the persistent worker fleet (``_RunnersFleet``), and ``_RunnerShipper`` (ships the runner
class to workers by value). The runner contract it executes (``MGPURunner`` / ``RunnerError``)
lives in ``runner.py``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Queue
from typing import Any

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.multiprocessing.api import LogsSpecs
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record

from ltx_pipelines.multigpu._broadcast import BroadcastCoordinator
from ltx_pipelines.multigpu.nccl_groups import create_local_nccl_groups
from ltx_pipelines.multigpu.runner import MGPURunner, RunnerError

logger = logging.getLogger(__name__)

_RELAY_RANK = 0
_POLL_S = 0.2  # how often the controller re-checks "did a worker die?" while waiting for a result


# =============================================================================
# The wire: what crosses the queues between controller and workers.
# =============================================================================
@dataclass
class _TensorPlaceholder:
    """Marker left in a kwargs/return dict where a tensor was lifted out for separate
    transport. Carries the shape/dtype so receivers can preallocate before NCCL fills it.
    Picklable and tiny -- this is what rides the object collective in the tensor's place.
    """

    idx: int  # position in the lifted-out tensor list
    shape: tuple[int, ...]
    dtype: Any  # torch.dtype


def _replace_tensors_by_placeholders(d: dict[str, Any]) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Split a dict into (skeleton, tensors): top-level Tensor values become _TensorPlaceholders.
    Top level only -- a tensor buried inside a list or nested dict is left alone and
    will ride the pickle path. Keep tensors as direct kwargs/return values.
    """
    skeleton: dict[str, Any] = {}
    tensors: list[torch.Tensor] = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            skeleton[k] = _TensorPlaceholder(len(tensors), tuple(v.shape), v.dtype)
            tensors.append(v)
        else:
            skeleton[k] = v
    return skeleton, tensors


def _fill_tensors_into_placeholders(skeleton: dict[str, Any], tensors: list[torch.Tensor]) -> dict[str, Any]:
    """Inverse of _replace_tensors_by_placeholders: put the tensors back where the _TensorPlaceholders are."""
    return {k: (tensors[v.idx] if isinstance(v, _TensorPlaceholder) else v) for k, v in skeleton.items()}


@dataclass
class _Job:
    """One dispatched job: the controller queues it, the relay (rank 0) broadcasts it to every
    rank, and all ranks run it. `kwargs` may hold top-level tensors -- broadcast over NCCL rather
    than pickled (see `_RankLink`). A `None` on the job queue is the shutdown sentinel.
    """

    job_id: int  # every rank echoes this back; a mismatch means the fleet desynced
    kwargs: dict[str, Any]


@dataclass
class _JobResult:
    """worker -> controller: one item from one rank -- a yielded value, a RunnerError, or the
    terminating StopIteration itself (carrying `.value`, the generator's `return`). A yielded value
    is forwarded to the caller as soon as it arrives. Terminals are collected per rank: once all
    `world_size` ranks have ended, the controller classifies them -- all StopIteration -> end
    iteration with `StopIteration([returns...])`; all/some RunnerError -> Symmetric/Asymmetric.
    Tensors in `value` ride the queue by shared memory / CUDA IPC, the same for every rank.
    """

    job_id: int  # the job this is for; the controller rejects a mismatch as a desync
    rank: int  # which rank produced this -- used to collect one terminal per rank (and order returns)
    value: Any  # a yielded value, a RunnerError, or the StopIteration that ended this rank


@dataclass
class _Channels:
    """The queues bridging the controller and the workers."""

    jobs: Queue  # type: ignore[type-arg]  # controller -> relay: _Job or None
    results: Queue  # type: ignore[type-arg]  # all ranks -> controller: chunk items + each rank's _JobResult
    ready: Queue  # type: ignore[type-arg]  # workers -> controller: rank, on setup-complete


# =============================================================================
# The NCCL core: the per-job input broadcast (relay -> all ranks).
# =============================================================================
class _RankLink:
    """One rank's handle to a job's *input* broadcast (relay -> all ranks, over NCCL). A job's data
    crosses in two directions and this covers only the inbound one: inputs arrive here, while each
    rank sends its results back out-of-band on the result queue -- never through a collective -- so
    the controller assembles them without a second NCCL gather.
    Between jobs the coordinator parks idle non-relay ranks on a cheap store signal instead of
    leaving them blocked inside a pending broadcast (which would trip the NCCL watchdog); a rank
    enters the collective only once the relay signals it has a job to send.
    """

    def __init__(self, coordinator: BroadcastCoordinator, device: torch.device) -> None:
        self._coordinator = coordinator
        self._device = device  # this rank's GPU; where received tensors land

    # ---- relay side
    def announce_job(self, job: _Job | None) -> _Job | None:
        """Broadcast the job to every rank, sending kwargs tensors over NCCL instead of
        pickling them. Returns the job with its kwargs tensors now on THIS rank's GPU,
        ready for the relay to run; None for the shutdown sentinel.
        """
        with self._coordinator.broadcast_context():
            if job is None:
                dist.broadcast_object_list([None], src=_RELAY_RANK)
                return None
            skeleton, tensors = _replace_tensors_by_placeholders(job.kwargs)
            gpu = [t.to(self._device).contiguous() for t in tensors]  # NCCL needs CUDA + contiguous
            dist.broadcast_object_list([_Job(job.job_id, skeleton)], src=_RELAY_RANK)
            for t in gpu:  # same order every rank, driven by the skeleton broadcast above
                dist.broadcast(t, src=_RELAY_RANK)
            job.kwargs = _fill_tensors_into_placeholders(skeleton, gpu)
            return job

    # ---- non-relay side
    def await_job(self) -> _Job | None:
        self._coordinator.wait_for_signal_change()
        payload: list[Any] = [None]
        dist.broadcast_object_list(payload, src=_RELAY_RANK)
        shell: _Job | None = payload[0]
        if shell is None:
            return None
        holes = sorted((v for v in shell.kwargs.values() if isinstance(v, _TensorPlaceholder)), key=lambda h: h.idx)
        tensors: list[Any] = []
        for h in holes:  # receive in idx order -- matches the relay's send order
            buf = torch.empty(h.shape, dtype=h.dtype, device=self._device)
            dist.broadcast(buf, src=_RELAY_RANK)
            tensors.append(buf)
        shell.kwargs = _fill_tensors_into_placeholders(shell.kwargs, tensors)
        return shell


def _run_job(runner: MGPURunner, kwargs: dict[str, Any], job_id: int, rank: int, result: Queue[_JobResult]) -> None:
    """Run one job on this rank and put each item on the result queue: every yield is a `_JobResult`
    (tagged by rank), forwarded straight to the caller; the rank's end -- a clean StopIteration or a
    raised RunnerError (recoverable) -- is itself a `_JobResult`, which the controller collects and
    classifies. Runners must be generators; a non-generator return is not iterable, so `next` raises
    (fatal).
    """
    out = runner(**kwargs)
    while True:
        try:
            value = next(out)
        except StopIteration as stop:  # clean finish: carries the (normally None) return value
            result.put(_JobResult(job_id, rank, stop))
            return
        except RunnerError as err:  # runner raised it explicitly -> recoverable; collect and classify
            result.put(_JobResult(job_id, rank, err))
            return
        except ValueError as err:  # input validation (symmetric) -> synthesize a recoverable RunnerError
            result.put(_JobResult(job_id, rank, RunnerError(str(err))))
            return
        result.put(_JobResult(job_id, rank, value))


# =============================================================================
# The worker process: entrypoint + the two loops.
# =============================================================================
def _relay_loop(runner: MGPURunner, link: _RankLink, channels: _Channels) -> None:
    """Rank 0: take a job, broadcast it (tensors over NCCL), run it, stream every item on the queue."""
    while True:
        job: _Job | None = channels.jobs.get()
        job = link.announce_job(job)  # returns the job with kwargs tensors on this GPU
        if job is None:  # shutdown sentinel (already broadcast to the others)
            return
        _run_job(runner, job.kwargs, job.job_id, _RELAY_RANK, channels.results)


def _worker_loop(runner: MGPURunner, link: _RankLink, channels: _Channels, rank: int) -> None:
    """Non-relay ranks: wait for the broadcast, run in SPMD, stream every item on the queue."""
    while True:
        job = link.await_job()
        if job is None:  # shutdown sentinel
            return
        _run_job(runner, job.kwargs, job.job_id, rank, channels.results)


def _shutdown_distributed() -> None:
    # Drop torch.compile state before tearing down NCCL: compiled artifacts hold CUDA
    # pool/stream refs that ncclCommDestroy waits on, and destroy_process_group can
    # deadlock without this.
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.destroy_process_group()


@record
def _worker_entrypoint(
    runner_cls: type[MGPURunner],
    setup_kwargs: dict[str, Any],
    init_timeout: timedelta,
    channels: _Channels,
    device_ids: list[int] | None = None,
) -> None:
    """Per-worker entry. @record turns a crash into a readable failure on the controller."""
    local_rank = int(os.environ["LOCAL_RANK"])
    # All GPUs stay visible; we bind this rank to its physical GPU by index. (Setting
    # CUDA_VISIBLE_DEVICES here is unreliable -- the spawn bootstrap can touch CUDA, freezing
    # the device list, before a late env override would apply.)
    device_index = local_rank if device_ids is None else device_ids[local_rank]
    torch.cuda.set_device(device_index)
    # Build the runner before NCCL init so its __init__ runs as the per-worker pre-init hook, for setup
    # that must precede init_process_group (e.g. tests set torch.use_deterministic_algorithms there).
    runner = runner_cls()
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", device_index),
            timeout=init_timeout,
        )

    is_relay = dist.get_rank() == _RELAY_RANK
    device = torch.device("cuda", device_index)
    groups = create_local_nccl_groups()
    store = dist.distributed_c10d.PrefixStore("ltx_pipeline_broadcast/", dist.distributed_c10d._get_default_store())
    link = _RankLink(BroadcastCoordinator(store=store, is_driver=is_relay), device)

    runner._groups = groups
    runner.setup(**setup_kwargs)
    channels.ready.put(dist.get_rank())

    try:
        if is_relay:
            _relay_loop(runner, link, channels)
        else:
            _worker_loop(runner, link, channels, dist.get_rank())
    finally:
        _shutdown_distributed()


# =============================================================================
# The fleet: spawn, poll for death, terminate. No job knowledge.
# =============================================================================
class _RunnerShipper:
    """Ships a runner CLASS to spawned workers by value (cloudpickle), not by reference.
    A runner defined in ``__main__`` (a plain script or ``python -m``) or in a test module is
    not importable under that name in a freshly spawned worker, so the stock by-reference pickle
    the elastic launcher uses would raise ModuleNotFoundError on every rank. This proxy serializes
    the class by value; its ``__reduce__`` targets ``cloudpickle.loads`` (importable everywhere),
    so only the runner crosses by value -- the controller's own channel payloads stay on the stock
    pickler. The worker unpickles it straight back to the runner class.
    """

    def __init__(self, runner_cls: type[MGPURunner]) -> None:
        module = sys.modules.get(runner_cls.__module__)
        if module is None:
            self._payload = cloudpickle.dumps(runner_cls)
            return
        # cloudpickle pickles a class by reference when its module looks importable; force
        # by-value so a __main__/test-module runner survives the spawn.
        cloudpickle.register_pickle_by_value(module)
        try:
            self._payload = cloudpickle.dumps(runner_cls)
        finally:
            cloudpickle.unregister_pickle_by_value(module)

    def __reduce__(self) -> tuple[object, tuple[bytes]]:
        return (cloudpickle.loads, (self._payload,))


def _format_failures(failures: dict[int, ProcessFailure]) -> RuntimeError:
    lines = [f"  rank {r} (pid {f.pid}) exit {f.exitcode}:\n{f.message}" for r, f in failures.items()]
    return RuntimeError("MGPU worker(s) failed:\n" + "\n".join(lines))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _RunnersFleet:
    """The worker processes. All three methods call torch.elastic from the controller
    thread only -- that is the whole reason there is no lock anywhere.
    """

    def __init__(self, pcontext: Any) -> None:  # noqa: ANN401
        self._pcontext = pcontext

    @classmethod
    def spawn(
        cls,
        *,
        runner_cls: type[MGPURunner],
        setup_kwargs: dict[str, Any],
        init_timeout: timedelta,
        channels: _Channels,
        num_gpus: int,
        logs_specs: LogsSpecs,
        device_ids: list[int] | None = None,
    ) -> _RunnersFleet:
        port = _find_free_port()
        envs = {
            r: {
                "RANK": str(r),
                "LOCAL_RANK": str(r),
                "WORLD_SIZE": str(num_gpus),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
            }
            for r in range(num_gpus)
        }
        # device_ids maps rank -> physical GPU (None = identity); the worker binds to it. The runner
        # class ships by value so a __main__/test-module runner survives the spawn (see _RunnerShipper).
        packed = (_RunnerShipper(runner_cls), setup_kwargs, init_timeout, channels, device_ids)
        args = dict.fromkeys(range(num_gpus), packed)
        logger.info("Spawning %d MGPU workers...", num_gpus)
        return cls(
            start_processes(
                name="ltx_mgpu_worker",
                entrypoint=_worker_entrypoint,
                args=args,
                envs=envs,
                logs_specs=logs_specs,
                start_method="spawn",
            )
        )

    def poll(self) -> RuntimeError | None:
        """None while all workers are alive. Once any has exited, an error describing it.
        Used only mid-job, where ANY exit is unexpected (workers only exit on the
        shutdown sentinel), so a clean exit is reported as an error too.
        """
        result = self._pcontext.wait(timeout=0)
        if result is None:
            return None
        if result.failures:
            return _format_failures(result.failures)
        return RuntimeError("MGPU workers exited unexpectedly")

    def drain(self, timeout: float) -> bool:
        """Wait up to `timeout` for all workers to exit on their own. True if they did."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._pcontext.wait(timeout=0) is not None:
                return True
            time.sleep(_POLL_S)
        return False

    def terminate(self) -> None:
        """SIGTERM -> SIGKILL. Never raises; safe to call more than once."""
        with contextlib.suppress(Exception):
            self._pcontext.close()
