"""Synchronous multi-GPU controller.
Spawns one worker process per GPU. Rank 0 is the *relay*: the controller hands it a job over a
queue and it NCCL-broadcasts the job to every rank. All ranks run the user's runner -- a
*generator* -- in SPMD lockstep. `stream(**kwargs)` dispatches and returns a `Stream` you iterate;
each yielded value is forwarded straight to you, one element per yield, as it comes off the result
queue (no gathering across ranks). One job at a time -- no job queue, no pipelining.
Constraints and contract:
- Single machine only: MASTER_ADDR is localhost, RANK == LOCAL_RANK, one rank per GPU. By default
  rank r runs on cuda:r; pass `devices=[...]` to place the fleet on a specific physical GPU subset
  (rank r -> cuda:devices[r]), e.g. to run two controllers side by side on disjoint GPUs.
- The runner is a generator and runs in SPMD LOCKSTEP. Yields are forwarded individually, not
  gathered, so each rank's yields appear as their own stream elements (in result-queue order). Only
  the ranks' terminals are collected: once all have ended, the controller classifies and ends iteration.
- Job kwargs (non-tensor parts) cross the queue and are pickled again through the NCCL broadcast;
  yielded values ride only the result queue (pickled once). All must be picklable and small.
  Tensors are the exception: pass them as top-level kwargs and the relay broadcasts them over NCCL
  instead of pickling (see `stream`); tensors inside a yielded value ride the result queue by
  shared memory / CUDA IPC.
- Each job belongs to the thread that dispatched it: only that thread may iterate it, enforced in
  `Stream`.
- One job in flight at a time: consume the `Stream` to the end before dispatching the next.
  Abandoning it is NOT cleaned up -- the next `stream()` raises `ControllerBusyError` until you
  `stream.drain()` or `shutdown()`.
- A runner that *raises* an unexpected exception kills the controller (a desynced NCCL collective
  cannot be unwound; make a new one). For a recoverable failure the runner raises a `RunnerError`
  (or a `ValueError`, which the controller turns into one) identically on every rank: the worker loop
  catches it, the fleet survives, and iterating the `Stream` re-raises it as `SymmetricRunnerError`
  -- or `AsymmetricRunnerError` if only some ranks raised.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.multiprocessing as torch_mp
from torch.distributed.elastic.multiprocessing.api import DefaultLogsSpecs, LogsSpecs, Std

from ltx_pipelines.multigpu.fleet import _POLL_S, _Channels, _Job, _RunnersFleet
from ltx_pipelines.multigpu.runner import MGPURunner, RunnerError

logger = logging.getLogger(__name__)

_DEFAULT_INIT_TIMEOUT = timedelta(minutes=30)


# =============================================================================
# Errors raised to the caller: a busy dispatch, or a runner failure classified across ranks.
# =============================================================================
class SymmetricRunnerError(Exception):
    """Every rank raised a RunnerError -- the contract was honored. Recoverable: fix the
    input and retry; the controller is still alive."""

    def __init__(self, errors: list[RunnerError]) -> None:
        self.errors = errors
        super().__init__(errors[0].message)


class AsymmetricRunnerError(Exception):
    """Some ranks raised a RunnerError and some finished cleanly. The fleet is provably healthy
    (a mix of terminals means every rank ran to completion and reported), but the runner's error
    path is non-deterministic across ranks -- a latent hang risk. Loud by design; does NOT kill
    the fleet."""

    def __init__(self, terminals: list[Any]) -> None:
        self.terminals = terminals  # the per-rank terminals (RunnerErrors mixed with clean StopIterations)
        super().__init__("runner raised RunnerError on some ranks but not all")


class ControllerBusyError(RuntimeError):
    """`stream()` was called while a job is still in flight (uncollected).
    `stream` returns IMMEDIATELY, so it raises rather than silently draining the in-flight job
    (which would block for the full job, possibly one owned by another thread). The in-flight job belongs
    to the thread that dispatched it: consume it (`try: ... finally: stream.drain()`) so you never
    wedge yourself, and let a concurrent caller that loses the dispatch race simply bounce.
    """

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id  # the in-flight job's id, for catch-site logging
        super().__init__(f"MGPU job {job_id} is still in flight; consume it (stream.drain()) first.")


# =============================================================================
# The caller's streaming handle: a thin wrapper over the collector generator.
# Iterating it drives the job and forwards each rank's yields as they arrive.
# =============================================================================
class Stream:
    """The iterable returned by `stream()` and the handle to one in-flight job. Each element is one
    rank's yielded value, forwarded as it comes off the result queue -- yields are NOT gathered
    across ranks. When every rank's generator has finished, iteration ends: all clean -> the
    `StopIteration` value is the per-rank list of returns (normally Nones, since results are yielded
    -- `result = yield from stream` to read it); all `RunnerError` -> `SymmetricRunnerError`; a mix
    -> `AsymmetricRunnerError`.
    The job belongs to the thread that dispatched it: only that thread may iterate (or `drain`) this
    Stream -- a second thread advancing the same ordered collection is the one hazard forbidden here.
    Consume the Stream to completion before the next `stream()`; abandoning it leaves the job in
    flight (the next `stream()` raises `ControllerBusyError` until it is drained or shut down). The
    controller does not clean up after you -- the recommended pattern is `try: ... finally:
    stream.drain()`.
    """

    def __init__(self, pump: Iterator[Any], job_thread: int, job_id: int) -> None:
        self._pump = pump  # generator: yields each rank's values, returns the per-rank returns
        self._job_thread = job_thread  # the thread that dispatched this job; only it may iterate/drain
        self.job_id = job_id  # this job's id (for messages/debugging)

    def __iter__(self) -> Stream:
        return self

    def __next__(self) -> Any:  # noqa: ANN401
        if threading.get_ident() != self._job_thread:
            raise RuntimeError(
                f"Stream for job {self.job_id} is single-threaded: dispatched on thread "
                f"{self._job_thread}, iterated from thread {threading.get_ident()} -- a job belongs "
                f"to its dispatching thread."
            )
        # The pump yields each rank's value directly; once all ranks have ended it raises
        # StopIteration(per-rank returns), or Symmetric/Asymmetric if any rank raised.
        return next(self._pump)

    def drain(self) -> None:
        """Exhaust the Stream and free the controller, discarding any unconsumed yields. drain() must
        be called from the same thread that called stream() (it iterates the Stream, so the owner
        check applies); the recommended pattern is `try: ... finally: stream.drain()`. A recoverable
        Symmetric/AsymmetricRunnerError is swallowed (cleanup); a dead/timed-out/desynced fleet still
        surfaces -- you need to know.
        """
        try:
            for _ in self:  # iterate via __next__, so the dispatching-thread check applies
                pass
        except (SymmetricRunnerError, AsymmetricRunnerError):
            pass  # cleanup: a recoverable runner error isn't worth surfacing when draining


# =============================================================================
# The controller.
# =============================================================================
class MGPUController:
    """Persistent one-job-at-a-time GPU fleet.
    controller = MGPUController(MyRunner, num_gpus=8)
    controller.start(**setup_kwargs)
    stream = controller.stream(prompt="...")
    try:
        for item in stream:  # one element per yield, as it arrives (not gathered)
            show(item)
    finally:
        stream.drain()  # free the controller even on early exit -- abandoning wedges it
    controller.shutdown()
    """

    def __init__(
        self,
        runner_cls: type[MGPURunner],
        *,
        num_gpus: int | None = None,
        devices: Sequence[int] | None = None,
        logs_specs: LogsSpecs | None = None,
    ) -> None:
        """`num_gpus` uses GPUs 0..num_gpus-1 (default: all). `devices` places the fleet on specific
        physical GPUs (e.g. `[2, 3]` -> rank r on cuda:devices[r]), so several controllers can share a
        box on disjoint GPU sets; the two are mutually exclusive. Indices are as the controller sees
        them; every GPU stays visible to each worker, which simply binds to its assigned one.
        """
        if devices is not None:
            if num_gpus is not None:
                raise ValueError("Pass either num_gpus or devices, not both.")
            if len(devices) == 0 or len(set(devices)) != len(devices):
                raise ValueError(f"devices must be non-empty and unique: {list(devices)}.")
        self._runner_cls = runner_cls
        self._devices = list(devices) if devices is not None else None
        self._num_gpus = num_gpus
        self._logs_specs = logs_specs
        self._spawn_ctx = torch_mp.get_context("spawn")

        self._fleet: _RunnersFleet | None = None
        self._channels: _Channels | None = None
        self._next_job_id = 0  # monotonic job-id generator; persists across jobs (powers desync detection)
        self._inflight: Stream | None = None  # the one in-flight job, as its handle; None between jobs

        # baton-lock: guards ONLY the _inflight check-and-set; never held across dispatch / iteration / _collect.
        self._lock = threading.Lock()
        self._fatal_error: BaseException | None = None  # one-way: set once, then every call raises
        self._started = False

    @property
    def is_alive(self) -> bool:
        """True while the fleet is up and unpoisoned -- a health check that needs no try/except."""
        return self._started and self._fatal_error is None

    # ---------------------------------------------------------------- lifecycle
    def start(
        self,
        *,
        timeout: timedelta = _DEFAULT_INIT_TIMEOUT,
        **setup_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Spawn the fleet, run setup() on every rank, block until all report ready.
        `timeout` bounds both the NCCL `init_process_group` and the controller's wait for
        every rank to finish CUDA init, `create_local_nccl_groups`, and `setup()`, so a
        worker wedged ALIVE turns into a clear error instead of an infinite hang (poll()
        only sees a process *exit*, never a wedge). It must comfortably exceed your slowest
        model load; pass a large value to effectively wait forever.
        """
        if self._started:
            raise RuntimeError("MGPUController.start() called twice")
        if self._devices is not None:
            num_gpus = len(self._devices)
            device_ids: list[int] | None = list(self._devices)
        else:
            num_gpus = torch.cuda.device_count() if self._num_gpus is None else self._num_gpus
            device_ids = None
        if num_gpus <= 0:
            raise ValueError(f"No GPUs available: num_gpus={num_gpus}.")
        self._num_gpus = num_gpus  # resolve "all GPUs" to the actual count == world size (one rank per GPU)

        self._channels = _Channels(
            jobs=self._spawn_ctx.Queue(),
            results=self._spawn_ctx.Queue(),
            ready=self._spawn_ctx.Queue(),
        )
        self._fleet = _RunnersFleet.spawn(
            runner_cls=self._runner_cls,
            setup_kwargs=setup_kwargs,
            init_timeout=timeout,
            channels=self._channels,
            num_gpus=num_gpus,
            logs_specs=self._logs_specs or DefaultLogsSpecs(tee=Std.ALL),
            device_ids=device_ids,
        )
        try:
            self._await_ready(num_gpus, timeout.total_seconds())
        except BaseException:
            self.shutdown()  # tear the half-up fleet down so the caller need not
            raise
        self._started = True
        logger.info("MGPU fleet ready (%d workers).", num_gpus)

    def _await_ready(self, num_gpus: int, timeout: float) -> None:
        assert self._channels is not None
        assert self._fleet is not None
        deadline = time.monotonic() + timeout
        seen: set[int] = set()  # which ranks have checked in (ready.put sends the rank)
        while len(seen) < num_gpus:
            died = self._fleet.poll()
            if died is not None:  # a worker exited -- catches crashes, not wedges
                raise died
            if time.monotonic() > deadline:  # catches the wedges poll() can't see
                missing = sorted(set(range(num_gpus)) - seen)
                raise TimeoutError(
                    f"MGPU startup: {len(seen)}/{num_gpus} workers ready after {timeout:.0f}s; "
                    f"ranks {missing} never checked in -- stuck in init_process_group / "
                    f"create_local_nccl_groups / setup()? Check the worker logs."
                )
            try:
                seen.add(self._channels.ready.get(timeout=_POLL_S))
            except Exception:
                continue  # queue.Empty: re-check death + deadline, then retry

    def shutdown(self, *, graceful_timeout: float = 60.0) -> None:
        """Tell the fleet to exit, give it a moment, then make sure it is gone.
        Both teardown and kill switch -- safe to call from another thread while a job is in flight: it
        force-kills the fleet, so a thread wedged on the Stream surfaces an error and unwedges (this is
        how you recover a job you can't drain). A mid-job shutdown waits out `graceful_timeout` before
        forcing; pass 0 to skip it.
        """
        if self._fleet is None:
            return
        if self._fatal_error is None and self._channels is not None:
            with contextlib.suppress(Exception):
                self._channels.jobs.put(None)  # relay broadcasts the sentinel to all ranks
            self._fleet.drain(graceful_timeout)
        self._fleet.terminate()
        self._fleet = None
        self._started = False

    # ---------------------------------------------------------------- dispatch
    def stream(self, *, timeout: float | None = None, **kwargs: Any) -> Stream:  # noqa: ANN401
        """Dispatch one job and return IMMEDIATELY; collect later by iterating the returned Stream.
        The workers run the job on their own while the controller does nothing; iterate the Stream
        for the runner's yields -- each rank's yield is forwarded as its own element, as it arrives
        (not gathered across ranks). Worker death or a blown timeout (measured from dispatch)
        surfaces when you come back to iterate, not before.
        TENSORS ARE TRANSPARENT. Pass them as top-level kwargs (`stream(latent=t, steps=30)`): the
        tensor rides the queue to the relay by shared memory / CUDA IPC and the relay broadcasts it
        to every rank over NCCL, so `__call__` receives it on the local GPU. (Only top-level kwargs
        are broadcast this way; tensors nested in a list/dict ride the pickle path.) To send one
        back, `yield` it from every rank; the yield rides the result queue, so its tensors come back
        without pickling.
        ONE JOB AT A TIME. Consume the Stream to the end before dispatching the next. Abandoning a
        Stream is NOT cleaned up: the job stays in flight and the next `stream()` raises
        `ControllerBusyError` until you `stream.drain()` or `shutdown()`. Use `try: ... finally:
        stream.drain()`.
        """
        if not self._started:
            raise RuntimeError("MGPUController not started; call start() first")
        if self._fatal_error is not None:
            raise RuntimeError("MGPUController is dead; create a new one.") from self._fatal_error
        assert self._channels is not None
        job_thread = threading.get_ident()  # this job belongs to the dispatching thread (checked in Stream)

        # The baton-lock guards exactly this check-and-set -- nothing else (see the lock's comment).
        with self._lock:
            if self._inflight is not None:
                raise ControllerBusyError(self._inflight.job_id)
            job_id = self._next_job_id
            self._next_job_id += 1
            deadline = None if timeout is None else time.monotonic() + timeout
            stream = Stream(self._collect(job_id, deadline, timeout), job_thread, job_id)
            self._inflight = stream  # the single source of "a job is in flight"

        self._channels.jobs.put(_Job(job_id=job_id, kwargs=kwargs))  # dispatch OUTSIDE the lock; fleet starts NOW
        return stream

    @staticmethod
    def _is_terminal(value: object) -> bool:
        """True if `value` is a rank's end marker on the wire: a clean StopIteration or a RunnerError."""
        return isinstance(value, (StopIteration, RunnerError))

    @staticmethod
    def _classify_terminal(values: list[Any]) -> list[Any]:
        """Classify every rank's terminal (a StopIteration or a RunnerError), ordered by rank. All
        RunnerError -> SymmetricRunnerError; a mix of RunnerError and clean StopIteration ->
        AsymmetricRunnerError; all clean -> the per-rank return values (the caller's StopIteration.value).
        Neither typed error kills the fleet."""
        errs = [v for v in values if isinstance(v, RunnerError)]
        if errs and len(errs) == len(values):
            raise SymmetricRunnerError(errs)
        if errs:
            logger.error("asymmetric RunnerError across ranks: %r", values)
            raise AsymmetricRunnerError(values)
        return [v.value for v in values]  # all StopIteration -> their return values

    def _collect(self, job_id: int, deadline: float | None, timeout: float | None) -> Iterator[Any]:
        """Drain the result queue for the in-flight job: forward each yielded value to the caller as
        soon as it arrives (no gathering), collecting each rank's terminal as it ends. Once every
        rank has ended, classify them -- all clean -> return the per-rank returns (the caller's
        StopIteration.value); all/some RunnerError -> Symmetric/Asymmetric. Watches for a dead worker
        / blown timeout meanwhile. Runs on the caller's thread when they come back to the Stream.
        """
        assert self._channels is not None
        assert self._fleet is not None
        assert self._num_gpus is not None
        channels, fleet = self._channels, self._fleet
        n = self._num_gpus
        terminals: dict[int, Any] = {}  # rank -> its end marker (StopIteration | RunnerError)
        while True:
            try:
                msg = channels.results.get(timeout=_POLL_S)
            except Exception:
                # Queue empty: nothing ready, so NOW (and only now) check for death / timeout.
                died = fleet.poll()
                if died is not None:
                    self._fatal_error = died
                    fleet.terminate()
                    raise died from None
                if deadline is not None and time.monotonic() > deadline:
                    self._fatal_error = TimeoutError(f"MGPU job {job_id} exceeded {timeout}s")
                    fleet.terminate()
                    raise self._fatal_error from None
                continue

            if msg.job_id != job_id:  # a rank ran a different job than we dispatched -> SPMD desync
                self._fatal_error = RuntimeError(
                    f"MGPU fleet desync: rank {msg.rank} sent job {msg.job_id}, expected {job_id}."
                )
                fleet.terminate()
                raise self._fatal_error from None

            if not self._is_terminal(msg.value):
                yield msg.value  # forward this rank's yield straight to the caller -- no gathering
                continue

            terminals[msg.rank] = msg.value  # a rank ended; hold its terminal for classification
            if len(terminals) == n:  # every rank has ended -> fleet free; classify and finish
                self._inflight = None
                return self._classify_terminal([terminals[r] for r in range(n)])
