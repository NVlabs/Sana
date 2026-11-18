"""Copy/compute synchronization for block streaming, abstracted across backends.
Weight streaming overlaps an H2D weight copy with block compute. The two
operations must be ordered both ways:
* copy -> compute: a block must not be read before its weights have landed.
* compute -> reuse: a GPU buffer slot must not be overwritten by the next
  copy until the compute that read it has finished.
On CUDA these are expressed with a dedicated copy stream and cross-stream
events. MPS exposes no user-facing streams (only ``torch.mps.Event`` on a single
implicit queue), and CPU is fully synchronous. :class:`StreamSync` hides those
differences behind one protocol; :func:`create_stream_sync` picks the backend
implementation. The event types (``torch.cuda.Event`` / ``torch.mps.Event``)
share the small :class:`StreamEvent` surface the pool and source rely on.
Kept internal to the streaming module -- nothing else needs stream coordination.
"""

from __future__ import annotations

import contextlib
from typing import Protocol, runtime_checkable

import torch

from ltx_core.devices import is_mps_available


@runtime_checkable
class StreamEvent(Protocol):
    """A device synchronization marker (``torch.cuda.Event`` / ``torch.mps.Event``)."""

    def wait(self) -> None:
        """Device-side: make subsequently queued work wait for this event."""
        ...

    def synchronize(self) -> None:
        """Host-side: block the calling thread until this event completes."""
        ...


class StreamSync(Protocol):
    """Coordinates the H2D copy against block compute for one streaming model."""

    @property
    def is_async_copy(self) -> bool:
        """Whether H2D copies may be enqueued asynchronously."""
        ...

    def copy_scope(self) -> contextlib.AbstractContextManager[None]:
        """Context to enqueue the H2D copy under (the copy stream on CUDA)."""
        ...

    def commit_copy(self) -> StreamEvent | None:
        """Record a copy-done event and make compute wait on it.
        Returns the event so the source can guard reuse of the CPU buffer (the
        disk path host-synchronizes on it), or ``None`` when copies are
        synchronous and no guard is needed.
        """
        ...

    def record_compute_done(self) -> StreamEvent | None:
        """Record an event marking the end of a block's compute, for slot reuse."""
        ...

    def reuse_barrier(self, event: StreamEvent | None) -> None:
        """Before a slot is overwritten by a new copy, wait for *event* (prior compute)."""
        ...

    def synchronize(self) -> None:
        """Drain all outstanding copy and compute work."""
        ...


class CudaStreamSync:
    """CUDA: a dedicated copy stream plus cross-stream events.
    H2D copies run on ``copy_stream`` so they overlap compute on the default
    stream; events order the two directions explicitly.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._copy_stream = torch.cuda.Stream(device=device)

    @property
    def is_async_copy(self) -> bool:
        return True

    def copy_scope(self) -> contextlib.AbstractContextManager[None]:
        return torch.cuda.stream(self._copy_stream)

    def commit_copy(self) -> StreamEvent:
        event = torch.cuda.Event()
        event.record(self._copy_stream)
        torch.cuda.current_stream(self._device).wait_event(event)
        return event

    def record_compute_done(self) -> StreamEvent:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self._device))
        return event

    def reuse_barrier(self, event: StreamEvent | None) -> None:
        if event is not None:
            self._copy_stream.wait_event(event)

    def synchronize(self) -> None:
        self._copy_stream.synchronize()
        torch.cuda.current_stream(self._device).synchronize()


class MpsStreamSync:
    """MPS: one implicit queue with ``torch.mps.Event`` markers.
    There is no user-facing copy stream, so copy and compute already serialize
    on the single default queue. The events make that ordering explicit -- and,
    crucially, let the buffer pool guard slot reuse on the compute-done event
    rather than relying on the implicit single-queue ordering. ``Event.wait``
    enqueues a device-side wait on the default queue (it does not block the
    host); ``Event.synchronize`` is the host-blocking variant.
    """

    @property
    def is_async_copy(self) -> bool:
        return False

    def copy_scope(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def commit_copy(self) -> StreamEvent:
        event = torch.mps.Event()
        event.record()
        event.wait()
        return event

    def record_compute_done(self) -> StreamEvent:
        event = torch.mps.Event()
        event.record()
        return event

    def reuse_barrier(self, event: StreamEvent | None) -> None:
        if event is not None:
            event.wait()

    def synchronize(self) -> None:
        torch.mps.synchronize()


class SynchronousStreamSync:
    """CPU (and any non-accelerator backend): copies are synchronous, no events."""

    @property
    def is_async_copy(self) -> bool:
        return False

    def copy_scope(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def commit_copy(self) -> None:
        return None

    def record_compute_done(self) -> None:
        return None

    def reuse_barrier(self, event: StreamEvent | None) -> None:  # noqa: ARG002
        return None

    def synchronize(self) -> None:
        return None


def create_stream_sync(device: torch.device) -> StreamSync:
    """Return the :class:`StreamSync` implementation for *device*'s backend."""
    if device.type == "cuda":
        return CudaStreamSync(device)
    if device.type == "mps" and is_mps_available():
        return MpsStreamSync()
    return SynchronousStreamSync()
