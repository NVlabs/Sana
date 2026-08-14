"""BlockFetcher: async disk reads on a worker thread."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import torch

from ltx_core.block_streaming.disk import DiskBlockReader

logger = logging.getLogger(__name__)

Buffer = dict[str, torch.Tensor]


@dataclass(slots=True)
class FetchHandle:
    """Caller-facing handle for an outstanding read returned by :meth:`BlockFetcher.submit`.
    Carries only what the caller needs: a completion event the worker sets and the
    read's error. The worker updates these once the read finishes.
    """

    _done: threading.Event
    _error: BaseException | None = None

    def wait(self) -> BaseException | None:
        """Block until the read finishes; return its error, or ``None`` on success."""
        self._done.wait()
        return self._error


@dataclass(slots=True)
class _ReadRequest:
    """One outstanding read, internal to :class:`BlockFetcher`.
    The fetcher's worker reads block ``idx`` into the caller-carved ``buffer`` and
    updates ``handle`` (its error, then its event) once the read has finished.
    """

    idx: int
    buffer: Buffer
    handle: FetchHandle


class BlockFetcher:
    """Fills caller-supplied buffers on a worker thread."""

    def __init__(self, reader: DiskBlockReader) -> None:
        self._reader = reader
        self._request_queue: queue.SimpleQueue[_ReadRequest | None] = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._run, name="BlockFetcher-IO", daemon=True)
        self._worker.start()

    def submit(self, idx: int, buffer: Buffer) -> FetchHandle:
        """Enqueue a read of block *idx* into the caller-carved *buffer*, return its handle."""
        handle = FetchHandle(_done=threading.Event())
        request = _ReadRequest(idx=idx, buffer=buffer, handle=handle)
        self._request_queue.put(request)
        return handle

    def cleanup(self) -> None:
        """Drain pending reads, join the worker, close the reader."""
        self._request_queue.put(None)
        self._worker.join()
        self._reader.cleanup()

    def _run(self) -> None:
        # Pinned buffers are allocated under the caller's inference_mode, so
        # in-place copy_ from this thread requires inference_mode here too.
        with torch.inference_mode():
            while True:
                request = self._request_queue.get()
                if request is None:
                    return

                try:
                    self._reader.read_into(request.buffer, request.idx)
                except Exception as exc:
                    logger.exception("BlockFetcher: fetch failed for item %d", request.idx)
                    request.handle._error = exc
                request.handle._done.set()
