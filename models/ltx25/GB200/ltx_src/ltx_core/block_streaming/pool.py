"""Raw buffer pool for block streaming."""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from ltx_core.block_streaming import utils
from ltx_core.block_streaming.stream_sync import StreamEvent


class BufferPool:
    """Fixed pool of pre-allocated raw buffer slots with event-based reuse.
    Slots are carved from a single contiguous ``uint8`` buffer; each is
    ``slot_nbytes`` long and handed out as a raw 1-D ``uint8`` tensor.
    Args:
        slot_nbytes: Byte size of each slot.
        capacity: Number of slots to pre-allocate.
        device: Device for allocation.
        reuse_barrier: Called with the pending event before a slot is reused.
        pin_memory: Pin buffers (for async H2D copies from CPU).
    """

    def __init__(
        self,
        slot_nbytes: int,
        capacity: int,
        device: torch.device,
        reuse_barrier: Callable[[StreamEvent], None],
        pin_memory: bool = False,
    ) -> None:
        self._slot_nbytes = slot_nbytes
        self._capacity = capacity
        self._free: deque[torch.Tensor] = deque()
        self._events: dict[int, StreamEvent] = {}
        self._reuse_barrier = reuse_barrier
        buffer = utils.alloc_buffer(max(slot_nbytes * capacity, 1), device, pin_memory)
        for slot in range(capacity):
            self._free.append(buffer[slot * slot_nbytes : (slot + 1) * slot_nbytes])

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def slot_nbytes(self) -> int:
        return self._slot_nbytes

    def acquire(self) -> torch.Tensor:
        """Take a free raw slot, waiting any pending event before returning.
        Raises :class:`RuntimeError` if every slot is currently in use.
        """
        if not self._free:
            raise RuntimeError(f"BufferPool exhausted: all {self._capacity} buffers are in use")
        buffer = self._free.popleft()
        event = self._events.pop(id(buffer), None)
        if event is not None:
            self._reuse_barrier(event)
        return buffer

    def release(self, buffer: torch.Tensor, event: StreamEvent | None = None) -> None:
        """Return a raw slot to the free list.
        The *buffer* must be the exact tensor object returned by :meth:`acquire`
        (reuse is keyed on its identity). If *event* is given it is waited on the
        next :meth:`acquire` of this slot, ensuring the prior operation finished.
        """
        if event is not None:
            self._events[id(buffer)] = event
        self._free.append(buffer)
