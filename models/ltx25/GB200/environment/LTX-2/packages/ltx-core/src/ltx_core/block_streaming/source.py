"""Weight sources for block streaming: protocol and implementations."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch

from ltx_core.block_streaming.block_fetcher import BlockFetcher, FetchHandle
from ltx_core.block_streaming.pool import BufferPool
from ltx_core.block_streaming.stream_sync import StreamEvent
from ltx_core.block_streaming.utils import carve_buffer, layout_nbytes
from ltx_core.loader.primitives import TensorLayout


class WeightSource(Protocol):
    """Provides pinned CPU weights for a given block index.
    Blocks may be heterogeneous: each has its own layout, so the source exposes a
    per-block layout and the byte size of the largest block (which sizes the
    pool slots -- a smaller block is carved into the front of a max-sized slot).
    The source is the single source of truth for each block's layout.
    """

    def block_layout(self, idx: int) -> TensorLayout:
        """Per-block buffer layout (shape + dtype for each param)."""
        ...

    @property
    def slot_nbytes(self) -> int:
        """Byte size of the largest block; sizes a pool slot (16-byte aligned)."""
        ...

    def get(self, idx: int) -> torch.Tensor:
        """Return one contiguous CPU buffer for block *idx*."""
        ...

    def release(self, idx: int, event: StreamEvent | None) -> None:
        """Signal that an async operation using these weights is guarded by *event*."""
        ...

    def cleanup(self) -> None:
        """Release all resources (buffers, readers, events)."""
        ...


class _Scheduled(NamedTuple):
    """A scheduled (possibly in-flight) read: the raw pool slot and its fetch handle."""

    raw: torch.Tensor
    status: FetchHandle


class DiskWeightSource(WeightSource):
    """WeightSource that streams blocks from disk via a :class:`BlockFetcher`.
    ``get(idx)`` must be paired with a ``release(idx)`` before *idx* is fetched
    again; getting a block that is already in flight raises. Each read acquires a
    raw pool slot, carves it to block *idx*'s layout, and hands the carved views to
    the fetcher to fill, so one max-sized slot can serve blocks of differing shapes.
    ``get`` returns that same contiguous slot.
    """

    def __init__(
        self,
        pool: BufferPool,
        fetcher: BlockFetcher,
        block_layouts: dict[int, TensorLayout],
        blocks_number: int,
        prefetch_depth: int = 0,
    ) -> None:
        if blocks_number <= 0:
            raise ValueError(f"blocks_number must be > 0, got {blocks_number}")
        if prefetch_depth < 0:
            raise ValueError(f"prefetch_depth must be >= 0, got {prefetch_depth}")
        max_layout_nbytes = max((layout_nbytes(layout) for layout in block_layouts.values()), default=0)
        if pool.slot_nbytes < max_layout_nbytes:
            raise ValueError(
                f"pool slot is too small for the largest block: slot {pool.slot_nbytes} bytes < {max_layout_nbytes}"
            )

        self._pool = pool
        self._blocks_number = blocks_number
        self._prefetch_depth = prefetch_depth
        self._fetcher = fetcher
        self._block_layouts = block_layouts
        self._scheduled: dict[int, _Scheduled] = {}
        self._in_flight: dict[int, torch.Tensor] = {}

    def block_layout(self, idx: int) -> TensorLayout:
        return self._block_layouts[idx]

    @property
    def slot_nbytes(self) -> int:
        return self._pool.slot_nbytes

    def get(self, idx: int) -> torch.Tensor:
        if idx in self._in_flight:
            raise RuntimeError(f"Block {idx} is already in flight; release it before getting it again")

        scheduled = self._scheduled.pop(idx, None)
        if scheduled is None:
            scheduled = self._schedule(idx)
        error = scheduled.status.wait()
        if error is not None:
            self._pool.release(scheduled.raw)
            raise error

        self._in_flight[idx] = scheduled.raw
        for k in range(1, self._prefetch_depth + 1):
            self._ensure_scheduled((idx + k) % self._blocks_number)
        return scheduled.raw

    def release(self, idx: int, event: StreamEvent | None) -> None:
        raw_buffer = self._in_flight.pop(idx)
        self._pool.release(raw_buffer, event=event)

    def cleanup(self) -> None:
        self._fetcher.cleanup()
        while self._in_flight:
            _, raw_buffer = self._in_flight.popitem()
            self._pool.release(raw_buffer)
        while self._scheduled:
            _, scheduled = self._scheduled.popitem()
            scheduled.status.wait()
            self._pool.release(scheduled.raw)

    def _ensure_scheduled(self, idx: int) -> None:
        """Schedule a read for *idx* if one is not already pending."""
        if idx not in self._scheduled:
            self._scheduled[idx] = self._schedule(idx)

    def _schedule(self, idx: int) -> _Scheduled:
        """Acquire a raw slot, carve it to block *idx*, enqueue a read, return the handle.
        The raw slot and its fetch status are returned together so the caller can
        track both as one unit; the fetcher only receives the carved views to fill.
        """
        raw_buffer = self._pool.acquire()
        carved = carve_buffer(raw_buffer, self._block_layouts[idx])
        status = self._fetcher.submit(idx, carved)
        return _Scheduled(raw_buffer, status)


class PinnedBlock(NamedTuple):
    """A pre-loaded pinned block: its single contiguous buffer and the layout to carve it with."""

    buffer: torch.Tensor
    layout: TensorLayout


class PinnedWeightSource(WeightSource):
    """Pre-loaded pinned CPU weights, one contiguous (possibly heterogeneous) buffer per block."""

    def __init__(self, blocks: dict[int, PinnedBlock]) -> None:
        if not blocks:
            raise ValueError("PinnedWeightSource requires at least one block")
        self._blocks = blocks
        self._slot_nbytes = max(layout_nbytes(block.layout) for block in blocks.values())

    def block_layout(self, idx: int) -> TensorLayout:
        return self._blocks[idx].layout

    @property
    def slot_nbytes(self) -> int:
        return self._slot_nbytes

    def get(self, idx: int) -> torch.Tensor:
        return self._blocks[idx].buffer

    def release(self, idx: int, event: StreamEvent | None) -> None:
        pass

    def cleanup(self) -> None:
        self._blocks.clear()

    def __len__(self) -> int:
        return len(self._blocks)
