"""Store-based broadcast signaling between the relay rank and workers.
Workers poll an incrementing counter in the process group's store instead of
blocking directly on a collective, so an idle fleet doesn't trip the NCCL
watchdog timeout while waiting for the next job.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch.distributed as dist


class BroadcastCoordinator:
    """Coordinates broadcast signaling between the relay and workers via a distributed store.
    Uses an incrementing counter to signal when a broadcast is ready. Workers
    poll for counter changes to avoid NCCL timeout issues during idle periods.
    The counter wraps at a large value to prevent overflow.
    """

    def __init__(self, store: dist.Store, is_driver: bool) -> None:
        """Initialize the broadcast coordinator.
        Args:
            store: Distributed store (TCPStore, PrefixStore, etc.) for coordination across ranks.
            is_driver: Whether this rank is the relay (signals broadcasts).
        """
        self.store = store
        self.is_driver = is_driver
        self.broadcast_key = "dist_pipeline_broadcast_ready"
        self.signal = 0
        self.signal_max = 2**31 - 1  # ~2.1 billion calls before wraparound

        if is_driver:
            self.store.set(self.broadcast_key, str(self.signal))

        self.last_seen_signal = self.get_current_signal() if not is_driver else 0

    @contextmanager
    def broadcast_context(self) -> Iterator[None]:
        """Signal a broadcast (driver only).
        Increments the signal counter on entry to notify workers that a
        broadcast is ready. The counter stays incremented for change detection.
        Raises:
            RuntimeError: If called on a non-driver rank.
        """
        if not self.is_driver:
            raise RuntimeError("broadcast_context can only be called on driver rank")
        self.signal = (self.signal + 1) % self.signal_max
        self.store.set(self.broadcast_key, str(self.signal))
        yield

    def wait_for_signal_change(self, poll_interval: float = 0.01) -> None:
        """Wait until the signal changes from the last seen value (worker only).
        Args:
            poll_interval: Seconds to sleep between polls (default: 0.01).
        """
        while True:
            current = self.get_current_signal()
            if current != self.last_seen_signal:
                self.last_seen_signal = current
                return
            time.sleep(poll_interval)

    def get_current_signal(self) -> int:
        """Get the current signal value from the store."""
        return int(self.store.get(self.broadcast_key).decode("utf-8"))
