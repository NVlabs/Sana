"""GPU weights provider for block streaming."""

from __future__ import annotations

from collections import OrderedDict
from typing import NamedTuple

import torch

from ltx_core.block_streaming.disk import LoraSource
from ltx_core.block_streaming.pool import BufferPool
from ltx_core.block_streaming.source import WeightSource
from ltx_core.block_streaming.stream_sync import StreamEvent, StreamSync
from ltx_core.block_streaming.utils import carve_buffer, layout_nbytes
from ltx_core.loader.fuse_loras import FuseRule, aggregate_lora_products, bf16_fuse_rule, device_fuse_rule
from ltx_core.loader.primitives import StateDict

_EMPTY_STATE_DICT = StateDict(sd={}, device=torch.device("cpu"), size=0, dtype=set())


class CachedBlock(NamedTuple):
    """A cached GPU block: the raw pool slot plus the carved per-key views.
    The raw slot is what is returned to the pool on eviction; the views are
    what callers consume.
    """

    raw: torch.Tensor
    views: dict[str, torch.Tensor]


class WeightsProvider:
    """Provides GPU-ready block weights via H2D copy from a pinned CPU weight source.
    Args:
        pool: Pre-allocated GPU weight buffer pool.
        sync: Coordinates copy-vs-compute ordering for the backend
            (see :class:`StreamSync`).
        target_device: device for compute.
        source: Pinned CPU weight source.
        lora_sources: LoRA adapters fused on H2D copy.
        blocks_prefix: State-dict prefix for LoRA key matching.
        fuse_rule: Per-policy LoRA merge rule (must be streaming-compatible:
            no companion-key emission). Defaults to ``bf16_fuse_rule``.
    """

    def __init__(
        self,
        pool: BufferPool,
        sync: StreamSync,
        target_device: torch.device,
        source: WeightSource,
        lora_sources: list[LoraSource] | None = None,
        blocks_prefix: str = "",
        fuse_rule: FuseRule = bf16_fuse_rule,
    ) -> None:
        self._sync = sync
        self._pool = pool
        self._cache: OrderedDict[int, CachedBlock] = OrderedDict()
        self._events: dict[int, StreamEvent | None] = {}
        self._target_device = target_device
        self._source = source
        self._lora_sources = lora_sources or []
        self._blocks_prefix = blocks_prefix
        self._fuse_rule = fuse_rule

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return GPU weights for block *idx*. Does H2D copy on miss."""
        if idx in self._cache:
            return self._cache[idx].views

        # Evict oldest GPU buffer if at capacity.
        if len(self._cache) >= self._pool.capacity:
            evicted_idx, evicted = self._cache.popitem(last=False)
            self._pool.release(evicted.raw, event=self._events.pop(evicted_idx, None))

        layout = self._source.block_layout(idx)
        raw = self._pool.acquire()
        gpu_weights = carve_buffer(raw, layout)
        cpu_buffer = self._source.get(idx)

        h2d_event = self._copy_to_gpu(idx, raw, gpu_weights, cpu_buffer, layout_nbytes(layout))
        self._source.release(idx, event=h2d_event)

        self._cache[idx] = CachedBlock(raw, gpu_weights)
        return gpu_weights

    def _copy_to_gpu(
        self,
        idx: int,
        raw: torch.Tensor,
        gpu_weights: dict[str, torch.Tensor],
        cpu_buffer: torch.Tensor,
        nbytes: int,
    ) -> StreamEvent | None:
        """Copy block weights to the target device and fuse LoRAs.
        *cpu_buffer* is one contiguous source buffer carved by the same layout as
        *raw*, so a single byte copy of its leading *nbytes* reproduces every view
        in *gpu_weights*.
        The copy + fusion run under :meth:`StreamSync.copy_scope`, then
        :meth:`StreamSync.commit_copy` orders the copy before compute and returns
        a guard event for the source to reuse (the ordering is committed inside
        this method so callers -- and instrumentation regions wrapping it --
        observe the full transfer time).
        """
        if not cpu_buffer.is_contiguous() or cpu_buffer.dtype != torch.uint8 or cpu_buffer.numel() < nbytes:
            raise ValueError(
                f"source buffer for block {idx} must be a contiguous uint8 buffer of >= {nbytes} bytes, "
                f"got {cpu_buffer.dim()}-D {cpu_buffer.dtype} with {cpu_buffer.numel()} elements"
            )
        with self._sync.copy_scope():
            raw[:nbytes].copy_(cpu_buffer[:nbytes], non_blocking=self._sync.is_async_copy)
            if self._lora_sources:
                self._fuse_block_loras(idx, gpu_weights)

        return self._sync.commit_copy()

    def release(self, idx: int, event: StreamEvent | None) -> None:
        """Attach a compute-done guard, waited before this buffer is recycled
        (``None`` when the backend needs no guard)."""
        self._events[idx] = event

    def mark_block_done(self, idx: int) -> None:
        """Record a compute-done guard for block *idx* and queue it for slot reuse.
        Called once the block's forward pass has been enqueued, so the buffer is
        not overwritten by a later copy until this compute completes."""
        self.release(idx, self._sync.record_compute_done())

    def cleanup(self) -> None:
        """Drain outstanding copy/compute work and release all resources."""
        self._sync.synchronize()
        self._cache.clear()
        self._events.clear()
        self._source.cleanup()
        for lora in self._lora_sources:
            lora.cleanup()

    def __len__(self) -> int:
        return len(self._cache)

    def _fuse_block_loras(self, idx: int, weights: dict[str, torch.Tensor]) -> None:
        """Fuse LoRA deltas directly into GPU block weights via ``fuse_rule``.
        The fusion device+dtype come from :func:`device_fuse_rule`: on MPS it
        aggregates the ``B@A`` on the GPU in fp32 (fast, and fp32 avoids the
        bf16-on-MPS unreliability), with the rule casting back to the weight
        dtype; CUDA/CPU keep the rule's dtype. ``get_ab`` places A/B on the
        target device for CUDA/MPS, so aggregation and the in-place fuse stay
        co-located there.
        """
        rule = device_fuse_rule(self._target_device, self._fuse_rule)
        for name, tensor in weights.items():
            if not name.endswith(".weight"):
                continue
            prefix = f"{self._blocks_prefix}.{idx}.{name}".removesuffix(".weight")
            products = (
                ab
                for ab in (
                    s.get_ab(prefix, device=self._target_device, dtype=rule.aggregation_dtype)
                    for s in self._lora_sources
                )
                if ab is not None
            )
            deltas = aggregate_lora_products(products, rule.aggregation_dtype)
            if deltas is None:
                continue
            fused = rule(name, tensor, deltas, _EMPTY_STATE_DICT)
            tensor.copy_(fused[name])
