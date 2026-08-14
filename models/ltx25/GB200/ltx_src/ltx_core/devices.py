"""Device abstraction for CUDA, Apple Silicon (MPS), and CPU backends.
Centralizes backend detection and the handful of APIs that genuinely differ
across accelerators (synchronization, allocator cache, memory queries, RNG
state). Selection order is CUDA -> MPS -> CPU.
CUDA-only optimizations (FlashAttention, Triton blockwise FP8/FP6,
bitsandbytes, NCCL) are gated at their call sites, not here. MPS in particular
has no ``float64`` support and no fp8 dtype support; use
:func:`highest_precision_float` to stay within what the backend can represent.
"""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)

DeviceSpec = torch.device | None


def is_mps_available() -> bool:
    """Return whether PyTorch can use the Apple Metal/MPS backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def get_preferred_device(local_rank: int | None = None) -> torch.device:
    """Prefer CUDA, then MPS, then CPU.
    ``local_rank`` is only meaningful for CUDA multi-process launches. MPS exposes
    a single logical device in PyTorch, so rank-based indexing is not used there.
    """
    if torch.cuda.is_available():
        index = torch.cuda.current_device() if local_rank is None else local_rank
        return torch.device("cuda", index)
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: DeviceSpec = None, *, local_rank: int | None = None) -> torch.device:
    """Return *device*, or the best available accelerator when it is ``None``."""
    if device is None:
        return get_preferred_device(local_rank=local_rank)
    return device


def supports_float64(device: DeviceSpec) -> bool:
    """Return whether *device* can represent ``torch.float64``.
    MPS has no double-precision support; CUDA and CPU do.
    """
    return resolve_device(device).type != "mps"


def highest_precision_float(device: DeviceSpec) -> torch.dtype:
    """Return the widest float the backend supports: ``float64`` on CUDA/CPU,
    ``float32`` on MPS.
    Use for numerically sensitive accumulators (e.g. sampler ODE math) that
    request double precision but must degrade gracefully on MPS.
    """
    return torch.float64 if supports_float64(device) else torch.float32


def synchronize_device(device: DeviceSpec = None) -> None:
    """Synchronize CUDA or MPS work if the selected backend supports it."""
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and is_mps_available():
        torch.mps.synchronize()


def empty_device_cache(device: DeviceSpec = None) -> None:
    """Release cached allocator memory for CUDA or MPS."""
    resolved = resolve_device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and is_mps_available():
        torch.mps.empty_cache()


def cuda_activation_budget_bytes(device: DeviceSpec = None) -> int:
    """Bytes still available for new CUDA allocations in this process.
    Honors ``torch.cuda.set_per_process_memory_fraction``: ``mem_get_info`` reports
    raw device free memory and does *not* shrink under a fraction cap, so we use
    ``fraction * device_total - memory_reserved`` (also capped by raw free).
    """
    if not torch.cuda.is_available():
        return 0
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return 0
    idx = resolved.index if resolved.index is not None else torch.cuda.current_device()
    free_raw, _total_raw = torch.cuda.mem_get_info(idx)
    props_total = torch.cuda.get_device_properties(idx).total_memory
    fraction = torch.cuda.get_per_process_memory_fraction(idx)
    reserved = torch.cuda.memory_reserved(idx)
    under_fraction = max(0, int(fraction * props_total) - int(reserved))
    return min(int(free_raw), under_fraction)


def cleanup_accelerator_memory(device: DeviceSpec = None) -> None:
    """Run Python GC and release CUDA/MPS allocator caches."""
    gc.collect()
    empty_device_cache(device)
    synchronize_device(device)
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logger.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)
