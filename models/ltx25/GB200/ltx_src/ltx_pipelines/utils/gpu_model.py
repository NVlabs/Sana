from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.devices import synchronize_device
from ltx_core.model.disposable import DisposableProtocol
from ltx_pipelines.utils.helpers import cleanup_memory

_M = TypeVar("_M", bound=DisposableProtocol)


@contextmanager
def gpu_model(model: _M, alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.
    Always calls ``model.dispose()`` so parameter / persistent-buffer storage
    moves to meta (shell-safe; fused LoRA weights do not linger on a cached
    module). ``TRIM`` (default) also synchronizes and ``cleanup_memory()`` to
    return cached blocks to the OS. ``DEFER`` skips sync / ``empty_cache`` so
    the CUDA caching allocator stays warm for the next build.
    Usage::
        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder -- typed as the concrete class
        # parameter storage released; TRIM also returns cached blocks to the OS
    """
    try:
        yield model
    finally:
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            synchronize_device()
            model.dispose()
            cleanup_memory()
        else:
            model.dispose()
