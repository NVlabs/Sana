"""Shared utilities for the block_streaming package."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import nn

from ltx_core.loader.primitives import TensorLayout

FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})

_BUFFER_ALIGN = 16


def make_block_key(blocks_prefix: str, block_idx: int, param_name: str) -> str:
    """Return the state-dict key for *param_name* under block *block_idx*."""
    return f"{blocks_prefix}.{block_idx}.{param_name}"


def resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


def assign_tensor_to_module(root: nn.Module, dotted_name: str, tensor: torch.Tensor) -> None:
    """Assign *tensor* to the parameter/buffer at *dotted_name* inside *root*.
    Unlike ``param.data = tensor``, this works even when the existing parameter
    lives on the ``meta`` device (which has an incompatible storage type).
    """
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = tensor
    else:
        raise AttributeError(f"{leaf} is not a parameter or buffer of {type(parent).__name__}")


def derive_layout(tensors: dict[str, torch.Tensor], dtype: torch.dtype | None = None) -> TensorLayout:
    """Derive a layout from a ``{name: tensor}`` dict.
    If ``dtype`` is given, non-FP8 dtypes are coerced to it (FP8 preserved). If
    ``None``, the source dtype is preserved as-is.
    """
    return {
        name: (t.shape, t.dtype if dtype is None or t.dtype in FP8_DTYPES else dtype) for name, t in tensors.items()
    }


def _align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def _alloc_pinned_exact(nbytes: int) -> torch.Tensor | None:
    """Allocate exactly ``nbytes`` of pinned host memory via ``cudaHostRegister``.
    Bypasses PyTorch's ``CachingHostAllocator``, which rounds every
    ``pin_memory=True`` request up to ``PowerOf2Ceil(N)`` (see
    ``aten/src/ATen/core/CachingHostAllocator.h``). Returns ``None`` if
    registration fails. The unregister hook is bound to the storage (not the
    tensor) so views of the buffer keep the registration alive until the
    memory is actually freed. Caller is responsible for ensuring CUDA is
    available.
    """
    cudart = torch.cuda.cudart()
    buf = torch.empty(nbytes, dtype=torch.uint8)
    ptr = buf.data_ptr()
    err = int(cudart.cudaHostRegister(ptr, nbytes, 0))
    if err != 0:
        return None
    weakref.finalize(buf.untyped_storage(), lambda p=ptr: cudart.cudaHostUnregister(p))
    return buf


def alloc_buffer(nbytes: int, device: torch.device | None, pin_memory: bool) -> torch.Tensor:
    """Allocate one ``uint8`` buffer for :func:`allocate_layout_views`.
    For pinned host buffers, prefer ``cudaHostRegister`` to dodge the caching
    allocator's power-of-2 rounding. Falls back to the caching allocator if
    registration fails. Pinning is fundamentally a CUDA driver operation; when
    requested without a CUDA runtime (e.g. on MPS/CPU, where H2D copies are
    synchronous and pinning is meaningless) it degrades to a normal allocation.
    """
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False  # pinning is CUDA-only; degrade gracefully off-CUDA
    if pin_memory and (device is None or torch.device(device).type == "cpu"):
        buf = _alloc_pinned_exact(nbytes)
        if buf is not None:
            return buf
    return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=pin_memory)


@dataclass(frozen=True)
class _TensorSlice:
    """Location of a single tensor view within the buffer."""

    offset: int
    shape: torch.Size
    dtype: torch.dtype

    def size(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


class LayoutSlices(NamedTuple):
    """Per-key tensor slices of a layout plus the total aligned buffer size."""

    slices: dict[str, _TensorSlice]
    nbytes: int


def _layout_slices(layout: TensorLayout) -> LayoutSlices:
    """Compute the byte offset of each key in *layout* and the total aligned size.
    The size is at least one byte so empty layouts still produce a valid buffer.
    """
    slices: dict[str, _TensorSlice] = {}
    cursor = 0
    for key, (shape, dtype) in layout.items():
        cursor = _align_up(cursor, _BUFFER_ALIGN)
        slices[key] = _TensorSlice(offset=cursor, shape=shape, dtype=dtype)
        cursor += slices[key].size()
    return LayoutSlices(slices, max(_align_up(cursor, _BUFFER_ALIGN), 1))


def layout_nbytes(layout: TensorLayout) -> int:
    """Byte size of one contiguous, 16-byte-aligned buffer holding *layout* (>= 1)."""
    return _layout_slices(layout).nbytes


def carve_buffer(buffer: torch.Tensor, layout: TensorLayout) -> dict[str, torch.Tensor]:
    """Carve per-key tensor views for *layout* into the front of *buffer*.
    *buffer* is a 1-D ``uint8`` tensor at least :func:`layout_nbytes` long. Each
    returned tensor is a non-overlapping slice of its leading bytes reinterpreted
    at the requested shape and dtype; any trailing bytes are left unused. That
    slack is what lets one max-sized pool slot hold a smaller (heterogeneous)
    block. The views keep *buffer*'s storage alive via PyTorch refcounting.
    """
    if buffer.dtype != torch.uint8 or buffer.dim() != 1:
        raise ValueError(f"carve_buffer expects a 1-D uint8 buffer, got {buffer.dim()}-D {buffer.dtype}")
    slices, nbytes = _layout_slices(layout)
    if buffer.numel() < nbytes:
        raise ValueError(f"buffer too small to carve layout: need {nbytes} bytes, got {buffer.numel()}")
    return {key: buffer[s.offset : s.offset + s.size()].view(s.dtype).view(s.shape) for key, s in slices.items()}


def allocate_layout_views(
    layout: TensorLayout,
    device: torch.device | None = None,
    pin_memory: bool = False,
) -> dict[str, torch.Tensor]:
    """Allocate a single ``uint8`` buffer and return per-key tensor views into it.
    All keys in *layout* live in one contiguous allocation; each returned
    tensor is a non-overlapping slice of that buffer reinterpreted at the
    requested shape and dtype. The views keep the underlying storage alive
    via PyTorch refcounting — drop them all to release the memory.
    """
    buffer = alloc_buffer(layout_nbytes(layout), device, pin_memory)
    return carve_buffer(buffer, layout)
