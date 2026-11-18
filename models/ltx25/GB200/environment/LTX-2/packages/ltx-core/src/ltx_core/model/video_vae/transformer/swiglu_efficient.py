# ruff: noqa: ANN001, N803, ANN202, PLR0913

"""SwiGLU kernels: plain ``swiglu`` (reference) and memory-efficient tiled / Triton paths.
``swiglu`` materializes full gate/up intermediates — tests only.
``swiglu_tiled`` / ``swiglu_triton`` take the full activation and stream tokens in
chunks with a single reusable ``(chunk, hidden)`` workspace (donor
``swiglu_memory_efficient`` pattern). Both wrap the chunk loop in
``torch.library.custom_op`` so ``torch.compile`` sees one opaque node (no Dynamo
unroll of hundreds of Python iterations).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F

from ltx_core.tiling import DimensionInterval, split_by_count, split_by_size

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

# Match visual_fidelity ``DEFAULT_CHUNK_TOKENS`` — hard token cap for peak VRAM.
DEFAULT_SWIGLU_TILE_SIZE: Final[int] = 16_384
DEFAULT_SWIGLU_TILES: Final[int] = 4


@dataclass(frozen=True)
class SwiGLUTileSpec:
    """Token tiling: provide exactly one of ``num_tiles`` or ``tile_size``."""

    num_tiles: int | None = None
    tile_size: int | None = None

    def __post_init__(self) -> None:
        if (self.num_tiles is None) == (self.tile_size is None):
            raise ValueError("Provide exactly one of num_tiles or tile_size")
        if self.num_tiles is not None and self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.tile_size is not None and self.tile_size < 1:
            raise ValueError(f"tile_size must be >= 1, got {self.tile_size}")

    @classmethod
    def by_count(cls, num_tiles: int = DEFAULT_SWIGLU_TILES) -> SwiGLUTileSpec:
        return cls(num_tiles=num_tiles)

    @classmethod
    def by_size(cls, tile_size: int = DEFAULT_SWIGLU_TILE_SIZE) -> SwiGLUTileSpec:
        return cls(tile_size=tile_size)


DEFAULT_SWIGLU_TILE_SPEC = SwiGLUTileSpec(tile_size=DEFAULT_SWIGLU_TILE_SIZE)


def triton_swiglu_available() -> bool:
    """True when a CUDA-capable Triton install can run the fused up-mul kernel."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def token_intervals(n_tok: int, tile: SwiGLUTileSpec) -> list[DimensionInterval]:
    """Split ``n_tok`` tokens per ``tile`` (count or size). Overlap is always 0."""
    if n_tok < 0:
        raise ValueError(f"n_tok must be >= 0, got {n_tok}")
    if n_tok == 0:
        return []
    if tile.num_tiles is not None:
        tiles = min(tile.num_tiles, n_tok)
        if tiles <= 1:
            return [DimensionInterval(start=0, end=n_tok, left_ramp=0, right_ramp=0)]
        return list(split_by_count(num_tiles=tiles, overlap=0)(n_tok).intervals)
    assert tile.tile_size is not None
    if n_tok <= tile.tile_size:
        return [DimensionInterval(start=0, end=n_tok, left_ramp=0, right_ramp=0)]
    return list(split_by_size(tile.tile_size, overlap=0)(n_tok).intervals)


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_up_mul_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_gm,
        stride_gn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``g = g * (x @ w.T)`` with ``w`` shaped ``(N, K)`` (nn.Linear layout)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w)

        g = tl.load(
            g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        out = g * acc.to(g.dtype)
        tl.store(
            g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            out,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _fused_up_mul_triton(x: torch.Tensor, w_up: torch.Tensor, silu_gate: torch.Tensor) -> None:
    """Inplace: ``silu_gate *= (x @ w_up.T)``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    m, k = x.shape
    n = w_up.shape[0]
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_up_mul_kernel[grid](
        x,
        w_up,
        silu_gate,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        w_up.stride(0),
        w_up.stride(1),
        silu_gate.stride(0),
        silu_gate.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )


def _fused_up_mul_torch(x: torch.Tensor, w_up: torch.Tensor, silu_gate: torch.Tensor) -> None:
    """PyTorch fallback: temporary ``up`` chunk, then inplace mul into ``silu_gate``."""
    up = F.linear(x, w_up)
    silu_gate.mul_(up)


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Reference: ``w_down(silu(x@W_gateᵀ) * (x@W_upᵀ))`` (full materialization)."""
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def _intervals_from_starts_ends(starts: torch.Tensor, ends: torch.Tensor) -> list[DimensionInterval]:
    return [
        DimensionInterval(start=int(s), end=int(e), left_ramp=0, right_ramp=0)
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True)
    ]


def _starts_ends_tensors(intervals: list[DimensionInterval], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.tensor([iv.start for iv in intervals], device=device, dtype=torch.int64)
    ends = torch.tensor([iv.end for iv in intervals], device=device, dtype=torch.int64)
    return starts, ends


def _tile_from_op_args(tile_size: int, num_tiles: int) -> SwiGLUTileSpec:
    """Decode custom_op int knobs: exactly one of ``tile_size`` / ``num_tiles`` is > 0."""
    if (tile_size > 0) == (num_tiles > 0):
        raise ValueError(f"need exactly one of tile_size>0 or num_tiles>0, got {tile_size=}, {num_tiles=}")
    if tile_size > 0:
        return SwiGLUTileSpec(tile_size=tile_size)
    return SwiGLUTileSpec(num_tiles=num_tiles)


def _swiglu_chunked_impl(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    intervals: list[DimensionInterval],
    *,
    use_triton: bool,
) -> torch.Tensor:
    """Chunked SwiGLU: one reusable ``(chunk, hidden)`` workspace (runs via custom_op)."""
    if use_triton and not triton_swiglu_available():
        raise RuntimeError("use_triton=True but Triton/CUDA is unavailable")

    leading = x.shape[:-1]
    dim = x.shape[-1]
    x_flat = x.reshape(-1, dim).contiguous()
    n_tok = x_flat.shape[0]
    hidden = w_gate.shape[0]
    if w_up.shape[0] != hidden or w_gate.shape[1] != dim or w_up.shape[1] != dim:
        raise ValueError(
            f"weight shapes incompatible with x[..., {dim}]: gate={tuple(w_gate.shape)} up={tuple(w_up.shape)}"
        )
    if w_down.shape[1] != hidden or w_down.shape[0] != dim:
        raise ValueError(f"w_down {tuple(w_down.shape)} incompatible with hidden={hidden} dim={dim}")
    if n_tok == 0:
        return x

    out_flat = torch.empty((n_tok, dim), device=x.device, dtype=x.dtype)
    max_chunk = max((iv.end - iv.start for iv in intervals), default=0)
    workspace = torch.empty((max_chunk, hidden), device=x.device, dtype=x.dtype)
    w_up_c = w_up.contiguous() if use_triton else w_up

    for iv in intervals:
        start, end = iv.start, iv.end
        if start >= end:
            continue
        xc = x_flat[start:end]
        ws = workspace[: end - start]
        torch.mm(xc, w_gate.t(), out=ws)
        F.silu(ws, inplace=True)
        if use_triton:
            _fused_up_mul_triton(xc, w_up_c, ws)
        else:
            _fused_up_mul_torch(xc, w_up, ws)
        torch.mm(ws, w_down.t(), out=out_flat[start:end])

    return out_flat.view(*leading, dim)


@torch.library.custom_op("ltx_core::swiglu_tiled", mutates_args=())
def _swiglu_tiled_op(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    num_tiles: int,
    use_triton: bool,
) -> torch.Tensor:
    """Opaque op: memory-efficient chunked SwiGLU (compile-friendly; no Dynamo unroll)."""
    tile = _tile_from_op_args(tile_size, num_tiles)
    n_tok = x.reshape(-1, x.shape[-1]).shape[0]
    return _swiglu_chunked_impl(x, w_gate, w_up, w_down, token_intervals(n_tok, tile), use_triton=use_triton)


@_swiglu_tiled_op.register_fake
def _swiglu_tiled_fake(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    num_tiles: int,
    use_triton: bool,
) -> torch.Tensor:
    del w_gate, w_up, w_down, tile_size, num_tiles, use_triton
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def swiglu_tiled(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile: SwiGLUTileSpec,
    *,
    use_triton: bool | None = None,
) -> torch.Tensor:
    """Memory-efficient SwiGLU on the full activation; tiling is internal (custom_op).
    Peak hidden activation is ``O(chunk · hidden)`` (one reusable workspace), not
    ``O(N · hidden)`` from materializing full gate/up tensors.
    """
    tile_size = int(tile.tile_size) if tile.tile_size is not None else 0
    num_tiles = int(tile.num_tiles) if tile.num_tiles is not None else 0
    if use_triton is None:
        use_triton = triton_swiglu_available() and x.is_cuda
    return _swiglu_tiled_op(x, w_gate, w_up, w_down, tile_size, num_tiles, bool(use_triton))


def _swiglu_triton_impl(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    intervals: list[DimensionInterval],
) -> torch.Tensor:
    """Triton-fused SwiGLU with internal token chunking (runs via custom_op)."""
    if not triton_swiglu_available():
        raise RuntimeError("Triton/CUDA is unavailable for SwiGLUMode.TRITON")
    return _swiglu_chunked_impl(x, w_gate, w_up, w_down, intervals, use_triton=True)


@torch.library.custom_op("ltx_core::swiglu_triton", mutates_args=())
def _swiglu_triton_op(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Opaque op: full-tensor Triton SwiGLU with internal chunks (compile-friendly)."""
    return _swiglu_triton_impl(x, w_gate, w_up, w_down, _intervals_from_starts_ends(starts, ends))


@_swiglu_triton_op.register_fake
def _swiglu_triton_fake(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    del w_gate, w_up, w_down, starts, ends
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def swiglu_triton(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile: SwiGLUTileSpec,
) -> torch.Tensor:
    """Triton SwiGLU on the full activation; tiling is internal to this path."""
    n_tok = x.reshape(-1, x.shape[-1]).shape[0]
    intervals = token_intervals(n_tok, tile)
    if not intervals:
        return x
    starts, ends = _starts_ends_tensors(intervals, x.device)
    return _swiglu_triton_op(x, w_gate, w_up, w_down, starts, ends)
