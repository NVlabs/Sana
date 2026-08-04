"""Public Sol-Attn interface."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

from .common import to_cute_tensor
from .preprocess import BLOCK_SIZE, prepare


_compiled = {}


def _validate(
    q,
    k,
    v,
    kv_splits,
    thresh_type,
    sink_tokens=0,
    sink_start=None,
):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous BTHD tensors")
    if kv_splits not in (1, 2, 4):
        raise ValueError("kv_splits must be 1, 2, or 4")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    if not isinstance(sink_tokens, int):
        raise TypeError("sink_tokens must be an integer")
    if not 0 <= sink_tokens <= q.shape[1]:
        raise ValueError("sink_tokens must be in [0, T]")
    if sink_start is not None:
        if not isinstance(sink_start, int):
            raise TypeError("sink_start must be an integer or None")
        if not 0 <= sink_start <= q.shape[1]:
            raise ValueError("sink_start must be in [0, T]")
        if sink_start + sink_tokens > q.shape[1]:
            raise ValueError("sink_start + sink_tokens must be <= T")

    arch = torch.cuda.get_device_capability(q.device)
    if arch not in ((9, 0), (10, 0), (12, 0)):
        raise RuntimeError(
            "Sol-Attn supports H100 (SM90), B200 (SM100), and RTX 5090 (SM120)"
        )
    if arch != (9, 0) and kv_splits != 1:
        raise ValueError("kv_splits=2/4 is currently available on SM90 only")
    route_groups = ((q.shape[1] + 63) // 64 + 63) // 64
    if kv_splits > route_groups:
        raise ValueError("each KV split must contain at least one N64 route group")
    return arch


def _stream(device):
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _sink_block_range(tokens, sink_start, sink_tokens):
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return (
        start // BLOCK_SIZE,
        (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )


def _compile_sm90(
    key,
    tensors,
    scale,
    tokens,
    kv_splits,
    sink_range,
    stream,
):
    from .sm90 import make_kernel

    operator = make_kernel(tokens, kv_splits)
    args = [to_cute_tensor(x) for x in tensors]
    compiled = cute.compile(
        operator,
        *args,
        scale,
        sink_range,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _compile_sm100(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
):
    from .sm100 import forward

    args = [to_cute_tensor(x) for x in tensors]
    compiled = cute.compile(
        forward,
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _compile_sm120(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
):
    from .sm120 import make_kernel

    operator = make_kernel()
    args = [to_cute_tensor(x) for x in tensors]
    compiled = cute.compile(
        operator,
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    kv_splits: int = 1,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> torch.Tensor:
    """Compute noncausal Sol-Attn for contiguous BF16 BTHD tensors.

    ``sink_start`` and ``sink_tokens`` keep every KV block overlapping the
    corresponding contiguous token range exact for all queries. Omitting
    ``sink_start`` places the range at the token suffix.
    """

    arch = _validate(
        q,
        k,
        v,
        kv_splits,
        thresh_type,
        sink_tokens,
        sink_start,
    )
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, _ = q.shape

    with torch.cuda.device(q.device):
        kc, vc, threshold = prepare(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
        )
        output = torch.empty_like(v)
        lse = torch.empty(
            (batch, tokens, heads),
            device=q.device,
            dtype=torch.float32,
        )
        stream = _stream(q.device)
        key = (q.device.index, arch, batch, tokens, heads, kv_splits)

        if arch == (9, 0):
            if sink_tokens:
                sink_start_block, sink_end_block = _sink_block_range(
                    tokens,
                    sink_start,
                    sink_tokens,
                )
                sink_range = sink_start_block | (sink_end_block << 16)
            else:
                sink_range = 0
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            if kv_splits > 1:
                tensors.extend(
                    [
                        torch.empty(
                            (batch, tokens, kv_splits * heads, 128),
                            device=q.device,
                            dtype=torch.bfloat16,
                        ),
                        torch.empty(
                            (batch, tokens, kv_splits * heads),
                            device=q.device,
                            dtype=torch.float32,
                        ),
                    ]
                )
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm90(
                    key,
                    tensors,
                    scale,
                    tokens,
                    kv_splits,
                    sink_range,
                    stream,
                )
            else:
                args = [to_cute_tensor(x) for x in tensors]
            compiled(
                *args,
                scale,
                sink_range,
                stream=stream,
            )
        elif arch == (10, 0):
            sink_start_block, sink_end_block = _sink_block_range(
                tokens,
                sink_start,
                sink_tokens,
            )
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm100(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                )
            else:
                args = [to_cute_tensor(x) for x in tensors]
            compiled(
                *args,
                scale,
                sink_start_block,
                sink_end_block,
                stream=stream,
            )
        else:
            sink_start_block, sink_end_block = _sink_block_range(
                tokens,
                sink_start,
                sink_tokens,
            )
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm120(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                )
            else:
                args = [to_cute_tensor(x) for x in tensors]
            compiled(
                *args,
                scale,
                sink_start_block,
                sink_end_block,
                stream=stream,
            )
    return output


__all__ = ["sol_attn"]
