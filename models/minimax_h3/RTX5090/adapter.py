"""Sol-Attn adapter for MiniMax-H3 packed multimodal attention."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F


_LAYER_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.attn$")
_LOG_PREFIX = "MINIMAX_H3_SOL_ATTN "
_MAIN_DIT_LAYER_INDICES: set[int] = set()


@dataclass(frozen=True)
class StepContext:
    request_epoch: str
    request_index: int
    step_index: int
    total_tokens: int
    live_tokens: int
    text_start: int
    text_tokens: int
    cu_seqlens_host: tuple[int, ...]
    num_layers: int
    timestep_max: float | None


@dataclass
class _RuntimeState:
    context: StepContext | None = None
    request_index: int = -1
    previous_epoch: str | None = None
    previous_timestep_max: float | None = None
    density_logged_shapes: set[tuple[int, int, int]] = field(default_factory=set)
    gate_shapes: set[tuple[int, int, int]] = field(default_factory=set)
    first_sparse_logged_shapes: set[tuple[int, int, int]] = field(default_factory=set)
    dense_backend_logged: bool = False


_STATE = _RuntimeState()


def parse_layer_index(prefix: str) -> int | None:
    """Return the main DiT block index, excluding token-refiner blocks."""

    if "token_refiner" in prefix:
        return None
    match = _LAYER_PATTERN.search(prefix)
    if match is None:
        return None
    layer_index = int(match.group(1))
    _MAIN_DIT_LAYER_INDICES.add(layer_index)
    return layer_index


def _host_positions(values: torch.Tensor | Iterable[int]) -> tuple[int, ...]:
    if torch.is_tensor(values):
        values = values.detach().reshape(-1).to(device="cpu").tolist()
    return tuple(int(value) for value in values if int(value) >= 0)


def _contiguous_range(values: tuple[int, ...], name: str) -> tuple[int, int]:
    if not values:
        return 0, 0
    ordered = tuple(sorted(set(values)))
    start = ordered[0]
    if ordered != tuple(range(start, start + len(ordered))):
        raise ValueError(f"MiniMax-H3 {name} positions must be contiguous")
    return start, len(ordered)


def _normalize_boundaries(
    boundaries: Iterable[int] | None,
    total_tokens: int,
) -> tuple[int, ...]:
    if boundaries is None:
        return (0, total_tokens)
    result = tuple(int(value) for value in boundaries)
    if not result or result[0] != 0:
        raise ValueError(f"cu_seqlens_host must start at 0, got {result}")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError(f"cu_seqlens_host must be nondecreasing, got {result}")
    if result[-1] > total_tokens:
        raise ValueError(
            f"cu_seqlens_host ends at {result[-1]}, beyond T={total_tokens}"
        )
    if result[-1] < total_tokens:
        result = (*result, total_tokens)
    return result


def _request_epoch() -> str | None:
    path = os.getenv("SOL_ATTN_REQUEST_EPOCH_FILE")
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _timestep_max(unique_timesteps: torch.Tensor) -> float | None:
    if not torch.is_tensor(unique_timesteps) or not unique_timesteps.numel():
        return None
    return float(unique_timesteps.detach().float().max().to(device="cpu").item())


def _rank() -> int:
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def _emit(event: str, **payload: Any) -> None:
    record = {"event": event, "rank": _rank(), **payload}
    line = json.dumps(record, sort_keys=True)
    print(_LOG_PREFIX + line, flush=True)
    path_template = os.getenv("SOL_ATTN_EVENT_LOG")
    if path_template:
        path = Path(path_template.format(rank=_rank()))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def emit_event(event: str, **payload: Any) -> None:
    """Write an integration event to the shared per-rank log."""

    _emit(event, **payload)


@torch.no_grad()
def begin_model_forward(
    *,
    unique_timesteps: torch.Tensor,
    text_positions: torch.Tensor | Iterable[int],
    image_positions: torch.Tensor | Iterable[int],
    audio_positions: torch.Tensor | Iterable[int],
    cu_seqlens_host: Iterable[int] | None,
    total_tokens: int,
    num_layers: int,
) -> StepContext:
    """Record request-static layout and advance the denoise-step counter."""

    registered_num_layers = max(_MAIN_DIT_LAYER_INDICES, default=-1) + 1
    num_layers = max(int(num_layers), registered_num_layers)

    text = _host_positions(text_positions)
    image = _host_positions(image_positions)
    audio = _host_positions(audio_positions)
    text_start, text_tokens = _contiguous_range(text, "text")
    live_positions = text + image + audio
    live_tokens = max(live_positions, default=-1) + 1
    if live_tokens <= 0 or live_tokens > total_tokens:
        raise ValueError(
            f"invalid MiniMax-H3 live token count {live_tokens} for T={total_tokens}"
        )
    if len(set(live_positions)) != len(live_positions):
        raise ValueError("MiniMax-H3 text/video/audio token positions overlap")

    boundaries = _normalize_boundaries(cu_seqlens_host, total_tokens)
    timestep_max = _timestep_max(unique_timesteps)
    epoch = _request_epoch()
    layout_key = (total_tokens, live_tokens, text_start, text_tokens, boundaries)
    previous = _STATE.context
    layout_changed = previous is None or (
        previous.total_tokens,
        previous.live_tokens,
        previous.text_start,
        previous.text_tokens,
        previous.cu_seqlens_host,
    ) != layout_key
    epoch_changed = epoch is not None and epoch != _STATE.previous_epoch
    # MiniMax-H3 traverses its denoise schedule in increasing timestep order.
    # A decrease therefore marks the next request; treating an increase as a
    # reset would keep every denoise iteration at step zero and force dense.
    timestep_reset = (
        epoch is None
        and previous is not None
        and timestep_max is not None
        and _STATE.previous_timestep_max is not None
        and timestep_max < _STATE.previous_timestep_max - 1.0e-6
    )
    new_request = layout_changed or epoch_changed or timestep_reset
    if new_request:
        _STATE.request_index += 1
        step_index = 0
    else:
        step_index = previous.step_index + 1 if previous is not None else 0

    request_epoch = epoch if epoch is not None else f"auto-{_STATE.request_index}"
    _STATE.context = StepContext(
        request_epoch=request_epoch,
        request_index=_STATE.request_index,
        step_index=step_index,
        total_tokens=total_tokens,
        live_tokens=live_tokens,
        text_start=text_start,
        text_tokens=text_tokens,
        cu_seqlens_host=boundaries,
        num_layers=num_layers,
        timestep_max=timestep_max,
    )
    _STATE.previous_epoch = epoch
    _STATE.previous_timestep_max = timestep_max

    if new_request or step_index < 2:
        _emit(
            "step_context",
            request_epoch=request_epoch,
            request_index=_STATE.request_index,
            step_index=step_index,
            timestep_max=timestep_max,
            total_tokens=total_tokens,
            live_tokens=live_tokens,
            padding_tokens=total_tokens - live_tokens,
            text_start=text_start,
            text_tokens=text_tokens,
            image_tokens=len(image),
            audio_tokens=len(audio),
            cu_seqlens_host=list(boundaries),
            num_layers=num_layers,
        )
    return _STATE.context


def _set_dense_backend(attention: Any, q: torch.Tensor) -> None:
    if attention._attention_impl is not None:
        return
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        get_attn_backend,
    )

    attention._set_attention_backend(
        get_attn_backend(
            attention.head_dim,
            q.dtype,
            supported_attention_backends=attention._supported_attention_backends,
        )
    )


def _dense_varlen(
    attention: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
) -> torch.Tensor:
    global _STATE
    _set_dense_backend(attention, q)
    if not _STATE.dense_backend_logged:
        _emit(
            "dense_backend",
            backend=type(attention._attention_impl).__name__,
            q_shape=list(q.shape),
        )
        _STATE.dense_backend_logged = True
    return attention._attention_impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        cu_seqlens_host=cu_seqlens_host,
    )


def _dense_text_queries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    start: int,
    tokens: int,
    scale: float,
) -> torch.Tensor:
    q_text = q[:, start : start + tokens].transpose(1, 2)
    k_heads = k.transpose(1, 2)
    v_heads = v.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        q_text,
        k_heads,
        v_heads,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return output.transpose(1, 2)


def _error_stats(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | int]:
    difference = actual.float() - expected.float()
    absolute = difference.abs()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12)
    flat_index = int(absolute.argmax().item())
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    return {
        "max_abs": float(absolute.reshape(-1)[flat_index].item()),
        "mean_abs": float(absolute.mean().item()),
        "rel_l2": float(
            (torch.linalg.vector_norm(difference) / denominator).item()
        ),
        "max_error_flat_index": flat_index,
        "actual_at_max_error": float(actual_flat[flat_index].item()),
        "expected_at_max_error": float(expected_flat[flat_index].item()),
    }


@torch.no_grad()
def _real_qkv_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> None:
    shape_key = (int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
    if (
        shape_key in _STATE.gate_shapes
        or os.getenv("SOL_ATTN_CORRECTNESS_GATE", "1") != "1"
    ):
        return
    from sol_attn import sol_attn

    gate_heads = min(int(os.getenv("SOL_ATTN_GATE_HEADS", "1")), q.shape[2])
    q_gate = q[:, :, :gate_heads].contiguous()
    k_gate = k[:, :, :gate_heads].contiguous()
    v_gate = v[:, :, :gate_heads].contiguous()
    # This one-shot validation follows component loading and dense warmup.
    # Release inactive allocator blocks before Triton's first autotune sweep.
    torch.cuda.synchronize(q.device)
    torch.cuda.empty_cache()
    transfeat = sol_attn(
        q_gate,
        k_gate,
        v_gate,
        scale=scale,
        tau=-1000.0,
        thresh_type="diag",
    )
    reference = F.scaled_dot_product_attention(
        q_gate.transpose(1, 2),
        k_gate.transpose(1, 2),
        v_gate.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).transpose(1, 2)
    torch.cuda.synchronize(q.device)
    stats = _error_stats(transfeat, reference)
    limits = {
        "max_abs": float(
            os.getenv(
                "SOL_ATTN_GATE_MAX_ABS",
                "0.15" if q.shape[1] >= 32768 else "0.08",
            )
        ),
        "mean_abs": float(os.getenv("SOL_ATTN_GATE_MEAN_ABS", "0.002")),
        "rel_l2": float(os.getenv("SOL_ATTN_GATE_REL_L2", "0.005")),
    }
    passed = all(stats[name] <= limit for name, limit in limits.items())
    _emit(
        "real_qkv_correctness_gate",
        passed=passed,
        shape=list(q_gate.shape),
        stats=stats,
        limits=limits,
    )
    if not passed:
        raise RuntimeError(f"MiniMax-H3 Sol-Attn correctness gate failed: {stats}")
    _STATE.gate_shapes.add(shape_key)


@torch.no_grad()
def _estimate_density(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    tau: float,
    thresh_type: str,
    sink_start: int,
    sink_tokens: int,
) -> dict[str, float | int]:
    from sol_attn.preprocess import BLOCK_SIZE, prepare

    sample_heads = min(int(os.getenv("SOL_ATTN_DENSITY_SAMPLE_HEADS", "1")), q.shape[2])
    q_sample = q[:, :, :sample_heads].contiguous()
    k_sample = k[:, :, :sample_heads].contiguous()
    v_sample = v[:, :, :sample_heads].contiguous()
    kc, _, threshold = prepare(
        q_sample,
        k_sample,
        v_sample,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
    )
    tokens = q_sample.shape[1]
    blocks = math.ceil(tokens / BLOCK_SIZE)
    pad = blocks * BLOCK_SIZE - tokens
    q_padded = F.pad(q_sample, (0, 0, 0, 0, 0, pad))
    q_sum = q_padded.view(
        q_sample.shape[0], blocks, BLOCK_SIZE, sample_heads, q_sample.shape[3]
    ).float().sum(dim=2)
    counts = torch.full(
        (blocks,),
        BLOCK_SIZE,
        device=q.device,
        dtype=torch.float32,
    )
    counts[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    q_bar = q_sum / counts.view(1, blocks, 1, 1)
    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float())
    scores.mul_(scale * math.log2(math.e))
    routed = scores > threshold[:, :, None, :]
    threshold_density = float(routed.float().mean().item())

    block_ids = torch.arange(blocks, device=q.device)
    local_band = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
    routed |= local_band[None, :, :, None]
    sink_blocks = 0
    if sink_tokens:
        sink_first = sink_start // BLOCK_SIZE
        sink_last = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
        routed[:, :, sink_first:sink_last, :] = True
        sink_blocks = sink_last - sink_first
    return {
        "sample_heads": sample_heads,
        "blocks": blocks,
        "sink_blocks": sink_blocks,
        "threshold_density": threshold_density,
        "effective_density": float(routed.float().mean().item()),
    }


def _dense_policy(layer_index: int | None, context: StepContext | None) -> bool:
    if layer_index is None or context is None:
        return True
    if os.getenv("SOL_ATTN_FORCE_DENSE", "0") == "1":
        return True
    first_steps = int(os.getenv("SOL_ATTN_FIRST_DENSE_STEPS", "10"))
    layer_ratio = float(os.getenv("SOL_ATTN_FIRST_DENSE_LAYER_RATIO", "0.03"))
    default_layers = max(1, math.ceil(context.num_layers * layer_ratio))
    first_layers = int(os.getenv("SOL_ATTN_FIRST_DENSE_LAYERS", str(default_layers)))
    return context.step_index < first_steps or layer_index < first_layers


@torch.no_grad()
def _sol_varlen(
    attention: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    boundaries: tuple[int, ...],
    context: StepContext,
) -> torch.Tensor:
    from sol_attn import sol_attn

    tau = float(os.getenv("SOL_ATTN_TAU", "1.0"))
    thresh_type = os.getenv("SOL_ATTN_THRESH_TYPE", "diag")
    output = torch.zeros_like(q)
    for raw_start, raw_end in zip(boundaries, boundaries[1:]):
        start = min(raw_start, context.live_tokens)
        end = min(raw_end, context.live_tokens)
        if end <= start:
            continue
        q_segment = q[start:end].unsqueeze(0).contiguous()
        k_segment = k[start:end].unsqueeze(0).contiguous()
        v_segment = v[start:end].unsqueeze(0).contiguous()
        text_global_start = context.text_start
        text_global_end = context.text_start + context.text_tokens
        sink_global_start = max(start, text_global_start)
        sink_global_end = min(end, text_global_end)
        sink_tokens = max(0, sink_global_end - sink_global_start)
        sink_start = sink_global_start - start if sink_tokens else 0

        _real_qkv_gate(
            q_segment,
            k_segment,
            v_segment,
            scale=attention.softmax_scale,
        )
        shape_key = (
            int(q_segment.shape[1]),
            int(q_segment.shape[2]),
            int(q_segment.shape[3]),
        )
        if shape_key not in _STATE.density_logged_shapes:
            density = _estimate_density(
                q_segment,
                k_segment,
                v_segment,
                scale=attention.softmax_scale,
                tau=tau,
                thresh_type=thresh_type,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            )
            _emit(
                "route_density",
                tau=tau,
                thresh_type=thresh_type,
                sequence_tokens=end - start,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
                **density,
            )
            _STATE.density_logged_shapes.add(shape_key)

        started = time.perf_counter()
        segment_output = sol_attn(
            q_segment,
            k_segment,
            v_segment,
            scale=attention.softmax_scale,
            tau=tau,
            thresh_type=thresh_type,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        if sink_tokens:
            segment_output[:, sink_start : sink_start + sink_tokens] = (
                _dense_text_queries(
                    q_segment,
                    k_segment,
                    v_segment,
                    start=sink_start,
                    tokens=sink_tokens,
                    scale=attention.softmax_scale,
                )
            )
        output[start:end].copy_(segment_output[0])

        if shape_key not in _STATE.first_sparse_logged_shapes:
            torch.cuda.synchronize(q.device)
            _emit(
                "first_sparse_forward",
                q_shape=list(q_segment.shape),
                elapsed_s=time.perf_counter() - started,
                tau=tau,
                thresh_type=thresh_type,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            )
            _STATE.first_sparse_logged_shapes.add(shape_key)
    return output


@torch.no_grad()
def forward_attention(
    attention: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    ulysses_active: bool,
) -> torch.Tensor:
    """Replace only the H3 main DiT attention core."""

    if ulysses_active:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all_packed_qkv,
            _usp_output_all_to_all,
        )

        q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

    layer_index = getattr(attention, "_sol_attn_layer_index", None)
    context = _STATE.context
    if _dense_policy(layer_index, context):
        output = _dense_varlen(
            attention,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
        )
    else:
        if context is None:
            raise RuntimeError("MiniMax-H3 Sol-Attn context was not initialized")
        boundaries = _normalize_boundaries(
            cu_seqlens_host or context.cu_seqlens_host,
            q.shape[0],
        )
        output = _sol_varlen(
            attention,
            q,
            k,
            v,
            boundaries=boundaries,
            context=context,
        )

    if ulysses_active:
        output = _usp_output_all_to_all(output[None], head_dim=2)[0]
    return output


__all__ = [
    "StepContext",
    "begin_model_forward",
    "emit_event",
    "forward_attention",
    "parse_layer_index",
]
