"""Sol-Engine attention adapter for SGLang's packed MiniMax-H3 DiT.

The adapter runs after the Ulysses all-to-all, where each rank owns all packed
tokens for a disjoint set of heads. MiniMax-H3 packs one live document as
``[text | references | target audio | target video]`` followed by padding.
Everything before the target-video suffix is therefore one exact KV sink; its
query rows are recomputed densely, as required for multimodal joint attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

import torch
import torch.nn.functional as F


_LAYER_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.attn$")
_LOG_PREFIX = "MINIMAX_H3_SOL_ENGINE "
_SPARSE_POLICIES = {
    "quality": (0.5, "diag", 15, 2),
    "balanced": (1.0, "diag", 10, 2),
    "aggressive": (1.0, "diag", 10, 2),
    "fullopt_exact": (1.0, "exact", 10, 2),
}


@dataclass(frozen=True)
class StepContext:
    request_epoch: str
    request_index: int
    step_index: int
    total_tokens: int
    live_tokens: int
    target_video_start: int
    prefix_tokens: int
    cu_seqlens_host: tuple[int, ...]
    num_layers: int
    timestep_max: float | None


@dataclass
class _RuntimeState:
    context: StepContext | None = None
    request_index: int = -1
    previous_epoch: str | None = None
    previous_timestep: float | None = None
    timestep_direction: int = 0
    dense_backend_logged: bool = False
    backend_logged: bool = False
    gated_shapes: set[tuple[int, int, int]] = field(default_factory=set)
    density_logged: set[tuple[int, int, int, int]] = field(default_factory=set)
    sparse_logged: set[tuple[int, int, int, int]] = field(default_factory=set)
    layout_signature: tuple[Any, ...] | None = None
    layout: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        int,
        int,
    ] | None = None


_STATE = _RuntimeState()


def enabled() -> bool:
    return (
        os.getenv("H3_SOL_ATTN", "0") == "1"
        or os.getenv("H3_FIRSTBLOCKCACHE", "0") == "1"
        or os.getenv("H3_EASYCACHE", "0") == "1"
    )


def parse_layer_index(prefix: str) -> int | None:
    if "token_refiner" in prefix:
        return None
    match = _LAYER_PATTERN.search(prefix)
    return None if match is None else int(match.group(1))


def _rank() -> int:
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def emit_event(event: str, **payload: Any) -> None:
    record = {"event": event, "rank": _rank(), **payload}
    line = json.dumps(record, sort_keys=True)
    print(_LOG_PREFIX + line, flush=True)
    path_template = os.getenv("H3_SOL_EVENT_LOG")
    if path_template:
        path = Path(path_template.format(rank=_rank()))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _host_positions(values: torch.Tensor | Iterable[int]) -> tuple[int, ...]:
    if torch.is_tensor(values):
        values = values.detach().reshape(-1).to(device="cpu").tolist()
    return tuple(int(value) for value in values if int(value) >= 0)


def _position_identity(values: torch.Tensor | Iterable[int]) -> tuple[Any, ...]:
    if torch.is_tensor(values):
        return (
            str(values.device),
            int(values.data_ptr()),
            int(values.numel()),
        )
    materialized = tuple(int(value) for value in values)
    return ("host", materialized)


def _last_contiguous_run_start(values: tuple[int, ...]) -> int:
    ordered = tuple(sorted(set(values)))
    if not ordered:
        raise ValueError("MiniMax-H3 has no video token positions")
    start_index = 0
    for index, (left, right) in enumerate(zip(ordered, ordered[1:]), start=1):
        if right != left + 1:
            start_index = index
    return ordered[start_index]


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
        raise ValueError(f"cu_seqlens_host ends beyond T={total_tokens}: {result}")
    if result[-1] < total_tokens:
        result = (*result, total_tokens)
    return result


def _request_epoch() -> str | None:
    path = os.getenv("H3_REQUEST_EPOCH_FILE")
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _timestep_max(unique_timesteps: torch.Tensor) -> float | None:
    if not torch.is_tensor(unique_timesteps) or not unique_timesteps.numel():
        return None
    return float(unique_timesteps.detach().float().max().cpu().item())


def _timestep_reversed(current: float | None) -> bool:
    previous = _STATE.previous_timestep
    _STATE.previous_timestep = current
    if current is None or previous is None:
        return False
    delta = current - previous
    if abs(delta) <= 1.0e-6:
        return False
    direction = 1 if delta > 0 else -1
    if _STATE.timestep_direction == 0:
        _STATE.timestep_direction = direction
        return False
    return direction != _STATE.timestep_direction


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
) -> StepContext | None:
    if not enabled():
        return None

    boundaries = _normalize_boundaries(cu_seqlens_host, total_tokens)
    layout_signature = (
        total_tokens,
        boundaries,
        _position_identity(text_positions),
        _position_identity(image_positions),
        _position_identity(audio_positions),
    )
    if layout_signature == _STATE.layout_signature and _STATE.layout is not None:
        text, image, audio, live_tokens, target_video_start = _STATE.layout
    else:
        text = _host_positions(text_positions)
        image = _host_positions(image_positions)
        audio = _host_positions(audio_positions)
        live_positions = text + image + audio
        live_tokens = max(live_positions, default=-1) + 1
        if live_tokens <= 0 or live_tokens > total_tokens:
            raise ValueError(
                f"invalid live token count {live_tokens} for T={total_tokens}"
            )
        if len(set(live_positions)) != len(live_positions):
            raise ValueError("MiniMax-H3 text/video/audio token positions overlap")
        if tuple(sorted(live_positions)) != tuple(range(live_tokens)):
            raise ValueError("MiniMax-H3 live packed rows must cover one contiguous prefix")

        target_video_start = _last_contiguous_run_start(image)
        image_set = set(image)
        if any(
            position not in image_set
            for position in range(target_video_start, live_tokens)
        ):
            raise ValueError(
                "MiniMax-H3 target video must be the final contiguous live suffix"
            )
        _STATE.layout_signature = layout_signature
        _STATE.layout = (text, image, audio, live_tokens, target_video_start)

    if len(boundaries) < 2 or boundaries[1] != live_tokens:
        raise ValueError(
            "MiniMax-H3 first packed document must end at the live-token boundary: "
            f"{boundaries}, live={live_tokens}"
        )

    epoch = _request_epoch()
    timestep = None if epoch is not None else _timestep_max(unique_timesteps)
    previous = _STATE.context
    layout_changed = previous is None or (
        previous.total_tokens,
        previous.live_tokens,
        previous.target_video_start,
        previous.cu_seqlens_host,
    ) != (total_tokens, live_tokens, target_video_start, boundaries)
    epoch_changed = epoch is not None and epoch != _STATE.previous_epoch
    reversal = epoch is None and _timestep_reversed(timestep)
    new_request = layout_changed or epoch_changed or reversal
    if new_request:
        _STATE.request_index += 1
        step_index = 0
        _STATE.timestep_direction = 0
        _STATE.previous_timestep = timestep
    else:
        step_index = 0 if previous is None else previous.step_index + 1

    request_epoch = epoch if epoch is not None else f"auto-{_STATE.request_index}"
    context = StepContext(
        request_epoch=request_epoch,
        request_index=_STATE.request_index,
        step_index=step_index,
        total_tokens=total_tokens,
        live_tokens=live_tokens,
        target_video_start=target_video_start,
        prefix_tokens=target_video_start,
        cu_seqlens_host=boundaries,
        num_layers=int(num_layers),
        timestep_max=timestep,
    )
    _STATE.context = context
    _STATE.previous_epoch = epoch
    if new_request or step_index < 2:
        emit_event(
            "step_context",
            request_epoch=request_epoch,
            request_index=context.request_index,
            step_index=step_index,
            total_tokens=total_tokens,
            live_tokens=live_tokens,
            padding_tokens=total_tokens - live_tokens,
            text_tokens=len(text),
            image_tokens=len(image),
            audio_tokens=len(audio),
            target_video_start=target_video_start,
            prefix_tokens=context.prefix_tokens,
            cu_seqlens_host=list(boundaries),
            num_layers=context.num_layers,
        )
    return context


def _set_dense_backend(attention: Any, q: torch.Tensor) -> None:
    if attention._attention_impl is not None:
        return
    from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend

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
    _set_dense_backend(attention, q)
    if not _STATE.dense_backend_logged:
        emit_event(
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


def _dense_queries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    start: int,
    tokens: int,
    scale: float,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q[:, start : start + tokens].transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).transpose(1, 2)


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rel_l2": float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12)
            ).item()
        ),
    }


def _kernel_and_backend(device: torch.device) -> tuple[Any, str]:
    from sol_attn import get_sol_attn_backend, sol_attn

    backend = get_sol_attn_backend(device)
    expected = os.getenv("H3_EXPECTED_SOL_BACKEND")
    if expected and backend != expected:
        raise RuntimeError(
            f"MiniMax-H3 expected Sol-Attn backend {expected!r}, selected {backend!r}"
        )
    if not _STATE.backend_logged:
        emit_event(
            "sol_backend",
            backend=backend,
            expected_backend=expected,
            capability=list(torch.cuda.get_device_capability(device)),
        )
        _STATE.backend_logged = True
    return sol_attn, backend


@torch.no_grad()
def _run_correctness_gate(
    kernel: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    thresh_type: str,
) -> None:
    shape = (int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
    if shape in _STATE.gated_shapes or os.getenv("H3_SOL_CORRECTNESS_GATE", "1") != "1":
        return
    candidate = kernel(
        q,
        k,
        v,
        scale=scale,
        tau=-1000.0,
        thresh_type=thresh_type,
        kv_splits=1,
    )
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).transpose(1, 2)
    torch.cuda.synchronize(q.device)
    stats = _error_stats(candidate, reference)
    limits = {
        "max_abs": float(os.getenv("H3_SOL_GATE_MAX_ABS", "0.15")),
        "mean_abs": float(os.getenv("H3_SOL_GATE_MEAN_ABS", "0.002")),
        "rel_l2": float(os.getenv("H3_SOL_GATE_REL_L2", "0.005")),
    }
    passed = all(stats[name] <= limit for name, limit in limits.items())
    emit_event(
        "real_qkv_correctness_gate",
        passed=passed,
        shape=list(q.shape),
        stats=stats,
        limits=limits,
    )
    if not passed:
        raise RuntimeError(f"MiniMax-H3 Sol-Attn correctness gate failed: {stats}")
    _STATE.gated_shapes.add(shape)


@torch.no_grad()
def _estimate_density(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: str,
    scale: float,
    tau: float,
    thresh_type: str,
    sink_start: int,
    sink_tokens: int,
) -> dict[str, float | int]:
    from sol_attn.preprocess import BLOCK_SIZE

    tokens = int(q.shape[1])
    blocks = math.ceil(tokens / BLOCK_SIZE)
    if backend == "triton" and torch.cuda.get_device_capability(q.device)[0] < 9:
        from sol_attn.triton_ref.preprocess import prepare

        kc, _, threshold = prepare(
            q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, tokens=tokens
        )
        kc = kc[:, :blocks]
    else:
        from sol_attn.preprocess import prepare

        kc, _, threshold = prepare(
            q, k, v, scale=scale, tau=tau, thresh_type=thresh_type
        )

    padded = F.pad(q, (0, 0, 0, 0, 0, blocks * BLOCK_SIZE - tokens))
    counts = torch.full(
        (blocks,), BLOCK_SIZE, device=q.device, dtype=torch.float32
    )
    counts[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    q_bar = padded.view(
        q.shape[0], blocks, BLOCK_SIZE, q.shape[2], q.shape[3]
    ).float().sum(dim=2)
    q_bar.div_(counts.view(1, blocks, 1, 1))
    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float())
    scores.mul_(scale * math.log2(math.e))
    routed = scores > threshold[:, :, None, :]
    threshold_density = float(routed.float().mean().item())

    ids = torch.arange(blocks, device=q.device)
    routed |= ((ids[:, None] - ids[None, :]).abs() <= 1)[None, :, :, None]
    sink_first = sink_start // BLOCK_SIZE
    sink_last = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
    if sink_tokens:
        routed[:, :, sink_first:sink_last, :] = True
    return {
        "blocks": blocks,
        "heads": int(q.shape[2]),
        "sink_blocks": max(0, sink_last - sink_first),
        "threshold_density": threshold_density,
        "effective_density": float(routed.float().mean().item()),
    }


def _use_dense(layer_index: int | None, context: StepContext | None) -> bool:
    if os.getenv("H3_SOL_ATTN", "0") != "1":
        return True
    if layer_index is None or context is None:
        return True
    dense_steps = int(os.getenv("H3_SOL_DENSE_STEPS", "10"))
    dense_layers = int(os.getenv("H3_SOL_DENSE_LAYERS", "2"))
    return context.step_index < dense_steps or layer_index < dense_layers


def _validate_sparse_policy(tau: float, thresh_type: str) -> str:
    name = os.getenv("H3_POLICY_NAME", "fullopt_exact")
    expected = _SPARSE_POLICIES.get(name)
    if expected is None:
        raise RuntimeError(f"unknown H100/A100 MiniMax-H3 policy {name!r}")
    actual = (
        tau,
        thresh_type,
        int(os.getenv("H3_SOL_DENSE_STEPS", "10")),
        int(os.getenv("H3_SOL_DENSE_LAYERS", "2")),
    )
    if actual != expected:
        raise RuntimeError(
            f"H100/A100 MiniMax-H3 policy {name!r} requires {expected}, got {actual}"
        )
    return name


def _should_measure_density(mode: str, context: StepContext) -> bool:
    if mode == "all":
        return True
    if mode != "warmup":
        return False
    if ":warmup:" in context.request_epoch:
        return True
    return (
        context.request_epoch == f"auto-{context.request_index}"
        and context.request_index == 0
    )


@torch.no_grad()
def _sparse_varlen(
    attention: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    boundaries: tuple[int, ...],
    context: StepContext,
) -> torch.Tensor:
    tau = float(os.getenv("H3_SOL_TAU", "1.0"))
    thresh_type = os.getenv("H3_SOL_THRESH_TYPE", "exact")
    policy_name = _validate_sparse_policy(tau, thresh_type)
    if os.getenv("H3_SOL_SINK_MODE", "prefix") != "prefix":
        raise RuntimeError("H100/A100 MiniMax-H3 requires the full prefix sink")

    output = torch.zeros_like(q)
    for raw_start, raw_end in zip(boundaries, boundaries[1:]):
        start = min(raw_start, context.live_tokens)
        end = min(raw_end, context.live_tokens)
        if end <= start:
            continue
        if start != 0:
            raise RuntimeError("MiniMax-H3 H100/A100 path expects one live packed document")
        qb = q[start:end].unsqueeze(0).contiguous()
        kb = k[start:end].unsqueeze(0).contiguous()
        vb = v[start:end].unsqueeze(0).contiguous()
        sink_start = 0
        sink_tokens = context.prefix_tokens
        kernel, backend = _kernel_and_backend(q.device)
        _run_correctness_gate(
            kernel,
            qb,
            kb,
            vb,
            scale=attention.softmax_scale,
            thresh_type=thresh_type,
        )

        shape_key = (
            context.request_index,
            int(qb.shape[1]),
            int(qb.shape[2]),
            int(qb.shape[3]),
        )
        density_mode = os.getenv("H3_SOL_DENSITY_MODE", "warmup")
        should_measure_density = _should_measure_density(density_mode, context)
        if should_measure_density and shape_key not in _STATE.density_logged:
            density = _estimate_density(
                qb,
                kb,
                vb,
                backend=backend,
                scale=attention.softmax_scale,
                tau=tau,
                thresh_type=thresh_type,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            )
            emit_event(
                "route_density",
                request_epoch=context.request_epoch,
                request_index=context.request_index,
                step_index=context.step_index,
                tau=tau,
                thresh_type=thresh_type,
                backend=backend,
                policy_name=policy_name,
                sequence_tokens=end - start,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
                **density,
            )
            _STATE.density_logged.add(shape_key)

        segment = kernel(
            qb,
            kb,
            vb,
            scale=attention.softmax_scale,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=1,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )
        if sink_tokens:
            segment[:, :sink_tokens] = _dense_queries(
                qb,
                kb,
                vb,
                start=0,
                tokens=sink_tokens,
                scale=attention.softmax_scale,
            )
        output[start:end].copy_(segment[0])
        if shape_key not in _STATE.sparse_logged:
            emit_event(
                "first_sparse_forward",
                request_epoch=context.request_epoch,
                request_index=context.request_index,
                step_index=context.step_index,
                q_shape=list(qb.shape),
                backend=backend,
                policy_name=policy_name,
                tau=tau,
                thresh_type=thresh_type,
                sink_start=sink_start,
                sink_tokens=sink_tokens,
            )
            _STATE.sparse_logged.add(shape_key)
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
    if ulysses_active:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all_packed_qkv,
            _usp_output_all_to_all,
        )

        q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

    context = _STATE.context
    layer_index = getattr(attention, "_sol_engine_layer_index", None)
    if _use_dense(layer_index, context):
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
            raise RuntimeError("MiniMax-H3 Sol-Engine context was not initialized")
        output = _sparse_varlen(
            attention,
            q,
            k,
            v,
            boundaries=_normalize_boundaries(
                cu_seqlens_host or context.cu_seqlens_host,
                int(q.shape[0]),
            ),
            context=context,
        )

    if ulysses_active:
        output = _usp_output_all_to_all(output[None], head_dim=2)[0]
    return output


__all__ = [
    "StepContext",
    "begin_model_forward",
    "emit_event",
    "enabled",
    "forward_attention",
    "parse_layer_index",
]
