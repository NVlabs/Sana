"""Experiment-local Sana adapter for authoritative Piecewise Sparse Attention.

Algorithm and Triton kernels copied from:
  /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/
  python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py

Source commit: 7546a4bd1d382923ef4876945172655a84d23686
Source SHA-256: bfad198d834d21254492676ad210e6d5393c88b236bd3b4b793c99a6ac960fb3

Only the inference-forward closure is copied here.  The centroid reduction,
Taylor-error router, TMA allocator, exact selected-block phase, and approximate
remainder are intentionally kept equivalent to the authoritative local source.
Sana-specific code below the kernels adds the explicit guard/policy and durable
dispatch/timing counters; it does not change the PISA calculation.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


AUTHORITATIVE_SOURCE = (
    "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/"
    "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/"
    "backends/piecewise_attn.py"
)
AUTHORITATIVE_COMMIT = "7546a4bd1d382923ef4876945172655a84d23686"
AUTHORITATIVE_SHA256 = "bfad198d834d21254492676ad210e6d5393c88b236bd3b4b793c99a6ac960fb3"


def _piecewise_num_stages() -> int:
    """Return the launch-only TMA pipeline depth for the copied PISA kernel."""

    raw = os.environ.get("SANA_PISA_KERNEL_NUM_STAGES", "2")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"SANA_PISA_KERNEL_NUM_STAGES must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise RuntimeError("SANA_PISA_KERNEL_NUM_STAGES must be at least 1")
    return value


def _make_tma_allocator():
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    return alloc_fn


@triton.jit
def chunk_reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    k_var,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))

    b_k = p_k.load([i_t * BT, 0])
    b_v = p_v.load([i_t * BT, 0])

    b_kc = tl.sum(b_k, axis=0) / block_size
    b_vc = tl.sum(b_v, axis=0)

    mean_norm = tl.sum(b_k * b_k) / block_size
    kc_norm = tl.sum(b_kc * b_kc, axis=0)
    b_k_var = tl.maximum(mean_norm - kc_norm, 0.0)

    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (0,), (BV,), (0,))

    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))
    tl.store(k_var + i_bh * N + i_t, b_k_var)


@triton.jit
def chunk_reduce_k_kernel(
    k,
    kc,
    k_var,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_k = p_k.load([i_t * BT, 0])
    b_kc = tl.sum(b_k, axis=0) / block_size
    mean_norm = tl.sum(b_k * b_k) / block_size
    kc_norm = tl.sum(b_kc * b_kc, axis=0)
    b_k_var = tl.maximum(mean_norm - kc_norm, 0.0)

    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(k_var + i_bh * N + i_t, b_k_var)


@triton.jit
def chunk_reduce_q_kernel(
    q,
    qc,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_q = p_q.load([i_t * BT, 0])
    b_qc = tl.sum(b_q, axis=0) / block_size

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))


def chunk_reduce_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    include_v_centroid: bool = True,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]

    N_Q = triton.cdiv(T_Q, block_size)
    N_KV = triton.cdiv(T_KV, block_size)
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    qc = torch.empty(B, H, N_Q, K, device=q.device, dtype=q.dtype)
    kc = torch.empty(B, H, N_KV, K, device=k.device, dtype=k.dtype)
    vc = (
        torch.empty(B, H, N_KV, V, device=v.device, dtype=v.dtype)
        if include_v_centroid
        else None
    )
    k_var = torch.empty(B, H, N_KV, device=k.device, dtype=k.dtype)

    chunk_reduce_q_kernel[(N_Q, B * H)](
        q=q,
        qc=qc,
        T=T_Q,
        N=N_Q,
        K=K,
        BT=block_size,
        BK=BK,
        num_warps=4,
        num_stages=2,
    )
    if include_v_centroid:
        chunk_reduce_kv_kernel[(N_KV, B * H)](
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            k_var=k_var,
            T=T_KV,
            N=N_KV,
            K=K,
            V=V,
            BT=block_size,
            BK=BK,
            BV=BV,
            num_warps=4,
            num_stages=3,
        )
    else:
        chunk_reduce_k_kernel[(N_KV, B * H)](
            k=k,
            kc=kc,
            k_var=k_var,
            T=T_KV,
            N=N_KV,
            K=K,
            BT=block_size,
            BK=BK,
            num_warps=4,
            num_stages=3,
        )
    return qc, kc, vc, k_var


@triton.jit
def piecewise_attn_fwd_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    kc,
    vc,
    lse,
    indices,
    scale,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    NS: tl.constexpr,
    B_NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    APPROX_REMAINDER: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    token_offsets = tl.arange(0, BT)
    token_offsets = tl.max_contiguous(token_offsets, BT)
    q_start = i_t * BT
    tl.multiple_of(q_start, BT)
    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])

    sm_scale = scale * 1.44269504
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        kv_start = i_n * BT
        tl.multiple_of(kv_start, BT)
        b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])
        b_s = tl.dot(b_q, b_k.T) * sm_scale
        b_s += tl.where((kv_start + token_offsets)[None, :] < T_KV, 0, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(b_s, axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        score = tl.math.exp2(b_s - new_m[:, None])
        b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])
        l_i = l_i * alpha + tl.sum(score, axis=1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m

    offs_n_idx = tl.arange(0, B_NS)
    selected = tl.load(
        indices + i_bh * NT_Q * NS + i_t * NS + offs_n_idx,
        mask=offs_n_idx < NS,
        other=-1,
    )

    if APPROX_REMAINDER:
        for start_n in range(0, NT_KV, GROUP_SIZE):
            p_kc = tl.make_tensor_descriptor(
                kc + i_bh * NT_KV * K,
                (NT_KV, K),
                (K, 1),
                (GROUP_SIZE, BK),
            )
            b_kc = p_kc.load([start_n, 0])
            chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
            is_selected = chunk_indices[:, None] == selected[None, :]
            selected_mask = tl.max(is_selected, axis=1)
            valid_mask = (chunk_indices < NT_KV) & (selected_mask == 0)
            current_lens = tl.minimum(BT, tl.maximum(0, T_KV - chunk_indices * BT)).to(tl.float32)

            b_s_mean = tl.dot(b_q, b_kc.T) * sm_scale
            b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))
            new_m = tl.maximum(m_i, tl.max(b_s_mean, axis=1))
            alpha = tl.math.exp2(m_i - new_m)
            prob_chunk = tl.math.exp2(b_s_mean - new_m[:, None])

            p_vc = tl.make_tensor_descriptor(
                vc + i_bh * NT_KV * V,
                (NT_KV, V),
                (V, 1),
                (GROUP_SIZE, BV),
            )
            b_vc = p_vc.load([start_n, i_v * BV])
            acc = acc * alpha[:, None] + tl.dot(prob_chunk.to(b_vc.dtype), b_vc)
            l_i = l_i * alpha + tl.sum(prob_chunk * current_lens[None, :], axis=1)
            m_i = new_m

    acc = acc / l_i[:, None]
    m_i += tl.math.log2(l_i)
    p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
    tl.store(p_lse, m_i, boundary_check=(0,))
    o_desc.store([i_bh, q_start, i_v * BV], acc[None, :, :])


def piecewise_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    scale: float,
    approx_remainder: bool = True,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]
    BT, NS = block_size, block_indices.shape[-1]
    o = torch.empty(B, H, T_Q, V, device=q.device, dtype=v.dtype)
    lse = torch.empty(B, H, T_Q, device=q.device, dtype=torch.float)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    B_NS = triton.next_power_of_2(NS)
    NT_Q = triton.cdiv(T_Q, BT)
    NT_KV = triton.cdiv(T_KV, BT)

    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T_Q, K), [1, block_size, BK])
    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T_Q, V), [1, block_size, BV])
    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T_KV, K), [1, block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, block_size, BV])

    grid = (triton.cdiv(V, BV), NT_Q, B * H)
    piecewise_attn_fwd_kernel[grid](
        q_desc=q_desc,
        k_desc=k_desc,
        v_desc=v_desc,
        o_desc=o_desc,
        kc=kc,
        vc=vc,
        lse=lse,
        indices=block_indices,
        scale=scale,
        T_Q=T_Q,
        T_KV=T_KV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NS=NS,
        B_NS=B_NS,
        NT_Q=NT_Q,
        NT_KV=NT_KV,
        GROUP_SIZE=64,
        APPROX_REMAINDER=approx_remainder,
        num_warps=4,
        # Pipeline staging does not change the copied PISA calculation.  The
        # authoritative default remains two; FP32 Sana block-64 candidates use
        # one stage so the same TMA kernel fits GB200 shared memory.
        num_stages=_piecewise_num_stages(),
    )
    return o, lse


@torch.no_grad()
def taylor_error_block_indices(
    qc: torch.Tensor,
    kc: torch.Tensor,
    k_var: torch.Tensor,
    density: float,
    scale: float,
    eps: float = 1e-8,
):
    NT_KV = kc.shape[2]
    top_k = max(1, int(density * NT_KV))
    top_k = min(top_k, NT_KV)

    route_score = torch.einsum("bhik,bhjk->bhij", qc, kc)
    route_score.mul_(scale)
    log_k_var = torch.log(k_var.clamp_min(eps)).unsqueeze(-2)
    route_score.add_(log_k_var)
    return torch.topk(route_score, k=top_k, dim=-1, sorted=False).indices.to(torch.int32)


def _parse_index_set(raw: str | None) -> set[int]:
    values: set[int] = set()
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    return values


def _json_index_set(value: Any, *, field: str, rule_name: str) -> frozenset[int]:
    """Parse an explicit layer/step selector from one density rule."""

    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset(_parse_index_set(value))
    if not isinstance(value, list):
        raise RuntimeError(
            f"SANA_PISA_DENSITY_RULES rule {rule_name!r} field {field!r} "
            "must be a range string or integer list"
        )
    parsed: set[int] = set()
    for item in value:
        if isinstance(item, int):
            parsed.add(item)
            continue
        if (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(endpoint, int) for endpoint in item)
        ):
            start, end = item
            if end < start:
                start, end = end, start
            parsed.update(range(start, end + 1))
            continue
        raise RuntimeError(
            f"SANA_PISA_DENSITY_RULES rule {rule_name!r} field {field!r} "
            f"contains invalid selector {item!r}"
        )
    return frozenset(parsed)


@lru_cache(maxsize=16)
def _parse_density_rules(raw: str) -> tuple[tuple[str, frozenset[int], frozenset[int], float], ...]:
    """Parse experiment policy rules without adding work to every attention call."""

    if not raw.strip():
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SANA_PISA_DENSITY_RULES is not valid JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("rules")
    if not isinstance(payload, list):
        raise RuntimeError("SANA_PISA_DENSITY_RULES must be a JSON list or {\"rules\": [...]}")

    rules: list[tuple[str, frozenset[int], frozenset[int], float]] = []
    for index, rule in enumerate(payload):
        if not isinstance(rule, dict):
            raise RuntimeError(f"SANA_PISA_DENSITY_RULES rule {index} must be an object")
        name = str(rule.get("name") or f"rule_{index}")
        attention_type = str(rule.get("attention_type") or "video_self_softmax")
        if attention_type != "video_self_softmax":
            raise RuntimeError(
                f"SANA_PISA_DENSITY_RULES rule {name!r} targets unsupported "
                f"attention_type={attention_type!r}"
            )
        layers = _json_index_set(rule.get("layers"), field="layers", rule_name=name)
        steps = _json_index_set(rule.get("steps"), field="steps", rule_name=name)
        try:
            density = float(rule["density"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"SANA_PISA_DENSITY_RULES rule {name!r} requires numeric density"
            ) from exc
        if not 0.0 < density <= 1.0:
            raise RuntimeError(
                f"SANA_PISA_DENSITY_RULES rule {name!r} density must be in (0, 1], "
                f"got {density}"
            )
        rules.append((name, layers, steps, density))
    return tuple(rules)


def _density_for_cell(layer_index: int, step_index: int) -> tuple[float, str]:
    default_density = float(os.environ.get("SANA_PISA_DENSITY", "0.75"))
    if not 0.0 < default_density <= 1.0:
        raise RuntimeError(f"SANA_PISA_DENSITY must be in (0, 1], got {default_density}")
    raw_rules = os.environ.get("SANA_PISA_DENSITY_RULES", "")
    matches: list[tuple[str, float]] = []
    for name, layers, steps, density in _parse_density_rules(raw_rules):
        if layers and layer_index not in layers:
            continue
        if steps and step_index not in steps:
            continue
        matches.append((name, density))
    if len(matches) > 1:
        names = ", ".join(name for name, _ in matches)
        raise RuntimeError(
            f"Overlapping SANA_PISA_DENSITY_RULES for layer={layer_index}, "
            f"step={step_index}: {names}"
        )
    if matches:
        return matches[0][1], matches[0][0]
    return default_density, "default"


def _density_rules_manifest() -> list[dict[str, Any]]:
    raw_rules = os.environ.get("SANA_PISA_DENSITY_RULES", "")
    return [
        {
            "name": name,
            "attention_type": "video_self_softmax",
            "layers": sorted(layers),
            "steps": sorted(steps),
            "density": density,
            "sparsity": 1.0 - density,
        }
        for name, layers, steps, density in _parse_density_rules(raw_rules)
    ]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def pisa_enabled() -> bool:
    return _env_flag("SANA_PISA_ENABLED", False)


_LOCK = threading.Lock()
_REGISTERED = False
_LAYER_CALLS: dict[int, int] = {}
_EXPLICIT_CONTEXT = threading.local()
_PENDING_EVENTS: list[tuple[Any, Any, Any, Any, int, int]] = []
_STATS: dict[str, Any] = {
    "schema_version": 1,
    "backend": "experiment_local_authoritative_piecewise_attn_tma",
    "authoritative_source": AUTHORITATIVE_SOURCE,
    "authoritative_commit": AUTHORITATIVE_COMMIT,
    "authoritative_sha256": AUTHORITATIVE_SHA256,
    "guard": "SANA_PISA_ENABLED",
    "route_mode": "score",
    "route_bias": False,
    "approx_remainder": True,
    "calls": {
        "total": 0,
        "pisa_dispatch": 0,
        "dense_policy": 0,
        "dense_fallback": 0,
        "exact_phase": 0,
        "approximate_remainder_phase": 0,
    },
    "by_layer": {},
    "by_step": {},
    "by_layer_step": {},
    "by_density": {},
    "by_shape": {},
    "timing": {
        "mask_selection_and_route_ms": 0.0,
        "piecewise_fused_exact_approx_kernel_ms": 0.0,
        "event_count": 0,
        "approximation_timing_note": (
            "Exact selected blocks and approximate remainder execute in one fused "
            "piecewise_attn_fwd kernel; their sub-times are not separable."
        ),
    },
}


def set_pisa_context(step_index: int | None, prompt_index: int = 0) -> None:
    """Set the authoritative denoising context for PISA dispatches.

    The original adapter inferred the step from per-layer call counts. That
    fallback is valid only when every block executes on every step. Wan's
    block Cache can skip a layer, so its driver supplies the actual step at the
    transformer-forward boundary instead.
    """

    if step_index is None:
        for name in ("step_index", "prompt_index"):
            if hasattr(_EXPLICIT_CONTEXT, name):
                delattr(_EXPLICIT_CONTEXT, name)
        return
    _EXPLICIT_CONTEXT.step_index = int(step_index)
    _EXPLICIT_CONTEXT.prompt_index = int(prompt_index)


def reset_pisa_layer_counters() -> None:
    """Reset the legacy call-counter fallback at a prompt boundary."""

    with _LOCK:
        _LAYER_CALLS.clear()


def _counter(bucket: dict[str, Any], key: str) -> dict[str, Any]:
    return bucket.setdefault(
        key,
        {
            "total": 0,
            "pisa_dispatch": 0,
            "dense_policy": 0,
            "dense_fallback": 0,
            "mask_selection_and_route_ms": 0.0,
            "piecewise_fused_exact_approx_kernel_ms": 0.0,
        },
    )


def _record_branch(
    layer_index: int,
    step_index: int,
    branch: str,
    shape_key: str,
    density: float,
) -> None:
    with _LOCK:
        _STATS["calls"]["total"] += 1
        _STATS["calls"][branch] += 1
        for bucket, key in (
            (_STATS["by_layer"], str(layer_index)),
            (_STATS["by_step"], str(step_index)),
            (_STATS["by_layer_step"], f"{layer_index}:{step_index}"),
            (_STATS["by_density"], f"{density:.9g}"),
            (_STATS["by_shape"], shape_key),
        ):
            item = _counter(bucket, key)
            item["total"] += 1
            item[branch] += 1
            density_key = f"{density:.9g}"
            density_counts = item.setdefault("configured_density_counts", {})
            density_counts[density_key] = int(density_counts.get(density_key) or 0) + 1


def _drain_timing_events() -> None:
    with _LOCK:
        pending = list(_PENDING_EVENTS)
        _PENDING_EVENTS.clear()
    if not pending:
        return
    torch.cuda.synchronize()
    for route_start, route_end, kernel_start, kernel_end, layer_index, step_index in pending:
        route_ms = float(route_start.elapsed_time(route_end))
        kernel_ms = float(kernel_start.elapsed_time(kernel_end))
        with _LOCK:
            timing = _STATS["timing"]
            timing["mask_selection_and_route_ms"] += route_ms
            timing["piecewise_fused_exact_approx_kernel_ms"] += kernel_ms
            timing["event_count"] += 1
            for bucket, key in (
                (_STATS["by_layer"], str(layer_index)),
                (_STATS["by_step"], str(step_index)),
                (_STATS["by_layer_step"], f"{layer_index}:{step_index}"),
            ):
                item = _counter(bucket, key)
                item["mask_selection_and_route_ms"] += route_ms
                item["piecewise_fused_exact_approx_kernel_ms"] += kernel_ms


def dump_pisa_stats() -> None:
    path_raw = os.environ.get("SANA_PISA_STATS_PATH", "")
    if not path_raw:
        return
    try:
        if torch.cuda.is_available():
            _drain_timing_events()
        with _LOCK:
            payload = json.loads(json.dumps(_STATS))
        payload["config"] = {
            "enabled": pisa_enabled(),
            "density": float(os.environ.get("SANA_PISA_DENSITY", "0.75")),
            "sparsity": 1.0 - float(os.environ.get("SANA_PISA_DENSITY", "0.75")),
            "default_density": float(os.environ.get("SANA_PISA_DENSITY", "0.75")),
            "default_sparsity": 1.0 - float(os.environ.get("SANA_PISA_DENSITY", "0.75")),
            "density_rules": _density_rules_manifest(),
            "block_size": int(os.environ.get("SANA_PISA_BLOCK_SIZE", "64")),
            "kernel_num_stages": _piecewise_num_stages(),
            "num_steps": int(os.environ.get("SANA_PISA_NUM_STEPS", "50")),
            "pisa_layers": sorted(_parse_index_set(os.environ.get("SANA_PISA_PISA_LAYERS"))),
            "dense_layers": sorted(_parse_index_set(os.environ.get("SANA_PISA_DENSE_LAYERS"))),
            "pisa_steps": sorted(_parse_index_set(os.environ.get("SANA_PISA_PISA_STEPS"))),
            "dense_steps": sorted(_parse_index_set(os.environ.get("SANA_PISA_DENSE_STEPS"))),
            "only_video_self_attention": True,
        }
        if torch.cuda.is_available():
            payload["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            payload["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            payload["device"] = torch.cuda.get_device_name(torch.cuda.current_device())
        path = Path(path_raw).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover - best-effort exit artifact
        print(f"[Sana PISA] failed to write stats: {type(exc).__name__}: {exc}", flush=True)


def _register_dump() -> None:
    global _REGISTERED
    if not _REGISTERED:
        atexit.register(dump_pisa_stats)
        _REGISTERED = True


def _policy_mode(layer_index: int, step_index: int) -> str:
    pisa_layers = _parse_index_set(os.environ.get("SANA_PISA_PISA_LAYERS"))
    dense_layers = _parse_index_set(os.environ.get("SANA_PISA_DENSE_LAYERS"))
    pisa_steps = _parse_index_set(os.environ.get("SANA_PISA_PISA_STEPS"))
    dense_steps = _parse_index_set(os.environ.get("SANA_PISA_DENSE_STEPS"))
    if layer_index in dense_layers or step_index in dense_steps:
        return "dense"
    if pisa_layers and layer_index not in pisa_layers:
        return "dense"
    if pisa_steps and step_index not in pisa_steps:
        return "dense"
    return "pisa"


@torch.no_grad()
def sana_pisa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    layer_index: int,
    dense_fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Dispatch one Sana video self-attention cell according to the PISA policy."""

    _register_dump()
    num_steps = int(os.environ.get("SANA_PISA_NUM_STEPS", "50"))
    explicit_step = getattr(_EXPLICIT_CONTEXT, "step_index", None)
    if explicit_step is None:
        with _LOCK:
            layer_call = _LAYER_CALLS.get(layer_index, 0)
            _LAYER_CALLS[layer_index] = layer_call + 1
        step_index = layer_call % num_steps
        prompt_index = layer_call // num_steps
    else:
        step_index = int(explicit_step)
        prompt_index = int(getattr(_EXPLICIT_CONTEXT, "prompt_index", 0))
    density, density_rule = _density_for_cell(layer_index, step_index)
    block_size = int(os.environ.get("SANA_PISA_BLOCK_SIZE", "64"))
    shape_key = (
        f"q={tuple(q.shape)};k={tuple(k.shape)};v={tuple(v.shape)};"
        f"dtype={q.dtype};layer={layer_index};step={step_index};prompt={prompt_index}"
    )

    if _policy_mode(layer_index, step_index) == "dense":
        _record_branch(layer_index, step_index, "dense_policy", shape_key, 1.0)
        return dense_fn()

    fallback_reason = ""
    if q.device.type != "cuda":
        fallback_reason = "non_cuda"
    elif q.shape != k.shape or q.shape != v.shape:
        fallback_reason = "qkv_shape_or_gqa_mismatch"
    elif q.ndim != 4:
        fallback_reason = "rank_not_four"
    elif density >= 1.0:
        fallback_reason = "density_ge_1"
    elif density <= 0.0:
        fallback_reason = "density_le_0"
    elif not _env_flag("SANA_PISA_APPROX_REMAINDER", True):
        raise RuntimeError("SANA PISA requires approx_remainder=True")

    if fallback_reason:
        _record_branch(layer_index, step_index, "dense_fallback", shape_key, density)
        with _LOCK:
            _STATS.setdefault("fallback_reasons", {}).setdefault(fallback_reason, 0)
            _STATS["fallback_reasons"][fallback_reason] += 1
        return dense_fn()

    block_size = min(block_size, q.shape[-2], k.shape[-2])
    triton.set_allocator(_make_tma_allocator())
    route_start = torch.cuda.Event(enable_timing=True)
    route_end = torch.cuda.Event(enable_timing=True)
    kernel_start = torch.cuda.Event(enable_timing=True)
    kernel_end = torch.cuda.Event(enable_timing=True)

    route_start.record()
    qc, kc, vc, k_var = chunk_reduce_qkv(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        include_v_centroid=True,
    )
    block_indices = taylor_error_block_indices(
        qc=qc,
        kc=kc,
        k_var=k_var,
        density=density,
        scale=scale,
    )
    route_end.record()

    kernel_start.record()
    output, _ = piecewise_attn_fwd(
        q=q,
        k=k,
        v=v,
        kc=kc,
        vc=vc,
        block_indices=block_indices,
        block_size=block_size,
        scale=scale,
        approx_remainder=True,
    )
    kernel_end.record()

    _record_branch(layer_index, step_index, "pisa_dispatch", shape_key, density)
    with _LOCK:
        _STATS["calls"]["exact_phase"] += 1
        _STATS["calls"]["approximate_remainder_phase"] += 1
        _PENDING_EVENTS.append(
            (route_start, route_end, kernel_start, kernel_end, layer_index, step_index)
        )
        exact_blocks = int(block_indices.shape[-1])
        key_blocks = int(kc.shape[-2])
        _STATS["last_dispatch"] = {
            "layer_index": layer_index,
            "step_index": step_index,
            "prompt_index": prompt_index,
            "block_size": block_size,
            "kernel_num_stages": _piecewise_num_stages(),
            "key_blocks": key_blocks,
            "exact_blocks_per_query": exact_blocks,
            "configured_density": density,
            "density_rule": density_rule,
            "actual_density": exact_blocks / key_blocks,
            "actual_sparsity": 1.0 - exact_blocks / key_blocks,
            "dtype": str(q.dtype),
            "shape": list(q.shape),
        }
    return output
