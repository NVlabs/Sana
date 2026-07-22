"""Wan-facing adapter for the evidence-bound Sol-Attn SM100 colmask kernel."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch


_GATE_DONE = False
_EVENT_COUNT = 0
_COMPILED_OPS: dict[tuple[str, int, int, int], dict[str, Any]] = {}


def _load_release() -> tuple[Any, Any]:
    from kernels import sol_attention_bf16_aligned as aligned
    from kernels import sol_attention as canonical

    return aligned, canonical


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int) -> None:
    if block_size != 64:
        raise ValueError(f"SM100 colmask fixes block_size=64, got {block_size}")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("SM100 colmask requires equal [B,H,T,128] Q/K/V")
    if q.shape[-1] != 128:
        raise ValueError(f"SM100 colmask requires head_dim=128, got {q.shape[-1]}")
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
        raise TypeError("SM100 colmask requires BF16 Q/K/V")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("SM100 colmask requires Q/K/V on one CUDA device")
    if tuple(torch.cuda.get_device_capability(q.device)) != (10, 0):
        raise RuntimeError("SM100 colmask requires a compute capability 10.0 GPU")


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / denominator).item()),
    }


def _run_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    threshold: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, dict[str, Any], bool]:
    """Compile once per Wan shape, then launch with fresh tensor pointers."""

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from experiments.sol_attn.native_bf16_claude50_colmask_full45_runner import (
        _validate_kernel_identity,
    )
    from experiments.sol_attn.native_bf16_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner import (
        _to_cute_tensor,
        _validate_prepared,
    )
    from kernels.sol_attn_sm100.native_bf16_claude49_g256_colmask_fwd import (
        build_sol_attn_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_fwd,
    )

    tokens = int(q.shape[2])
    _validate_prepared(tokens, q, k, v, kc, vc, threshold, scale)
    output = torch.full_like(v, float("nan"))
    lse = torch.full(
        (q.shape[0], q.shape[1], tokens),
        float("nan"),
        device=q.device,
        dtype=torch.float32,
    )
    op_args = [
        _to_cute_tensor(q),
        _to_cute_tensor(k),
        _to_cute_tensor(v),
        _to_cute_tensor(output),
        _to_cute_tensor(kc),
        _to_cute_tensor(vc),
        _to_cute_tensor(threshold),
        _to_cute_tensor(lse),
        float(scale),
    ]
    _cur_stream = torch.cuda.current_stream()
    _raw_stream = getattr(_cur_stream, "cuda_stream", None)
    if _raw_stream is None:
        try:
            _raw_stream = torch.cuda.Stream(
                stream_id=_cur_stream.stream_id,
                device_index=_cur_stream.device_index,
                device_type=_cur_stream.device_type,
            ).cuda_stream
        except Exception:
            _raw_stream = torch.cuda.default_stream().cuda_stream
    stream = cuda.CUstream(_raw_stream)
    key = (str(q.device), tokens, int(q.shape[0]), int(q.shape[1]))
    entry = _COMPILED_OPS.get(key)
    cache_hit = entry is not None
    if entry is None:
        kernel_sha256 = _validate_kernel_identity()
        op = build_sol_attn_sm100_lean6_routeidx_g256_cursor_ballotscatter_packedgmemo_tworowhoist_inlineu32_fusedroute_n128_pair_tmemp_bf16_fwd(
            tokens
        )
        started = time.perf_counter()
        compiled_op = cute.compile(
            op,
            *op_args,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        entry = {
            "compiled_op": compiled_op,
            "compile_s": time.perf_counter() - started,
            "kernel_sha256": kernel_sha256,
        }
        _COMPILED_OPS[key] = entry
    entry["compiled_op"](*op_args, stream=stream)
    return output, lse, entry, cache_hit


def _emit_event(payload: dict[str, Any]) -> None:
    global _EVENT_COUNT
    _EVENT_COUNT += 1
    event = {"event": "wan_sol_attn_sm100_colmask", "index": _EVENT_COUNT, **payload}
    print("WAN_SOL_ATTN_SM100 " + json.dumps(event, sort_keys=True), flush=True)
    path = os.getenv("SOL_ATTN_SM100_EVENT_LOG")
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


@torch.no_grad()
def calibrate_tau(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    target_density: float,
    block_size: int = 64,
    iterations: int = 12,
) -> dict[str, Any]:
    """Calibrate BF16 route density on a deterministic head sample."""

    _validate_qkv(q, k, v, block_size)
    if not 0.0 < target_density <= 1.0:
        raise ValueError(f"target_density must be in (0, 1], got {target_density}")
    aligned, canonical = _load_release()

    sample_heads = max(1, int(os.getenv("SOL_ATTN_CALIBRATE_SAMPLE_HEADS", "2")))
    q_sample = q[:1, :sample_heads].contiguous()
    k_sample = k[:1, :sample_heads].contiguous()
    v_sample = v[:1, :sample_heads].contiguous()
    scale = q.shape[-1] ** -0.5
    kc, _, threshold1, unit_scale, _ = aligned.prepare_qkv(
        q_sample,
        k_sample,
        v_sample,
        tau=1.0,
        block_size=block_size,
        scale=scale,
    )
    threshold2 = canonical.compute_global_qck_threshold(
        q_sample,
        unit_scale,
        kc,
        unit_scale,
        scale,
        block_size,
        2.0,
    )
    slope = threshold2 - threshold1
    trials: list[dict[str, float]] = []

    def evaluate(tau: float) -> float:
        tau = float(torch.tensor(tau, dtype=torch.float32).item())
        threshold = (threshold1 + (tau - 1.0) * slope).contiguous()
        route = aligned.materialize_route_mask(
            q_sample,
            kc,
            threshold,
            group_size=64,
            block_size=block_size,
            scale=scale,
        )
        density = float(route.float().mean().item())
        trials.append({"tau": tau, "density": density})
        return density

    low = float(os.getenv("SOL_ATTN_CALIBRATE_COARSE_START", "0.0"))
    high = float(os.getenv("SOL_ATTN_CALIBRATE_COARSE_END", "4.0"))
    evaluate(low)
    evaluate(high)
    for _ in range(iterations):
        middle = (low + high) * 0.5
        density = evaluate(middle)
        if density > target_density:
            low = middle
        else:
            high = middle

    best = min(trials, key=lambda row: abs(row["density"] - target_density))
    result = {
        "backend": "sol_attn_sm100_colmask_bf16",
        "threshold": best["tau"],
        "density": best["density"],
        "density_delta": best["density"] - target_density,
        "target_density": float(target_density),
        "sample_batch": 1,
        "sample_heads": int(q_sample.shape[1]),
        "sequence_length": int(q.shape[2]),
        "trials": trials,
    }
    _emit_event({"phase": "calibration", **result})
    return result


@torch.no_grad()
def _correctness_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    block_size: int,
) -> None:
    global _GATE_DONE
    if _GATE_DONE or os.getenv("SOL_ATTN_SM100_CORRECTNESS_GATE", "1") != "1":
        return

    aligned, _ = _load_release()
    q_gate = q[:1, :1].contiguous()
    k_gate = k[:1, :1].contiguous()
    v_gate = v[:1, :1].contiguous()
    scale = q.shape[-1] ** -0.5
    kc, vc, threshold, _, _ = aligned.prepare_qkv(
        q_gate, k_gate, v_gate, tau=tau, block_size=block_size, scale=scale
    )
    candidate_output, _candidate_lse, candidate, cache_hit = _run_compiled(
        q_gate, k_gate, v_gate, kc, vc, threshold, scale
    )
    reference = aligned.make_prepared_runner(
        q_gate,
        k_gate,
        v_gate,
        kc,
        vc,
        threshold,
        group_size=64,
        block_size=block_size,
        scale=scale,
    )
    reference_output = reference()
    torch.cuda.synchronize(q.device)
    stats = _error_stats(candidate_output, reference_output)
    limits = {"max_abs": 0.08, "mean_abs": 0.01, "rel_l2": 0.01}
    passed = all(math.isfinite(stats[name]) and stats[name] <= limit for name, limit in limits.items())
    _emit_event(
        {
            "phase": "correctness_gate",
            "passed": passed,
            "shape": list(q_gate.shape),
            "tau": float(tau),
            "compile_s": float(candidate["compile_s"]),
            "compile_cache_hit": cache_hit,
            "kernel_sha256": candidate["kernel_sha256"],
            "stats": stats,
            "limits": limits,
        }
    )
    if not passed:
        raise RuntimeError(f"SM100 colmask correctness gate failed: {stats} vs {limits}")
    _GATE_DONE = True


@torch.no_grad()
def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    block_size: int = 64,
    return_lse: bool = False,
) -> torch.Tensor:
    """Run the release SM100 colmask kernel on Wan BHTD tensors.

    When ``return_lse=True`` returns ``(output, lse)`` where ``lse`` is the
    per-query log-sum-exp over the routed keys ([B, H, T], fp32). The kernel
    already computes it; exposing it lets a caller online-softmax-merge the
    sparse video x video result with a dense video x text tail (HunyuanVideo),
    which is how SOL stays exact under a joint video+text sequence with a
    text-padding mask the kernel itself does not consume.
    """

    _validate_qkv(q, k, v, block_size)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    _correctness_gate(q, k, v, tau=tau, block_size=block_size)

    aligned, _ = _load_release()
    scale = q.shape[-1] ** -0.5
    kc, vc, threshold, _, _ = aligned.prepare_qkv(
        q, k, v, tau=tau, block_size=block_size, scale=scale
    )
    output, lse, runner, cache_hit = _run_compiled(q, k, v, kc, vc, threshold, scale)
    if _EVENT_COUNT < 4:
        _emit_event(
            {
                "phase": "forward",
                "shape": list(q.shape),
                "tau": float(tau),
                "compile_s": float(runner["compile_s"]),
                "compile_cache_hit": cache_hit,
                "kernel_sha256": runner["kernel_sha256"],
                "logical_group_size": 256,
                "physical_route_tile_size": 128,
            }
        )
    if return_lse:
        return output, lse
    return output


__all__ = ["calibrate_tau", "run"]
