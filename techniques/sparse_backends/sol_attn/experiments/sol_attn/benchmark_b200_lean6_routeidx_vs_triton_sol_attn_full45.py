#!/usr/bin/env python3
"""Source-bound SM100 full45 timing harness with separate semantic and timing controls."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from experiments.sol_attn.bf16_triton_sol_attn_contract import (
    BATCH_HEAD_PAIRS,
    CONTRACT_PATH,
    DENSITIES,
    G_CANDIDATES,
    OUTPUT_LIMITS,
    TOKENS,
    load_and_validate_contract,
)
from experiments.sol_attn.check_b200_lean6_routeidx_full45_correctness import (
    _calibrate_density as _shared_calibrate_density,
    _prepare_qkv as _shared_prepare_qkv,
    _runtime_signature as _shared_runtime_signature,
    _tensor_sha256,
    _normalize_route_trace,
    _unpack_route_trace,
    case_id,
    summarize_correctness_rows,
)
from experiments.sol_attn.full45_candidate_backends import (
    BACKENDS as FULL45_CANDIDATE_BACKENDS,
    DEFAULT_BACKEND,
    get_backend,
    load_runner_factories,
    source_paths as candidate_source_paths,
)


CONTRACT = "b200_lean6_routeidx_vs_triton_sol_attn_full45_v1"
BACKENDS = ("lean6_routeidx", "triton_sol_attn_R_perf")
RATIO_KEY = "candidate_over_triton_sol_attn"
BLOCK_SIZE = 64
HEAD_DIM = 128
GROUP_SIZE = 64
G_TUNE_WARMUP = 3
G_TUNE_REPS = 7
FORMAL_WARMUP = 20
REPS_PER_BLOCK = 30
ROUNDS = 2
SELECTION_RULE = (
    "minimum_7_launch_cuda_event_median_after_3_warmups_smallest_g_tiebreak"
)
MAX_GPU_TEMPERATURE_C = 85.0
MAX_POWER_LIMIT_OVERSHOOT_PCT = 5.0
CRITICAL_CLOCK_REASONS = (
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "sw_thermal_slowdown",
)
TELEMETRY_FIELDS = (
    "index",
    "uuid",
    "pstate",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.sm",
    "clocks.current.memory",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)

ROOT = Path(__file__).resolve().parents[2]
HARNESS_SOURCE = Path(__file__).resolve()
CORRECTNESS_SOURCE = (
    ROOT / "experiments/sol_attn/check_b200_lean6_routeidx_full45_correctness.py"
)
SOURCE_PATHS = candidate_source_paths(
    ROOT,
    DEFAULT_BACKEND,
    correctness_harness=CORRECTNESS_SOURCE,
    timing_harness=HARNESS_SOURCE,
)
KERNEL_SOURCE = SOURCE_PATHS["kernel"]
RUNNER_SOURCE = SOURCE_PATHS["runner"]
SEMANTIC_SOURCE = SOURCE_PATHS["semantic_kernel"]
PARENT_KERNEL_SOURCE = SOURCE_PATHS["parent_kernel"]
PARENT_RUNNER_SOURCE = SOURCE_PATHS["parent_runner"]
REFERENCE_SOURCE = SOURCE_PATHS["prepared_reference"]
SEMANTIC_HELPER_SOURCE = SOURCE_PATHS["semantic_helper"]
LEGACY_SOURCE = SOURCE_PATHS["legacy_kernel"]
ROUTING_SOURCE = SOURCE_PATHS["routing_kernel"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_source_identity(
    paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    selected = SOURCE_PATHS if paths is None else paths
    return {role: _sha256_file(path) for role, path in selected.items()}


def _prepare_qkv(
    torch: Any,
    aligned: Any,
    *,
    tokens: int,
    batch: int,
    heads: int,
    seed: int,
) -> tuple[Any, ...]:
    return _shared_prepare_qkv(
        torch, aligned, tokens=tokens, batch=batch, heads=heads, seed=seed
    )


def _calibrate_density(
    torch: Any,
    aligned: Any,
    canonical: Any,
    q: Any,
    kc: Any,
    unit_scale: Any,
    target_density: float,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[float, Any, int, float]:
    return _shared_calibrate_density(
        torch, aligned, canonical, q, kc, unit_scale, target_density,
        group_size=group_size,
    )


def _time_samples(
    torch: Any, call: Callable[[], Any], repetitions: int
) -> list[float]:
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        measured = float(start.elapsed_time(stop))
        if not math.isfinite(measured) or measured <= 0.0:
            raise RuntimeError(f"invalid CUDA event duration {measured!r}")
        samples.append(measured)
    return samples


def _config_record(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    kwargs = getattr(config, "kwargs", {})
    return {
        "kwargs": {
            str(name): value
            if isinstance(value, (bool, int, float, str)) or value is None
            else repr(value)
            for name, value in sorted(dict(kwargs).items())
        },
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
        "num_ctas": getattr(config, "num_ctas", None),
        "maxnreg": getattr(config, "maxnreg", None),
    }


def _autotune_triton_sol_attn(
    torch: Any,
    aligned: Any,
    q: Any,
    k: Any,
    v: Any,
    kc: Any,
    vc: Any,
    threshold: Any,
    scale: float,
    g_candidates: tuple[int, ...],
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Select the timing-only reference by replayable seven-launch medians."""

    cache = getattr(aligned.single_pass_dynamic_routing_kernel, "cache", None)
    if not isinstance(cache, dict):
        raise RuntimeError("cannot reset Triton performance-reference tune cache")
    cache_entries_cleared = len(cache)
    cache.clear()
    runners: dict[int, Callable[[], Any]] = {}
    records: list[dict[str, Any]] = []
    for group_size in g_candidates:
        runner = aligned.make_prepared_runner(
            q, k, v, kc, vc, threshold,
            group_size=group_size, block_size=BLOCK_SIZE, scale=scale,
        )
        started = time.perf_counter()
        output = runner()
        torch.cuda.synchronize()
        first_launch_s = time.perf_counter() - started
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError(f"Triton G{group_size} returned non-finite output")
        for _ in range(G_TUNE_WARMUP):
            runner()
        torch.cuda.synchronize()
        samples = _time_samples(torch, runner, G_TUNE_REPS)
        best_config = _config_record(
            getattr(aligned.single_pass_dynamic_routing_kernel, "best_config", None)
        )
        if best_config is None:
            raise RuntimeError(f"Triton G{group_size} best_config is unavailable")
        records.append(
            {
                "group_size": group_size,
                "warmup": G_TUNE_WARMUP,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "first_launch_s": first_launch_s,
                "best_config": best_config,
            }
        )
        runners[group_size] = runner
    selected_record = min(
        records, key=lambda record: (record["median_ms"], record["group_size"])
    )
    selected_group = int(selected_record["group_size"])
    runners[selected_group]()
    torch.cuda.synchronize()
    selected_config = _config_record(
        getattr(aligned.single_pass_dynamic_routing_kernel, "best_config", None)
    )
    if selected_config != selected_record["best_config"]:
        raise RuntimeError("Triton selected config changed after selected-G replay")
    return runners[selected_group], {
        "cache_entries_cleared": cache_entries_cleared,
        "tuning": records,
        "selected_group_size": selected_group,
        "selected_config": selected_config,
        "selection_rule": SELECTION_RULE,
    }


def _input_hashes(torch: Any, tensors: Mapping[str, Any]) -> dict[str, str]:
    return {name: _tensor_sha256(torch, tensor) for name, tensor in tensors.items()}


def _error_stats(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    denominator = torch.linalg.vector_norm(expected_f).clamp_min(1.0e-12)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(diff) / denominator).item()),
    }


def _stats_pass(stats: Mapping[str, float]) -> bool:
    return all(
        math.isfinite(float(stats[name]))
        and 0.0 <= float(stats[name]) <= limit
        for name, limit in OUTPUT_LIMITS.items()
    )


def _device_uuid(torch: Any) -> str:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    value = getattr(properties, "uuid", None)
    if value:
        text = str(value)
        return text if text.startswith("GPU-") else f"GPU-{text}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.startswith("GPU-"):
        return visible
    command = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    physical_index = int(visible) if visible.isdigit() else torch.cuda.current_device()
    if not 0 <= physical_index < len(uuids):
        raise RuntimeError("cannot bind CUDA device to an nvidia-smi UUID")
    uuid = uuids[physical_index]
    if not uuid.startswith("GPU-"):
        raise RuntimeError("nvidia-smi returned an invalid GPU UUID")
    return uuid


def _canonical_uuid(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text.replace("-", "")


def _parse_metric(value: str, label: str) -> float:
    try:
        measured = float(value.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid nvidia-smi {label}: {value!r}") from exc
    if not math.isfinite(measured):
        raise RuntimeError(f"non-finite nvidia-smi {label}: {value!r}")
    return measured


def _gpu_compute_processes(target_uuid: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
            f"--id={target_uuid}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi compute-process query failed: "
            f"returncode={result.returncode} stderr={result.stderr.strip()!r}"
        )
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]
    if any(len(row) != 3 for row in rows):
        raise RuntimeError(f"invalid nvidia-smi process rows: {rows}")
    target = _canonical_uuid(target_uuid)
    processes: list[dict[str, Any]] = []
    for gpu_uuid, pid, process_name in rows:
        if _canonical_uuid(gpu_uuid) != target:
            raise RuntimeError("nvidia-smi process UUID mismatch")
        try:
            numeric_pid = int(pid)
        except ValueError as exc:
            raise RuntimeError(f"invalid compute process PID {pid!r}") from exc
        processes.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": numeric_pid,
                "process_name": process_name,
            }
        )
    return sorted(processes, key=lambda process: (process["pid"], process["process_name"]))


def gpu_telemetry(torch: Any, target_uuid: str) -> dict[str, Any]:
    """Capture UUID-bound health telemetry outside all timed launches."""

    del torch
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
            "--format=csv,noheader,nounits",
            f"--id={target_uuid}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]
    matches = [
        row for row in rows
        if len(row) == len(TELEMETRY_FIELDS)
        and _canonical_uuid(row[1]) == _canonical_uuid(target_uuid)
    ]
    if len(matches) != 1:
        raise RuntimeError("nvidia-smi telemetry UUID binding failed")
    values = dict(zip(TELEMETRY_FIELDS, matches[0]))
    reasons: dict[str, bool] = {}
    for field in TELEMETRY_FIELDS:
        if not field.startswith("clocks_event_reasons."):
            continue
        state = values[field].strip().lower()
        if state not in {"active", "not active"}:
            raise RuntimeError(f"invalid nvidia-smi {field}: {values[field]!r}")
        reasons[field.rsplit(".", 1)[1]] = state == "active"
    controller_pid = os.getpid()
    processes = _gpu_compute_processes(target_uuid)
    foreign = sorted(
        {process["pid"] for process in processes if process["pid"] != controller_pid}
    )
    gpu_utilization = _parse_metric(values["utilization.gpu"], "utilization.gpu")
    memory_utilization = _parse_metric(
        values["utilization.memory"], "utilization.memory"
    )
    sm_clock = _parse_metric(values["clocks.current.sm"], "clocks.current.sm")
    memory_clock = _parse_metric(
        values["clocks.current.memory"], "clocks.current.memory"
    )
    temperature = _parse_metric(values["temperature.gpu"], "temperature.gpu")
    power = _parse_metric(values["power.draw"], "power.draw")
    power_limit = _parse_metric(values["power.limit"], "power.limit")
    invalid: list[str] = []
    if _canonical_uuid(values["uuid"]) != _canonical_uuid(target_uuid):
        invalid.append("telemetry_uuid_mismatch")
    try:
        index = int(_parse_metric(values["index"], "index"))
    except RuntimeError:
        index = -1
    if index < 0:
        invalid.append("invalid_gpu_index")
    if values["pstate"] != "P0":
        invalid.append(f"pstate:{values['pstate']}")
    invalid.extend(
        f"critical_clock_reason:{name}"
        for name in CRITICAL_CLOCK_REASONS
        if reasons[name]
    )
    if not 0.0 <= gpu_utilization <= 100.0:
        invalid.append("gpu_utilization_out_of_range")
    if not 0.0 <= memory_utilization <= 100.0:
        invalid.append("memory_utilization_out_of_range")
    if sm_clock <= 0.0 or memory_clock <= 0.0:
        invalid.append("nonpositive_clock")
    if temperature > MAX_GPU_TEMPERATURE_C:
        invalid.append(f"temperature_c>{MAX_GPU_TEMPERATURE_C:g}")
    if power <= 0.0 or power_limit <= 0.0:
        invalid.append("nonpositive_power_or_limit")
    elif power > power_limit * (1.0 + MAX_POWER_LIMIT_OVERSHOOT_PCT / 100.0):
        invalid.append(
            f"power_limit_overshoot_pct>{MAX_POWER_LIMIT_OVERSHOOT_PCT:g}"
        )
    if foreign:
        invalid.append(f"foreign_compute_pids:{foreign}")
    valid = not invalid
    return {
        "index": index,
        "uuid": values["uuid"],
        "target_uuid": target_uuid,
        "pstate": values["pstate"],
        "gpu_utilization_pct": gpu_utilization,
        "memory_utilization_pct": memory_utilization,
        "sm_clock_mhz": sm_clock,
        "memory_clock_mhz": memory_clock,
        "temperature_c": temperature,
        "power_w": power,
        "power_limit_w": power_limit,
        "clocks_event_reasons": reasons,
        "controller_pid": controller_pid,
        "compute_processes": processes,
        "foreign_compute_pids": foreign,
        "snapshot_invalid_reasons": invalid,
        "snapshot_valid": valid,
        "telemetry_valid": valid,
    }


def _snapshot(
    torch: Any,
    trace_runner: Any,
    immutable: Mapping[str, Any],
    expected_sources: Mapping[str, str],
    expected_route_sha: str,
    target_uuid: str,
    num_blocks: int,
    group_size: int,
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    trace_runner.route_mask_trace.fill_(-123456789)
    trace_runner()
    torch.cuda.synchronize()
    normalized_trace = _normalize_route_trace(
        torch,
        trace_runner.route_mask_trace,
        num_blocks,
        group_size,
    )
    snapshot = {
        "source_identity": _current_source_identity(source_paths),
        "input_hashes": _input_hashes(torch, immutable),
        "route_trace_sha256": _tensor_sha256(torch, normalized_trace),
        "telemetry": gpu_telemetry(torch, target_uuid),
    }
    if snapshot["source_identity"] != dict(expected_sources):
        raise RuntimeError("source identity changed during timing")
    if snapshot["route_trace_sha256"] != expected_route_sha:
        raise RuntimeError("route trace changed during timing")
    if snapshot["telemetry"]["telemetry_valid"] is not True:
        raise RuntimeError("GPU telemetry failed health or UUID binding")
    return snapshot


def _correctness_gate(
    torch: Any,
    candidate: Any,
    trace_runner: Any,
    semantic_runner: Callable[[], Any],
    expected_route: Any,
    semantic: Any,
    group_size: int = GROUP_SIZE,
) -> dict[str, Any]:
    candidate()
    torch.cuda.synchronize()
    first_o, first_lse = candidate.output.clone(), candidate.lse.clone()
    candidate()
    torch.cuda.synchronize()
    second_o, second_lse = candidate.output.clone(), candidate.lse.clone()
    trace_runner.route_mask_trace.fill_(-123456789)
    trace_runner()
    torch.cuda.synchronize()
    normalized_trace = _normalize_route_trace(
        torch,
        trace_runner.route_mask_trace,
        int(expected_route.shape[-2]),
        group_size,
    )
    trace_o = trace_runner.output.clone()
    trace_lse = trace_runner.lse.clone()
    if group_size == GROUP_SIZE:
        traced_route, trace_evidence = semantic._unpack_trace(
            torch, trace_runner.route_mask_trace, expected_route.shape[-2]
        )
    else:
        traced_route, trace_evidence = _unpack_route_trace(
            torch,
            normalized_trace,
            expected_route.shape[-2],
            group_size,
        )
    semantic_o = semantic_runner().clone()
    torch.cuda.synchronize()
    error = _error_stats(torch, second_o, semantic_o)
    evidence = {
        "route_bitwise": bool(torch.equal(traced_route, expected_route)),
        "route_packet_valid": bool(trace_evidence["passes"]),
        "output_pass": _stats_pass(error),
        "output_repeatable": bool(torch.equal(first_o, second_o)),
        "lse_repeatable": bool(torch.equal(first_lse, second_lse)),
        "output_finite": bool(torch.isfinite(second_o).all().item()),
        "lse_finite": bool(torch.isfinite(second_lse).all().item()),
        "trace_ordinary_output_bitwise": bool(torch.equal(trace_o, second_o)),
        "trace_ordinary_lse_bitwise": bool(torch.equal(trace_lse, second_lse)),
        "output_error": error,
    }
    evidence["passes"] = all(
        evidence[name]
        for name in (
            "route_bitwise", "route_packet_valid", "output_pass",
            "output_repeatable", "lse_repeatable", "output_finite", "lse_finite",
            "trace_ordinary_output_bitwise", "trace_ordinary_lse_bitwise",
        )
    )
    return evidence


def _run_case(
    torch: Any,
    aligned: Any,
    canonical: Any,
    semantic: Any,
    make_candidate: Any,
    *,
    correctness: Mapping[str, Any],
    correctness_summary_sha256: str,
    full45_contract_sha256: str,
    source_manifest_sha256: str,
    runtime_signature: Mapping[str, Any],
    source_paths: Mapping[str, Path] | None = None,
    candidate_backend: str = DEFAULT_BACKEND,
) -> dict[str, Any]:
    backend = get_backend(candidate_backend)
    group_size = backend.route_group_size
    correctness_backend = correctness.get(
        "candidate_backend", candidate_backend
    )
    if correctness_backend != candidate_backend:
        raise RuntimeError("timing candidate does not match correctness backend")
    tokens = int(correctness["T"])
    batch = int(correctness["B"])
    heads = int(correctness["H"])
    density = float(correctness["target_density"])
    seed = int(correctness["seed"])
    q, k, v, kc, vc, unit_scale = _prepare_qkv(
        torch, aligned, tokens=tokens, batch=batch, heads=heads, seed=seed
    )
    tau, threshold, exact_count, realized = _calibrate_density(
        torch, aligned, canonical, q, kc, unit_scale, density,
        group_size=group_size,
    )
    immutable = {
        "q": q, "k": k, "v": v, "kc": kc, "vc": vc, "threshold": threshold,
    }
    hashes = _input_hashes(torch, immutable)
    sources = _current_source_identity(source_paths)
    if hashes != correctness["input_hashes_before"]:
        raise RuntimeError("prepared inputs do not match all-position correctness row")
    if sources != correctness["source_identity"]:
        raise RuntimeError("timing sources do not match all-position correctness row")
    if dict(runtime_signature) != correctness["runtime_signature"]:
        raise RuntimeError("timing runtime does not match all-position correctness row")
    if abs(realized - float(correctness["realized_density"])) > 0.0:
        raise RuntimeError("density calibration does not reproduce correctness row")
    scale = HEAD_DIM**-0.5
    candidate = make_candidate(
        tokens, q, k, v, kc, vc, threshold, scale, trace_route_masks=False
    )
    trace_runner = make_candidate(
        tokens, q, k, v, kc, vc, threshold, scale, trace_route_masks=True
    )
    semantic_runner = aligned.make_prepared_runner(
        q, k, v, kc, vc, threshold,
        group_size=group_size, block_size=BLOCK_SIZE, scale=scale,
    )
    expected_route = aligned.materialize_route_mask(
        q, kc, threshold,
        group_size=group_size, block_size=BLOCK_SIZE, scale=scale,
    ).to(torch.uint8)
    expected_route_sha = correctness["route"]["trace_sha256"]
    target_uuid = str(runtime_signature["uuid"])
    identity_before = _snapshot(
        torch,
        trace_runner,
        immutable,
        sources,
        expected_route_sha,
        target_uuid,
        int(expected_route.shape[-2]),
        group_size,
        source_paths,
    )
    first_gate = _correctness_gate(
        torch, candidate, trace_runner, semantic_runner, expected_route,
        semantic, group_size,
    )
    if not first_gate["passes"]:
        raise RuntimeError("fresh candidate semantic gate failed before timing")
    if _input_hashes(torch, immutable) != hashes:
        raise RuntimeError("prepared inputs changed during correctness gate")

    performance_runner, tuning = _autotune_triton_sol_attn(
        torch, aligned, q, k, v, kc, vc, threshold, scale, G_CANDIDATES
    )
    runners = {BACKENDS[0]: candidate, BACKENDS[1]: performance_runner}
    for backend in BACKENDS:
        for _ in range(FORMAL_WARMUP):
            runners[backend]()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    blocks: list[dict[str, Any]] = []
    orders = [list(BACKENDS), list(reversed(BACKENDS))]
    for round_index, order in enumerate(orders):
        for backend in order:
            before = _snapshot(
                torch,
                trace_runner,
                immutable,
                sources,
                expected_route_sha,
                target_uuid,
                int(expected_route.shape[-2]),
                group_size,
                source_paths,
            )
            block_samples = _time_samples(
                torch, runners[backend], REPS_PER_BLOCK
            )
            after = _snapshot(
                torch,
                trace_runner,
                immutable,
                sources,
                expected_route_sha,
                target_uuid,
                int(expected_route.shape[-2]),
                group_size,
                source_paths,
            )
            samples[backend].extend(block_samples)
            blocks.append(
                {
                    "round_index": round_index,
                    "backend": backend,
                    "samples_ms": block_samples,
                    "median_ms": statistics.median(block_samples),
                    "before": before,
                    "after": after,
                }
            )
    medians = {
        backend: statistics.median(samples[backend]) for backend in BACKENDS
    }
    final_gate = _correctness_gate(
        torch, candidate, trace_runner, semantic_runner, expected_route,
        semantic, group_size,
    )
    identity_after = _snapshot(
        torch,
        trace_runner,
        immutable,
        sources,
        expected_route_sha,
        target_uuid,
        int(expected_route.shape[-2]),
        group_size,
        source_paths,
    )
    if not final_gate["passes"] or _input_hashes(torch, immutable) != hashes:
        raise RuntimeError("final correctness or immutable-input gate failed")
    ratio = medians[BACKENDS[0]] / medians[BACKENDS[1]]
    row = {
        "contract": CONTRACT,
        "candidate_backend": candidate_backend,
        "case_id": correctness["case_id"],
        "T": tokens, "B": batch, "H": heads, "D": HEAD_DIM,
        "target_density": density,
        "realized_density": realized,
        "seed": seed,
        "source_identity": sources,
        "runtime_signature": dict(runtime_signature),
        "input_hashes": hashes,
        "route_trace_sha256": expected_route_sha,
        "edge_receipt_sha256": correctness["edge_receipt_sha256"],
        "correctness_summary_sha256": correctness_summary_sha256,
        "full45_contract_sha256": full45_contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "correctness_row_sha256": correctness["case_row_sha256"],
        "r_sem": {
            "group_size": group_size,
            "route_bitwise": bool(first_gate["route_bitwise"] and final_gate["route_bitwise"]),
            "output_pass": bool(first_gate["output_pass"] and final_gate["output_pass"]),
            "trace_ordinary_output_bitwise": bool(
                first_gate["trace_ordinary_output_bitwise"]
                and final_gate["trace_ordinary_output_bitwise"]
            ),
            "trace_ordinary_lse_bitwise": bool(
                first_gate["trace_ordinary_lse_bitwise"]
                and final_gate["trace_ordinary_lse_bitwise"]
            ),
            "pre_timing_pass": bool(first_gate["passes"]),
            "post_timing_pass": bool(final_gate["passes"]),
        },
        "r_perf": tuning,
        "timing": {
            "warmup": FORMAL_WARMUP,
            "repetitions": ROUNDS * REPS_PER_BLOCK,
            "orders": orders,
            "blocks": blocks,
            "samples_ms": samples,
            "medians_ms": medians,
        },
        "ratios": {RATIO_KEY: ratio},
        "identity_before_correctness": identity_before,
        "identity_after_correctness": identity_after,
        "correctness_pass": True,
        "calibration": {
            "tau": tau,
            "exact_count": exact_count,
            "realized_density": realized,
        },
        "runtime_diagnostics": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_uuid": target_uuid,
            "candidate_compile_s": float(candidate.compile_s),
            "trace_compile_s": float(trace_runner.compile_s),
        },
    }
    del runners, candidate, trace_runner, performance_runner, semantic_runner
    del q, k, v, kc, vc, unit_scale, threshold, immutable, expected_route
    torch.cuda.empty_cache()
    gc.collect()
    return row


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _parse_pairs(raw: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in raw.split(","):
        fields = item.strip().lower().split("x")
        if len(fields) != 2:
            raise ValueError(f"invalid BxH pair {item!r}")
        result.append((int(fields[0]), int(fields[1])))
    return tuple(result)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-summary", type=Path, required=True)
    parser.add_argument("--correctness-summary-sha256", required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    parser.add_argument("--batch-head-pairs", default=",".join(f"{b}x{h}" for b, h in BATCH_HEAD_PAIRS))
    parser.add_argument("--densities", default=",".join(map(str, DENSITIES)))
    parser.add_argument("--g-candidates", default=",".join(map(str, G_CANDIDATES)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=FORMAL_WARMUP)
    parser.add_argument("--reps-per-round", type=int, default=REPS_PER_BLOCK)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument(
        "--candidate-backend",
        choices=tuple(FULL45_CANDIDATE_BACKENDS),
        default=DEFAULT_BACKEND,
    )
    args = parser.parse_args(argv)
    args.tokens = _parse_ints(args.tokens)
    args.batch_head_pairs = _parse_pairs(args.batch_head_pairs)
    args.densities = _parse_floats(args.densities)
    args.g_candidates = _parse_ints(args.g_candidates)
    if not set(args.tokens).issubset(TOKENS):
        parser.error("tokens are outside the frozen full45 grid")
    if args.batch_head_pairs != BATCH_HEAD_PAIRS or args.densities != DENSITIES:
        parser.error("formal shards require the frozen B/H and density axes")
    if args.g_candidates != G_CANDIDATES:
        parser.error("formal timing requires G16/G32/G64/G128")
    if (args.seed, args.warmup, args.reps_per_round, args.rounds) != (0, 20, 30, 2):
        parser.error("formal timing requires seed=0, 20 warmups, and two 30-launch blocks")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    backend = get_backend(args.candidate_backend)
    selected_sources = candidate_source_paths(
        ROOT,
        backend.name,
        correctness_harness=CORRECTNESS_SOURCE,
        timing_harness=HARNESS_SOURCE,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_summary_sha = args.correctness_summary_sha256
    if len(expected_summary_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_summary_sha):
        raise ValueError("invalid correctness summary SHA256")
    if _sha256_file(args.correctness_summary) != expected_summary_sha:
        raise RuntimeError("correctness summary SHA256 mismatch")
    contract = load_and_validate_contract(args.contract)
    del contract
    contract_sha256 = _sha256_file(args.contract)
    raw_summary = json.loads(args.correctness_summary.read_text(encoding="utf-8"))
    correctness = summarize_correctness_rows(
        raw_summary.get("cases", []),
        edge_receipt_sha256=raw_summary.get("edge_receipt_sha256"),
        contract_sha256=raw_summary.get("full45_contract_sha256"),
        edge_receipt_binding=raw_summary.get("edge_receipt_binding"),
        source_manifest_sha256=raw_summary.get("source_manifest_sha256"),
    )
    if (
        raw_summary.get("passes") is not True
        or raw_summary.get("case_count") != 45
        or raw_summary.get("candidate_backend", correctness["candidate_backend"])
        != backend.name
        or correctness["candidate_backend"] != backend.name
        or correctness["passes"] is not True
        or raw_summary.get("source_identity") != correctness["source_identity"]
        or raw_summary.get("runtime_signatures") != correctness["runtime_signatures"]
        or raw_summary.get("edge_receipt_binding") != correctness["edge_receipt_binding"]
        or raw_summary.get("source_manifest_sha256")
        != correctness["source_manifest_sha256"]
        or correctness["full45_contract_sha256"] != contract_sha256
    ):
        raise RuntimeError("all-position correctness summary did not pass")
    selected = [row for row in correctness["cases"] if row["T"] in args.tokens]
    expected_count = len(args.tokens) * len(BATCH_HEAD_PAIRS) * len(DENSITIES)
    if len(selected) != expected_count:
        raise RuntimeError("correctness summary does not cover requested shard")
    import torch
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (10, 0):
        raise RuntimeError("SM100 CUDA device required")
    import triton
    def allocate(size: int, alignment: int, stream: Any) -> Any:
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocate)
    from experiments.sol_attn import check_bf16_cutedsl_semantics as semantic
    from kernels import sol_attention_bf16_aligned as aligned
    from kernels import sol_attention as canonical
    make_candidate, _ = load_runner_factories(backend.name)
    runtime = _shared_runtime_signature(torch, triton)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for correctness_row in selected:
            row = _run_case(
                torch, aligned, canonical, semantic,
                make_candidate,
                correctness=correctness_row,
                correctness_summary_sha256=expected_summary_sha,
                full45_contract_sha256=contract_sha256,
                source_manifest_sha256=correctness["source_manifest_sha256"],
                runtime_signature=runtime,
                source_paths=selected_sources,
                candidate_backend=backend.name,
            )
            handle.write(json.dumps({"event": "case", **row}, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()


__all__ = [
    "BACKENDS",
    "CONTRACT",
    "KERNEL_SOURCE",
    "RATIO_KEY",
    "RUNNER_SOURCE",
]
