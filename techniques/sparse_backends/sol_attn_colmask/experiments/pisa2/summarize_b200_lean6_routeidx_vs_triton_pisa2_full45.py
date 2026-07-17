#!/usr/bin/env python3
"""Replay and summarize the source-bound route-index full45 timing matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from experiments.pisa2.bf16_triton_pisa2_contract import (
    BATCH_HEAD_PAIRS,
    DENSITIES,
    G_CANDIDATES,
    TOKENS,
    validate_contract,
)
from experiments.pisa2.check_b200_lean6_routeidx_full45_correctness import (
    _canonical_uuid,
    _validate_runtime_signature,
    case_id,
    summarize_correctness_rows,
)
from experiments.pisa2.full45_candidate_backends import (
    BACKENDS as FULL45_CANDIDATE_BACKENDS,
)


SUMMARY_CONTRACT = "b200_lean6_routeidx_vs_triton_pisa2_full45_summary_v1"
TIMING_CONTRACT = "b200_lean6_routeidx_vs_triton_pisa2_full45_v1"
CANDIDATE = "lean6_routeidx"
BASELINE = "triton_pisa2_R_perf"
BACKENDS = (CANDIDATE, BASELINE)
RATIO_KEY = "candidate_over_triton_pisa2"
SELECTION_RULE = (
    "minimum_7_launch_cuda_event_median_after_3_warmups_smallest_g_tiebreak"
)
EXPECTED_ORDERS = [list(BACKENDS), list(reversed(BACKENDS))]
MAX_BLOCK_CLOCK_SPREAD_PCT = 1.0
MAX_MATCHED_CLOCK_DELTA_PCT = 1.0
MAX_MATCHED_TEMPERATURE_DELTA_C = 5.0
MAX_MATCHED_POWER_DELTA_PCT = 10.0
MAX_MATCHED_POWER_LIMIT_DELTA_PCT = 1.0
MAX_GPU_TEMPERATURE_C = 85.0
MAX_POWER_LIMIT_OVERSHOOT_PCT = 5.0
CRITICAL_CLOCK_REASONS = (
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "sw_thermal_slowdown",
)
CLOCK_REASON_FIELDS = {
    "sw_power_cap",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "sw_thermal_slowdown",
}


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label}: invalid value {result!r}")
    return result


def _same_number(actual: Any, expected: float, label: str) -> None:
    if _number(actual, label) != expected:
        raise ValueError(f"{label}: stored value does not replay exactly")


def _case_key(row: Mapping[str, Any]) -> tuple[int, int, int, float]:
    try:
        return (
            int(row["T"]), int(row["B"]), int(row["H"]),
            float(row["target_density"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid timing case key: {exc}") from exc


def _expected_keys() -> set[tuple[int, int, int, float]]:
    return {
        (tokens, batch, heads, density)
        for tokens in TOKENS
        for batch, heads in BATCH_HEAD_PAIRS
        for density in DENSITIES
    }


def _require_exact_mapping(
    actual: Any, expected: Mapping[str, Any], label: str
) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ValueError(f"{label}: identity mismatch")


def _percent_delta(first: float, second: float) -> float:
    if min(first, second) <= 0.0:
        raise ValueError("telemetry comparison requires positive values")
    return 100.0 * abs(first - second) / max(first, second)


def _telemetry_number(value: Any, label: str) -> float:
    measured = _number(value, label)
    if not math.isfinite(measured):
        raise ValueError(f"{label}: telemetry value is not finite")
    return measured


def _validate_telemetry(
    value: Any, *, runtime_uuid: str, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: telemetry object missing")
    required = {
        "index", "uuid", "target_uuid", "pstate",
        "gpu_utilization_pct", "memory_utilization_pct",
        "sm_clock_mhz", "memory_clock_mhz", "temperature_c",
        "power_w", "power_limit_w", "clocks_event_reasons",
        "controller_pid", "compute_processes", "foreign_compute_pids",
        "snapshot_invalid_reasons", "snapshot_valid", "telemetry_valid",
    }
    if set(value) != required:
        raise ValueError(f"{label}: telemetry fields mismatch")
    invalid: list[str] = []
    target = _canonical_uuid(value["target_uuid"])
    observed = _canonical_uuid(value["uuid"])
    runtime = _canonical_uuid(runtime_uuid)
    if not target or target != observed or target != runtime:
        invalid.append("telemetry_uuid_mismatch")
    index = value["index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        invalid.append("invalid_gpu_index")
    if value["pstate"] != "P0":
        invalid.append(f"pstate:{value['pstate']}")
    reasons = value["clocks_event_reasons"]
    if (
        not isinstance(reasons, Mapping)
        or set(reasons) != CLOCK_REASON_FIELDS
        or any(not isinstance(state, bool) for state in reasons.values())
    ):
        raise ValueError(f"{label}: telemetry clock reasons malformed")
    invalid.extend(
        f"critical_clock_reason:{name}"
        for name in CRITICAL_CLOCK_REASONS
        if reasons[name]
    )
    gpu_util = _telemetry_number(
        value["gpu_utilization_pct"], f"{label}.gpu_utilization_pct"
    )
    memory_util = _telemetry_number(
        value["memory_utilization_pct"], f"{label}.memory_utilization_pct"
    )
    sm_clock = _telemetry_number(value["sm_clock_mhz"], f"{label}.sm_clock_mhz")
    memory_clock = _telemetry_number(
        value["memory_clock_mhz"], f"{label}.memory_clock_mhz"
    )
    temperature = _telemetry_number(
        value["temperature_c"], f"{label}.temperature_c"
    )
    power = _telemetry_number(value["power_w"], f"{label}.power_w")
    power_limit = _telemetry_number(
        value["power_limit_w"], f"{label}.power_limit_w"
    )
    if not 0.0 <= gpu_util <= 100.0:
        invalid.append("gpu_utilization_out_of_range")
    if not 0.0 <= memory_util <= 100.0:
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
    controller_pid = value["controller_pid"]
    if (
        isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 0
    ):
        raise ValueError(f"{label}: telemetry controller PID invalid")
    processes = value["compute_processes"]
    if not isinstance(processes, list):
        raise ValueError(f"{label}: telemetry process list invalid")
    process_pids: list[int] = []
    for process in processes:
        if not isinstance(process, Mapping):
            raise ValueError(f"{label}: telemetry process record invalid")
        pid = process.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"{label}: telemetry process PID invalid")
        if _canonical_uuid(process.get("gpu_uuid")) != target:
            raise ValueError(f"{label}: telemetry process UUID mismatch")
        if not isinstance(process.get("process_name"), str):
            raise ValueError(f"{label}: telemetry process name invalid")
        process_pids.append(pid)
    foreign = sorted({pid for pid in process_pids if pid != controller_pid})
    if value["foreign_compute_pids"] != foreign:
        raise ValueError(f"{label}: telemetry foreign PID replay mismatch")
    if foreign:
        invalid.append(f"foreign_compute_pids:{foreign}")
    if value["snapshot_invalid_reasons"] != invalid:
        raise ValueError(f"{label}: telemetry invalid reasons do not replay")
    valid = not invalid
    if value["snapshot_valid"] is not valid or value["telemetry_valid"] is not valid:
        raise ValueError(f"{label}: telemetry validity boolean does not replay")
    if not valid:
        raise ValueError(f"{label}: telemetry is not healthy: {invalid}")
    return {
        "uuid": value["uuid"],
        "sm_clock_mhz": sm_clock,
        "memory_clock_mhz": memory_clock,
        "temperature_c": temperature,
        "power_w": power,
        "power_limit_w": power_limit,
    }


def _validate_snapshot(
    snapshot: Any,
    *,
    sources: Mapping[str, Any],
    inputs: Mapping[str, Any],
    route_trace_sha256: str,
    runtime_uuid: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"{label}: missing identity snapshot")
    _require_exact_mapping(snapshot.get("source_identity"), sources, f"{label}.source")
    _require_exact_mapping(snapshot.get("input_hashes"), inputs, f"{label}.inputs")
    if snapshot.get("route_trace_sha256") != route_trace_sha256:
        raise ValueError(f"{label}: route trace identity mismatch")
    return _validate_telemetry(
        snapshot.get("telemetry"), runtime_uuid=runtime_uuid,
        label=f"{label}.telemetry",
    )


def _replay_tuning(value: Any, label: str) -> int:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: missing performance-reference tuning")
    if value.get("selection_rule") != SELECTION_RULE:
        raise ValueError(f"{label}: selection rule mismatch")
    records = value.get("tuning")
    if not isinstance(records, list) or len(records) != len(G_CANDIDATES):
        raise ValueError(f"{label}: tuning must contain all four G candidates")
    medians: dict[int, float] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label}.tuning[{index}]: expected object")
        group = record.get("group_size")
        if isinstance(group, bool) or group not in G_CANDIDATES or group in medians:
            raise ValueError(f"{label}.tuning[{index}]: invalid or duplicate G")
        if record.get("warmup") != 3:
            raise ValueError(f"{label}.tuning[{index}]: warmup must be 3")
        samples = record.get("samples_ms")
        if not isinstance(samples, list) or len(samples) != 7:
            raise ValueError(f"{label}.tuning[{index}]: exactly seven samples required")
        replay_samples = [
            _number(sample, f"{label}.tuning[{index}].samples", positive=True)
            for sample in samples
        ]
        median = statistics.median(replay_samples)
        _same_number(record.get("median_ms"), median, f"{label}.tuning[{index}].median")
        medians[int(group)] = median
    if set(medians) != set(G_CANDIDATES):
        raise ValueError(f"{label}: missing tuning group")
    selected = min((median, group) for group, median in medians.items())[1]
    if value.get("selected_group_size") != selected:
        raise ValueError(f"{label}: selected G does not replay")
    return selected


def _replay_timing(
    value: Any,
    *,
    sources: Mapping[str, Any],
    inputs: Mapping[str, Any],
    route_trace_sha256: str,
    runtime_uuid: str,
    label: str,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: missing timing evidence")
    if value.get("warmup") != 20 or value.get("repetitions") != 60:
        raise ValueError(f"{label}: formal warmup/repetition contract mismatch")
    if value.get("orders") != EXPECTED_ORDERS:
        raise ValueError(f"{label}: both timing directions are required")
    blocks = value.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 4:
        raise ValueError(f"{label}: exactly four alternating timing blocks required")
    expected_sequence = [
        (round_index, backend)
        for round_index, order in enumerate(EXPECTED_ORDERS)
        for backend in order
    ]
    replay_samples: dict[str, list[float]] = {name: [] for name in BACKENDS}
    observed_uuid: str | None = None
    block_representatives: list[dict[str, float]] = []
    for index, (block, expected) in enumerate(zip(blocks, expected_sequence)):
        if not isinstance(block, Mapping):
            raise ValueError(f"{label}.blocks[{index}]: expected object")
        if (block.get("round_index"), block.get("backend")) != expected:
            raise ValueError(f"{label}.blocks[{index}]: timing direction mismatch")
        samples = block.get("samples_ms")
        if not isinstance(samples, list) or len(samples) != 30:
            raise ValueError(f"{label}.blocks[{index}]: exactly 30 samples required")
        numeric = [
            _number(sample, f"{label}.blocks[{index}].samples", positive=True)
            for sample in samples
        ]
        _same_number(
            block.get("median_ms"), statistics.median(numeric),
            f"{label}.blocks[{index}].median",
        )
        replay_samples[str(block["backend"])].extend(numeric)
        boundary: dict[str, dict[str, Any]] = {}
        for side in ("before", "after"):
            telemetry = _validate_snapshot(
                block.get(side), sources=sources, inputs=inputs,
                route_trace_sha256=route_trace_sha256,
                runtime_uuid=runtime_uuid,
                label=f"{label}.blocks[{index}].{side}",
            )
            boundary[side] = telemetry
            uuid = str(telemetry["uuid"])
            if observed_uuid is None:
                observed_uuid = uuid
            elif uuid != observed_uuid:
                raise ValueError(f"{label}: GPU UUID changed within case")
        before, after = boundary["before"], boundary["after"]
        for name in ("sm_clock_mhz", "memory_clock_mhz"):
            if _percent_delta(float(before[name]), float(after[name])) > MAX_BLOCK_CLOCK_SPREAD_PCT:
                raise ValueError(f"{label}: telemetry {name} spread exceeds limit")
        block_representatives.append(
            {
                "sm_clock_mhz": math.sqrt(
                    float(before["sm_clock_mhz"]) * float(after["sm_clock_mhz"])
                ),
                "memory_clock_mhz": math.sqrt(
                    float(before["memory_clock_mhz"])
                    * float(after["memory_clock_mhz"])
                ),
                "temperature_c": statistics.fmean(
                    [float(before["temperature_c"]), float(after["temperature_c"])]
                ),
                "power_w": math.sqrt(float(before["power_w"]) * float(after["power_w"])),
                "power_limit_w": math.sqrt(
                    float(before["power_limit_w"]) * float(after["power_limit_w"])
                ),
            }
        )
    for round_index in range(2):
        pair = block_representatives[round_index * 2 : round_index * 2 + 2]
        for name in ("sm_clock_mhz", "memory_clock_mhz"):
            if _percent_delta(pair[0][name], pair[1][name]) > MAX_MATCHED_CLOCK_DELTA_PCT:
                raise ValueError(f"{label}: matched telemetry {name} delta exceeds limit")
        if abs(pair[0]["temperature_c"] - pair[1]["temperature_c"]) > MAX_MATCHED_TEMPERATURE_DELTA_C:
            raise ValueError(f"{label}: matched telemetry temperature delta exceeds limit")
        if _percent_delta(pair[0]["power_limit_w"], pair[1]["power_limit_w"]) > MAX_MATCHED_POWER_LIMIT_DELTA_PCT:
            raise ValueError(f"{label}: matched telemetry power-limit delta exceeds limit")
    # Instantaneous power draw is both kernel-dependent and too noisy to gate a
    # short CUDA-event block.  Its absolute health and power-limit overshoot are
    # still validated in every snapshot; clock, temperature, and configured
    # power-limit matching remain hard ABBA environment checks.
    stored_samples = value.get("samples_ms")
    stored_medians = value.get("medians_ms")
    if not isinstance(stored_samples, Mapping) or set(stored_samples) != set(BACKENDS):
        raise ValueError(f"{label}: backend sample sets mismatch")
    if not isinstance(stored_medians, Mapping) or set(stored_medians) != set(BACKENDS):
        raise ValueError(f"{label}: backend medians mismatch")
    medians: dict[str, float] = {}
    for backend in BACKENDS:
        if len(replay_samples[backend]) != 60:
            raise ValueError(f"{label}: {backend} must have 60 samples")
        stored = stored_samples[backend]
        if not isinstance(stored, list):
            raise ValueError(f"{label}: {backend} sample list missing")
        stored_numeric = [
            _number(sample, f"{label}.{backend}.samples", positive=True)
            for sample in stored
        ]
        if stored_numeric != replay_samples[backend]:
            raise ValueError(f"{label}: {backend} raw block/sample mismatch")
        medians[backend] = statistics.median(replay_samples[backend])
        _same_number(
            stored_medians[backend], medians[backend],
            f"{label}.{backend}.formal_median",
        )
    return replay_samples, medians


def _validated_timing_row(
    raw: Mapping[str, Any],
    correctness: Mapping[str, Any],
    *,
    correctness_summary_sha256: str,
    full45_contract_sha256: str,
    source_manifest_sha256: str,
    expected_candidate_backend: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("timing row must be an object")
    row = dict(raw)
    if row.get("contract") != TIMING_CONTRACT:
        raise ValueError("timing contract mismatch")
    candidate_backend = row.get(
        "candidate_backend", expected_candidate_backend
    )
    if candidate_backend not in FULL45_CANDIDATE_BACKENDS:
        raise ValueError("unknown full45 candidate backend")
    if candidate_backend != expected_candidate_backend:
        raise ValueError("timing/correctness candidate backend mismatch")
    backend = FULL45_CANDIDATE_BACKENDS[candidate_backend]
    key = _case_key(row)
    expected_id = case_id(*key)
    if row.get("case_id") != expected_id:
        raise ValueError(f"{expected_id}: case identifier mismatch")
    for field in ("T", "B", "H", "D", "target_density", "realized_density", "seed"):
        if row.get(field) != correctness.get(field):
            raise ValueError(f"{expected_id}: timing/correctness {field} mismatch")
    sources = correctness["source_identity"]
    inputs = correctness["input_hashes_before"]
    route_sha = correctness["route"]["trace_sha256"]
    runtime = _validate_runtime_signature(
        row.get("runtime_signature"), f"{expected_id}.runtime_signature"
    )
    if runtime != correctness.get("runtime_signature"):
        raise ValueError(f"{expected_id}: runtime signature mismatch")
    _require_exact_mapping(row.get("source_identity"), sources, f"{expected_id}.source")
    _require_exact_mapping(row.get("input_hashes"), inputs, f"{expected_id}.inputs")
    if row.get("route_trace_sha256") != route_sha:
        raise ValueError(f"{expected_id}: route trace mismatch")
    if row.get("edge_receipt_sha256") != correctness.get("edge_receipt_sha256"):
        raise ValueError(f"{expected_id}: edge receipt mismatch")
    if row.get("correctness_summary_sha256") != correctness_summary_sha256:
        raise ValueError(f"{expected_id}: full correctness receipt mismatch")
    if row.get("full45_contract_sha256") != full45_contract_sha256:
        raise ValueError(f"{expected_id}: full45 contract SHA256 mismatch")
    if row.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError(f"{expected_id}: source manifest SHA256 mismatch")
    if row.get("correctness_row_sha256") != correctness.get("case_row_sha256"):
        raise ValueError(f"{expected_id}: correctness row SHA256 mismatch")
    if row.get("correctness_pass") is not True:
        raise ValueError(f"{expected_id}: correctness gate did not pass")
    semantic = row.get("r_sem")
    if not isinstance(semantic, Mapping) or dict(semantic) != {
        "group_size": backend.route_group_size,
        "route_bitwise": True,
        "output_pass": True,
        "trace_ordinary_output_bitwise": True,
        "trace_ordinary_lse_bitwise": True,
        "pre_timing_pass": True,
        "post_timing_pass": True,
    }:
        raise ValueError(f"{expected_id}: candidate semantic evidence invalid")
    selected = _replay_tuning(row.get("r_perf"), expected_id)
    _, medians = _replay_timing(
        row.get("timing"), sources=sources, inputs=inputs,
        route_trace_sha256=route_sha, runtime_uuid=str(runtime["uuid"]),
        label=expected_id,
    )
    telemetry_records = [
        _validate_snapshot(
            row.get(field), sources=sources, inputs=inputs,
            route_trace_sha256=route_sha, runtime_uuid=str(runtime["uuid"]),
            label=f"{expected_id}.{field}",
        )
        for field in ("identity_before_correctness", "identity_after_correctness")
    ]
    uuids = {record["uuid"] for record in telemetry_records}
    for block in row["timing"]["blocks"]:
        uuids.add(block["before"]["telemetry"]["uuid"])
        uuids.add(block["after"]["telemetry"]["uuid"])
    if len(uuids) != 1:
        raise ValueError(f"{expected_id}: identity snapshots span multiple GPUs")
    ratio = medians[CANDIDATE] / medians[BASELINE]
    ratios = row.get("ratios")
    if not isinstance(ratios, Mapping) or set(ratios) != {RATIO_KEY}:
        raise ValueError(f"{expected_id}: ratio field mismatch")
    _same_number(ratios[RATIO_KEY], ratio, f"{expected_id}.ratio")
    return {
        "case_id": expected_id,
        "candidate_backend": candidate_backend,
        "T": key[0], "B": key[1], "H": key[2],
        "target_density": key[3],
        "realized_density": float(row["realized_density"]),
        "selected_group_size": selected,
        "candidate_median_ms": medians[CANDIDATE],
        "triton_pisa2_median_ms": medians[BASELINE],
        RATIO_KEY: ratio,
        "correctness_row_sha256": row["correctness_row_sha256"],
        "input_hashes": dict(inputs),
        "route_trace_sha256": route_sha,
        "runtime_signature": runtime,
    }


def _geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _group_record(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row[RATIO_KEY]) for row in rows]
    return {
        "count": len(values),
        "geomean": _geomean(values),
        "min": min(values),
        "max": max(values),
        "wins": sum(value < 1.0 for value in values),
    }


def _groups(
    rows: Sequence[Mapping[str, Any]],
    key: Any,
    label: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    members: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        members.setdefault(label(key(row)), []).append(row)
    return (
        {name: _group_record(group) for name, group in sorted(members.items())},
        members,
    )


def summarize_full45(
    rows: Iterable[Mapping[str, Any]],
    correctness_summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    correctness_summary_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay all raw evidence and compute the strict initial R0 gate."""

    validate_contract(contract)
    if not isinstance(correctness_summary, Mapping):
        raise ValueError("correctness summary must be an object")
    correctness = summarize_correctness_rows(
        correctness_summary.get("cases", []),
        edge_receipt_sha256=correctness_summary.get("edge_receipt_sha256"),
        contract_sha256=correctness_summary.get("full45_contract_sha256"),
        edge_receipt_binding=correctness_summary.get("edge_receipt_binding"),
        source_manifest_sha256=correctness_summary.get("source_manifest_sha256"),
    )
    stored_candidate_backend = correctness_summary.get("candidate_backend")
    if (
        correctness_summary.get("passes") is not True
        or correctness_summary.get("case_count") != 45
        or correctness_summary.get("source_identity") != correctness["source_identity"]
        or correctness_summary.get("edge_receipt_sha256")
        != correctness["edge_receipt_sha256"]
        or correctness_summary.get("full45_contract_sha256")
        != correctness["full45_contract_sha256"]
        or correctness_summary.get("source_manifest_sha256")
        != correctness["source_manifest_sha256"]
        or correctness_summary.get("edge_receipt_binding")
        != correctness["edge_receipt_binding"]
        or correctness_summary.get("runtime_signatures")
        != correctness["runtime_signatures"]
        or (
            stored_candidate_backend is not None
            and stored_candidate_backend != correctness["candidate_backend"]
        )
    ):
        raise ValueError("full45 correctness summary did not pass")
    correctness_by_id = {row["case_id"]: row for row in correctness["cases"]}
    materialized = list(rows)
    if correctness_summary_sha256 is None:
        receipt_values = {
            row.get("correctness_summary_sha256") for row in materialized
        }
        if len(receipt_values) != 1:
            raise ValueError("timing rows do not share one correctness receipt")
        correctness_summary_sha256 = receipt_values.pop()
    if (
        not isinstance(correctness_summary_sha256, str)
        or len(correctness_summary_sha256) != 64
        or any(character not in "0123456789abcdef" for character in correctness_summary_sha256)
    ):
        raise ValueError("invalid correctness summary SHA256")
    full45_contract_sha256 = correctness["full45_contract_sha256"]
    source_manifest_sha256 = correctness["source_manifest_sha256"]
    if len(materialized) != 45:
        raise ValueError(f"timing matrix must contain exactly 45 rows, got {len(materialized)}")
    raw_ids = [row.get("case_id") for row in materialized]
    if len(set(raw_ids)) != 45 or set(raw_ids) != set(correctness_by_id):
        raise ValueError("timing matrix must cover the unique exact 45-case grid")
    replayed = [
        _validated_timing_row(
            row,
            correctness_by_id[str(row["case_id"])],
            correctness_summary_sha256=correctness_summary_sha256,
            full45_contract_sha256=full45_contract_sha256,
            source_manifest_sha256=source_manifest_sha256,
            expected_candidate_backend=correctness["candidate_backend"],
        )
        for row in materialized
    ]
    if {_case_key(row) for row in replayed} != _expected_keys():
        raise ValueError("timing matrix does not match frozen full45 grid")
    replayed.sort(key=lambda row: (row["T"], row["B"], row["H"], row["target_density"]))
    candidate_backends = {row["candidate_backend"] for row in replayed}
    if len(candidate_backends) != 1:
        raise ValueError("timing matrix mixes full45 candidate backends")
    candidate_backend = candidate_backends.pop()
    overall = _group_record(replayed)
    by_t, t_members = _groups(replayed, lambda row: row["T"], lambda value: str(value))
    by_bh, bh_members = _groups(
        replayed, lambda row: (row["B"], row["H"]),
        lambda value: f"B{value[0]}H{value[1]}",
    )
    by_density, density_members = _groups(
        replayed, lambda row: row["target_density"],
        lambda value: f"{value:.2f}",
    )
    initial_gate = {
        "all_45_correct": len(replayed) == 45,
        "overall_geomean_lt_1": overall["geomean"] < 1.0,
        "all_T_geomeans_lt_1": all(group["geomean"] < 1.0 for group in by_t.values()),
        "all_BH_geomeans_lt_1": all(group["geomean"] < 1.0 for group in by_bh.values()),
        "all_density_geomeans_lt_1": all(group["geomean"] < 1.0 for group in by_density.values()),
        "max_ratio_le_1_03": max(row[RATIO_KEY] for row in replayed) <= 1.03,
    }
    reasons: dict[str, set[str]] = {}
    for row in replayed:
        if row[RATIO_KEY] >= 0.98:
            reasons.setdefault(row["case_id"], set()).add("case_ratio")
    for reason, summaries, members in (
        ("T", by_t, t_members),
        ("BH", by_bh, bh_members),
        ("density", by_density, density_members),
    ):
        for name, group in summaries.items():
            if group["geomean"] >= 0.98:
                for row in members[name]:
                    reasons.setdefault(row["case_id"], set()).add(reason)
    marginal_cases = [
        {"case_id": cid, "marginal_reasons": sorted(values)}
        for cid, values in sorted(reasons.items())
    ]
    decision = (
        "ABBA_REQUIRED" if marginal_cases
        else "R0_INITIAL_GATE_PASS_NO_ABBA" if all(initial_gate.values())
        else "R0_MEASURED"
    )
    histogram = {
        str(group): sum(row["selected_group_size"] == group for row in replayed)
        for group in G_CANDIDATES
        if any(row["selected_group_size"] == group for row in replayed)
    }
    worst = max(replayed, key=lambda row: (row[RATIO_KEY], row["case_id"]))
    return {
        "contract": SUMMARY_CONTRACT,
        "timing_contract": TIMING_CONTRACT,
        "case_count": len(replayed),
        "ratio_key": RATIO_KEY,
        "candidate_backend": candidate_backend,
        "overall": overall,
        "by_T": by_t,
        "by_BH": by_bh,
        "by_density": by_density,
        "selected_g_histogram": histogram,
        "initial_gate": initial_gate,
        "marginal_cases": marginal_cases,
        "decision": decision,
        "promotion_allowed": False,
        "worst_case": {
            "case_id": worst["case_id"],
            RATIO_KEY: worst[RATIO_KEY],
        },
        "edge_receipt_sha256": correctness["edge_receipt_sha256"],
        "edge_receipt_binding": correctness["edge_receipt_binding"],
        "full45_contract_sha256": correctness["full45_contract_sha256"],
        "correctness_summary_sha256": correctness_summary_sha256,
        "source_manifest_sha256": correctness["source_manifest_sha256"],
        "source_identity": correctness["source_identity"],
        "runtime_signatures": correctness["runtime_signatures"],
        "telemetry_contract": {
            "pstate": "P0",
            "critical_slowdowns_allowed": False,
            "foreign_compute_processes_allowed": False,
            "max_gpu_temperature_c": MAX_GPU_TEMPERATURE_C,
            "max_power_limit_overshoot_pct": MAX_POWER_LIMIT_OVERSHOOT_PCT,
            "block_clock_spread_pct": MAX_BLOCK_CLOCK_SPREAD_PCT,
            "matched_clock_delta_pct": MAX_MATCHED_CLOCK_DELTA_PCT,
            "matched_temperature_delta_c": MAX_MATCHED_TEMPERATURE_DELTA_C,
            "matched_power_delta_pct": MAX_MATCHED_POWER_DELTA_PCT,
            "matched_power_delta_scope": "diagnostic_only_not_gating",
            "matched_power_limit_delta_pct": MAX_MATCHED_POWER_LIMIT_DELTA_PCT,
        },
        "cases": replayed,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if value.get("event") == "case":
                value = {key: item for key, item in value.items() if key not in {"event", "time"}}
                rows.append(value)
            elif value.get("contract") == TIMING_CONTRACT and "case_id" in value:
                rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--correctness-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    correctness_summary = json.loads(
        args.correctness_summary.read_text(encoding="utf-8")
    )
    if correctness_summary.get("full45_contract_sha256") != _sha256_file(args.contract):
        raise ValueError("full45 contract file SHA256 mismatch")
    summary = summarize_full45(
        _load_jsonl(args.timings),
        correctness_summary,
        json.loads(args.contract.read_text(encoding="utf-8")),
        correctness_summary_sha256=_sha256_file(args.correctness_summary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = ["SUMMARY_CONTRACT", "summarize_full45"]
