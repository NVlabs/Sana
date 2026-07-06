#!/usr/bin/env python3
"""Validate a component delivery draft and publish the stable delivery panel."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


EXPECTED_COMPONENT = "pisa"
EXPECTED_TIERS = ("conservative", "balanced", "aggressive")
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
MAX_TIER_SEVERITY = {"exact_fastest": 0, "conservative": 1, "balanced": 2, "aggressive": 3}
DRAFT_NAME = "DELIVERY-DRAFT.json"
DELIVERY_NAME = "DELIVERY.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(ctx: NodeContext, path: Path) -> str:
    return ctx.rel_to_worktree(path.resolve())


def resolve_inside(ctx: NodeContext, raw: str, issues: list[str], label: str) -> Path | None:
    if not raw:
        issues.append(f"{label}_path_missing")
        return None
    path = Path(raw)
    path = (path if path.is_absolute() else ctx.worktree / path).resolve()
    try:
        path.relative_to(ctx.worktree.resolve())
    except ValueError:
        issues.append(f"{label}_outside_worktree:{raw}")
        return None
    if not path.is_file():
        issues.append(f"{label}_file_missing:{raw}")
        return None
    return path


def baseline_record(ctx: NodeContext, issues: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = ctx.worktree / "BASELINE-LOCK.json"
    lock = read_json(path)
    if lock.get("status") != "locked" or lock.get("successful_baseline_runs") != 1:
        issues.append("baseline_lock_invalid")
        return {}, {}
    if not isinstance(lock.get("baseline_total_s"), (int, float)) or float(lock["baseline_total_s"]) <= 0:
        issues.append("baseline_total_invalid")
    if not str(lock.get("timing_scope") or ""):
        issues.append("baseline_timing_scope_missing")
    for item in [*(lock.get("artifacts") or []), *(lock.get("source_files") or [])]:
        if not isinstance(item, dict):
            issues.append("baseline_hash_record_invalid")
            continue
        artifact_path = resolve_inside(ctx, str(item.get("path") or ""), issues, "baseline_artifact")
        if artifact_path and sha256(artifact_path) != item.get("sha256"):
            issues.append(f"baseline_artifact_hash_mismatch:{item.get('path')}")
    if not path.is_file():
        issues.append("baseline_lock_missing")
        return lock, {}
    return lock, {
        "lock_path": relative(ctx, path),
        "lock_sha256": sha256(path),
        "workload_id": str(lock.get("workload_id") or ""),
        "timing_scope": str(lock.get("timing_scope") or ""),
        "baseline_total_s": lock.get("baseline_total_s"),
        "run_dir": lock.get("run_dir"),
    }


def path_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("path") or "")
    return ""


def normalize_files(
    ctx: NodeContext,
    values: Any,
    issues: list[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        issues.append(f"{label}_files_missing")
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        raw = path_value(value)
        path = resolve_inside(ctx, raw, issues, f"{label}_{index}")
        if not path:
            continue
        rel = relative(ctx, path)
        if rel in seen:
            continue
        seen.add(rel)
        records.append({"path": rel, "sha256": sha256(path), "size": path.stat().st_size})
    if not records:
        issues.append(f"{label}_files_empty")
    return records


def assessment_path(ctx: NodeContext, point: dict[str, Any], issues: list[str], tier: str) -> Path | None:
    evidence = point.get("runtime_evidence") if isinstance(point.get("runtime_evidence"), dict) else {}
    raw = str(evidence.get("assessment_path") or point.get("assess_verdict") or "")
    return resolve_inside(ctx, raw, issues, f"{tier}_assessment")


def normalize_point(
    ctx: NodeContext,
    raw: Any,
    expected_tier: str,
    lock: dict[str, Any],
    issues: list[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        issues.append(f"{expected_tier}_point_missing")
        return {}
    tier = str(raw.get("tier") or "")
    if tier != expected_tier:
        issues.append(f"{expected_tier}_tier_mismatch:{tier}")
    candidate_id = str(raw.get("candidate_id") or "")
    if not candidate_id:
        issues.append(f"{expected_tier}_candidate_id_missing")
    run_raw = str(raw.get("run_dir") or "")
    if not run_raw:
        issues.append(f"{expected_tier}_run_dir_missing")
    run_dir = (ctx.worktree / run_raw).resolve() if run_raw and not Path(run_raw).is_absolute() else Path(run_raw).resolve()
    try:
        run_dir.relative_to(ctx.worktree.resolve())
    except (ValueError, OSError):
        issues.append(f"{expected_tier}_run_dir_outside_worktree")
    if not run_dir.is_dir():
        issues.append(f"{expected_tier}_run_dir_missing")

    manifest_raw = path_value(raw.get("implementation_manifest"))
    manifest = resolve_inside(ctx, manifest_raw, issues, f"{expected_tier}_manifest")
    assess_path = assessment_path(ctx, raw, issues, expected_tier)
    assess = read_json(assess_path) if assess_path else {}
    required_numbers = ("baseline_total_s", "candidate_total_s", "speedup")
    for field in required_numbers:
        if not isinstance(assess.get(field), (int, float)):
            issues.append(f"{expected_tier}_assessment_{field}_missing")
    baseline_total = assess.get("baseline_total_s")
    candidate_total = assess.get("candidate_total_s")
    measured_speedup = assess.get("speedup")
    locked_total = lock.get("baseline_total_s")
    if isinstance(baseline_total, (int, float)) and isinstance(locked_total, (int, float)):
        if not math.isclose(float(baseline_total), float(locked_total), rel_tol=1e-5, abs_tol=1e-5):
            issues.append(f"{expected_tier}_baseline_total_not_locked")
    if isinstance(baseline_total, (int, float)) and isinstance(candidate_total, (int, float)) and candidate_total > 0:
        calculated = float(baseline_total) / float(candidate_total)
        if not isinstance(measured_speedup, (int, float)) or not math.isclose(
            calculated, float(measured_speedup), rel_tol=5e-4, abs_tol=5e-4
        ):
            issues.append(f"{expected_tier}_speedup_inconsistent")
    timing_scope = str(assess.get("timing_scope") or "")
    if timing_scope != str(lock.get("timing_scope") or ""):
        issues.append(f"{expected_tier}_timing_scope_not_locked")
    workload_id = str(assess.get("workload_id") or "")
    if workload_id and workload_id != str(lock.get("workload_id") or ""):
        issues.append(f"{expected_tier}_workload_id_mismatch")

    quality_raw = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}
    severity = str(assess.get("max_artifact_severity") or quality_raw.get("max_artifact_severity") or "")
    if severity not in SEVERITY_ORDER:
        issues.append(f"{expected_tier}_quality_severity_invalid")
    elif SEVERITY_ORDER[severity] > MAX_TIER_SEVERITY[expected_tier]:
        issues.append(f"{expected_tier}_quality_severity_exceeds_tier")
    relation = str(quality_raw.get("candidate_relation") or "")
    if not relation:
        issues.append(f"{expected_tier}_candidate_relation_missing")
    visual_overall = str(assess.get("codex_visual_overall") or assess.get("gemini_overall") or "").lower()
    if visual_overall not in {"pass", "fail"}:
        issues.append(f"{expected_tier}_visual_review_incomplete")
    if EXPECTED_COMPONENT == "kernel" and quality_raw.get("lossless") is not True:
        issues.append("exact_fastest_kernel_point_must_be_lossless")
    blockers = assess.get("quality_blockers") or []
    if not isinstance(blockers, list):
        issues.append(f"{expected_tier}_quality_blockers_invalid")

    activation = raw.get("activation") if isinstance(raw.get("activation"), dict) else {}
    compute_budget = raw.get("compute_budget") if isinstance(raw.get("compute_budget"), dict) else {}
    if not activation:
        issues.append(f"{expected_tier}_activation_missing")
    if not compute_budget:
        issues.append(f"{expected_tier}_compute_budget_missing")
    artifacts = normalize_files(ctx, raw.get("artifacts"), issues, f"{expected_tier}_artifact")
    if assess_path:
        assess_record = {"path": relative(ctx, assess_path), "sha256": sha256(assess_path), "size": assess_path.stat().st_size}
        if not any(item["path"] == assess_record["path"] for item in artifacts):
            artifacts.append(assess_record)
    benchmark = run_dir / "outputs" / "benchmark.json"
    if benchmark.is_file() and not any(item["path"] == relative(ctx, benchmark) for item in artifacts):
        artifacts.append({"path": relative(ctx, benchmark), "sha256": sha256(benchmark), "size": benchmark.stat().st_size})

    evidence = raw.get("runtime_evidence") if isinstance(raw.get("runtime_evidence"), dict) else {}
    evidence = {**evidence}
    if assess_path:
        evidence["assessment_path"] = relative(ctx, assess_path)
        evidence["assessment_sha256"] = sha256(assess_path)
    return {
        "tier": expected_tier,
        "candidate_id": candidate_id,
        "run_dir": relative(ctx, run_dir) if run_dir.is_dir() else run_raw,
        "implementation_manifest": {
            "path": relative(ctx, manifest) if manifest else manifest_raw,
            "sha256": sha256(manifest) if manifest else "",
        },
        "activation": activation,
        "compute_budget": compute_budget,
        "performance": {
            "timing_scope": timing_scope,
            "baseline_total_s": baseline_total,
            "candidate_total_s": candidate_total,
            "speedup": measured_speedup,
        },
        "quality": {
            **quality_raw,
            "review_status": "complete",
            "candidate_relation": relation,
            "max_artifact_severity": severity,
            "visual_overall": visual_overall,
            "lpips_max": assess.get("lpips_max"),
            "quality_blockers": blockers,
        },
        "runtime_evidence": evidence,
        "artifacts": [item["path"] for item in artifacts],
        "artifact_hashes": artifacts,
    }


def run(ctx: NodeContext) -> NodeResult:
    issues: list[str] = []
    draft_path = ctx.worktree / DRAFT_NAME
    draft = read_json(draft_path)
    if not draft:
        return NodeResult("invalid", updates={"delivery_issues": ["delivery_draft_missing_or_invalid"]}, message="delivery_draft_missing_or_invalid")
    if draft.get("component") != EXPECTED_COMPONENT:
        issues.append("delivery_component_mismatch")
    if draft.get("model_id") not in {None, "sana_video"}:
        issues.append("delivery_model_id_mismatch")
    lock, baseline = baseline_record(ctx, issues)
    package = draft.get("implementation_package") if isinstance(draft.get("implementation_package"), dict) else {}
    package_files = normalize_files(ctx, package.get("files"), issues, "implementation_package")
    build_smoke = package.get("build_smoke") if isinstance(package.get("build_smoke"), dict) else {}
    if build_smoke.get("status") not in {"passed", "not_required"}:
        issues.append("implementation_build_smoke_invalid")

    raw_points = draft.get("frontier_points")
    if not isinstance(raw_points, list) or len(raw_points) != len(EXPECTED_TIERS):
        issues.append(f"frontier_point_count_must_equal_{len(EXPECTED_TIERS)}")
        raw_points = raw_points if isinstance(raw_points, list) else []
    by_tier = {
        str(point.get("tier") or ""): point
        for point in raw_points
        if isinstance(point, dict)
    }
    points = [normalize_point(ctx, by_tier.get(tier), tier, lock, issues) for tier in EXPECTED_TIERS]
    candidate_ids = [point.get("candidate_id") for point in points if point]
    run_dirs = [point.get("run_dir") for point in points if point]
    if len(candidate_ids) == len(EXPECTED_TIERS) and len(set(candidate_ids)) != len(candidate_ids):
        issues.append("frontier_candidate_ids_must_be_distinct")
    if len(run_dirs) == len(EXPECTED_TIERS) and len(set(run_dirs)) != len(run_dirs):
        issues.append("frontier_run_dirs_must_be_distinct")
    speedups = [point.get("performance", {}).get("speedup") for point in points]
    if all(isinstance(value, (int, float)) for value in speedups):
        if any(float(left) >= float(right) for left, right in zip(speedups, speedups[1:])):
            issues.append("frontier_speedups_must_strictly_increase")

    pareto = draft.get("pareto_assessment") if isinstance(draft.get("pareto_assessment"), dict) else {}
    if pareto.get("status") != "nondominated":
        issues.append("pareto_assessment_status_invalid")
    if pareto.get("objective") != "maximize_quality_subject_to_measured_compute_budget":
        issues.append("pareto_assessment_objective_invalid")
    if not isinstance(pareto.get("evidence"), list) or not pareto.get("evidence"):
        issues.append("pareto_assessment_evidence_missing")

    state_path = ctx.worktree / "state" / "delivery-gate.json"
    gate = {
        "schema_version": 1,
        "component": EXPECTED_COMPONENT,
        "checked_at_utc": utc_now(),
        "status": "invalid" if issues else "passed",
        "issues": issues,
        "draft": relative(ctx, draft_path),
        "draft_sha256": sha256(draft_path),
    }
    write_json(state_path, gate)
    if issues:
        return NodeResult("invalid", updates={"delivery_issues": issues, "delivery_gate": relative(ctx, state_path)}, artifacts=[relative(ctx, state_path)], message=";".join(issues))

    delivery = {
        "schema_version": 2,
        "delivery_kind": "component_frontier" if EXPECTED_COMPONENT != "integrator" else "integrated_frontier",
        "workflow_uid": ctx.state.get("workflow_uid"),
        "experiment_uid": ctx.state.get("experiment_uid"),
        "model_id": "sana_video",
        "component": EXPECTED_COMPONENT,
        "status": "complete",
        "created_at_utc": utc_now(),
        "baseline": baseline,
        "implementation_package": {"files": package_files, "build_smoke": build_smoke},
        "frontier_points": points,
        "pareto_assessment": pareto,
    }
    delivery_path = ctx.worktree / DELIVERY_NAME
    write_json(delivery_path, delivery)
    return NodeResult(
        "published",
        updates={"delivery": relative(ctx, delivery_path), "delivery_gate": relative(ctx, state_path), "delivery_issues": []},
        artifacts=[relative(ctx, delivery_path), relative(ctx, state_path)],
        message="delivery_published",
    )

