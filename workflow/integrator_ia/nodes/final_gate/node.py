#!/usr/bin/env python3
"""Final gate for three warm-sample recipes and tiered visual evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from nodes.integration_gate.node import RECIPE_TIERS, TIMING_SCOPE, inspect_integration
from workflow_types import NodeContext, NodeResult


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


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve(ctx: NodeContext, raw: Any) -> Path:
    path = Path(str(raw or ""))
    if not path.is_absolute():
        path = ctx.worktree / path
    return path.resolve()


def validate_hashed_path(
    ctx: NodeContext,
    payload: dict[str, Any],
    label: str,
    issues: list[str],
) -> None:
    path = resolve(ctx, payload.get("path"))
    try:
        path.relative_to(ctx.worktree.resolve())
    except ValueError:
        issues.append(f"delivery_{label}_outside_worktree")
        return
    if not path.is_file():
        issues.append(f"delivery_{label}_missing")
    elif payload.get("sha256") != sha256(path):
        issues.append(f"delivery_{label}_hash_mismatch")


def close_number(actual: Any, expected: Any) -> bool:
    return numeric(actual) and numeric(expected) and math.isclose(
        float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6
    )


def validate_assess(
    ctx: NodeContext,
    tier: str,
    run_dir: Path,
    issues: list[str],
) -> dict[str, Any]:
    assess = read_json(run_dir / "assess_verdict.json")
    if not assess:
        issues.append(f"final_assess_verdict_missing:{tier}")
        return {}
    if assess.get("visual_provider") != "codex":
        issues.append(f"final_visual_provider_not_codex:{tier}")
    if assess.get("quality_tier") != tier:
        issues.append(f"final_visual_tier_mismatch:{tier}")
    if assess.get("timing_scope") != TIMING_SCOPE:
        issues.append(f"final_timing_scope_mismatch:{tier}")
    if assess.get("codex_visual_overall") != "pass":
        issues.append(f"final_codex_visual_not_pass:{tier}")
    if tier not in (assess.get("eligible_tiers") or []):
        issues.append(f"final_visual_tier_not_eligible:{tier}")
    severity = assess.get("max_artifact_severity")
    if tier == "conservative" and severity not in {"none", "low"}:
        issues.append("final_conservative_visual_too_severe")
    elif tier == "balanced" and severity == "medium" and assess.get("medium_affected_prompt_count", 99) > 1:
        issues.append("final_balanced_medium_not_isolated")
    elif tier in {"balanced", "aggressive"} and severity not in {"none", "low", "medium"}:
        issues.append(f"final_{tier}_visual_too_severe")
    if not numeric(assess.get("lpips_max")):
        issues.append(f"final_lpips_missing:{tier}")
    if assess.get("quality_blockers") not in ([], None):
        issues.append(f"final_quality_blockers_present:{tier}")
    verdict = resolve(ctx, assess.get("codex_visual_verdict"))
    if not verdict.is_file():
        issues.append(f"final_codex_visual_verdict_missing:{tier}")
    return assess


def validate_delivery(
    ctx: NodeContext,
    delivery: dict[str, Any],
    status: dict[str, Any],
    assessments: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    if delivery.get("schema_version") != 2:
        issues.append("delivery_schema_version_invalid")
    if delivery.get("workflow_uid") != "integrator_ia":
        issues.append("delivery_workflow_uid_invalid")
    if delivery.get("experiment_uid") != ctx.state.get("experiment_uid"):
        issues.append("delivery_experiment_uid_mismatch")

    source_lock = delivery.get("source_lock") if isinstance(delivery.get("source_lock"), dict) else {}
    validate_hashed_path(ctx, source_lock, "source_lock", issues)
    expected_source_lock = (ctx.worktree / "INTEGRATION-SOURCES.lock.json").resolve()
    if resolve(ctx, source_lock.get("path")) != expected_source_lock:
        issues.append("delivery_source_lock_path_mismatch")

    manifest = delivery.get("integrated_manifest") if isinstance(delivery.get("integrated_manifest"), dict) else {}
    validate_hashed_path(ctx, manifest, "integrated_manifest", issues)

    implementation_files = delivery.get("implementation_files")
    implementation_files = implementation_files if isinstance(implementation_files, list) else []
    if not implementation_files:
        issues.append("delivery_implementation_files_missing")
    delivered_paths: set[Path] = set()
    for index, item in enumerate(implementation_files):
        if not isinstance(item, dict):
            issues.append(f"delivery_implementation_file_{index}_invalid")
            continue
        validate_hashed_path(ctx, item, f"implementation_file_{index}", issues)
        delivered_paths.add(resolve(ctx, item.get("path")))
    lock_payload = read_json(expected_source_lock)
    lock_files = lock_payload.get("files") if isinstance(lock_payload.get("files"), list) else []
    expected_paths = {
        resolve(ctx, item.get("destination"))
        for item in lock_files
        if isinstance(item, dict) and item.get("destination")
    }
    if not expected_paths.issubset(delivered_paths):
        issues.append("delivery_implementation_files_do_not_cover_source_lock")

    timing = delivery.get("timing_contract") if isinstance(delivery.get("timing_contract"), dict) else {}
    if timing.get("scope") != TIMING_SCOPE:
        issues.append("delivery_timing_scope_invalid")

    delivered_recipes = delivery.get("recipes") if isinstance(delivery.get("recipes"), dict) else {}
    status_recipes = status.get("recipes") if isinstance(status.get("recipes"), dict) else {}
    if set(delivered_recipes) != set(RECIPE_TIERS):
        issues.append("delivery_recipe_set_invalid")
    speedups: list[float] = []
    for tier in RECIPE_TIERS:
        delivered = delivered_recipes.get(tier) if isinstance(delivered_recipes.get(tier), dict) else {}
        expected = status_recipes.get(tier) if isinstance(status_recipes.get(tier), dict) else {}
        assess = assessments.get(tier) or {}
        if delivered.get("candidate_id") != expected.get("candidate_id"):
            issues.append(f"delivery_candidate_id_mismatch:{tier}")
        activation = delivered.get("activation_env")
        if not isinstance(activation, dict) or not activation or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in activation.items()
        ):
            issues.append(f"delivery_activation_env_invalid:{tier}")
        if delivered.get("components") != expected.get("components"):
            issues.append(f"delivery_component_evidence_mismatch:{tier}")
        if delivered.get("settings") != expected.get("settings"):
            issues.append(f"delivery_settings_mismatch:{tier}")

        expected_run_dir = str(expected.get("run_dir") or "").rstrip("/")
        run = delivered.get("run") if isinstance(delivered.get("run"), dict) else {}
        run_dir = resolve(ctx, expected_run_dir)
        expected_benchmark = run_dir / "outputs" / "benchmark.json"
        expected_stats = run_dir / "outputs" / "integration_stats.json"
        if resolve(ctx, run.get("run_dir")) != run_dir or not run_dir.is_dir():
            issues.append(f"delivery_run_dir_mismatch:{tier}")
        if resolve(ctx, run.get("benchmark")) != expected_benchmark or not expected_benchmark.is_file():
            issues.append(f"delivery_benchmark_path_mismatch:{tier}")
        if resolve(ctx, run.get("integration_stats")) != expected_stats or not expected_stats.is_file():
            issues.append(f"delivery_integration_stats_path_mismatch:{tier}")

        performance = delivered.get("performance") if isinstance(delivered.get("performance"), dict) else {}
        expected_performance = expected.get("performance") if isinstance(expected.get("performance"), dict) else {}
        for key in ("baseline_sample_total_s", "candidate_sample_total_s", "speedup"):
            if not close_number(performance.get(key), expected_performance.get(key)):
                issues.append(f"delivery_performance_mismatch:{tier}:{key}")
        if numeric(performance.get("speedup")):
            speedups.append(float(performance["speedup"]))

        quality = delivered.get("quality") if isinstance(delivered.get("quality"), dict) else {}
        expected_assess = expected_run_dir + "/assess_verdict.json"
        if quality.get("tier") != tier:
            issues.append(f"delivery_quality_tier_mismatch:{tier}")
        if quality.get("assess_verdict") != expected_assess or not resolve(ctx, expected_assess).is_file():
            issues.append(f"delivery_assess_path_mismatch:{tier}")
        if quality.get("codex_visual_overall") != "pass":
            issues.append(f"delivery_visual_quality_not_pass:{tier}")
        if quality.get("max_artifact_severity") != assess.get("max_artifact_severity"):
            issues.append(f"delivery_visual_severity_mismatch:{tier}")
        if not close_number(quality.get("lpips_max"), assess.get("lpips_max")):
            issues.append(f"delivery_lpips_mismatch:{tier}")
    if len(speedups) == len(RECIPE_TIERS) and any(a >= b for a, b in zip(speedups, speedups[1:])):
        issues.append("delivery_recipe_speedups_not_strictly_increasing")


def run(ctx: NodeContext) -> NodeResult:
    integration_report, integration_issues, first_run = inspect_integration(ctx)
    issues = list(integration_issues)
    status = read_json(ctx.worktree / "INTEGRATION-STATUS.json")
    recipes = status.get("recipes") if isinstance(status.get("recipes"), dict) else {}
    assessments: dict[str, dict[str, Any]] = {}
    for tier in RECIPE_TIERS:
        recipe = recipes.get(tier) if isinstance(recipes.get(tier), dict) else {}
        run_dir = resolve(ctx, recipe.get("run_dir"))
        assessments[tier] = validate_assess(ctx, tier, run_dir, issues)
        expected_performance = recipe.get("performance") if isinstance(recipe.get("performance"), dict) else {}
        for key in ("baseline_sample_total_s", "candidate_sample_total_s", "speedup"):
            if assessments[tier] and not close_number(assessments[tier].get(key), expected_performance.get(key)):
                issues.append(f"final_assess_performance_mismatch:{tier}:{key}")

    delivery = read_json(ctx.worktree / "INTEGRATION-DELIVERY.json")
    status_value = status.get("status")
    if status_value == "complete":
        if not delivery:
            issues.append("integration_delivery_missing")
        else:
            validate_delivery(ctx, delivery, status, assessments, issues)

    report = {
        "schema_version": 2,
        "ok": not issues and status_value == "complete",
        "issues": issues,
        "integration_report": integration_report,
        "integration_status": status_value,
        "timing_scope": TIMING_SCOPE,
        "recipe_quality": {
            tier: {
                "overall": assessments.get(tier, {}).get("codex_visual_overall"),
                "severity": assessments.get(tier, {}).get("max_artifact_severity"),
                "lpips_max": assessments.get(tier, {}).get("lpips_max"),
            }
            for tier in RECIPE_TIERS
        },
    }
    path = ctx.worktree / "state" / "final-gate.json"
    write_json(path, report)
    if issues:
        outcome = "needs_retry"
        reason = ";".join(issues)
    elif status_value != "complete":
        outcome = "needs_finalize"
        reason = "three_recipe_visual_evidence_passed_delivery_reconciliation_required"
    else:
        outcome = "smooth"
        reason = "three_recipe_integrated_delivery_complete"
    return NodeResult(
        outcome,
        updates={
            "final_reason": reason,
            "final_issues": issues,
            "final_run": ctx.rel_to_worktree(first_run),
            "final_recipe_quality": report["recipe_quality"],
        },
        artifacts=[ctx.rel_to_worktree(path)],
        message=reason,
    )
