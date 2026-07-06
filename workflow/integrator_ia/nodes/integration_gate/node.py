#!/usr/bin/env python3
"""Programmatic gate for warm-sample composition and tiered recipes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


REPORT_NAME = "integration-gate.json"
TIMING_SCOPE = "warm_single_sample_text_encoder_through_vae_decode"
RECIPE_TIERS = ("conservative", "balanced", "aggressive")
CONDITION_BITS = {
    "baseline": "000",
    "kernel_only": "100",
    "pisa_only": "010",
    "cache_only": "001",
    "kernel_pisa": "110",
    "kernel_cache": "101",
    "pisa_cache": "011",
    "full_stack": "111",
}
REQUIRED_INCLUDED_STAGES = {"text_encoder_compute", "dit_denoise", "vae_decode"}
REQUIRED_EXCLUDED_STAGES = {
    "process_startup",
    "model_load",
    "text_encoder_load",
    "vae_load",
    "compile",
    "warmup",
    "video_encode",
    "video_write",
}


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


def worktree_path(ctx: NodeContext, raw: Any) -> Path:
    path = Path(str(raw or ""))
    if not path.is_absolute():
        path = ctx.worktree / path
    return path.resolve()


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def inventory_artifact(source: dict[str, Any], role: str) -> dict[str, Any]:
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, list):
        return {}
    return next(
        (item for item in artifacts if isinstance(item, dict) and item.get("role") == role),
        {},
    )


def validate_inventory_snapshots(
    ctx: NodeContext,
    inventory: dict[str, Any],
    issues: list[str],
) -> None:
    sources = inventory.get("sources") if isinstance(inventory.get("sources"), dict) else {}
    for component in ("kernel", "pisa", "cache"):
        source = sources.get(component)
        source = source if isinstance(source, dict) else {}
        snapshot_root = worktree_path(ctx, source.get("snapshot_root"))
        if not under(snapshot_root, ctx.worktree):
            issues.append(f"source_snapshot_root_invalid:{component}")
            continue
        implementations = source.get("implementation_files")
        implementations = implementations if isinstance(implementations, list) else []
        if not implementations:
            issues.append(f"source_snapshot_implementations_missing:{component}")
        entries = []
        for key in ("artifacts", "implementation_files"):
            value = source.get(key)
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, dict))
        for index, item in enumerate(entries):
            snapshot = worktree_path(ctx, item.get("snapshot_path"))
            if not snapshot.is_file() or not under(snapshot, snapshot_root):
                issues.append(f"source_snapshot_file_invalid:{component}:{index}")
                continue
            actual = sha256(snapshot)
            if actual != item.get("sha256") or actual != item.get("snapshot_sha256"):
                issues.append(f"source_snapshot_hash_mismatch:{component}:{index}")


def validate_source_lock(
    ctx: NodeContext,
    inventory: dict[str, Any],
    lock: dict[str, Any],
    issues: list[str],
) -> None:
    inventory_path = ctx.worktree / "state" / "integration-source-inventory.json"
    if inventory.get("status") != "ready":
        issues.append("source_inventory_not_ready")
    if lock.get("workflow_uid") != "integrator_ia":
        issues.append("source_lock_workflow_uid_invalid")
    if lock.get("inventory_sha256") != sha256(inventory_path):
        issues.append("source_lock_inventory_hash_mismatch")
    inv_sources = inventory.get("sources") if isinstance(inventory.get("sources"), dict) else {}
    locked_sources = lock.get("sources") if isinstance(lock.get("sources"), dict) else {}

    for component in ("kernel", "pisa", "cache"):
        source = inv_sources.get(component) if isinstance(inv_sources.get(component), dict) else {}
        pinned = locked_sources.get(component) if isinstance(locked_sources.get(component), dict) else {}
        expected_ids = (source.get("selection") or {}).get("candidate_ids")
        if pinned.get("candidate_ids") != expected_ids:
            issues.append(f"source_lock_{component}_candidate_ids_mismatch")
        delivery = inventory_artifact(source, f"{component}_delivery")
        if pinned.get("delivery_sha256") != delivery.get("sha256"):
            issues.append(f"source_lock_{component}_delivery_hash_mismatch")

    files = lock.get("files")
    files = files if isinstance(files, list) else []
    covered: set[str] = set()
    covered_sources: dict[str, set[Path]] = {"kernel": set(), "pisa": set(), "cache": set()}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            issues.append(f"source_lock_file_{index}_invalid")
            continue
        component = str(item.get("component") or "")
        source_info = inv_sources.get(component)
        source_info = source_info if isinstance(source_info, dict) else {}
        source_root = worktree_path(ctx, source_info.get("snapshot_root"))
        source = worktree_path(ctx, item.get("source"))
        destination = worktree_path(ctx, item.get("destination"))
        if component not in {"kernel", "pisa", "cache"}:
            issues.append(f"source_lock_file_{index}_component_invalid")
            continue
        covered.add(component)
        covered_sources[component].add(source)
        implementations = source_info.get("implementation_files")
        implementations = implementations if isinstance(implementations, list) else []
        allowed_sources = {
            worktree_path(ctx, candidate.get("snapshot_path"))
            for candidate in implementations
            if isinstance(candidate, dict) and candidate.get("snapshot_path")
        }
        if not source.is_file() or not under(source, source_root) or source not in allowed_sources:
            issues.append(f"source_lock_file_{index}_source_invalid")
        elif item.get("source_sha256") != sha256(source):
            issues.append(f"source_lock_file_{index}_source_hash_mismatch")
        if not destination.is_file() or not under(destination, ctx.worktree):
            issues.append(f"source_lock_file_{index}_destination_invalid")
        elif item.get("destination_sha256") != sha256(destination):
            issues.append(f"source_lock_file_{index}_destination_hash_mismatch")
    for component in ("kernel", "pisa", "cache"):
        if component not in covered:
            issues.append(f"source_lock_component_files_missing:{component}")
            continue
        source_info = inv_sources.get(component)
        source_info = source_info if isinstance(source_info, dict) else {}
        implementations = source_info.get("implementation_files")
        implementations = implementations if isinstance(implementations, list) else []
        expected_sources = {
            worktree_path(ctx, candidate.get("snapshot_path"))
            for candidate in implementations
            if isinstance(candidate, dict) and candidate.get("snapshot_path")
        }
        missing = expected_sources - covered_sources[component]
        if missing:
            issues.append(f"source_lock_implementation_files_missing:{component}:{len(missing)}")


def validate_timing_descriptor(timing: Any, label: str, issues: list[str]) -> None:
    timing = timing if isinstance(timing, dict) else {}
    if timing.get("scope") != TIMING_SCOPE:
        issues.append(f"{label}_timing_scope_invalid")
    if timing.get("warmup_completed") is not True:
        issues.append(f"{label}_warmup_not_completed")
    if timing.get("cuda_synchronized") is not True:
        issues.append(f"{label}_cuda_not_synchronized")
    included = set(timing.get("included_stages") or [])
    excluded = set(timing.get("excluded_stages") or [])
    if not REQUIRED_INCLUDED_STAGES.issubset(included):
        issues.append(f"{label}_included_stages_incomplete")
    if not REQUIRED_EXCLUDED_STAGES.issubset(excluded):
        issues.append(f"{label}_excluded_stages_incomplete")
    if not numeric(timing.get("sample_count")) or timing.get("sample_count", 0) <= 0:
        issues.append(f"{label}_sample_count_invalid")
    if not str(timing.get("aggregation") or ""):
        issues.append(f"{label}_aggregation_missing")


def validate_component_evidence(
    components: Any,
    inventory: dict[str, Any],
    label: str,
    issues: list[str],
) -> None:
    components = components if isinstance(components, dict) else {}
    sources = inventory.get("sources") if isinstance(inventory.get("sources"), dict) else {}

    kernel = components.get("kernel") if isinstance(components.get("kernel"), dict) else {}
    allowed_kernel = ((sources.get("kernel") or {}).get("selection") or {}).get("candidate_ids") or []
    declared_kernel = kernel.get("candidate_ids")
    if declared_kernel is None and kernel.get("candidate_id"):
        declared_kernel = [kernel.get("candidate_id")]
    if declared_kernel != allowed_kernel:
        issues.append(f"{label}_kernel_candidate_ids_mismatch")
    kernel_dispatches = kernel.get("dispatches")
    if kernel.get("enabled") is True:
        if not numeric(kernel_dispatches) or kernel_dispatches <= 0:
            issues.append(f"{label}_kernel_dispatches_missing")
    elif kernel.get("enabled") is False:
        if kernel_dispatches != 0:
            issues.append(f"{label}_kernel_disabled_activity_nonzero")
    else:
        issues.append(f"{label}_kernel_enabled_invalid")
    if kernel.get("fallbacks") != 0:
        issues.append(f"{label}_kernel_fallbacks_nonzero")

    pisa = components.get("pisa") if isinstance(components.get("pisa"), dict) else {}
    allowed_pisa = ((sources.get("pisa") or {}).get("selection") or {}).get("candidate_ids") or []
    if pisa.get("candidate_id") not in allowed_pisa:
        issues.append(f"{label}_pisa_candidate_mismatch")
    pisa_dispatches = pisa.get("dispatches")
    if pisa.get("enabled") is True:
        if not numeric(pisa_dispatches) or pisa_dispatches <= 0:
            issues.append(f"{label}_pisa_dispatches_missing")
    elif pisa.get("enabled") is False:
        if pisa_dispatches != 0:
            issues.append(f"{label}_pisa_disabled_activity_nonzero")
    else:
        issues.append(f"{label}_pisa_enabled_invalid")
    if pisa.get("fallbacks") != 0:
        issues.append(f"{label}_pisa_fallbacks_nonzero")

    cache = components.get("cache") if isinstance(components.get("cache"), dict) else {}
    allowed_cache = ((sources.get("cache") or {}).get("selection") or {}).get("candidate_ids") or []
    if cache.get("candidate_id") not in allowed_cache:
        issues.append(f"{label}_cache_candidate_mismatch")
    if cache.get("enabled") is True:
        if cache.get("calls") != 250:
            issues.append(f"{label}_cache_calls_must_equal_250")
        if not numeric(cache.get("hits")) or cache.get("hits", 0) <= 0:
            issues.append(f"{label}_cache_hits_missing")
    elif cache.get("enabled") is False:
        if cache.get("calls") != 0 or cache.get("hits") != 0:
            issues.append(f"{label}_cache_disabled_activity_nonzero")
    else:
        issues.append(f"{label}_cache_enabled_invalid")


def validate_component_status(
    status: dict[str, Any],
    inventory: dict[str, Any],
    issues: list[str],
) -> None:
    if status.get("workflow_uid") != "integrator_ia":
        issues.append("integration_status_workflow_uid_invalid")
    if status.get("status") not in {"ready_for_visual", "complete"}:
        issues.append("integration_status_not_ready_for_visual")
    if status.get("owned_jobs") not in ([], None):
        issues.append("integration_owned_jobs_not_empty")
    recipes = status.get("recipes") if isinstance(status.get("recipes"), dict) else {}
    if set(recipes) != set(RECIPE_TIERS):
        issues.append("integration_recipe_set_invalid")
    candidate_ids: list[str] = []
    run_dirs: list[str] = []
    speedups: list[float] = []
    for tier in RECIPE_TIERS:
        recipe = recipes.get(tier) if isinstance(recipes.get(tier), dict) else {}
        candidate_id = str(recipe.get("candidate_id") or "")
        run_dir = str(recipe.get("run_dir") or "")
        if not candidate_id:
            issues.append(f"integration_recipe_candidate_missing:{tier}")
        if not run_dir:
            issues.append(f"integration_recipe_run_missing:{tier}")
        candidate_ids.append(candidate_id)
        run_dirs.append(run_dir)
        validate_component_evidence(recipe.get("components"), inventory, f"integration_recipe_{tier}", issues)
        if not isinstance(recipe.get("settings"), dict):
            issues.append(f"integration_recipe_settings_invalid:{tier}")
        performance = recipe.get("performance") if isinstance(recipe.get("performance"), dict) else {}
        for key in ("baseline_sample_total_s", "candidate_sample_total_s", "speedup"):
            if not numeric(performance.get(key)) or performance.get(key, 0) <= 0:
                issues.append(f"integration_recipe_performance_invalid:{tier}:{key}")
        if numeric(performance.get("speedup")):
            speedups.append(float(performance["speedup"]))
    if len(set(candidate_ids)) != len(RECIPE_TIERS):
        issues.append("integration_recipe_candidate_ids_not_distinct")
    if len(set(run_dirs)) != len(RECIPE_TIERS):
        issues.append("integration_recipe_run_dirs_not_distinct")
    if len(speedups) == len(RECIPE_TIERS) and any(a >= b for a, b in zip(speedups, speedups[1:])):
        issues.append("integration_recipe_speedups_not_strictly_increasing")


def validate_matrix(matrix: dict[str, Any], status: dict[str, Any], issues: list[str]) -> None:
    validate_timing_descriptor(matrix.get("timing_contract"), "composition", issues)
    conditions = matrix.get("conditions") if isinstance(matrix.get("conditions"), dict) else {}
    baseline_total: float | None = None
    baseline = conditions.get("baseline") if isinstance(conditions.get("baseline"), dict) else {}
    if numeric(baseline.get("sample_total_s")) and baseline.get("sample_total_s", 0) > 0:
        baseline_total = float(baseline["sample_total_s"])
    for name, bits in CONDITION_BITS.items():
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            issues.append(f"composition_condition_missing:{name}")
            continue
        if condition.get("bits") != bits:
            issues.append(f"composition_condition_bits_invalid:{name}")
        if condition.get("status") != "measured":
            issues.append(f"composition_condition_not_measured:{name}")
        total = condition.get("sample_total_s")
        speedup = condition.get("speedup")
        if not numeric(total) or total <= 0:
            issues.append(f"composition_condition_sample_total_invalid:{name}")
        if not numeric(speedup) or speedup <= 0:
            issues.append(f"composition_condition_speedup_invalid:{name}")
        if baseline_total and numeric(total) and total > 0 and numeric(speedup):
            expected = baseline_total / float(total)
            if not math.isclose(float(speedup), expected, rel_tol=1e-4, abs_tol=1e-4):
                issues.append(f"composition_condition_speedup_mismatch:{name}")
    if matrix.get("all_off_identity") is not True:
        issues.append("composition_all_off_identity_missing")

    matrix_recipes = matrix.get("recipes") if isinstance(matrix.get("recipes"), dict) else {}
    status_recipes = status.get("recipes") if isinstance(status.get("recipes"), dict) else {}
    if set(matrix_recipes) != set(RECIPE_TIERS):
        issues.append("composition_recipe_set_invalid")
    for tier in RECIPE_TIERS:
        measured = matrix_recipes.get(tier) if isinstance(matrix_recipes.get(tier), dict) else {}
        declared = status_recipes.get(tier) if isinstance(status_recipes.get(tier), dict) else {}
        for key in ("candidate_id", "run_dir"):
            if measured.get(key) != declared.get(key):
                issues.append(f"composition_recipe_mismatch:{tier}:{key}")
        measured_perf = measured.get("performance") if isinstance(measured.get("performance"), dict) else {}
        declared_perf = declared.get("performance") if isinstance(declared.get("performance"), dict) else {}
        for key in ("baseline_sample_total_s", "candidate_sample_total_s", "speedup"):
            actual = measured_perf.get(key)
            expected = declared_perf.get(key)
            if not numeric(actual) or not numeric(expected) or not math.isclose(
                float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6
            ):
                issues.append(f"composition_recipe_performance_mismatch:{tier}:{key}")
        if baseline_total and numeric(declared_perf.get("baseline_sample_total_s")) and not math.isclose(
            float(declared_perf["baseline_sample_total_s"]), baseline_total, rel_tol=1e-6, abs_tol=1e-6
        ):
            issues.append(f"composition_recipe_baseline_mismatch:{tier}")


def config_value(config: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return None


def validate_benchmark_timing(benchmark: dict[str, Any], label: str, issues: list[str]) -> float | None:
    validate_timing_descriptor(benchmark.get("timing"), label, issues)
    aggregate = benchmark.get("aggregate") if isinstance(benchmark.get("aggregate"), dict) else {}
    for key in ("text_encoder_s", "dit_denoise_s", "vae_decode_s", "sample_total_s"):
        if not numeric(aggregate.get(key)) or aggregate.get(key, 0) <= 0:
            issues.append(f"{label}_aggregate_invalid:{key}")
    total = aggregate.get("sample_total_s")
    return float(total) if numeric(total) and total > 0 else None


def validate_baseline_run(
    ctx: NodeContext,
    matrix: dict[str, Any],
    declared_total: float | None,
    issues: list[str],
) -> float | None:
    lock = read_json(ctx.worktree / "BASELINE-LOCK.json")
    conditions = matrix.get("conditions") if isinstance(matrix.get("conditions"), dict) else {}
    baseline = conditions.get("baseline") if isinstance(conditions.get("baseline"), dict) else {}
    run_dir = worktree_path(ctx, baseline.get("run_dir"))
    locked_run = worktree_path(ctx, lock.get("run_dir"))
    if run_dir != locked_run:
        issues.append("composition_baseline_must_use_locked_run")
    locked_total = lock.get("baseline_total_s")
    if lock.get("successful_baseline_runs") != 1:
        issues.append("composition_baseline_run_count_invalid")
    if lock.get("timing_scope") != TIMING_SCOPE:
        issues.append("composition_baseline_lock_timing_scope_invalid")
    if numeric(locked_total) and declared_total is not None and not math.isclose(
        float(locked_total), declared_total, rel_tol=1e-6, abs_tol=1e-6
    ):
        issues.append("composition_baseline_declared_total_not_locked")
    if not under(run_dir, ctx.worktree) or not run_dir.is_dir():
        issues.append("composition_baseline_run_invalid")
        return declared_total
    benchmark = read_json(run_dir / "outputs" / "benchmark.json")
    if not benchmark:
        issues.append("composition_baseline_benchmark_missing")
        return declared_total
    measured = validate_benchmark_timing(benchmark, "composition_baseline", issues)
    if measured is not None and declared_total is not None and not math.isclose(
        measured, declared_total, rel_tol=1e-6, abs_tol=1e-6
    ):
        issues.append("composition_baseline_benchmark_mismatch")
    if measured is not None and numeric(locked_total) and not math.isclose(
        measured, float(locked_total), rel_tol=1e-6, abs_tol=1e-6
    ):
        issues.append("composition_baseline_benchmark_not_locked")
    return measured if measured is not None else declared_total


def validate_stats_components(
    stats: dict[str, Any],
    declared: dict[str, Any],
    tier: str,
    issues: list[str],
) -> None:
    components = stats.get("components") if isinstance(stats.get("components"), dict) else {}
    for component in ("kernel", "pisa", "cache"):
        observed = components.get(component) if isinstance(components.get(component), dict) else {}
        expected = declared.get(component) if isinstance(declared.get(component), dict) else {}
        if observed.get("enabled") is not expected.get("enabled"):
            issues.append(f"integration_stats_enabled_mismatch:{tier}:{component}")
        keys = {
            "kernel": ("dispatches", "fallbacks"),
            "pisa": ("dispatches", "fallbacks"),
            "cache": ("calls", "hits"),
        }[component]
        for key in keys:
            if observed.get(key) != expected.get(key):
                issues.append(f"integration_stats_counter_mismatch:{tier}:{component}:{key}")

    pisa = components.get("pisa") if isinstance(components.get("pisa"), dict) else {}
    if pisa.get("enabled") is True and (
        not numeric(pisa.get("exact_phase_calls"))
        or pisa.get("exact_phase_calls", 0) <= 0
        or not numeric(pisa.get("approximate_remainder_phase_calls"))
        or pisa.get("approximate_remainder_phase_calls", 0) <= 0
    ):
        issues.append(f"integration_stats_pisa_phase_activity_invalid:{tier}")
    if pisa.get("enabled") is False and (
        pisa.get("exact_phase_calls", 0) != 0 or pisa.get("approximate_remainder_phase_calls", 0) != 0
    ):
        issues.append(f"integration_stats_pisa_disabled_phase_activity:{tier}")


def validate_run(
    ctx: NodeContext,
    tier: str,
    recipe: dict[str, Any],
    inventory: dict[str, Any],
    baseline_total: float | None,
    issues: list[str],
) -> tuple[Path, dict[str, Any]]:
    run_dir = worktree_path(ctx, recipe.get("run_dir"))
    if not under(run_dir, ctx.worktree) or not run_dir.is_dir():
        issues.append(f"integration_run_dir_invalid:{tier}")
        return run_dir, {}
    outputs = run_dir / "outputs"
    benchmark = read_json(outputs / "benchmark.json")
    run_config = read_json(outputs / "run_config.json")
    stats = read_json(outputs / "integration_stats.json")
    if not benchmark:
        issues.append(f"integration_benchmark_missing:{tier}")
    if not run_config:
        issues.append(f"integration_run_config_missing:{tier}")
    if not stats:
        issues.append(f"integration_stats_missing:{tier}")
    measured_total = validate_benchmark_timing(benchmark, f"integration_run_{tier}", issues) if benchmark else None
    config = benchmark.get("config") if isinstance(benchmark.get("config"), dict) else {}
    config = {**config, **run_config}
    expected = {
        "sample_nums": (5, ("sample_nums", "prompt_count")),
        "image_size": (720, ("image_size",)),
        "num_frames": (193, ("num_frames", "frames")),
        "fps": (24, ("fps",)),
        "steps": (50, ("steps", "num_inference_steps")),
        "cfg_scale": (8, ("cfg_scale",)),
        "flow_shift": (12, ("flow_shift",)),
        "motion_score": (20, ("motion_score",)),
    }
    for label, (wanted, aliases) in expected.items():
        if config_value(config, *aliases) != wanted:
            issues.append(f"integration_config_mismatch:{tier}:{label}")
    videos = list(outputs.rglob("*.mp4"))
    frames = list(outputs.rglob("*.png"))
    if len(videos) < 5 and len(frames) < 5 * 193:
        issues.append(f"integration_outputs_incomplete:{tier}")

    declared_components = recipe.get("components") if isinstance(recipe.get("components"), dict) else {}
    if stats:
        validate_stats_components(stats, declared_components, tier, issues)
    validate_component_evidence(declared_components, inventory, f"integration_run_{tier}", issues)

    performance = recipe.get("performance") if isinstance(recipe.get("performance"), dict) else {}
    if measured_total is not None and (
        not numeric(performance.get("candidate_sample_total_s"))
        or not math.isclose(
            measured_total,
            float(performance["candidate_sample_total_s"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        issues.append(f"integration_recipe_benchmark_mismatch:{tier}")
    if baseline_total is not None and measured_total is not None:
        expected_speedup = baseline_total / measured_total
        if not numeric(performance.get("speedup")) or not math.isclose(
            float(performance["speedup"]), expected_speedup, rel_tol=1e-4, abs_tol=1e-4
        ):
            issues.append(f"integration_recipe_speedup_mismatch:{tier}")
        if expected_speedup <= 1.0:
            issues.append(f"integration_recipe_not_faster_than_baseline:{tier}")
    return run_dir, benchmark


def inspect_integration(ctx: NodeContext) -> tuple[dict[str, Any], list[str], Path]:
    issues: list[str] = []
    inventory_path = ctx.worktree / "state" / "integration-source-inventory.json"
    inventory = read_json(inventory_path)
    lock = read_json(ctx.worktree / "INTEGRATION-SOURCES.lock.json")
    status = read_json(ctx.worktree / "INTEGRATION-STATUS.json")
    matrix = read_json(ctx.worktree / "COMPOSITION-MATRIX.json")
    if not inventory:
        issues.append("source_inventory_missing")
    if not lock:
        issues.append("integration_source_lock_missing")
    if not status:
        issues.append("integration_status_missing")
    if not matrix:
        issues.append("composition_matrix_missing")
    if inventory:
        validate_inventory_snapshots(ctx, inventory, issues)
    if inventory and lock:
        validate_source_lock(ctx, inventory, lock, issues)
    if status and inventory:
        validate_component_status(status, inventory, issues)
    if matrix and status:
        validate_matrix(matrix, status, issues)
    conditions = matrix.get("conditions") if isinstance(matrix.get("conditions"), dict) else {}
    baseline = conditions.get("baseline") if isinstance(conditions.get("baseline"), dict) else {}
    baseline_total = (
        float(baseline["sample_total_s"])
        if numeric(baseline.get("sample_total_s")) and baseline.get("sample_total_s", 0) > 0
        else None
    )
    if matrix:
        baseline_total = validate_baseline_run(ctx, matrix, baseline_total, issues)
    recipes = status.get("recipes") if isinstance(status.get("recipes"), dict) else {}
    run_dirs: dict[str, str] = {}
    recipe_totals: dict[str, Any] = {}
    first_run = ctx.worktree / "runs"
    for index, tier in enumerate(RECIPE_TIERS):
        recipe = recipes.get(tier) if isinstance(recipes.get(tier), dict) else {}
        run_dir, benchmark = (
            validate_run(ctx, tier, recipe, inventory, baseline_total, issues)
            if status and inventory and recipe
            else (worktree_path(ctx, recipe.get("run_dir")), {})
        )
        if index == 0:
            first_run = run_dir
        run_dirs[tier] = ctx.rel_to_worktree(run_dir)
        aggregate = benchmark.get("aggregate") if isinstance(benchmark.get("aggregate"), dict) else {}
        recipe_totals[tier] = aggregate.get("sample_total_s")
    report = {
        "schema_version": 2,
        "ok": not issues,
        "issues": issues,
        "integration_status": status.get("status"),
        "timing_scope": TIMING_SCOPE,
        "baseline_sample_total_s": baseline_total,
        "recipe_run_dirs": run_dirs,
        "recipe_sample_total_s": recipe_totals,
    }
    return report, issues, first_run


def run(ctx: NodeContext) -> NodeResult:
    report, issues, run_dir = inspect_integration(ctx)
    path = ctx.worktree / "state" / REPORT_NAME
    write_json(path, report)
    outcome = "needs_retry" if issues else "ready"
    reason = ";".join(issues) if issues else "integration_contract_ready_for_visual"
    status = read_json(ctx.worktree / "INTEGRATION-STATUS.json")
    return NodeResult(
        outcome,
        updates={
            "integration_reason": reason,
            "integration_issues": issues,
            "integration_run": ctx.rel_to_worktree(run_dir),
            "integration_recipe_runs": report.get("recipe_run_dirs", {}),
            "executor_status": status.get("status", "missing"),
            "executor_terminal_reason": status.get("terminal_reason", ""),
        },
        artifacts=[ctx.rel_to_worktree(path)],
        message=reason,
    )
