#!/usr/bin/env python3
"""Workflow-local full diffusion, Codex visual, and recipe gate for PISA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


ASSESS_NAME = "assess_verdict.json"
RECIPES_NAME = "PISA-RECIPES.json"
REQUIRED_RECIPE_TIERS = ("conservative", "balanced", "aggressive")
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
MAX_TIER_SEVERITY = {"conservative": 1, "balanced": 2, "aggressive": 3}
REQUIRED_PROMPTS = "models/sana_video/prompts/dpo_holdout_qwen35_val64_concrete40_first5.txt"
REQUIRED_CONFIG = {
    "num_frames": 193,
    "fps": 24,
    "image_size": 720,
    "steps": 50,
    "cfg_scale": 8.0,
    "flow_shift": 12.0,
    "motion_score": 20,
}
INFRA_BLOCKER_HINTS = (
    "baseline_frame",
    "candidate_frame",
    "ffmpeg",
    "lpips",
    "codex_visual_inconclusive",
    "codex_visual_missing",
    "codex_visual_session",
    "codex_visual_launch",
    "codex_autorun",
    "missing_frame",
    "missing_video",
    "missing_benchmark",
    "no_output",
    "no-output",
    "slurm",
    "cancelled",
    "allocation",
    "quota",
    "filesystem",
    "heartbeat",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(root: Path, raw: str, run_dir: str | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [root / path]
    if run_dir:
        base = Path(run_dir)
        if not base.is_absolute():
            base = root / base
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def blocker_is_infra(blocker: Any) -> bool:
    text = blocker if isinstance(blocker, str) else json.dumps(blocker, sort_keys=True)
    lowered = text.lower()
    return any(hint in lowered for hint in INFRA_BLOCKER_HINTS)


def assess_paths_from_status(root: Path) -> list[Path]:
    status = read_json(root / "AGENT-STATUS.json")
    records: list[dict[str, Any]] = []
    if isinstance(status.get("evidence"), list):
        records.append(status)
    for collection in ("candidates", "frontier_candidates"):
        records.extend(row for row in status.get(collection, []) or [] if isinstance(row, dict))
    paths: list[Path] = []
    for record in records:
        run_dir = record.get("run_dir") if isinstance(record.get("run_dir"), str) else None
        for raw in record.get("evidence", []) or []:
            if isinstance(raw, str) and Path(raw).name == ASSESS_NAME:
                paths.append(resolve_path(root, raw, run_dir))
    return paths


def discover_assess_paths(root: Path) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for path in [*assess_paths_from_status(root), *root.glob(f"runs/*/{ASSESS_NAME}")]:
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)


def run_config(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "outputs" / "run_config.json")
    if config:
        return config
    benchmark = read_json(run_dir / "outputs" / "benchmark.json")
    value = benchmark.get("config")
    return value if isinstance(value, dict) else {}


def full_run_contract_issues(run_dir: Path) -> list[str]:
    config = run_config(run_dir)
    issues: list[str] = []
    for key, expected in REQUIRED_CONFIG.items():
        actual = config.get(key)
        try:
            ok = abs(float(actual) - float(expected)) <= 1e-6
        except (TypeError, ValueError):
            ok = False
        if not ok:
            issues.append(f"{key}={actual!r},expected={expected!r}")
    prompts_path = str(config.get("prompts_path") or "")
    if not prompts_path.endswith(REQUIRED_PROMPTS):
        issues.append(f"prompts_path={prompts_path!r},expected_suffix={REQUIRED_PROMPTS!r}")
    return issues


def visual_contract(root: Path, data: dict[str, Any], assess_path: Path) -> tuple[bool, str, dict[str, Any]]:
    if data.get("visual_provider") != "codex":
        return False, "assess_visual_provider_must_be_codex", {}
    overall = str(data.get("codex_visual_overall") or "").lower()
    if overall not in {"pass", "fail"}:
        return False, f"assess_codex_visual_inconclusive:{overall or 'missing'}", {}
    raw_path = str(data.get("codex_visual_verdict") or "")
    if not raw_path:
        return False, "codex_visual_missing", {}
    run_dir = str(data.get("run_dir") or assess_path.parent)
    path = resolve_path(root, raw_path, run_dir)
    visual = read_json(path)
    if visual.get("provider") != "codex" or visual.get("status") != "complete":
        return False, "codex_visual_verdict_invalid", visual
    if str(visual.get("overall") or "").lower() != overall:
        return False, "codex_visual_assess_mismatch", visual
    severity = str(data.get("max_artifact_severity") or "")
    if severity not in {"none", "low", "medium", "high", "critical"}:
        return False, "codex_visual_severity_invalid", visual
    return True, "", visual


def assess_gate(root: Path, path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "assess_missing", {}
    data = read_json(path)
    if not data:
        return False, "assess_invalid_json", {}
    missing = [
        key
        for key in ("baseline_total_s", "candidate_total_s", "speedup", "lpips_max")
        if not isinstance(data.get(key), (int, float))
    ]
    if missing:
        return False, "assess_missing_numeric_fields:" + ",".join(missing), data
    visual_ok, visual_reason, _ = visual_contract(root, data, path)
    if not visual_ok:
        return False, visual_reason, data
    blockers = list(data.get("quality_blockers") or [])
    infra = [blocker for blocker in blockers if blocker_is_infra(blocker)]
    if infra:
        return False, "assess_has_infrastructure_blockers:" + ",".join(map(str, infra)), data
    run_dir = resolve_path(root, str(data.get("run_dir") or path.parent))
    issues = full_run_contract_issues(run_dir)
    if issues:
        return False, "full_run_contract_mismatch:" + ";".join(issues), data
    quality = "pass" if data.get("codex_visual_overall") == "pass" and not blockers else "fail"
    return True, f"full_codex_visual_assess_complete_quality_{quality}", data


def recipe_contract(root: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    search_path = root / "PISA-SEARCH-STATE.json"
    search = read_json(search_path)
    if not search_path.exists():
        issues.append("missing:PISA-SEARCH-STATE.json")
    elif not search:
        issues.append("invalid_json:PISA-SEARCH-STATE.json")
    else:
        if search.get("schema_version") != 1:
            issues.append("pisa_search_state.schema_version_must_be_1")
        if not isinstance(search.get("trials"), list) or not search.get("trials"):
            issues.append("pisa_search_state.trials_missing")
        attention_map = search.get("attention_map")
        if not isinstance(attention_map, str) or not attention_map:
            issues.append("pisa_search_state.attention_map_missing")
        elif not resolve_path(root, attention_map).exists():
            issues.append("pisa_search_state.attention_map_not_found")

    path = root / RECIPES_NAME
    recipes_file = read_json(path)
    if not path.exists():
        return False, [*issues, f"missing:{RECIPES_NAME}"]
    if not recipes_file:
        return False, [*issues, f"invalid_json:{RECIPES_NAME}"]
    if recipes_file.get("schema_version") != 1:
        issues.append("schema_version_must_be_1")
    if recipes_file.get("model_id") != "sana_video":
        issues.append("model_id_must_be_sana_video")
    if recipes_file.get("workflow_uid") != "attention_pa":
        issues.append("workflow_uid_must_be_attention_pa")
    recipes = recipes_file.get("recipes")
    if not isinstance(recipes, dict):
        return False, [*issues, "recipes_must_be_object"]

    candidate_ids: list[str] = []
    assess_paths: list[str] = []
    for tier in REQUIRED_RECIPE_TIERS:
        recipe = recipes.get(tier)
        prefix = f"recipes.{tier}"
        if not isinstance(recipe, dict):
            issues.append(f"{prefix}_missing")
            continue
        if recipe.get("status") != "measured":
            issues.append(f"{prefix}.status_must_be_measured")
        for key in (
            "candidate_id",
            "source_hash",
            "backend",
            "route_mode",
            "dense_fallback",
            "run_dir",
            "assess_verdict",
        ):
            if not isinstance(recipe.get(key), str) or not recipe.get(key):
                issues.append(f"{prefix}.{key}_missing")
        if isinstance(recipe.get("candidate_id"), str) and recipe.get("candidate_id"):
            candidate_ids.append(str(recipe["candidate_id"]))
        density = recipe.get("density")
        sparsity = recipe.get("sparsity")
        if not isinstance(density, (int, float)) or not 0.0 < float(density) <= 1.0:
            issues.append(f"{prefix}.density_invalid")
        if not isinstance(sparsity, (int, float)) or not 0.0 <= float(sparsity) < 1.0:
            issues.append(f"{prefix}.sparsity_invalid")
        if isinstance(density, (int, float)) and isinstance(sparsity, (int, float)):
            if abs(float(density) + float(sparsity) - 1.0) > 1e-6:
                issues.append(f"{prefix}.density_sparsity_mismatch")
        if not isinstance(recipe.get("block_size"), list) or len(recipe.get("block_size")) != 2:
            issues.append(f"{prefix}.block_size_invalid")
        for key in ("route_bias", "only_video_self_attention"):
            if not isinstance(recipe.get(key), bool):
                issues.append(f"{prefix}.{key}_missing")
        for key in ("layer_policy", "step_policy", "attention_types", "dispatch"):
            if not isinstance(recipe.get(key), dict):
                issues.append(f"{prefix}.{key}_missing")
        for key in ("speedup", "full_e2e_total_s", "lpips_max"):
            if not isinstance(recipe.get(key), (int, float)):
                issues.append(f"{prefix}.{key}_missing")
        overall = str(recipe.get("codex_visual_overall") or "").lower()
        severity = str(recipe.get("max_artifact_severity") or "").lower()
        if overall not in {"pass", "fail"}:
            issues.append(f"{prefix}.codex_visual_overall_invalid")
        if severity not in {"none", "low", "medium", "high", "critical"}:
            issues.append(f"{prefix}.max_artifact_severity_invalid")
        if not isinstance(recipe.get("artifacts"), list):
            issues.append(f"{prefix}.artifacts_missing")
        if severity in SEVERITY_ORDER and SEVERITY_ORDER[severity] > MAX_TIER_SEVERITY[tier]:
            issues.append(f"{prefix}.quality_class_mismatch")

        assess_raw = recipe.get("assess_verdict")
        if isinstance(assess_raw, str) and assess_raw:
            assess_path = resolve_path(root, assess_raw, recipe.get("run_dir"))
            if not assess_path.exists() or assess_path.name != ASSESS_NAME:
                issues.append(f"{prefix}.assess_verdict_not_found")
            else:
                assess_paths.append(str(assess_path.resolve()))
                assess_ok, assess_reason, assess = assess_gate(root, assess_path)
                if not assess_ok:
                    issues.append(f"{prefix}.assessment_not_complete:{assess_reason}")
                else:
                    comparisons = (
                        ("speedup", recipe.get("speedup"), assess.get("speedup")),
                        ("full_e2e_total_s", recipe.get("full_e2e_total_s"), assess.get("candidate_total_s")),
                        ("lpips_max", recipe.get("lpips_max"), assess.get("lpips_max")),
                        ("codex_visual_overall", overall, str(assess.get("codex_visual_overall") or "").lower()),
                        ("max_artifact_severity", severity, str(assess.get("max_artifact_severity") or "").lower()),
                    )
                    for key, recipe_value, assess_value in comparisons:
                        if recipe_value != assess_value:
                            issues.append(f"{prefix}.{key}_does_not_match_assessment")
    if len(candidate_ids) == len(REQUIRED_RECIPE_TIERS) and len(set(candidate_ids)) != len(candidate_ids):
        issues.append("recipe_candidate_ids_must_be_distinct")
    if len(assess_paths) == len(REQUIRED_RECIPE_TIERS) and len(set(assess_paths)) != len(assess_paths):
        issues.append("recipe_assess_verdicts_must_be_distinct")
    return not issues, issues


def run(ctx: NodeContext) -> NodeResult:
    agent_status = read_json(ctx.worktree / "AGENT-STATUS.json")
    executor_status = str(agent_status.get("status") or "running")
    executor_terminal_reason = str(agent_status.get("terminal_reason") or "")
    recipes_ok, recipe_issues = recipe_contract(ctx.worktree)
    gates: list[dict[str, Any]] = []
    for path in discover_assess_paths(ctx.worktree):
        ok, reason, data = assess_gate(ctx.worktree, path)
        gate = {
            "path": rel_to(ctx.worktree, path),
            "ok": ok,
            "reason": reason,
            "speedup": data.get("speedup"),
            "lpips_max": data.get("lpips_max"),
            "codex_visual_overall": data.get("codex_visual_overall"),
            "max_artifact_severity": data.get("max_artifact_severity"),
            "quality_pass": str(data.get("codex_visual_overall") or "").lower() == "pass"
            and not bool(data.get("quality_blockers")),
        }
        gates.append(gate)
        if ok:
            return NodeResult(
                "smooth",
                updates={
                    "eval_gate": gate,
                    "eval_reason": reason,
                    "eval_all_gates": gates,
                    "executor_status": executor_status,
                    "executor_terminal_reason": executor_terminal_reason,
                    "recipe_contract_ok": recipes_ok,
                    "recipe_contract_issues": recipe_issues,
                },
                artifacts=[gate["path"]],
                message=reason,
            )
    reason = gates[-1]["reason"] if gates else "no_codex_visual_assess"
    if reason in {"assess_missing", "no_codex_visual_assess"}:
        outcome = "missing"
    elif blocker_is_infra(reason):
        outcome = "infra_blocked"
    else:
        outcome = "quality_failed"
    return NodeResult(
        outcome,
        updates={
            "eval_reason": reason,
            "eval_all_gates": gates,
            "executor_status": executor_status,
            "executor_terminal_reason": executor_terminal_reason,
            "recipe_contract_ok": recipes_ok,
            "recipe_contract_issues": recipe_issues,
        },
        message=reason,
    )
