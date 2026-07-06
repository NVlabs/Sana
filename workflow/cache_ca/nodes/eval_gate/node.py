#!/usr/bin/env python3
"""Workflow-local full diffusion + LPIPS + Codex visual gate for cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


ASSESS_NAME = "assess_verdict.json"
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
    quality = "pass" if data.get("codex_visual_overall") == "pass" and not blockers else "shifted"
    return True, f"full_codex_visual_assess_complete_quality_{quality}", data


def run(ctx: NodeContext) -> NodeResult:
    agent_status = read_json(ctx.worktree / "AGENT-STATUS.json")
    executor_status = str(agent_status.get("status") or "running")
    executor_terminal_reason = str(agent_status.get("terminal_reason") or "")
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
        },
        message=reason,
    )
