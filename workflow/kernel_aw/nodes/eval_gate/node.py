#!/usr/bin/env python3
"""Workflow-local DiT evaluation gate node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


AUTHORITATIVE_GATE_NAMES = {"dit_eval.json", "gate_assess.json", "verdict.json"}
INFRA_BLOCKER_HINTS = (
    "no_output",
    "no-output",
    "slurm",
    "cancelled",
    "allocation",
    "quota",
    "filesystem",
    "stdout",
    "stderr",
    "heartbeat",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_gate_path(root: Path, raw: str, run_dir: str | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [root / path]
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = root / run_path
        candidates.append(run_path / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def gate_paths_from_status(root: Path) -> list[Path]:
    status = read_json(root / "AGENT-STATUS.json")
    paths: list[Path] = []
    active_gate = status.get("active_gate")
    if isinstance(active_gate, dict):
        active_gate = active_gate.get("path")
    if isinstance(active_gate, str) and Path(active_gate).name in AUTHORITATIVE_GATE_NAMES:
        paths.append(resolve_gate_path(root, active_gate))

    records = [record for record in status.get("candidates", []) or [] if isinstance(record, dict)]
    active_id = str(status.get("active_candidate_id") or "")
    if active_id:
        records = [record for record in records if str(record.get("candidate_id") or "") == active_id]
    elif records:
        # Legacy status files have no explicit active id. The last candidate is
        # the least-stale fallback; never scan older smooth candidates first.
        records = [records[-1]]

    for record in records:
        run_dir = record.get("run_dir") if isinstance(record.get("run_dir"), str) else None
        for raw in record.get("evidence", []) or []:
            if isinstance(raw, str) and Path(raw).name in AUTHORITATIVE_GATE_NAMES:
                paths.append(resolve_gate_path(root, raw, run_dir))
    return paths


def executor_discard_violations(root: Path) -> list[dict[str, Any]]:
    status = read_json(root / "AGENT-STATUS.json")
    violations: list[dict[str, Any]] = []
    for collection in ("discarded_candidates", "rejected_candidates"):
        for record in status.get(collection, []) or []:
            if isinstance(record, dict):
                violations.append(
                    {
                        "collection": collection,
                        "candidate_id": record.get("candidate_id"),
                        "decision": record.get("decision"),
                        "reason": record.get("reason"),
                    }
                )
    forbidden_prefixes = ("discard", "reject")
    for record in status.get("candidates", []) or []:
        if not isinstance(record, dict):
            continue
        decision = str(record.get("decision") or "").lower()
        if decision.startswith(forbidden_prefixes):
            violations.append(
                {
                    "collection": "candidates",
                    "candidate_id": record.get("candidate_id"),
                    "decision": record.get("decision"),
                    "reason": record.get("reason"),
                }
            )
    return violations


def discover_gate_paths(root: Path) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    status = read_json(root / "AGENT-STATUS.json")
    status_paths = gate_paths_from_status(root)
    has_current_selection = bool(
        status.get("active_gate") or status.get("active_candidate_id") or status.get("candidates")
    )
    globbed: list[Path] = []
    if not has_current_selection:
        for name in AUTHORITATIVE_GATE_NAMES:
            globbed.extend(root.glob(f"runs/*/{name}"))
        globbed.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in [*status_paths, *globbed]:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def blocker_is_infra(blocker: Any) -> bool:
    text = json.dumps(blocker, sort_keys=True) if not isinstance(blocker, str) else blocker
    lowered = text.lower()
    return any(hint in lowered for hint in INFRA_BLOCKER_HINTS)


def extract_speedup(data: dict[str, Any]) -> float | None:
    direct = data.get("speedup")
    if isinstance(direct, (int, float)):
        return float(direct)
    baseline = data.get("baseline_total_s")
    candidate = data.get("candidate_total_s")
    if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)) and candidate:
        return float(baseline) / float(candidate)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    for key in ("median_speedup_median", "speedup_median", "best_speedup_median", "worst_speedup_median"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def smooth_dit_gate(path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "gate_missing", {}
    if path.stat().st_size == 0:
        return False, "gate_empty", {}
    data = read_json(path)
    if not data:
        return False, "gate_invalid_json", {}
    blockers = []
    for key in ("blockers", "quality_blockers", "infra_blockers"):
        raw = data.get(key)
        if isinstance(raw, list):
            blockers.extend(raw)
    for key in ("reason", "message", "stderr_tail", "stdout_tail"):
        raw = data.get(key)
        if isinstance(raw, str):
            blockers.append(raw)
    infra_blockers = [item for item in blockers if blocker_is_infra(item)]
    if infra_blockers:
        return False, "gate_has_infrastructure_blockers:" + ",".join(map(str, infra_blockers)), data
    status = str(data.get("status") or "").lower()
    decision = str(data.get("decision") or "").lower()
    smooth_statuses = {"ok", "pass", "passed", "smooth", "complete", "completed"}
    smooth_decisions = {
        "keep",
        "passed",
        "promote",
        "needs_reviewer_judgment",
        "needs_full_diffusion_eval",
        "needs_final_review",
    }
    if status in smooth_statuses or decision in smooth_decisions:
        return True, "smooth_dit_gate", data
    if status or decision:
        return False, f"dit_gate_not_smooth:{status or decision}", data
    return False, "dit_gate_missing_status", data


def run(ctx: NodeContext) -> NodeResult:
    violations = executor_discard_violations(ctx.worktree)
    if violations:
        return NodeResult(
            "policy_violation",
            updates={
                "eval_reason": "executor_discard_decision_forbidden",
                "executor_discard_violations": violations,
            },
            message="executor_discard_decision_forbidden",
        )

    status = read_json(ctx.worktree / "AGENT-STATUS.json")
    active_id = str(status.get("active_candidate_id") or "")
    gate_results = []
    for path in discover_gate_paths(ctx.worktree):
        ok, reason, data = smooth_dit_gate(path)
        gate_candidate_id = str(data.get("candidate_id") or "")
        if ok and active_id and gate_candidate_id and gate_candidate_id != active_id:
            ok = False
            reason = f"gate_candidate_mismatch:{gate_candidate_id}!={active_id}"
        gate = {
            "path": rel_to(ctx.worktree, path),
            "ok": ok,
            "reason": reason,
            "speedup": extract_speedup(data),
            "candidate_id": gate_candidate_id or active_id or None,
            "measurement_scope": data.get("measurement_scope") or data.get("scope"),
        }
        gate_results.append(gate)
        if ok:
            return NodeResult(
                "smooth",
                updates={"eval_gate": gate, "eval_reason": "smooth_dit_gate", "eval_all_gates": gate_results},
                artifacts=[gate["path"]],
                message="smooth_dit_gate",
            )

    reason = gate_results[-1]["reason"] if gate_results else "no_smooth_dit_gate"
    outcome = "infra_blocked" if blocker_is_infra(reason) else "missing"
    return NodeResult(
        outcome,
        updates={"eval_reason": reason, "eval_all_gates": gate_results},
        message=reason,
    )
