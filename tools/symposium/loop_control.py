#!/usr/bin/env python3
"""Runtime state machine for fan-out and integration goal loops."""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
VALID_STATUS = {"running", "terminal_pending_review", "blocked", "complete"}
KEEP_DECISIONS = {
    "quality_improved",
    "speed_improved",
    "quality_and_speed_improved",
    "kept_frontier",
    # Backward-compatible alias for older goal prompts.
    "pareto_improved",
}
DISCARD_DECISIONS = {
    "discarded_regression",
    # Backward-compatible alias for older goal prompts.
    "non_improving_pass",
}
HARD_REJECT_DECISIONS = {"rejected"}
VALID_RECORD_DECISIONS = {
    *KEEP_DECISIONS,
    *DISCARD_DECISIONS,
    *HARD_REJECT_DECISIONS,
    "blocked",
    "structured_negative",
    "orchestrator_release",
}
VALID_REVIEW_ACTIONS = {
    "select_tiers_for_integration",
    "accept_frontier_for_integration",
    "restart_with_new_direction",
    "request_validation",
    "mark_blocked",
    "drop_dimension",
    "integrate",
    "stop",
}
AUTHORITATIVE_GATE_ARTIFACTS = {
    "assess_verdict.json",
    "verdict.json",
    "gate_assess.json",
    "reject_note.json",
}
DEFAULT_INTEGRATION_OBJECTIVE = (
    "Integrate fan-out winners into gated composed low, medium, and high delivery "
    "profiles for the 1.5x, 2.0x, and 3.0x speed targets. Read each dimension "
    "status and durable run artifacts, build target plans, merge one composed "
    "profile per iteration, launch GPU generation, run the authoritative aligned "
    "gate, rank quality with Gemini and LPIPS together, and loop on failures "
    "until every target has a composed artifact or explicit blocker."
)
REQUIRED_STATUS_FIELDS = {
    "schema_version": int,
    "dimension": str,
    "status": str,
    "iters_used": int,
    "max_iters": int,
    "early_stop_patience": int,
    "loop_mode": str,
    "no_improve_count": int,
    "best_per_tier": dict,
    "frontier_candidates": list,
    "discarded_candidates": list,
    "rejected_candidates": list,
    "candidates": list,
    "failure_signatures": list,
    "remaining_hypotheses": list,
    "agent_recommendation": str,
    "next_commands": list,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing status file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def shell_join(items: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def run_checked(cmd: list[str | Path], cwd: Path, env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(
        [str(item) for item in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"command failed: {shell_join(cmd)}"
        raise SystemExit(message)


def status_template(args: argparse.Namespace) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": now,
        "updated_at_utc": now,
        "dimension": args.dimension,
        "goal_id": args.goal_id or args.dimension,
        "status": "running",
        "iters_used": 0,
        "max_iters": args.max_iters,
        "early_stop_patience": args.early_stop_patience,
        "loop_mode": args.loop_mode,
        "retention_rule": "keep_candidate_if_quality_improves_or_speed_improves",
        "discard_rule": "discard_if_no_quality_improvement_and_no_speed_improvement_or_speed_regresses",
        "tier_selection_policy": "after_budget_select_1p5x_2x_3x_speed_targets_by_best_gemini_and_lpips_quality",
        "no_improve_count": 0,
        "best_per_tier": {},
        "frontier_candidates": [],
        "discarded_candidates": [],
        "rejected_candidates": [],
        "candidates": [],
        "failure_signatures": [],
        "remaining_hypotheses": [],
        "agent_recommendation": "",
        "next_commands": [
            "propose_next_hypothesis",
            "implement_one_candidate",
            "preflight_launch_collect_gate",
            "record_candidate_with_loop_control",
        ],
        "early_stop_reason": "",
        "terminal_reason": "",
        "blocker": {},
    }


def resolve_evidence_path(raw_path: str, base_dir: Path | None, run_dir: str | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or base_dir is None:
        return path

    candidates = [base_dir / path]
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = base_dir / run_path
        candidates.append(run_path / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def record_requires_authoritative_gate(record: dict[str, Any]) -> bool:
    if record.get("decision") in {"blocked", "orchestrator_release"}:
        return False
    return bool(record.get("run_dir"))


def validate_record_evidence(record: dict[str, Any], base_dir: Path | None = None, label: str = "record") -> list[str]:
    if not record_requires_authoritative_gate(record):
        return []

    evidence = record.get("evidence") or []
    if not isinstance(evidence, list):
        return [f"{label}.evidence must be list"]

    authoritative = [
        item
        for item in evidence
        if isinstance(item, str) and Path(item).name in AUTHORITATIVE_GATE_ARTIFACTS
    ]
    if not authoritative:
        names = ", ".join(sorted(AUTHORITATIVE_GATE_ARTIFACTS))
        return [f"{label}.evidence must include authoritative gate artifact ({names})"]

    errors: list[str] = []
    for item in authoritative:
        path = resolve_evidence_path(item, base_dir, record.get("run_dir") or None)
        if not path.exists():
            errors.append(f"{label}.evidence artifact does not exist: {item}")
            continue
        if path.stat().st_size == 0:
            errors.append(f"{label}.evidence artifact is empty: {item}")
            continue
        try:
            json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{label}.evidence artifact is not valid JSON: {item}: {exc}")
    return errors


def validate_status_payload(
    payload: dict[str, Any],
    base_dir: Path | None = None,
    *,
    require_evidence: bool = True,
) -> list[str]:
    errors: list[str] = []
    for field, expected_type in REQUIRED_STATUS_FIELDS.items():
        if field not in payload:
            errors.append(f"missing field: {field}")
            continue
        if not isinstance(payload[field], expected_type):
            errors.append(f"{field} must be {expected_type.__name__}")

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("status") not in VALID_STATUS:
        errors.append(f"status must be one of {sorted(VALID_STATUS)}")
    if payload.get("max_iters", 0) < 1:
        errors.append("max_iters must be >= 1")
    if payload.get("early_stop_patience", 0) < 0:
        errors.append("early_stop_patience must be >= 0")
    if payload.get("iters_used", 0) < 0:
        errors.append("iters_used must be >= 0")
    if payload.get("no_improve_count", 0) < 0:
        errors.append("no_improve_count must be >= 0")
    if payload.get("iters_used", 0) > payload.get("max_iters", 0):
        errors.append("iters_used cannot exceed max_iters")
    if payload.get("status") == "blocked" and not payload.get("blocker"):
        errors.append("blocked status requires blocker details")
    if payload.get("status") == "terminal_pending_review":
        if (
            not payload.get("early_stop_reason")
            and not payload.get("terminal_reason")
            and not payload.get("agent_recommendation")
        ):
            errors.append("terminal_pending_review requires terminal_reason or agent_recommendation")
    if payload.get("agent_recommendation") and payload["agent_recommendation"] not in VALID_REVIEW_ACTIONS:
        errors.append(f"agent_recommendation must be one of {sorted(VALID_REVIEW_ACTIONS)}")
    for collection_name in ("candidates", "frontier_candidates"):
        records = payload.get(collection_name)
        if not isinstance(records, list):
            continue
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"{collection_name}[{idx}] must be object")
                continue
            decision = record.get("decision", "")
            axis = record.get("improvement_axis", "")
            needs_speedup = decision in {"speed_improved", "quality_and_speed_improved"} or axis in {"speed", "both"}
            if needs_speedup and not isinstance(record.get("speedup"), (int, float)):
                errors.append(f"{collection_name}[{idx}].speedup must be numeric for speed-improved records")
            if require_evidence:
                errors.extend(validate_record_evidence(record, base_dir, f"{collection_name}[{idx}]"))

    candidate_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, record in enumerate(payload.get("candidates", [])):
        if isinstance(record, dict):
            key = (str(record.get("candidate_id", "")), str(record.get("run_dir", "")))
            if key[0]:
                candidate_by_key[key] = record

    non_frontier_keys: dict[tuple[str, str], str] = {}
    for collection_name in ("discarded_candidates", "rejected_candidates"):
        for record in payload.get(collection_name, []):
            if isinstance(record, dict):
                key = (str(record.get("candidate_id", "")), str(record.get("run_dir", "")))
                if key[0]:
                    non_frontier_keys[key] = collection_name

    for idx, record in enumerate(payload.get("frontier_candidates", [])):
        if not isinstance(record, dict):
            continue
        key = (str(record.get("candidate_id", "")), str(record.get("run_dir", "")))
        if record.get("decision") not in KEEP_DECISIONS:
            errors.append(f"frontier_candidates[{idx}].decision must be a keep decision")
        source = candidate_by_key.get(key)
        if source is None:
            errors.append(f"frontier_candidates[{idx}] has no matching candidates record")
        elif source.get("decision") not in KEEP_DECISIONS:
            errors.append(
                f"frontier_candidates[{idx}] source candidate decision must be a keep decision"
            )
        elif source.get("improvement_axis") != record.get("improvement_axis"):
            errors.append(
                f"frontier_candidates[{idx}] improvement_axis differs from candidates record"
            )
        if key in non_frontier_keys:
            errors.append(
                f"frontier_candidates[{idx}] also appears in {non_frontier_keys[key]}"
            )
    return errors


def validate_or_exit(payload: dict[str, Any], path: Path) -> None:
    errors = validate_status_payload(payload, path.parent)
    if errors:
        raise SystemExit(f"{path} failed schema validation:\n- " + "\n- ".join(errors))


def append_journal(path: Path, entry: dict[str, Any]) -> None:
    lines = [
        "",
        f"## Iter {entry['iter']}: {entry['candidate_id']}",
        "",
        f"- decision: `{entry['decision']}`",
        f"- reason: {entry.get('reason') or 'n/a'}",
    ]
    if entry.get("tier"):
        lines.append(f"- tier: `{entry['tier']}`")
    if entry.get("improvement_axis"):
        lines.append(f"- improvement_axis: `{entry['improvement_axis']}`")
    if entry.get("run_dir"):
        lines.append(f"- run_dir: `{entry['run_dir']}`")
    if entry.get("manifest"):
        lines.append(f"- manifest: `{entry['manifest']}`")
    if entry.get("evidence"):
        lines.append("- evidence: " + ", ".join(f"`{item}`" for item in entry["evidence"]))
    if entry.get("next_decision"):
        lines.append(f"- next_decision: `{entry['next_decision']}`")
    path.write_text((path.read_text() if path.exists() else "# Search Journal\n") + "\n".join(lines) + "\n")


def decide_status(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["status"] == "blocked":
        return {
            "decision": "blocked",
            "reason": payload.get("blocker", {}).get("reason", "blocked"),
            "status": "blocked",
        }
    if payload["status"] in {"terminal_pending_review", "complete"}:
        return {
            "decision": payload["status"],
            "reason": payload.get("terminal_reason") or payload.get("early_stop_reason") or payload["status"],
            "status": payload["status"],
        }
    if payload["iters_used"] >= payload["max_iters"]:
        return {
            "decision": "terminal_pending_review",
            "reason": "max_iters_reached",
            "status": "terminal_pending_review",
        }
    if (
        payload.get("loop_mode") != "fixed_budget_frontier"
        and payload.get("early_stop_patience", 0) > 0
        and payload["no_improve_count"] >= payload["early_stop_patience"]
    ):
        return {
            "decision": "terminal_pending_review",
            "reason": "early_stop_patience_reached",
            "status": "terminal_pending_review",
        }
    return {
        "decision": "continue",
        "reason": "budget_remaining",
        "status": "running",
    }


def default_recommendation(payload: dict[str, Any]) -> str:
    if payload["status"] == "blocked":
        return "mark_blocked"
    if payload.get("frontier_candidates"):
        return "select_tiers_for_integration"
    if payload.get("best_per_tier"):
        return "accept_frontier_for_integration"
    if payload.get("remaining_hypotheses"):
        return "restart_with_new_direction"
    if payload.get("failure_signatures"):
        return "drop_dimension"
    return "request_validation"


def cmd_init(args: argparse.Namespace) -> int:
    status_path = Path(args.status_file)
    if status_path.exists() and not args.force:
        raise SystemExit(f"{status_path} already exists; use --force to overwrite")
    payload = status_template(args)
    write_json(status_path, payload)
    journal_path = Path(args.journal_file)
    if not journal_path.exists() or args.force:
        journal_path.write_text(f"# Search Journal — {args.dimension}\n")
    print(json.dumps(decide_status(payload), sort_keys=True))
    return 0


def record_entry(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "iter": payload["iters_used"] + 1,
        "candidate_id": args.candidate_id,
        "decision": args.decision,
        "reason": args.reason,
        "tier": args.tier,
        "run_dir": args.run_dir,
        "manifest": args.manifest,
        "evidence": args.evidence or [],
        "speedup": args.speedup,
        "quality": args.quality,
        "improvement_axis": args.improvement_axis,
        "recorded_at_utc": utc_now(),
    }
    return entry


def retained_candidate(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": entry["candidate_id"],
        "run_dir": entry["run_dir"],
        "manifest": entry["manifest"],
        "decision": entry["decision"],
        "improvement_axis": entry.get("improvement_axis"),
        "reason": entry["reason"],
        "tier": entry["tier"],
        "speedup": entry["speedup"],
        "quality": entry["quality"],
        "evidence": entry["evidence"],
        "updated_at_utc": entry["recorded_at_utc"],
    }


def cmd_record_candidate(args: argparse.Namespace) -> int:
    status_path = Path(args.status_file)
    payload = read_json(status_path)
    validate_or_exit(payload, status_path)
    if payload["status"] != "running":
        raise SystemExit(f"Cannot record candidate while status={payload['status']}")
    if args.decision not in VALID_RECORD_DECISIONS:
        raise SystemExit(f"Invalid decision: {args.decision}")

    entry = record_entry(args, payload)
    evidence_errors = validate_record_evidence(entry, status_path.parent, "new candidate")
    if evidence_errors:
        raise SystemExit("Cannot record candidate:\n- " + "\n- ".join(evidence_errors))

    payload["iters_used"] += 1
    payload["candidates"].append(entry)
    payload["updated_at_utc"] = utc_now()

    if args.remaining_hypothesis:
        payload["remaining_hypotheses"].extend(args.remaining_hypothesis)

    if args.decision in KEEP_DECISIONS:
        payload["no_improve_count"] = 0
        payload["frontier_candidates"].append(retained_candidate(entry))
    elif args.decision == "blocked":
        payload["status"] = "blocked"
        payload["blocker"] = {
            "candidate_id": args.candidate_id,
            "reason": args.reason or "blocked",
            "evidence": args.evidence or [],
        }
    elif args.decision == "orchestrator_release":
        payload["status"] = "terminal_pending_review"
        payload["early_stop_reason"] = "orchestrator_release"
        payload["terminal_reason"] = "orchestrator_release"
    else:
        payload["no_improve_count"] += 1
        rejected = {
            "candidate_id": args.candidate_id,
            "run_dir": args.run_dir,
            "manifest": args.manifest,
            "decision": args.decision,
            "reason": args.reason,
            "evidence": args.evidence or [],
            "updated_at_utc": payload["updated_at_utc"],
        }
        if args.decision in DISCARD_DECISIONS:
            payload["discarded_candidates"].append(rejected)
        if args.decision in HARD_REJECT_DECISIONS:
            payload["rejected_candidates"].append(rejected)
            payload["failure_signatures"].append(rejected)
        if args.decision == "structured_negative":
            rejected["decision"] = "structured_negative_proposal"
            payload["failure_signatures"].append(rejected)
            payload["remaining_hypotheses"].extend(
                args.remaining_hypothesis
                or [
                    "Continue fixed-budget search; structured-negative proposals require max_iters, real blocker, or explicit orchestrator release."
                ]
            )

    decision = decide_status(payload)
    if decision["decision"] == "terminal_pending_review":
        payload["status"] = "terminal_pending_review"
        payload["early_stop_reason"] = payload.get("early_stop_reason") or decision["reason"]
        payload["terminal_reason"] = payload.get("terminal_reason") or decision["reason"]
        payload["agent_recommendation"] = args.recommendation or payload.get("agent_recommendation") or default_recommendation(payload)
        payload["next_commands"] = [f"main_agent_review:{payload['agent_recommendation']}"]
    elif decision["decision"] == "continue":
        payload["status"] = "running"
        payload["agent_recommendation"] = args.recommendation or payload.get("agent_recommendation", "")
        payload["next_commands"] = [
            "propose_next_hypothesis",
            "implement_one_candidate",
            "preflight_launch_collect_gate",
            "record_candidate_with_loop_control",
        ]
    elif decision["decision"] == "blocked":
        payload["agent_recommendation"] = "mark_blocked"
        payload["next_commands"] = ["main_agent_review:mark_blocked"]

    entry["next_decision"] = decision["decision"]
    validate_or_exit(payload, status_path)
    write_json(status_path, payload)
    append_journal(Path(args.journal_file), entry)
    print(json.dumps(decision, sort_keys=True))
    return 0


def cmd_add_evidence(args: argparse.Namespace) -> int:
    status_path = Path(args.status_file)
    payload = read_json(status_path)
    structural_errors = validate_status_payload(payload, status_path.parent, require_evidence=False)
    if structural_errors:
        raise SystemExit(f"{status_path} failed schema validation:\n- " + "\n- ".join(structural_errors))

    touched = 0
    collections = (
        "candidates",
        "frontier_candidates",
        "discarded_candidates",
        "rejected_candidates",
        "failure_signatures",
    )
    for collection_name in collections:
        for record in payload.get(collection_name, []):
            if not isinstance(record, dict) or record.get("candidate_id") != args.candidate_id:
                continue
            record.setdefault("evidence", [])
            for item in args.evidence:
                if item not in record["evidence"]:
                    record["evidence"].append(item)
            if args.reason:
                record["evidence_backfill_reason"] = args.reason
            record["updated_at_utc"] = utc_now()
            touched += 1

    if touched == 0:
        raise SystemExit(f"No records found for candidate_id={args.candidate_id}")

    payload["updated_at_utc"] = utc_now()
    structural_errors = validate_status_payload(payload, status_path.parent, require_evidence=False)
    if structural_errors:
        raise SystemExit(f"{status_path} failed schema validation:\n- " + "\n- ".join(structural_errors))

    target_errors: list[str] = []
    for collection_name in collections:
        for idx, record in enumerate(payload.get(collection_name, [])):
            if isinstance(record, dict) and record.get("candidate_id") == args.candidate_id:
                target_errors.extend(validate_record_evidence(record, status_path.parent, f"{collection_name}[{idx}]"))
    if target_errors:
        raise SystemExit("Backfilled candidate still has invalid evidence:\n- " + "\n- ".join(target_errors))

    write_json(status_path, payload)
    remaining_errors = validate_status_payload(payload, status_path.parent)
    print(json.dumps({
        "candidate_id": args.candidate_id,
        "evidence_added": args.evidence,
        "records_touched": touched,
        "status_ok": not remaining_errors,
        "remaining_errors": remaining_errors,
    }, sort_keys=True))
    return 0


def cmd_decide_next(args: argparse.Namespace) -> int:
    status_path = Path(args.status_file)
    payload = read_json(status_path)
    validate_or_exit(payload, status_path)
    decision = decide_status(payload)
    if decision["decision"] == "terminal_pending_review" and payload["status"] == "running":
        payload["status"] = "terminal_pending_review"
        payload["early_stop_reason"] = decision["reason"]
        payload["terminal_reason"] = decision["reason"]
        payload["agent_recommendation"] = payload.get("agent_recommendation") or default_recommendation(payload)
        payload["next_commands"] = [f"main_agent_review:{payload['agent_recommendation']}"]
        payload["updated_at_utc"] = utc_now()
        write_json(status_path, payload)
    print(json.dumps(decision, sort_keys=True))
    return 0


def cmd_validate_status(args: argparse.Namespace) -> int:
    status_path = Path(args.status_file)
    payload = read_json(status_path)
    errors = validate_status_payload(payload, status_path.parent)
    result = {"ok": not errors, "errors": errors, "status_file": str(status_path)}
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


def review_one(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    errors = validate_status_payload(payload, path.parent)
    if errors:
        return {"path": str(path), "valid": False, "errors": errors, "action": "request_validation"}
    action = payload.get("agent_recommendation") or default_recommendation(payload)
    if action == "integrate":
        action = "accept_frontier_for_integration"
    return {
        "path": str(path),
        "valid": True,
        "dimension": payload["dimension"],
        "status": payload["status"],
        "iters_used": payload["iters_used"],
        "no_improve_count": payload["no_improve_count"],
        "best_tiers": sorted(payload.get("best_per_tier", {}).keys()),
        "frontier_count": len(payload.get("frontier_candidates", [])),
        "discarded_count": len(payload.get("discarded_candidates", [])),
        "action": action,
        "next_commands": payload.get("next_commands", []),
    }


def cmd_review_dimensions(args: argparse.Namespace) -> int:
    paths = collect_status_paths(args.status_file, args.glob)
    if not paths:
        raise SystemExit("No status files provided")
    reviews = [review_one(path) for path in paths]
    global_decision = global_decision_for_reviews(reviews)
    print(json.dumps({"global_decision": global_decision, "dimensions": reviews}, indent=2, sort_keys=True))
    return 0


def collect_status_paths(status_files: list[str], glob_patterns: list[str]) -> list[Path]:
    paths = [Path(item) for item in status_files]
    for pattern in glob_patterns:
        paths.extend(Path(item) for item in sorted(globlib.glob(pattern)))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def global_decision_for_reviews(reviews: list[dict[str, Any]]) -> str:
    if any(not item["valid"] for item in reviews):
        return "request_validation"
    if any(item["status"] == "blocked" for item in reviews):
        return "blocked"
    if any(item["status"] == "running" for item in reviews):
        return "continue_monitoring"
    if any(item["action"] == "restart_with_new_direction" for item in reviews):
        return "reopen_dimension_loop"
    if any(item["action"] == "select_tiers_for_integration" for item in reviews):
        return "tier_selection_pending"
    if any(item["action"] == "accept_frontier_for_integration" for item in reviews):
        return "integration_pending"
    return "no_eligible_dimensions"


def infer_run_id_from_fanout_root(fanout_root: Path, explicit: str = "") -> str:
    if explicit:
        return explicit
    for name in ("SYMPOSIUM_CURRENT_RUN_ID", "AUTO_VIDEO_RUN_ID", "RUN_ID"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return fanout_root.resolve().name


def integration_state(integration_dir: Path) -> tuple[str, dict[str, Any]]:
    status_path = integration_dir / "INTEGRATION-STATUS.json"
    if status_path.exists():
        payload = read_json(status_path)
        if payload.get("status") == "complete":
            return "complete", payload
        return "status_present", payload
    agent_path = integration_dir / "AGENT-STATUS.json"
    if agent_path.exists():
        payload = read_json(agent_path)
        if payload.get("status") in {"running", "terminal_pending_review", "complete"}:
            return str(payload.get("status")), payload
    if integration_dir.exists():
        return "worktree_present", {}
    return "missing", {}


def integration_plan(args: argparse.Namespace, review: dict[str, Any]) -> dict[str, Any]:
    fanout_root = Path(args.fanout_root).resolve()
    run_id = infer_run_id_from_fanout_root(fanout_root, args.run_id)
    integration_dir = (fanout_root / args.integration_dir).resolve()
    branch = args.branch or f"codex/{run_id}-integration"
    codex_home = Path(args.codex_home).resolve() if args.codex_home else integration_dir / ".codex-home"
    session_name = args.name or f"{run_id}-integration"
    objective = args.objective or DEFAULT_INTEGRATION_OBJECTIVE
    root = project_root()
    commands = [
        ["git", "worktree", "add", "-b", branch, integration_dir, args.base],
        [
            sys.executable,
            "tools/symposium/prepare_goal.py",
            "--clean-stale-records",
            "--run-id",
            run_id,
        ],
        [
            sys.executable,
            "tools/symposium/prepare_goal.py",
            "--goal-id",
            "integration",
            "--candidate",
            args.candidate,
            "--dimension",
            "integration",
            "--role",
            "integration",
            "--run-id",
            run_id,
            "--root-branch",
            branch,
            "--submodule-branch",
            f"{branch}-sol",
            "--objective",
            objective,
            "--overwrite",
        ],
    ]
    if args.start_session:
        commands.append(
            [
                sys.executable,
                root / "tools/symposium/codex_goal_session.py",
                "start",
                "--worktree",
                integration_dir,
                "--name",
                session_name,
                "goals/integration",
            ]
        )
    return {
        "fanout_root": str(fanout_root),
        "run_id": run_id,
        "integration_dir": str(integration_dir),
        "branch": branch,
        "codex_home": str(codex_home),
        "session_name": session_name,
        "review": review,
        "commands": [shell_join(command) for command in commands],
        "command_argv": [[str(item) for item in command] for command in commands],
    }


def prepare_codex_home(codex_home: Path, source: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    for name in ("auth.json", "config.toml"):
        src = source / name
        if src.exists():
            shutil.copy2(src, codex_home / name)


def execute_integration_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    root = project_root()
    fanout_root = Path(plan["fanout_root"])
    integration_dir = Path(plan["integration_dir"])
    codex_home = Path(plan["codex_home"])
    command_argv = [[str(item) for item in command] for command in plan["command_argv"]]

    if not integration_dir.exists():
        run_checked(command_argv[0], cwd=root)
    elif not (integration_dir / ".git").exists():
        raise SystemExit(f"Integration directory exists but is not a git worktree: {integration_dir}")

    prepare_codex_home(codex_home, Path(args.codex_home_source).expanduser())
    run_checked(command_argv[1], cwd=integration_dir)
    run_checked(command_argv[2], cwd=integration_dir)
    if args.start_session:
        env = dict(os.environ)
        env["CODEX_HOME"] = str(codex_home)
        env["SYMPOSIUM_CURRENT_RUN_ID"] = plan["run_id"]
        env["AUTO_VIDEO_RUN_ID"] = plan["run_id"]
        env["RUN_ID"] = plan["run_id"]
        run_checked(command_argv[3], cwd=fanout_root, env=env)


def cmd_ensure_integration(args: argparse.Namespace) -> int:
    fanout_root = Path(args.fanout_root)
    if not fanout_root.exists():
        raise SystemExit(f"Fanout root does not exist: {fanout_root}")
    glob_patterns = list(args.glob)
    if not args.status_file and not glob_patterns:
        glob_patterns.append(str(fanout_root / "*" / "AGENT-STATUS.json"))
    paths = [
        path
        for path in collect_status_paths(args.status_file, glob_patterns)
        if Path(path).parent.name != args.integration_dir
    ]
    if not paths:
        raise SystemExit("No fan-out dimension status files provided")
    reviews = [review_one(path) for path in paths]
    review = {"global_decision": global_decision_for_reviews(reviews), "dimensions": reviews}
    state, state_payload = integration_state(fanout_root / args.integration_dir)
    result: dict[str, Any] = {
        "decision": "no_op",
        "reason": review["global_decision"],
        "integration_state": state,
        "review": review,
    }
    if state == "complete":
        result.update({"decision": "already_complete", "status": state_payload.get("status")})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if state in {"running", "terminal_pending_review", "status_present"}:
        result.update({"decision": "already_started", "status": state_payload.get("status")})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if review["global_decision"] not in {"tier_selection_pending", "integration_pending"}:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    plan = integration_plan(args, review)
    result.update(
        {
            "decision": "start_integration",
            "reason": review["global_decision"],
            "plan": {k: v for k, v in plan.items() if k != "command_argv"},
            "dry_run": args.dry_run,
        }
    )
    if not args.dry_run:
        execute_integration_plan(plan, args)
        result["started"] = True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create AGENT-STATUS.json and SEARCH_JOURNAL.md")
    init.add_argument("--dimension", required=True)
    init.add_argument("--goal-id", default="")
    init.add_argument("--max-iters", type=int, default=40)
    init.add_argument("--early-stop-patience", type=int, default=0)
    init.add_argument("--loop-mode", choices=("fixed_budget_frontier", "patience_frontier"), default="fixed_budget_frontier")
    init.add_argument("--status-file", default="AGENT-STATUS.json")
    init.add_argument("--journal-file", default="SEARCH_JOURNAL.md")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    record = sub.add_parser("record-candidate", help="record one candidate verdict and update loop state")
    record.add_argument("--candidate-id", required=True)
    record.add_argument("--decision", required=True, choices=sorted(VALID_RECORD_DECISIONS))
    record.add_argument("--status-file", default="AGENT-STATUS.json")
    record.add_argument("--journal-file", default="SEARCH_JOURNAL.md")
    record.add_argument("--tier", default="")
    record.add_argument("--run-dir", default="")
    record.add_argument("--manifest", default="")
    record.add_argument("--reason", default="")
    record.add_argument("--evidence", action="append", default=[])
    record.add_argument("--remaining-hypothesis", action="append", default=[])
    record.add_argument("--recommendation", choices=sorted(VALID_REVIEW_ACTIONS), default="")
    record.add_argument("--speedup", type=float)
    record.add_argument("--quality", default="")
    record.add_argument("--improvement-axis", choices=("quality", "speed", "both", "none"), default="")
    record.set_defaults(func=cmd_record_candidate)

    add_evidence = sub.add_parser("add-evidence", help="append evidence to an existing candidate record")
    add_evidence.add_argument("--candidate-id", required=True)
    add_evidence.add_argument("--status-file", default="AGENT-STATUS.json")
    add_evidence.add_argument("--evidence", action="append", required=True)
    add_evidence.add_argument("--reason", default="")
    add_evidence.set_defaults(func=cmd_add_evidence)

    decide = sub.add_parser("decide-next", help="print continue/terminal/blocker decision")
    decide.add_argument("--status-file", default="AGENT-STATUS.json")
    decide.set_defaults(func=cmd_decide_next)

    validate = sub.add_parser("validate-status", help="validate AGENT-STATUS.json schema")
    validate.add_argument("--status-file", default="AGENT-STATUS.json")
    validate.set_defaults(func=cmd_validate_status)

    review = sub.add_parser("review-dimensions", help="main-agent review over multiple AGENT-STATUS.json files")
    review.add_argument("--status-file", action="append", default=[])
    review.add_argument("--glob", action="append", default=[])
    review.set_defaults(func=cmd_review_dimensions)

    ensure = sub.add_parser("ensure-integration", help="start the mandatory fan-in integration goal when fan-out is terminal")
    ensure.add_argument("--fanout-root", required=True)
    ensure.add_argument("--status-file", action="append", default=[])
    ensure.add_argument("--glob", action="append", default=[])
    ensure.add_argument("--run-id", default="")
    ensure.add_argument("--integration-dir", default="integration")
    ensure.add_argument("--base", default="HEAD")
    ensure.add_argument("--branch", default="")
    ensure.add_argument("--candidate", default="candidates/baseline.toml")
    ensure.add_argument("--objective", default="")
    ensure.add_argument("--name", default="")
    ensure.add_argument("--codex-home", default="")
    ensure.add_argument("--codex-home-source", default=str(Path.home() / ".codex"))
    ensure.add_argument("--start-session", action=argparse.BooleanOptionalAction, default=True)
    ensure.add_argument("--dry-run", action="store_true")
    ensure.set_defaults(func=cmd_ensure_integration)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "max_iters") and args.max_iters < 1:
        raise SystemExit("--max-iters must be >= 1")
    if hasattr(args, "early_stop_patience") and args.early_stop_patience < 0:
        raise SystemExit("--early-stop-patience must be >= 0")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
