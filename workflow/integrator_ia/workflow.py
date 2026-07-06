#!/usr/bin/env python3
"""Explicit integration loop for tiered kernel, PISA, and cache recipes.

The workflow owns all runtime nodes under ``workflow/integrator_ia/nodes``.
Donor workflow directories and donor experiment worktrees are read-only inputs;
the integration experiment is the only mutable implementation target.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKFLOW_DIR = Path(__file__).resolve().parent
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from workflow_types import NodeContext, NodeResult  # noqa: E402
from nodes.baseline_gate.node import run as run_baseline_gate  # noqa: E402
from nodes.baseline_run.node import run as run_baseline  # noqa: E402
from nodes.codex_executor.node import run as run_executor  # noqa: E402
from nodes.codex_visual_reviewer.node import run as run_visual_reviewer  # noqa: E402
from nodes.delivery_gate.node import run as run_delivery_gate  # noqa: E402
from nodes.final_gate.node import run as run_final_gate  # noqa: E402
from nodes.integration_gate.node import run as run_integration_gate  # noqa: E402
from nodes.resume_prompt.node import run as run_resume_prompt  # noqa: E402
from nodes.source_gate.node import run as run_source_gate  # noqa: E402


WORKFLOW_UID = "integrator_ia"
WORKFLOW_UID_RE = re.compile(r"^([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*)_([A-Za-z]{2})$")
EXPERIMENT_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TERMINAL_PHASES = {"done", "blocked", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {path}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"at_utc": utc_now(), **event}, sort_keys=True) + "\n")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(base: Path, raw: str | None, default: Path | None = None) -> Path:
    if not raw:
        if default is None:
            raise SystemExit("Missing required path")
        return default.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_experiment(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = Path(path)
    if not raw.is_absolute():
        raw = project_root() / raw
    if raw.is_dir():
        raw = raw / "experiment.json"
    if not raw.exists():
        raise SystemExit(f"Experiment metadata does not exist: {raw}")
    return read_json(raw)


def validate_uids(workflow_uid: str, experiment_uid: str, allow_legacy: bool) -> None:
    if not WORKFLOW_UID_RE.fullmatch(workflow_uid):
        raise SystemExit(f"Workflow uid must be <aspect>_<two_letter_code>: {workflow_uid}")
    if workflow_uid != WORKFLOW_UID:
        raise SystemExit(f"This workflow implements uid {WORKFLOW_UID}; got {workflow_uid}")
    if allow_legacy:
        return
    marker = f"-{workflow_uid}-"
    if marker not in experiment_uid:
        raise SystemExit(
            "Experiment uid must be <task_code>-<workflow_uid>-<0000>; "
            f"got {experiment_uid}"
        )
    prefix, sequence = experiment_uid.rsplit(marker, 1)
    if not EXPERIMENT_PREFIX_RE.fullmatch(prefix):
        raise SystemExit(f"Invalid experiment uid prefix: {prefix!r}")
    if not re.fullmatch(r"[0-9]{4}", sequence):
        raise SystemExit(f"Experiment uid must end with a four-digit sequence: {experiment_uid}")


def initial_state(
    args: argparse.Namespace,
    meta: dict[str, Any],
    worktree: Path,
    goal_dir: Path,
) -> dict[str, Any]:
    experiment_uid = args.experiment_uid or str(meta.get("experiment_id") or "")
    if not experiment_uid:
        raise SystemExit("--experiment-uid is required when metadata has no experiment_id")
    validate_uids(args.workflow_uid, experiment_uid, args.allow_legacy_experiment_id)
    return {
        "schema_version": 1,
        "workflow_uid": args.workflow_uid,
        "experiment_uid": experiment_uid,
        "phase": "baseline_run",
        "status": "running",
        "cycles": 0,
        "max_cycles": args.max_cycles,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "worktree": str(worktree),
        "goal_dir": str(goal_dir),
        "last_node": "",
        "last_outcome": "",
        "resume_target": "",
        "resume_reason": "",
        "resume_followups": [],
        "artifacts": [],
    }


def load_or_create_state(
    args: argparse.Namespace,
    meta: dict[str, Any],
    worktree: Path,
    goal_dir: Path,
    state_path: Path,
) -> dict[str, Any]:
    existing = read_json(state_path)
    if existing and not args.reset:
        return existing
    return initial_state(args, meta, worktree, goal_dir)


def make_context(
    args: argparse.Namespace,
    state: dict[str, Any],
    root: Path,
    worktree: Path,
    goal_dir: Path,
    state_path: Path,
    event_log: Path,
) -> NodeContext:
    config = {
        "workflow_uid": args.workflow_uid,
        "experiment_uid": state["experiment_uid"],
        "model_id": args.model_id,
        "kernel_delivery": args.kernel_delivery,
        "pisa_delivery": args.pisa_delivery,
        "cache_delivery": args.cache_delivery,
        "autorun_model": args.autorun_model,
        "autorun_poll_sec": args.autorun_poll_sec,
        "visual_review_timeout_sec": args.visual_review_timeout_sec,
        "baseline_frames": args.baseline_frames,
        "baseline_manifest": getattr(args, "baseline_manifest", "candidates/sana_video_baseline.toml"),
        "baseline_poll_sec": getattr(args, "baseline_poll_sec", 30.0),
        "baseline_startup_timeout_sec": float(os.environ.get("BASELINE_STARTUP_TIMEOUT_SEC", "900")),
        "baseline_max_infra_retries": int(os.environ.get("BASELINE_MAX_INFRA_RETRIES", "2")),
    }
    return NodeContext(
        root=root,
        workflow_dir=WORKFLOW_DIR,
        worktree=worktree,
        goal_dir=goal_dir,
        state_path=state_path,
        event_log=event_log,
        state=state,
        config=config,
        env=os.environ.copy(),
        dry_run=args.dry_run,
    )


def run_node(phase: str, ctx: NodeContext) -> NodeResult:
    if phase == "baseline_run":
        return run_baseline(ctx)
    if phase == "baseline_gate":
        return run_baseline_gate(ctx)
    if phase == "check_sources":
        return run_source_gate(ctx)
    if phase == "executor":
        return run_executor(ctx)
    if phase == "check_integration":
        return run_integration_gate(ctx)
    if phase == "visual_review":
        return run_visual_reviewer(ctx)
    if phase == "check_final":
        return run_final_gate(ctx)
    if phase == "delivery_gate":
        return run_delivery_gate(ctx)
    if phase == "write_resume":
        return run_resume_prompt(ctx)
    raise SystemExit(f"Unknown workflow phase: {phase}")


def apply_result(state: dict[str, Any], phase: str, result: NodeResult) -> None:
    if result.outcome != "waiting":
        state["cycles"] = int(state.get("cycles") or 0) + 1
    state["updated_at_utc"] = utc_now()
    state["last_node"] = phase
    state["last_outcome"] = result.outcome
    state.setdefault("artifacts", [])
    for artifact in result.artifacts:
        if artifact not in state["artifacts"]:
            state["artifacts"].append(artifact)
    state.update(result.updates)


def set_resume(state: dict[str, Any], reason: str, followups: list[str]) -> str:
    state["resume_target"] = "executor"
    state["resume_reason"] = reason
    state["resume_followups"] = followups
    return "write_resume"


def transition(state: dict[str, Any], phase: str, result: NodeResult) -> str:
    if result.outcome in {"failed", "node_error"}:
        state["status"] = "failed"
        state["terminal_reason"] = result.message or result.outcome
        return "failed"

    if phase == "baseline_run":
        if result.outcome == "completed":
            return "baseline_gate"
        if result.outcome == "waiting":
            return "baseline_run"
        state["status"] = "blocked"
        state["terminal_reason"] = result.message or "baseline_run_blocked"
        return "blocked"

    if phase == "baseline_gate":
        if result.outcome == "ready":
            return "check_sources"
        state["status"] = "blocked"
        state["terminal_reason"] = result.message or "baseline_gate_failed"
        return "blocked"

    if phase == "check_sources":
        if result.outcome == "ready":
            return "executor"
        state["status"] = "blocked"
        state["terminal_reason"] = result.message or "source_delivery_contract_blocked"
        return "blocked"

    if phase == "executor":
        return "check_integration" if result.outcome == "exited" else "failed"

    if phase == "check_integration":
        if result.outcome == "ready":
            return "visual_review"
        if str(state.get("executor_status") or "") == "blocked":
            state["status"] = "blocked"
            state["terminal_reason"] = str(
                state.get("executor_terminal_reason") or result.message or "executor_blocked"
            )
            return "blocked"
        return set_resume(
            state,
            str(state.get("integration_reason") or result.message or result.outcome),
            [
                "Repair the integration source lock, file-level provenance, or composition matrix without mutating any donor experiment.",
                "Keep kernel, PISA, and cache implementations materialized and independently toggleable; runtime enablement is recipe-specific.",
                "Measure all eight toggle combinations and run conservative, balanced, and aggressive recipes under the warm-sample timing contract.",
                "Exclude process/load/compile/warmup/video-write time from every reported speedup and preserve stage timings and integration_stats.json.",
                "Treat Slurm, filesystem, collection, and evaluator failures as retryable infrastructure, not component evidence.",
            ],
        )

    if phase == "visual_review":
        if result.outcome == "reviewed":
            return "delivery_gate"
        return set_resume(
            state,
            result.message or result.outcome,
            [
                "Repair the completed recipe frame/video artifacts required by the blind Codex reviewer.",
                "Preserve the selected recipe and its declared quality tier while repairing evaluation infrastructure.",
                "Treat image attachment, LPIPS, frame extraction, malformed verdict, or reviewer launch failure as infrastructure.",
            ],
        )

    if phase == "delivery_gate":
        if result.outcome == "published":
            state["status"] = "done"
            state["terminal_reason"] = "integrated_frontier_delivery_published"
            return "done"
        return set_resume(
            state,
            result.message or "integrator_delivery_invalid",
            [
                "Repair DELIVERY-DRAFT.json against the unified frontier contract; do not edit DELIVERY.json directly.",
                "Expose conservative, balanced, and aggressive as three distinct, measured integrated points.",
                "Use this experiment's immutable baseline for every absolute time and speedup.",
            ],
        )

    if phase == "write_resume":
        state["resume_target"] = "executor"
        return "executor"

    state["status"] = "failed"
    state["terminal_reason"] = f"no_transition_for_{phase}_{result.outcome}"
    return "failed"


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    meta = load_experiment(args.experiment_json)
    worktree = resolve_path(root, args.worktree, Path(meta.get("worktree") or root))
    goal_default = Path(meta["goal_dir"]) if meta.get("goal_dir") else worktree / "goals" / WORKFLOW_UID
    goal_dir = resolve_path(worktree, args.goal_dir, goal_default)
    state_path = resolve_path(
        worktree,
        args.state_file,
        worktree / "state" / "workflow-integrator_ia-state.json",
    )
    event_log = resolve_path(
        worktree,
        args.event_log,
        worktree / "state" / "workflow-integrator_ia-events.jsonl",
    )
    state = load_or_create_state(args, meta, worktree, goal_dir, state_path)

    if getattr(args, "reset_baseline", False):
        for path in (worktree / "BASELINE-LOCK.json", worktree / "state" / "baseline-run.json"):
            path.unlink(missing_ok=True)
        state = initial_state(args, meta, worktree, goal_dir)

    if args.command == "status":
        return {"state": state, "state_file": str(state_path), "event_log": str(event_log)}

    if args.reset and event_log.exists():
        event_log.unlink()
    if args.reset or not state_path.exists():
        write_json(state_path, state)

    while state.get("phase") not in TERMINAL_PHASES:
        if int(state.get("cycles") or 0) >= int(state.get("max_cycles") or args.max_cycles):
            state["phase"] = "blocked"
            state["status"] = "blocked"
            state["terminal_reason"] = "workflow_max_cycles_reached"
            write_json(state_path, state)
            append_event(event_log, {"phase": "blocked", "outcome": "workflow_max_cycles_reached"})
            break

        phase = str(state.get("phase") or "baseline_run")
        ctx = make_context(args, state, root, worktree, goal_dir, state_path, event_log)
        append_event(event_log, {"phase": phase, "event": "node_start"})
        result = run_node(phase, ctx)
        apply_result(state, phase, result)
        next_phase = transition(state, phase, result)
        append_event(
            event_log,
            {
                "phase": phase,
                "outcome": result.outcome,
                "next_phase": next_phase,
                "message": result.message,
                "artifacts": result.artifacts,
            },
        )
        state["phase"] = next_phase
        write_json(state_path, state)
        if args.once:
            break

    return {"state": state, "state_file": str(state_path), "event_log": str(event_log)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--experiment-json", help="Path to experiment.json or experiment directory")
    parser.add_argument("--experiment-uid", help="UID like sana-integrator_ia-0001")
    parser.add_argument("--workflow-uid", default=WORKFLOW_UID)
    parser.add_argument("--allow-legacy-experiment-id", action="store_true")
    parser.add_argument("--worktree")
    parser.add_argument("--goal-dir")
    parser.add_argument("--state-file")
    parser.add_argument("--event-log")
    parser.add_argument("--max-cycles", type=int, default=100)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-id", default="sana_video")
    parser.add_argument("--kernel-delivery", default="")
    parser.add_argument("--pisa-delivery", default="")
    parser.add_argument("--cache-delivery", default="")
    parser.add_argument("--autorun-model", default=os.environ.get("CODEX_AUTORUN_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--autorun-poll-sec", type=float, default=5.0)
    parser.add_argument("--visual-review-timeout-sec", type=int, default=1800)
    parser.add_argument("--baseline-frames", default=os.environ.get("CANONICAL_BASELINE_FRAMES", ""))
    parser.add_argument("--baseline-manifest", default="candidates/sana_video_baseline.toml")
    parser.add_argument("--baseline-poll-sec", type=float, default=30.0)
    parser.add_argument("--reset-baseline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_loop(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
