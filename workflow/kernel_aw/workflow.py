#!/usr/bin/env python3
"""Direct executor/eval/reviewer loop for the Sana kernel workflow.

This is intentionally not a generic workflow language. It is the first-stage
centralized state machine for workflow uid `kernel_aw`; all node implementations
are owned by this workflow under `workflow/kernel_aw/nodes/`.
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
from nodes.delivery_gate.node import run as run_delivery_gate  # noqa: E402
from nodes.eval_gate.node import run as run_eval_gate  # noqa: E402
from nodes.final_full_eval.node import run as run_final_full_eval  # noqa: E402
from nodes.resume_prompt.node import run as run_resume_prompt  # noqa: E402
from nodes.reviewer.node import run as run_reviewer  # noqa: E402


WORKFLOW_UID = "kernel_aw"
WORKFLOW_UID_RE = re.compile(r"^([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*)_([A-Za-z]{2})$")
EXPERIMENT_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TERMINAL_PHASES = {"done", "blocked", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"at_utc": utc_now(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


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


def initial_state(args: argparse.Namespace, meta: dict[str, Any], worktree: Path, goal_dir: Path) -> dict[str, Any]:
    workflow_uid = args.workflow_uid
    experiment_uid = args.experiment_uid or str(meta.get("experiment_id") or "")
    if not experiment_uid:
        raise SystemExit("--experiment-uid is required when --experiment-json has no experiment_id")
    validate_uids(workflow_uid, experiment_uid, args.allow_legacy_experiment_id)
    return {
        "schema_version": 1,
        "workflow_uid": workflow_uid,
        "experiment_uid": experiment_uid,
        "phase": "baseline_run",
        "status": "running",
        "cycles": 0,
        "node_cycles": 0,
        "max_cycles": args.max_cycles,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "worktree": str(worktree),
        "goal_dir": str(goal_dir),
        "reviewer_goal_dir": str(goal_dir.parent / f"{goal_dir.name}-reviewer"),
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
        "autorun_model": getattr(args, "autorun_model", os.environ.get("CODEX_AUTORUN_MODEL", "gpt-5.6-sol")),
        "autorun_poll_sec": getattr(args, "autorun_poll_sec", 5.0),
        "assess_timeout_sec": args.assess_timeout_sec,
        "baseline_frames": args.baseline_frames,
        "baseline_manifest": getattr(args, "baseline_manifest", "candidates/sana_video_baseline.toml"),
        "baseline_poll_sec": getattr(args, "baseline_poll_sec", 30.0),
        "baseline_startup_timeout_sec": float(os.environ.get("BASELINE_STARTUP_TIMEOUT_SEC", "900")),
        "baseline_max_infra_retries": int(os.environ.get("BASELINE_MAX_INFRA_RETRIES", "2")),
        "model_id": args.model_id,
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
    if phase == "executor":
        return run_executor(ctx)
    if phase == "check_eval":
        return run_eval_gate(ctx)
    if phase == "final_full_eval":
        return run_final_full_eval(ctx)
    if phase == "delivery_gate":
        return run_delivery_gate(ctx)
    if phase == "write_resume":
        return run_resume_prompt(ctx)
    if phase == "reviewer":
        return run_reviewer(ctx)
    raise SystemExit(f"Unknown workflow phase: {phase}")


def apply_result(state: dict[str, Any], phase: str, result: NodeResult) -> None:
    if result.outcome != "waiting":
        state["cycles"] = int(state.get("cycles") or 0) + 1
    state["node_cycles"] = state["cycles"]
    state["updated_at_utc"] = utc_now()
    state["last_node"] = phase
    state["last_outcome"] = result.outcome
    state.setdefault("artifacts", [])
    for artifact in result.artifacts:
        if artifact not in state["artifacts"]:
            state["artifacts"].append(artifact)
    state.update(result.updates)


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
            return "executor"
        state["status"] = "blocked"
        state["terminal_reason"] = result.message or "baseline_gate_failed"
        return "blocked"

    if phase == "executor":
        return "check_eval" if result.outcome == "exited" else "failed"

    if phase == "check_eval":
        if result.outcome == "smooth":
            state["resume_target"] = ""
            state["resume_reason"] = ""
            state["resume_followups"] = []
            return "reviewer"
        state["resume_target"] = "executor"
        state["resume_reason"] = state.get("eval_reason") or result.message or result.outcome
        state["resume_followups"] = [
            "Produce, repair, or retry the current candidate's gate before exiting again; do not reuse an older smooth gate.",
            "Set AGENT-STATUS.json.active_candidate_id and active_gate to the current invocation's authoritative evidence.",
            "Use a registry-resolved full-DiT gate for preflight and composition checkpoints; label synthetic/module evidence screening_only.",
            "Do not launch full diffusion as part of the ordinary executor/eval/reviewer loop.",
            "Treat infra/no-output/cancelled Slurm runs as retryable method-owned failures, not discard evidence.",
            "If AGENT-STATUS.json contains executor-written rejected/discarded records, rewrite them as retained_parked, needs_retry, needs_rewrite, needs_operator_refinement, or needs_reviewer_judgment.",
            "Do not record a final discard from executor; only the reviewer may write a discard decision.",
        ]
        return "write_resume"

    if phase == "write_resume":
        target = str(state.get("resume_target") or "executor")
        return "reviewer" if target == "reviewer" else "executor"

    if phase == "reviewer":
        if result.outcome == "accepted":
            state["status"] = "pending_final_full_eval"
            state["requested_final_decision"] = "accepted"
            state["terminal_reason"] = ""
            return "final_full_eval"
        if result.outcome == "discarded":
            state["status"] = "pending_final_full_eval"
            state["requested_final_decision"] = "discarded_by_reviewer"
            state["terminal_reason"] = ""
            return "final_full_eval"
        if result.outcome == "needs_executor_resume":
            state["resume_target"] = "executor"
            state["resume_reason"] = state.get("reviewer_reason") or "reviewer_requested_executor_resume"
            state["resume_followups"] = state.get("reviewer_followups") or []
            return "write_resume"
        state["resume_target"] = "reviewer"
        state["resume_reason"] = state.get("reviewer_reason") or "missing_invalid_or_stale_reviewer_status"
        state["resume_followups"] = [
            "Write REVIEWER-STATUS.json for the current executor invocation with status=accepted, status=discarded, or status=needs_executor_resume."
        ]
        return "write_resume"

    if phase == "final_full_eval":
        if result.outcome == "passed":
            if state.get("requested_final_decision") == "discarded_by_reviewer":
                state["status"] = "done"
                state["terminal_reason"] = "discarded_by_reviewer_after_terminal_validation"
                state["final_decision"] = "discarded_by_reviewer"
                return "done"
            return "delivery_gate"
        state["status"] = "running"
        state["resume_target"] = "reviewer"
        state["resume_reason"] = state.get("final_eval_reason") or result.message or result.outcome
        state["resume_followups"] = [
            "Inspect the terminal full diffusion/Gemini failure or missing final-run evidence.",
            "If the candidate still merits work, write REVIEWER-STATUS.json with status=needs_executor_resume and concrete executor follow-ups.",
            "The executor may run full_diffusion_eval only for this terminal validation repair, not as ordinary loop evaluation.",
            "If visual quality is blocked by infra such as Gemini credentials or collection failure, request a retry or repair instead of discarding.",
        ]
        return "write_resume"

    if phase == "delivery_gate":
        if result.outcome == "published":
            state["status"] = "done"
            state["terminal_reason"] = "exact_fastest_lossless_kernel_delivery_published"
            state["final_decision"] = "accepted"
            return "done"
        state["status"] = "running"
        state["resume_target"] = "reviewer"
        state["resume_reason"] = result.message or "kernel_delivery_invalid"
        state["resume_followups"] = [
            "Require the executor to repair DELIVERY-DRAFT.json; DELIVERY.json is graph-owned and must not be edited directly.",
            "Kernel must deliver exactly one point named exact_fastest, never conservative/balanced/aggressive variants.",
            "The point must be mathematically lossless and backed by the accepted terminal full evaluation.",
            "Use only the immutable BASELINE-LOCK.json workload and matching timing scope.",
        ]
        return "write_resume"

    return "failed"


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    meta = load_experiment(args.experiment_json)
    worktree = resolve_path(root, args.worktree, Path(meta.get("worktree") or root))
    goal_default = Path(meta["goal_dir"]) if meta.get("goal_dir") else worktree / "goals" / WORKFLOW_UID
    goal_dir = resolve_path(worktree, args.goal_dir, goal_default)
    state_path = resolve_path(worktree, args.state_file, worktree / "state" / "workflow-kernel_aw-state.json")
    event_log = resolve_path(worktree, args.event_log, worktree / "state" / "workflow-kernel_aw-events.jsonl")
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
    parser.add_argument("--experiment-uid", help="UID like sana-kernel_aw-0001")
    parser.add_argument("--workflow-uid", default=WORKFLOW_UID)
    parser.add_argument("--allow-legacy-experiment-id", action="store_true")
    parser.add_argument("--worktree")
    parser.add_argument("--goal-dir")
    parser.add_argument("--state-file")
    parser.add_argument("--event-log")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=400,
        help="Maximum workflow node transitions, not candidate iterations.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--autorun-model", default=os.environ.get("CODEX_AUTORUN_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--autorun-poll-sec", type=float, default=5.0)
    parser.add_argument("--baseline-frames", default=os.environ.get("CANONICAL_BASELINE_FRAMES", ""))
    parser.add_argument("--baseline-manifest", default="candidates/sana_video_baseline.toml")
    parser.add_argument("--baseline-poll-sec", type=float, default=30.0)
    parser.add_argument("--reset-baseline", action="store_true")
    parser.add_argument("--model-id", default="sana_video")
    parser.add_argument("--assess-timeout-sec", type=int, default=1800)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_loop(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
