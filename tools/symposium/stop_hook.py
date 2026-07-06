#!/usr/bin/env python3
"""Lifecycle stop hook for managed goal agents.

The hook runs after a managed Codex exec process exits. It prevents silent
executor exits that skipped a full evaluation, starts a reviewer once an
executor has a smooth full gate, and lets the reviewer either accept the
workflow or wake the executor with concrete follow-up work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTHORITATIVE_GATE_NAMES = {
    "assess_verdict.json",
    "gate_assess.json",
    "verdict.json",
}
INFRA_BLOCKER_HINTS = (
    "baseline_frame_missing",
    "candidate_frame_missing",
    "baseline_frames_missing",
    "candidate_frames_missing",
    "ffmpeg_missing",
    "api_key_missing",
    "missing_api_key",
    "missing_frame",
    "missing_video",
    "missing_benchmark",
)
RESUME_FILE = "STOP_HOOK_RESUME.md"
REVIEWER_STATUS = "REVIEWER-STATUS.json"
LIFECYCLE_STATUS = "STOP-HOOK-STATUS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_context(goal_dir: Path) -> dict[str, Any]:
    return read_json(goal_dir / "context.json")


def stop_hook_dir(root: Path) -> Path:
    return root / ".symposium" / "scratch" / "stop-hook"


def role_for(goal_dir: Path) -> str:
    return str(load_context(goal_dir).get("role") or "implementation")


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
    for collection in ("candidates", "frontier_candidates", "discarded_candidates", "rejected_candidates"):
        for record in status.get(collection, []) or []:
            if not isinstance(record, dict):
                continue
            run_dir = record.get("run_dir") if isinstance(record.get("run_dir"), str) else None
            for raw in record.get("evidence", []) or []:
                if isinstance(raw, str) and Path(raw).name in AUTHORITATIVE_GATE_NAMES:
                    paths.append(resolve_gate_path(root, raw, run_dir))
    return paths


def discover_gate_paths(root: Path) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in [*gate_paths_from_status(root), *root.glob("runs/*/assess_verdict.json")]:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def blocker_is_infra(blocker: Any) -> bool:
    text = json.dumps(blocker, sort_keys=True) if not isinstance(blocker, str) else blocker
    lowered = text.lower()
    return any(hint in lowered for hint in INFRA_BLOCKER_HINTS)


def smooth_gate(path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "gate_missing", {}
    if path.stat().st_size == 0:
        return False, "gate_empty", {}
    data = read_json(path)
    if not data:
        return False, "gate_invalid_json", {}
    required_numeric = ("baseline_total_s", "candidate_total_s", "speedup")
    missing = [key for key in required_numeric if not isinstance(data.get(key), (int, float))]
    if missing:
        return False, "gate_missing_numeric_fields:" + ",".join(missing), data
    blockers = list(data.get("quality_blockers") or [])
    collector_blockers = list(data.get("collector_quality_blockers") or [])
    infra_blockers = [item for item in [*blockers, *collector_blockers] if blocker_is_infra(item)]
    if infra_blockers:
        return False, "gate_has_infrastructure_blockers:" + ",".join(map(str, infra_blockers)), data
    return True, "smooth_gate", data


def latest_runnable_run(root: Path) -> Path | None:
    candidates = []
    for run_dir in sorted((root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        if (run_dir / "assess_verdict.json").exists():
            continue
        if (run_dir / "outputs/benchmark.json").exists() and (
            (run_dir / "outputs/frames").exists() or (run_dir / "outputs/out.mp4").exists()
        ):
            candidates.append(run_dir)
    return candidates[-1] if candidates else None


def baseline_frames_from_context(root: Path, context: dict[str, Any]) -> str:
    loop = context.get("loop_contract") if isinstance(context.get("loop_contract"), dict) else {}
    raw = str(loop.get("canonical_baseline_frames") or "")
    if raw:
        return raw
    model_id = str(context.get("model_id") or "hunyuan_diffusers")
    profile = read_toml_like_baseline_run(root / "models" / f"{model_id}.toml")
    if profile:
        local = root / "runs" / profile / "outputs" / "frames"
        if local.exists():
            return str(local)
        coord = coordinator_root(root) / "runs" / profile / "outputs" / "frames"
        return str(coord)
    return ""


def read_toml_like_baseline_run(path: Path) -> str:
    if not path.exists():
        return ""
    in_baseline = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "[baseline]":
            in_baseline = True
            continue
        if stripped.startswith("[") and in_baseline:
            return ""
        if in_baseline and stripped.startswith("run_id"):
            return stripped.split("=", 1)[1].strip().strip('"')
    return ""


def coordinator_root(root: Path) -> Path:
    parts = root.resolve().parts
    if "output" in parts:
        idx = parts.index("output")
        return Path(*parts[:idx])
    return root


def maybe_run_assess(root: Path, goal_dir: Path, dry_run: bool) -> dict[str, Any]:
    context = load_context(goal_dir)
    loop = context.get("loop_contract") if isinstance(context.get("loop_contract"), dict) else {}
    run_dir = latest_runnable_run(root)
    if not run_dir:
        return {"attempted": False, "reason": "no_completed_run_without_assess"}
    baseline_frames = baseline_frames_from_context(root, context)
    if not baseline_frames or not Path(baseline_frames).exists():
        return {
            "attempted": False,
            "reason": "baseline_frames_missing",
            "run_dir": rel_to(root, run_dir),
            "baseline_frames": baseline_frames,
        }
    model_id = str(context.get("model_id") or "hunyuan_diffusers")
    pybin = str(loop.get("authoritative_python") or sys.executable)
    out = run_dir / "assess_verdict.json"
    cmd = [
        pybin,
        "search/plan_eval.py",
        "--assess",
        rel_to(root, run_dir),
        "--baseline-frames",
        baseline_frames,
        "--model",
        model_id,
        "--out",
        rel_to(root, out),
    ]
    if dry_run:
        return {"attempted": False, "reason": "dry_run", "command": cmd, "run_dir": rel_to(root, run_dir)}
    timeout = int(os.environ.get("SYMPOSIUM_STOP_HOOK_ASSESS_TIMEOUT_SEC", "1800"))
    proc = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "attempted": True,
        "returncode": proc.returncode,
        "run_dir": rel_to(root, run_dir),
        "out": rel_to(root, out),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def write_resume(goal_dir: Path, title: str, body: str) -> Path:
    path = goal_dir / RESUME_FILE
    path.write_text(f"# {title}\n\n{body.strip()}\n")
    return path


def executor_resume_body(reason: str, assess_attempt: dict[str, Any] | None = None) -> str:
    detail = json.dumps(assess_attempt or {}, indent=2, sort_keys=True)
    return f"""
The previous executor turn exited before a smooth full evaluation was available.

Required before exiting again:
- run or finish the full denoise/full candidate validation for the best current candidate;
- run `search/plan_eval.py --assess <run_dir> --baseline-frames <canonical_frames> --model <model_id> --out <run_dir>/assess_verdict.json`;
- fix implementation, launch, frame extraction, baseline-frame, or quality-gate bugs instead of treating them as optimization failure;
- record the candidate with `tools/symposium/loop_control.py record-candidate` or `add-evidence` once `assess_verdict.json` exists;
- only then stop and let the stop hook start the reviewer.

Current stop-hook reason: `{reason}`.

Assess attempt:

```json
{detail}
```
"""


def reviewer_goal_id(executor_goal_dir: Path) -> str:
    return f"{executor_goal_dir.name}-reviewer"


def reviewer_goal_dir(executor_goal_dir: Path) -> Path:
    return executor_goal_dir.parent / reviewer_goal_id(executor_goal_dir)


def create_reviewer_goal(root: Path, executor_goal_dir: Path) -> Path:
    context = load_context(executor_goal_dir)
    target_id = executor_goal_dir.name
    review_id = reviewer_goal_id(executor_goal_dir)
    goal_dir = reviewer_goal_dir(executor_goal_dir)
    candidate = context.get("candidate_manifest") or "candidates/baseline.toml"
    objective = (
        f"Review executor goal {target_id}: inspect implementation diff, run artifacts, "
        "microbench/full-evaluation evidence, and remaining kernel/module optimization space. "
        "Only accept when the executor has a smooth full evaluation and no credible "
        "high-value local optimization remains; otherwise request executor resume."
    )
    cmd = [
        sys.executable,
        "tools/symposium/prepare_goal.py",
        "--goal-id",
        review_id,
        "--candidate",
        str(candidate),
        "--objective",
        objective,
        "--dimension",
        str(context.get("dimension") or "general"),
        "--role",
        "gate",
        "--model-id",
        str(context.get("model_id") or "hunyuan_diffusers"),
        "--run-id",
        str(context.get("run_id") or os.environ.get("SYMPOSIUM_CURRENT_RUN_ID", "")),
        "--root-branch",
        str(context.get("root_branch") or ""),
        "--submodule-branch",
        str(context.get("submodule_branch") or ""),
        "--goals-root",
        rel_to(root, executor_goal_dir.parent),
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    extra = f"""

## Reviewer Stop-Hook Contract

You are the only role allowed to accept this executor workflow. Inspect:

- target executor goal: `{target_id}`
- executor status: `AGENT-STATUS.json`
- executor journal and summary: `SEARCH_JOURNAL.md`, `SUMMARY.md`
- implementation diff and candidate manifests
- microbench artifacts and full `assess_verdict.json`

Write `{REVIEWER_STATUS}` at the repository root before exiting:

```json
{{
  "schema_version": 1,
  "reviewer_goal_id": "{review_id}",
  "target_goal_id": "{target_id}",
  "status": "accepted",
  "decision": "accept",
  "reason": "smooth full evaluation exists and no credible local optimization remains",
  "required_followups": [],
  "evidence": ["runs/<run-id>/assess_verdict.json"]
}}
```

If there is credible remaining module/kernel optimization space, or if the
evaluation is incomplete/buggy, write `"status": "needs_executor_resume"` and
put concrete follow-up instructions in `required_followups`. Do not modify
implementation code as reviewer.
"""
    goal_md = goal_dir / "goal.md"
    goal_md.write_text(goal_md.read_text() + extra)
    reviewer_context = read_json(goal_dir / "context.json")
    reviewer_context.update(
        {
            "review_target_goal_id": target_id,
            "review_target_goal_dir": rel_to(root, executor_goal_dir),
            "reviewer_status_file": REVIEWER_STATUS,
        }
    )
    write_json(goal_dir / "context.json", reviewer_context)
    return goal_dir


def tmux_alive(session: str) -> bool:
    if not session:
        return False
    try:
        return (
            subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            .returncode
            == 0
        )
    except FileNotFoundError:
        return False


def start_goal_session(root: Path, goal_dir: Path, session_name: str, dry_run: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "tools/symposium/codex_goal_session.py",
        "start",
        "--worktree",
        str(root),
        "--name",
        session_name,
        rel_to(root, goal_dir),
    ]
    if dry_run:
        return {"started": False, "reason": "dry_run", "session": session_name, "command": cmd}
    if tmux_alive(session_name):
        return {"started": False, "reason": "session_already_alive", "session": session_name, "command": cmd}
    proc = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "started": proc.returncode == 0,
        "returncode": proc.returncode,
        "session": session_name,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "command": cmd,
    }


def session_names(current_session: str, goal_dir: Path) -> tuple[str, str]:
    if current_session.endswith("-reviewer"):
        return current_session[: -len("-reviewer")], current_session
    executor = current_session or f"autovideo-{goal_dir.name}"
    return executor, f"{executor}-reviewer"


def evaluate_executor(root: Path, goal_dir: Path, dry_run: bool) -> dict[str, Any]:
    gate_results = []
    for path in discover_gate_paths(root):
        ok, reason, data = smooth_gate(path)
        gate_results.append({"path": rel_to(root, path), "ok": ok, "reason": reason, "speedup": data.get("speedup")})
        if ok:
            return {"smooth": True, "gate": gate_results[-1], "all_gates": gate_results}

    assess_attempt = maybe_run_assess(root, goal_dir, dry_run=dry_run)
    if assess_attempt.get("attempted"):
        for path in discover_gate_paths(root):
            ok, reason, data = smooth_gate(path)
            gate_results.append({"path": rel_to(root, path), "ok": ok, "reason": reason, "speedup": data.get("speedup")})
            if ok:
                return {"smooth": True, "gate": gate_results[-1], "all_gates": gate_results, "assess_attempt": assess_attempt}
    reason = gate_results[-1]["reason"] if gate_results else assess_attempt.get("reason", "no_authoritative_gate")
    write_resume(goal_dir, "Executor Resume Required", executor_resume_body(reason, assess_attempt))
    return {"smooth": False, "reason": reason, "all_gates": gate_results, "assess_attempt": assess_attempt}


def after_executor(args: argparse.Namespace, root: Path, goal_dir: Path) -> tuple[int, dict[str, Any]]:
    evaluation = evaluate_executor(root, goal_dir, dry_run=args.dry_run)
    lifecycle_path = stop_hook_dir(root) / f"{goal_dir.name}.json"
    if not evaluation["smooth"]:
        payload = {
            "action": "resume_current_agent",
            "reason": evaluation.get("reason"),
            "evaluation": evaluation,
            "resume_file": rel_to(root, goal_dir / RESUME_FILE),
            "updated_at_utc": utc_now(),
        }
        write_json(lifecycle_path, payload)
        return 10, payload

    reviewer_status = read_json(root / REVIEWER_STATUS)
    reviewer_target = str(reviewer_status.get("target_goal_id") or goal_dir.name)
    if reviewer_status.get("status") == "accepted" and reviewer_target == goal_dir.name:
        payload = {
            "action": "accepted",
            "reason": "reviewer_accepted",
            "evaluation": evaluation,
            "reviewer_status": reviewer_status,
            "updated_at_utc": utc_now(),
        }
        write_json(root / LIFECYCLE_STATUS, payload)
        write_json(lifecycle_path, payload)
        return 0, payload

    review_goal = create_reviewer_goal(root, goal_dir)
    executor_session, reviewer_session = session_names(args.session_name or "", goal_dir)
    start = start_goal_session(root, review_goal, reviewer_session, args.dry_run)
    payload = {
        "action": "start_reviewer",
        "reason": "smooth_evaluation_requires_reviewer_acceptance",
        "evaluation": evaluation,
        "reviewer_goal_dir": rel_to(root, review_goal),
        "reviewer_session": reviewer_session,
        "executor_session": executor_session,
        "start": start,
        "updated_at_utc": utc_now(),
    }
    write_json(lifecycle_path, payload)
    return 0, payload


def reviewer_resume_body(reason: str) -> str:
    return f"""
The previous reviewer turn exited without a valid `{REVIEWER_STATUS}`.

Before exiting again, inspect the executor implementation and artifacts, then
write `{REVIEWER_STATUS}` with either:

- `status="accepted"` and `decision="accept"` if the workflow is accepted;
- `status="needs_executor_resume"` and `decision="resume_executor"` with
  concrete `required_followups` if more executor work is needed.

Current stop-hook reason: `{reason}`.
"""


def after_reviewer(args: argparse.Namespace, root: Path, goal_dir: Path) -> tuple[int, dict[str, Any]]:
    status = read_json(root / REVIEWER_STATUS)
    lifecycle_path = stop_hook_dir(root) / f"{goal_dir.name}.json"
    if status.get("status") == "accepted" and status.get("decision") == "accept":
        payload = {
            "action": "accepted",
            "reason": status.get("reason") or "reviewer_accepted",
            "reviewer_status": status,
            "updated_at_utc": utc_now(),
        }
        write_json(root / LIFECYCLE_STATUS, payload)
        write_json(lifecycle_path, payload)
        return 0, payload

    if status.get("status") == "needs_executor_resume":
        target = str(status.get("target_goal_id") or goal_dir.name.removesuffix("-reviewer"))
        executor_goal = goal_dir.parent / target
        followups = status.get("required_followups") or []
        body = "Reviewer requested executor resume.\n\nRequired follow-ups:\n"
        body += "\n".join(f"- {item}" for item in followups) if followups else "- Reviewer's reason did not include explicit follow-ups."
        body += f"\n\nReviewer reason: {status.get('reason') or 'n/a'}\n"
        write_resume(executor_goal, "Reviewer Requested Executor Resume", body)
        executor_session, _reviewer_session = session_names(args.session_name or "", executor_goal)
        start = start_goal_session(root, executor_goal, executor_session, args.dry_run)
        payload = {
            "action": "start_executor",
            "reason": "reviewer_requested_executor_resume",
            "executor_goal_dir": rel_to(root, executor_goal),
            "executor_session": executor_session,
            "start": start,
            "reviewer_status": status,
            "updated_at_utc": utc_now(),
        }
        write_json(lifecycle_path, payload)
        return 0, payload

    reason = "missing_or_invalid_reviewer_status"
    write_resume(goal_dir, "Reviewer Resume Required", reviewer_resume_body(reason))
    payload = {
        "action": "resume_current_agent",
        "reason": reason,
        "resume_file": rel_to(root, goal_dir / RESUME_FILE),
        "reviewer_status": status,
        "updated_at_utc": utc_now(),
    }
    write_json(lifecycle_path, payload)
    return 10, payload


def cmd_after_agent(args: argparse.Namespace) -> int:
    root = project_root()
    goal_dir = Path(args.goal_dir)
    if not goal_dir.is_absolute():
        goal_dir = root / goal_dir
    goal_dir = goal_dir.resolve()
    if not (goal_dir / "context.json").exists():
        raise SystemExit(f"Goal context does not exist: {goal_dir}")
    role = role_for(goal_dir)
    if role == "gate":
        code, payload = after_reviewer(args, root, goal_dir)
    else:
        code, payload = after_executor(args, root, goal_dir)
    payload.update(
        {
            "role": role,
            "goal_dir": rel_to(root, goal_dir),
            "codex_exit_code": args.codex_exit_code,
            "dry_run": args.dry_run,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    after = sub.add_parser("after-agent", help="Run lifecycle hook after a goal agent exits")
    after.add_argument("goal_dir")
    after.add_argument("--codex-exit-code", type=int, default=0)
    after.add_argument("--session-name", default=os.environ.get("SYMPOSIUM_SESSION_NAME", ""))
    after.add_argument("--dry-run", action="store_true")
    after.set_defaults(func=cmd_after_agent)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
