#!/usr/bin/env python3
"""Workflow-local reviewer node."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from nodes.managed_codex import run_managed_codex
from workflow_types import NodeContext, NodeResult


REVIEWER_STATUS = "REVIEWER-STATUS.json"
RESUME_FILE = "STOP_HOOK_RESUME.md"


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


def reviewer_goal_dir(ctx: NodeContext) -> Path:
    return Path(str(ctx.state.get("reviewer_goal_dir") or ctx.goal_dir.parent / f"{ctx.goal_dir.name}-reviewer"))


def status_result(ctx: NodeContext, status: dict[str, Any]) -> NodeResult | None:
    if status.get("status") == "accepted" and status.get("decision") == "accept":
        return NodeResult(
            "accepted",
            updates={
                "reviewer_status": status,
                "reviewer_reason": status.get("reason") or "reviewer_accepted",
            },
            artifacts=[REVIEWER_STATUS],
            message=status.get("reason") or "reviewer_accepted",
        )
    if status.get("status") == "needs_executor_resume":
        followups = status.get("required_followups") or []
        return NodeResult(
            "needs_executor_resume",
            updates={
                "reviewer_status": status,
                "reviewer_reason": status.get("reason") or "reviewer_requested_executor_resume",
                "reviewer_followups": followups,
            },
            artifacts=[REVIEWER_STATUS],
            message=status.get("reason") or "reviewer_requested_executor_resume",
        )
    return None


def create_reviewer_goal(ctx: NodeContext, goal_dir: Path) -> None:
    if (goal_dir / "goal.md").exists() and (goal_dir / "context.json").exists():
        return
    if ctx.dry_run:
        return
    context = read_json(ctx.goal_dir / "context.json")
    review_id = goal_dir.name
    target_id = ctx.goal_dir.name
    candidate = context.get("candidate_manifest") or "candidates/hunyuan_diffusers_baseline.toml"
    objective = (
        f"Review executor goal {target_id}: inspect implementation diff, run artifacts, "
        "microbench/full-evaluation evidence, and remaining kernel/module optimization space. "
        "Only accept when the executor has a smooth full evaluation and no credible high-value "
        "local optimization remains; otherwise request executor resume."
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
        str(context.get("dimension") or "kwl_fusion"),
        "--role",
        "gate",
        "--model-id",
        str(context.get("model_id") or "hunyuan_diffusers"),
        "--run-id",
        str(context.get("run_id") or ctx.state.get("experiment_uid") or ""),
        "--root-branch",
        str(context.get("root_branch") or ""),
        "--submodule-branch",
        str(context.get("submodule_branch") or ""),
        "--goals-root",
        rel_to(ctx.worktree, ctx.goal_dir.parent),
        "--overwrite",
    ]
    proc = subprocess.run(cmd, cwd=ctx.worktree, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "prepare reviewer goal failed")

    extra = f"""

## Reviewer Workflow Contract

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

If credible module/kernel optimization remains, or if evaluation is incomplete
or buggy, write `"status": "needs_executor_resume"` and put concrete follow-ups
in `required_followups`. Do not modify implementation code as reviewer.
"""
    goal_md = goal_dir / "goal.md"
    goal_md.write_text(goal_md.read_text() + extra)
    reviewer_context = read_json(goal_dir / "context.json")
    reviewer_context.update(
        {
            "review_target_goal_id": target_id,
            "review_target_goal_dir": rel_to(ctx.worktree, ctx.goal_dir),
            "reviewer_status_file": REVIEWER_STATUS,
        }
    )
    write_json(goal_dir / "context.json", reviewer_context)


def build_prompt(goal_dir: Path) -> str:
    prompt = (goal_dir / "goal.md").read_text() if (goal_dir / "goal.md").exists() else ""
    resume = goal_dir / RESUME_FILE
    if resume.exists():
        prompt += "\n\n## Workflow Resume\n\n" + resume.read_text()
        resume.replace(goal_dir / "STOP_HOOK_RESUME.last.md")
    return prompt


def run_codex_reviewer(ctx: NodeContext, goal_dir: Path) -> tuple[int, str, str, Path]:
    goal_dir.mkdir(parents=True, exist_ok=True)
    out = goal_dir / "agent_last.md"
    if ctx.dry_run:
        out.write_text("[dry-run] reviewer node skipped\n")
        return 0, "", "", out
    prompt = build_prompt(goal_dir)
    result = run_managed_codex(
        ctx,
        source_goal_dir=goal_dir,
        node_name="reviewer",
        role="gate",
        prompt=prompt,
        required_artifacts=[REVIEWER_STATUS],
        output_path=out,
    )
    return result.returncode, result.stdout_tail[-2000:], result.stderr_tail[-2000:], out


def run(ctx: NodeContext) -> NodeResult:
    status_path = ctx.worktree / REVIEWER_STATUS
    existing = status_result(ctx, read_json(status_path))
    if existing:
        return existing

    goal_dir = reviewer_goal_dir(ctx)
    try:
        create_reviewer_goal(ctx, goal_dir)
    except RuntimeError as exc:
        return NodeResult("failed", updates={"reviewer_error": str(exc)}, message=str(exc))

    code, stdout_tail, stderr_tail, out = run_codex_reviewer(ctx, goal_dir)
    status = read_json(status_path)
    result = status_result(ctx, status)
    if result:
        result.updates.update(
            {
                "reviewer_exit_code": code,
                "reviewer_output": rel_to(ctx.worktree, out),
                "reviewer_stdout_tail": stdout_tail,
                "reviewer_stderr_tail": stderr_tail,
            }
        )
        result.artifacts.append(rel_to(ctx.worktree, out))
        return result

    return NodeResult(
        "invalid_status",
        updates={
            "reviewer_exit_code": code,
            "reviewer_output": rel_to(ctx.worktree, out),
            "reviewer_stdout_tail": stdout_tail,
            "reviewer_stderr_tail": stderr_tail,
            "reviewer_status": status,
            "reviewer_reason": "missing_or_invalid_reviewer_status",
        },
        artifacts=[rel_to(ctx.worktree, out)] if out.exists() else [],
        message="missing_or_invalid_reviewer_status",
    )
