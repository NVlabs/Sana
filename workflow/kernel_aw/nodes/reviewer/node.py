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
SYSTEM_PROMPT_FILE = "system_prompt.md"


def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


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


def executor_invocation_id(ctx: NodeContext) -> str:
    completion = ctx.state.get("executor_completion")
    if not isinstance(completion, dict):
        return ""
    return str(completion.get("invocation_id") or "")


def status_matches_executor(ctx: NodeContext, status: dict[str, Any]) -> bool:
    expected = executor_invocation_id(ctx)
    if not expected:
        return True
    return str(status.get("reviewed_executor_invocation_id") or "") == expected


def archive_stale_status(path: Path, invocation_id: str) -> Path:
    suffix = invocation_id[:12] if invocation_id else "unknown"
    target = path.with_name(f"REVIEWER-STATUS.stale-{suffix}.json")
    index = 1
    while target.exists():
        target = path.with_name(f"REVIEWER-STATUS.stale-{suffix}-{index}.json")
        index += 1
    path.replace(target)
    return target


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
    if status.get("status") == "discarded" and status.get("decision") == "discard":
        return NodeResult(
            "discarded",
            updates={
                "reviewer_status": status,
                "reviewer_reason": status.get("reason") or "reviewer_discarded",
            },
            artifacts=[REVIEWER_STATUS],
            message=status.get("reason") or "reviewer_discarded",
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
    candidate = context.get("candidate_manifest") or "candidates/sana_video_baseline.toml"
    objective = (
        f"Review executor goal {target_id}: inspect implementation diff, run artifacts, "
        "single-DiT/module evaluation evidence, infra reliability, numerical drift semantics, "
        "and remaining kernel/module optimization space. Only this reviewer may request exit "
        "or discard. Full diffusion/Gemini validation is a terminal workflow gate after the "
        "reviewer requests exit; otherwise request executor resume."
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
        str(context.get("dimension") or "kernel_fusion"),
        "--role",
        "gate",
        "--model-id",
        str(context.get("model_id") or "sana_video"),
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

    reviewer_context = read_json(goal_dir / "context.json")
    reviewer_context.update(
        {
            "review_target_goal_id": target_id,
            "review_target_goal_dir": rel_to(ctx.worktree, ctx.goal_dir),
            "reviewer_status_file": REVIEWER_STATUS,
        }
    )
    write_json(goal_dir / "context.json", reviewer_context)


def build_prompt(goal_dir: Path, ctx: NodeContext | None = None) -> str:
    sections: list[str] = []
    system_prompt = read_text(Path(__file__).with_name(SYSTEM_PROMPT_FILE)).strip()
    if system_prompt:
        sections.append(system_prompt)

    assignment = read_text(goal_dir / "goal.md").strip()
    if assignment:
        sections.extend(["## Current Review Assignment", assignment])

    if ctx is not None:
        invocation_id = executor_invocation_id(ctx)
        target = ctx.goal_dir.name
        review_id = goal_dir.name
        sections.append(
            "## Current Review Target\n\n"
            f"- reviewer goal id: `{review_id}`\n"
            f"- target executor goal id: `{target}`\n"
            f"- executor invocation id: `{invocation_id or '<missing>'}`\n\n"
            f"Write these exact identifiers to `{REVIEWER_STATUS}`. "
            "An older status file or invocation id is stale."
        )

    resume = goal_dir / RESUME_FILE
    if resume.exists():
        sections.extend(["## Workflow Resume", resume.read_text().strip()])
        resume.replace(goal_dir / "STOP_HOOK_RESUME.last.md")
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def run_codex_reviewer(ctx: NodeContext, goal_dir: Path) -> tuple[int, str, str, Path]:
    goal_dir.mkdir(parents=True, exist_ok=True)
    out = goal_dir / "agent_last.md"
    if ctx.dry_run:
        out.write_text("[dry-run] reviewer node skipped\n")
        return 0, "", "", out
    prompt = build_prompt(goal_dir, ctx)
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
    stale_status = ""
    existing_status = read_json(status_path)
    existing = status_result(ctx, existing_status)
    if existing and status_matches_executor(ctx, existing_status):
        return existing
    if existing and status_path.exists():
        stale_status = rel_to(
            ctx.worktree,
            archive_stale_status(status_path, executor_invocation_id(ctx)),
        )

    goal_dir = reviewer_goal_dir(ctx)
    try:
        create_reviewer_goal(ctx, goal_dir)
    except RuntimeError as exc:
        return NodeResult("failed", updates={"reviewer_error": str(exc)}, message=str(exc))

    code, stdout_tail, stderr_tail, out = run_codex_reviewer(ctx, goal_dir)
    status = read_json(status_path)
    result = status_result(ctx, status)
    if result and status_matches_executor(ctx, status):
        result.updates.update(
            {
                "reviewer_exit_code": code,
                "reviewer_output": rel_to(ctx.worktree, out),
                "reviewer_stdout_tail": stdout_tail,
                "reviewer_stderr_tail": stderr_tail,
                "archived_stale_reviewer_status": stale_status,
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
            "reviewer_reason": "missing_invalid_or_stale_reviewer_status",
            "archived_stale_reviewer_status": stale_status,
        },
        artifacts=[rel_to(ctx.worktree, out)] if out.exists() else [],
        message="missing_invalid_or_stale_reviewer_status",
    )
