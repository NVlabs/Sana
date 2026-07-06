#!/usr/bin/env python3
"""Workflow-local resume prompt node."""

from __future__ import annotations

from pathlib import Path

from workflow_types import NodeContext, NodeResult


def target_goal_dir(ctx: NodeContext) -> Path:
    target = str(ctx.state.get("resume_target") or "executor")
    if target == "reviewer":
        return Path(str(ctx.state.get("reviewer_goal_dir") or ctx.goal_dir.parent / f"{ctx.goal_dir.name}-reviewer"))
    return ctx.goal_dir


def run(ctx: NodeContext) -> NodeResult:
    target = str(ctx.state.get("resume_target") or "executor")
    goal_dir = target_goal_dir(ctx)
    goal_dir.mkdir(parents=True, exist_ok=True)
    reason = str(ctx.state.get("resume_reason") or "resume_required")
    followups = ctx.state.get("resume_followups") or []
    lines = [
        f"# {'Reviewer' if target == 'reviewer' else 'Executor'} Resume Required",
        "",
        f"Workflow requested `{target}` resume.",
        "",
        f"Reason: `{reason}`.",
        "",
        "Required follow-ups:",
    ]
    if followups:
        lines.extend(f"- {item}" for item in followups)
    else:
        lines.append("- Inspect workflow state and produce the required durable artifact before exiting again.")
    lines.extend(
        [
            "",
            "Do not treat this resume prompt as completion. Exit only after the required artifacts or status files exist.",
            "",
        ]
    )
    path = goal_dir / "STOP_HOOK_RESUME.md"
    path.write_text("\n".join(lines))
    return NodeResult(
        "written",
        updates={"resume_file": ctx.rel_to_worktree(path), "resume_target": target},
        artifacts=[ctx.rel_to_worktree(path)],
        message=f"wrote resume prompt for {target}",
    )
