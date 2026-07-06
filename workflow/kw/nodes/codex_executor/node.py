#!/usr/bin/env python3
"""Workflow-local Codex executor node."""

from __future__ import annotations

from pathlib import Path

from nodes.managed_codex import run_managed_codex
from workflow_types import NodeContext, NodeResult


RESUME_FILE = "STOP_HOOK_RESUME.md"


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def callable_contracts() -> str:
    base = Path(__file__).resolve().parents[1] / "callable"
    sections: list[str] = []
    for node_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        interface = read_text(node_dir / "interface.toml").strip()
        prompt = read_text(node_dir / "prompt.md").strip()
        body = []
        if interface:
            body.extend(["Interface:", "", "```toml", interface, "```"])
        if prompt:
            body.extend(["Usage:", "", prompt])
        if body:
            sections.extend([f"### {node_dir.name}", "", *body, ""])
    if not sections:
        return ""
    return "## Callable Node Contracts\n\n" + "\n".join(sections).strip()


def build_prompt(ctx: NodeContext) -> str:
    goal = read_text(ctx.goal_dir / "goal.md")
    callable_nodes = read_text(Path(__file__).with_name("callable_nodes.md"))
    contracts = callable_contracts()
    prompt = goal
    if callable_nodes:
        prompt += "\n\n" + callable_nodes
    if contracts:
        prompt += "\n\n" + contracts
    resume = ctx.goal_dir / RESUME_FILE
    if resume.exists():
        prompt += "\n\n## Workflow Resume\n\n" + resume.read_text()
        resume.replace(ctx.goal_dir / "STOP_HOOK_RESUME.last.md")
    return prompt


def run(ctx: NodeContext) -> NodeResult:
    out = ctx.goal_dir / "agent_last.md"
    prompt = build_prompt(ctx)
    if ctx.dry_run:
        out.write_text("[dry-run] managed executor node skipped\n")
        return NodeResult(
            "exited",
            updates={"codex_exit_code": 0, "executor_output": ctx.rel_to_worktree(out)},
            artifacts=[ctx.rel_to_worktree(out)],
            message="dry_run",
        )

    result = run_managed_codex(
        ctx,
        source_goal_dir=ctx.goal_dir,
        node_name="executor",
        role="implementation",
        prompt=prompt,
        required_artifacts=["AGENT-STATUS.json"],
        output_path=out,
    )
    message = "managed Codex executor completed"
    if result.returncode != 0:
        message = result.stderr_tail or result.stdout_tail or "managed Codex executor failed"
    return NodeResult(
        "exited",
        updates={
            "codex_exit_code": result.returncode,
            "codex_executor": "autorun_tui",
            "codex_session": result.session,
            "executor_output": ctx.rel_to_worktree(out),
            "executor_stdout_tail": result.stdout_tail[-2000:],
            "executor_stderr_tail": result.stderr_tail[-2000:],
            "executor_completion": result.completion or {},
        },
        artifacts=[ctx.rel_to_worktree(out)],
        message=message,
    )
