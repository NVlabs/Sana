#!/usr/bin/env python3
"""Workflow-local Codex executor for three-component integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nodes.managed_codex import run_managed_codex
from workflow_types import NodeContext, NodeResult


RESUME_FILE = "STOP_HOOK_RESUME.md"


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def callable_contracts() -> str:
    base = Path(__file__).resolve().parents[1] / "callable"
    sections: list[str] = []
    for node_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        interface = read_text(node_dir / "interface.toml").strip()
        prompt = read_text(node_dir / "prompt.md").strip()
        body: list[str] = []
        if interface:
            body.extend(["Interface:", "", "```toml", interface, "```"])
        if prompt:
            body.extend(["Usage:", "", prompt])
        if body:
            sections.extend([f"### {node_dir.name}", "", *body, ""])
    return "## Callable Node Contracts\n\n" + "\n".join(sections).strip() if sections else ""


def build_prompt(ctx: NodeContext) -> str:
    parts = [
        read_text(ctx.goal_dir / "goal.md"),
        read_text(Path(__file__).with_name("system_prompt.md")),
        read_text(Path(__file__).with_name("delivery_contract.md")),
        read_text(Path(__file__).with_name("composition_policy.md")),
        read_text(Path(__file__).with_name("callable_nodes.md")),
        callable_contracts(),
    ]
    inventory_path = ctx.worktree / "state" / "integration-source-inventory.json"
    if inventory_path.exists():
        parts.extend(
            [
                "## Pinned Source Inventory",
                "",
                "```json",
                inventory_path.read_text().strip(),
                "```",
            ]
        )
    resume = ctx.goal_dir / RESUME_FILE
    if resume.exists():
        parts.extend(["## Workflow Resume", "", resume.read_text().strip()])
        previous = ctx.goal_dir / "STOP_HOOK_RESUME.last.md"
        if previous.exists():
            previous.unlink()
        resume.replace(previous)
    return "\n\n".join(part for part in parts if part.strip()) + "\n"


def run(ctx: NodeContext) -> NodeResult:
    out = ctx.goal_dir / "agent_last.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if ctx.dry_run:
        out.write_text("[dry-run] managed integration executor skipped\n")
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
        prompt=build_prompt(ctx),
        required_artifacts=[
            "INTEGRATION-SOURCES.lock.json",
            "INTEGRATION-STATUS.json",
            "COMPOSITION-MATRIX.json",
            "INTEGRATION-SUMMARY.md",
        ],
        output_path=out,
    )
    integration_status = read_json(ctx.worktree / "INTEGRATION-STATUS.json")
    message = "managed integration executor completed"
    if result.returncode != 0:
        message = result.stderr_tail or result.stdout_tail or "managed integration executor failed"
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
            "executor_status": integration_status.get("status", "missing"),
            "executor_terminal_reason": integration_status.get("terminal_reason", ""),
        },
        artifacts=[ctx.rel_to_worktree(out)],
        message=message,
    )
