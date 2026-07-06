#!/usr/bin/env python3
"""Workflow-local types for the KR kernel retention workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NodeContext:
    root: Path
    workflow_dir: Path
    worktree: Path
    goal_dir: Path
    state_path: Path
    event_log: Path
    state: dict[str, Any]
    config: dict[str, Any]
    env: dict[str, str]
    dry_run: bool = False

    def rel_to_worktree(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.worktree))
        except ValueError:
            return str(path)


@dataclass
class NodeResult:
    outcome: str
    updates: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    message: str = ""
