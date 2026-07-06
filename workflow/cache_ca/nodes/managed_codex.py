#!/usr/bin/env python3
"""Workflow-local bridge from a node to a managed Codex autorun TUI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ManagedCodexResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str
    output_path: Path
    session: str = ""
    completion: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def manager_command(
    ctx: Any,
    action: str,
    runtime_goal: Path,
    session_name: str,
    *extra: str,
) -> list[str]:
    manager = ctx.worktree / "tools/symposium/codex_goal_session.py"
    return [
        sys.executable,
        str(manager),
        action,
        "--worktree",
        str(ctx.worktree),
        "--name",
        session_name,
        rel_to(ctx.worktree, runtime_goal),
        *extra,
    ]


def run_manager(
    ctx: Any,
    env: dict[str, str],
    action: str,
    runtime_goal: Path,
    session_name: str,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        manager_command(ctx, action, runtime_goal, session_name, *extra),
        cwd=ctx.worktree,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def completion_contract(
    ctx: Any,
    invocation_id: str,
    marker: Path,
    required_artifacts: Iterable[str],
) -> str:
    required = "\n".join(f"- `{item}`" for item in required_artifacts)
    marker_rel = rel_to(ctx.worktree, marker)
    return f"""

## Managed Autorun Node Completion

This is invocation `{invocation_id}` in a persistent, managed Codex TUI.
Complete the node's assigned work without starting another Codex session.
Before your final response, ensure these durable artifacts are current:

{required}

As the final filesystem action of this invocation, write `{marker_rel}`:

```json
{{
  "schema_version": 1,
  "invocation_id": "{invocation_id}",
  "status": "complete",
  "completed_at_utc": "<ISO-8601 UTC>",
  "summary": "<brief factual result>"
}}
```

Do not write the marker while a command or Slurm job owned by this node is
still running, or before the required status/evidence artifacts are durable.
Do not request full-access mode or bypass the sandbox. Make reasonable
in-scope decisions without asking ordinary follow-up questions. If a genuine
external blocker remains after diagnosis and retry, record it in the required
status artifact, then write the completion marker so the workflow can route it.

Slurm CLI and external evaluation APIs require host network access. If they
fail with address-family, DNS, controller-DOWN, or network-isolation errors,
request a narrowly scoped command/network escalation through the normal
approval overlay. Do not repeatedly retry those commands inside the isolated
sandbox, and do not misclassify sandbox reachability as cluster or method
failure.
"""


def managed_env(ctx: Any, runtime_goal: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(getattr(ctx, "env", {}) or {})
    env.update(
        {
            "CODEX_AUTORUN_MODEL": str(
                ctx.config.get("autorun_model") or env.get("CODEX_AUTORUN_MODEL") or "gpt-5.6-sol"
            ),
            "CODEX_AUTORUN_SANDBOX": "workspace-write",
            "CODEX_AUTORUN_AUTO_TRUST_DIRECTORY": "1",
            "CODEX_AUTORUN_RUNTIME_DIR": str(runtime_goal / "runtime"),
            "SYMPOSIUM_PRESERVE_HISTORY_RECORDS": "1",
            "SYMPOSIUM_ALLOW_HISTORY_RECORDS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def archive_previous_marker(marker: Path) -> None:
    if not marker.exists():
        return
    previous = marker.with_name("NODE-COMPLETE.last.json")
    if previous.exists():
        previous.unlink()
    marker.replace(previous)


def capture_session(
    ctx: Any,
    env: dict[str, str],
    runtime_goal: Path,
    session_name: str,
) -> str:
    proc = run_manager(ctx, env, "capture", runtime_goal, session_name, "--lines", "240")
    if proc.returncode == 0:
        return proc.stdout
    return proc.stderr or proc.stdout


def stop_session(
    ctx: Any,
    env: dict[str, str],
    runtime_goal: Path,
    session_name: str,
) -> subprocess.CompletedProcess[str]:
    return run_manager(ctx, env, "stop", runtime_goal, session_name)


def run_managed_codex(
    ctx: Any,
    *,
    source_goal_dir: Path,
    node_name: str,
    role: str,
    prompt: str,
    required_artifacts: Iterable[str],
    output_path: Path,
) -> ManagedCodexResult:
    """Run one node invocation and wait for its unique durable completion marker."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_goal = source_goal_dir.parent / f"{source_goal_dir.name}-{node_name}-autorun"
    runtime_goal.mkdir(parents=True, exist_ok=True)
    marker = runtime_goal / "NODE-COMPLETE.json"
    invocation_file = runtime_goal / "NODE-INVOCATION.json"
    experiment_uid = str(ctx.state.get("experiment_uid") or "experiment")
    session_name = f"{experiment_uid}-{node_name}"
    env = managed_env(ctx, runtime_goal)

    manager = ctx.worktree / "tools/symposium/codex_goal_session.py"
    launcher = ctx.worktree / "tools/symposium/start_codex_goal.sh"
    if not manager.exists() or not launcher.exists():
        detail = f"managed autorun launcher missing: manager={manager.exists()} launcher={launcher.exists()}"
        output_path.write_text(detail + "\n")
        return ManagedCodexResult(1, "", detail, output_path)

    status_proc = run_manager(ctx, env, "status", runtime_goal, session_name)
    status = read_json_from_text(status_proc.stdout) if status_proc.returncode == 0 else {}
    invocation = read_json(invocation_file)
    alive = bool(status.get("alive"))

    launch_stdout = ""
    launch_stderr = ""
    if alive:
        invocation_id = str(invocation.get("invocation_id") or "")
        if not invocation_id:
            detail = "managed session is alive but NODE-INVOCATION.json is missing"
            output_path.write_text(detail + "\n")
            return ManagedCodexResult(1, status_proc.stdout[-4000:], detail, output_path, str(status.get("session") or ""))
    else:
        archive_previous_marker(marker)
        invocation_id = uuid.uuid4().hex
        context = read_json(source_goal_dir / "context.json")
        context.update(
            {
                "role": role,
                "run_id": experiment_uid,
                "experiment_uid": experiment_uid,
                "managed_node": node_name,
                "managed_invocation_id": invocation_id,
            }
        )
        write_json(runtime_goal / "context.json", context)
        (runtime_goal / "goal.md").write_text(
            prompt + completion_contract(ctx, invocation_id, marker, required_artifacts)
        )
        write_json(
            invocation_file,
            {
                "schema_version": 1,
                "invocation_id": invocation_id,
                "node_name": node_name,
                "session_name": session_name,
                "started_at_utc": utc_now(),
                "status": "starting",
            },
        )
        start_proc = run_manager(ctx, env, "start", runtime_goal, session_name)
        launch_stdout = start_proc.stdout
        launch_stderr = start_proc.stderr
        if start_proc.returncode != 0:
            detail = start_proc.stderr.strip() or start_proc.stdout.strip() or "managed autorun start failed"
            output_path.write_text(detail + "\n")
            return ManagedCodexResult(start_proc.returncode, start_proc.stdout[-4000:], detail[-4000:], output_path)
        status = read_json_from_text(start_proc.stdout)

    session = str(status.get("session") or "")
    poll_sec = max(float(ctx.config.get("autorun_poll_sec") or 5.0), 1.0)
    while True:
        completion = read_json(marker)
        if (
            completion.get("invocation_id") == invocation_id
            and completion.get("status") == "complete"
        ):
            screen = capture_session(ctx, env, runtime_goal, session_name)
            output_path.write_text(screen or completion.get("summary", "") + "\n")
            stopped = stop_session(ctx, env, runtime_goal, session_name)
            invocation.update(
                {
                    "invocation_id": invocation_id,
                    "status": "complete",
                    "completed_at_utc": utc_now(),
                    "session": session,
                    "completion": completion,
                }
            )
            write_json(invocation_file, invocation)
            stop_error = "" if stopped.returncode == 0 else (stopped.stderr or stopped.stdout)[-4000:]
            return ManagedCodexResult(
                0 if stopped.returncode == 0 else stopped.returncode,
                screen[-4000:],
                (launch_stderr + stop_error)[-4000:],
                output_path,
                session,
                completion,
            )

        status_proc = run_manager(ctx, env, "status", runtime_goal, session_name)
        if status_proc.returncode == 0:
            current = read_json_from_text(status_proc.stdout)
            if not current.get("alive"):
                detail = "managed Codex session exited before writing its completion marker"
                output_path.write_text((launch_stdout + "\n" + launch_stderr + "\n" + detail).strip() + "\n")
                return ManagedCodexResult(1, launch_stdout[-4000:], detail, output_path, session)
            session = str(current.get("session") or session)
        time.sleep(poll_sec)


def read_json_from_text(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
