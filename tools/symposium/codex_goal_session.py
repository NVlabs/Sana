#!/usr/bin/env python3
"""Manage interactive Codex goal sessions through tmux.

This keeps Codex in real interactive mode while giving the orchestration layer
cheap status, capture, and send controls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def state_root(root: Path) -> Path:
    return root / ".symposium" / "scratch" / "codex-goal-sessions"


def require_tmux() -> str:
    tmux = shutil.which("tmux")
    if not tmux:
        raise SystemExit("tmux is required for managed Codex goal sessions.")
    return tmux


def sanitize(value: str) -> str:
    cleaned = VALID_NAME.sub("-", value.strip())
    return cleaned.strip("-") or "goal"


def goal_id(goal_dir: Path) -> str:
    return sanitize(goal_dir.name)


def session_name_for(goal_dir: Path, name: str | None = None) -> str:
    return sanitize(name) if name else f"autovideo-{goal_id(goal_dir)}"


def state_path(root: Path, goal_dir: Path, name: str | None = None) -> Path:
    return state_root(root) / f"{session_name_for(goal_dir, name)}.json"


def run_tmux(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [require_tmux(), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or f"tmux failed: {args}")
    return proc


def tmux_alive(session: str) -> bool:
    return run_tmux(["has-session", "-t", session], check=False).returncode == 0


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_context(goal_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((goal_dir / "context.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_goal_dir(root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"Goal directory does not exist: {path}")
    if not (path / "goal.md").exists() or not (path / "context.json").exists():
        raise SystemExit(f"Goal directory must contain goal.md and context.json: {path}")
    return path


def resolve_worktree(root: Path, raw: str | None) -> Path:
    if not raw:
        return root
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Worktree directory does not exist: {path}")
    if not (path / "tools/symposium/start_codex_goal.sh").exists():
        raise SystemExit(f"Worktree does not look like autovideo: {path}")
    return path


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def infer_run_id(worktree: Path, context: dict[str, Any]) -> str:
    raw = context.get("run_id")
    if isinstance(raw, str) and raw:
        return raw
    for env_name in ("SYMPOSIUM_CURRENT_RUN_ID", "AUTO_VIDEO_RUN_ID", "RUN_ID"):
        raw = os.environ.get(env_name, "")
        if raw:
            return raw
    parts = worktree.resolve().parts
    for idx, part in enumerate(parts[:-2]):
        if part == "output" and parts[idx + 1] == "fanout_runs":
            return parts[idx + 2]
    return ""


def start(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    worktree = resolve_worktree(root, args.worktree)
    goal_dir = resolve_goal_dir(worktree, args.goal_dir)
    context = load_context(goal_dir)
    run_id = infer_run_id(worktree, context)
    session = session_name_for(goal_dir, args.name)
    state_file = state_path(root, goal_dir, args.name)

    if tmux_alive(session):
        if not args.force:
            raise SystemExit(
                f"Session already exists: {session}. Use status/capture/send/attach or --force."
            )
        run_tmux(["kill-session", "-t", session])

    launcher = worktree / "tools/symposium/start_codex_goal.sh"
    goal_arg = relative_to_root(worktree, goal_dir)
    goal_file = relative_to_root(worktree, goal_dir / "goal.md")
    exports = ["export TERM=xterm-256color"]
    if run_id:
        quoted_run_id = shlex_quote(run_id)
        exports.append(
            "export "
            f"SYMPOSIUM_CURRENT_RUN_ID={quoted_run_id} "
            f"AUTO_VIDEO_RUN_ID={quoted_run_id} "
            f"RUN_ID={quoted_run_id}"
        )
    command = (
        "; ".join(exports)
        + "; "
        f"exec {shlex_quote(str(launcher))} "
        f"{shlex_quote(goal_arg)}"
    )
    run_tmux(
        [
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            str(args.cols),
            "-y",
            str(args.rows),
            "-c",
            str(worktree),
            f"bash -lc {shlex_quote(command)}",
        ]
    )
    time.sleep(args.startup_delay)
    follow_command = f"/goal follow {goal_file}"
    run_tmux(["send-keys", "-t", session, "--", follow_command])
    run_tmux(["send-keys", "-t", session, "Enter"])

    data = {
        "session": session,
        "tmux_session": session,
        "goal_dir": str(goal_dir),
        "goal_id": goal_id(goal_dir),
        "role": context.get("role", "implementation"),
        "dimension": context.get("dimension", "general"),
        "run_id": run_id,
        "worktree": str(worktree),
        "branch": context.get("root_branch"),
        "submodule_branch": context.get("submodule_branch"),
        "status": "starting",
        "resource_state": "active",
        "goal_follow_command": follow_command,
        "last_capture_at_utc": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_file": str(state_file),
        "root": str(root),
    }
    write_state(state_file, data)
    return {**data, "alive": tmux_alive(session)}


def shlex_quote(value: str) -> str:
    # Local tiny quote helper to avoid importing shlex just for this path.
    return "'" + value.replace("'", "'\"'\"'") + "'"


def session_from_args(args: argparse.Namespace) -> tuple[Path, Path, str, dict[str, Any]]:
    root = project_root()
    worktree = resolve_worktree(root, getattr(args, "worktree", None))
    goal_dir = resolve_goal_dir(worktree, args.goal_dir)
    state_file = state_path(root, goal_dir, getattr(args, "name", None))
    state = load_state(state_file)
    session = state.get("session") or session_name_for(goal_dir, getattr(args, "name", None))
    return root, goal_dir, session, state


def status(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive = tmux_alive(session)
    pane = {}
    if alive:
        pane_proc = run_tmux(
            [
                "display-message",
                "-p",
                "-t",
                session,
                "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}\t#{pane_dead_status}",
            ],
            check=False,
        )
        if pane_proc.returncode == 0 and pane_proc.stdout.strip():
            parts = pane_proc.stdout.rstrip("\n").split("\t")
            if len(parts) >= 5:
                pane = {
                    "session_name": parts[0],
                    "pane_id": parts[1],
                    "pane_pid": parts[2],
                    "pane_current_command": parts[3],
                    "pane_dead_status": parts[4],
                }
    return {
        "alive": alive,
        "session": session,
        "goal_dir": str(goal_dir),
        "state": state,
        "pane": pane,
        "state_file": str(state_path(root, goal_dir, getattr(args, "name", None))),
    }


def capture(args: argparse.Namespace) -> str:
    root, goal_dir, session, state = session_from_args(args)
    if not tmux_alive(session):
        raise SystemExit(f"Session is not running: {session}")
    proc = run_tmux(
        [
            "capture-pane",
            "-p",
            "-J",
            "-t",
            session,
            "-S",
            f"-{args.lines}",
        ]
    )
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "last_capture_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return proc.stdout.rstrip("\n")


def send(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    if not tmux_alive(session):
        raise SystemExit(f"Session is not running: {session}")
    if args.text:
        run_tmux(["send-keys", "-t", session, "--", args.text])
    if args.enter:
        run_tmux(["send-keys", "-t", session, "Enter"])
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "last_sent_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_sent_text": args.text,
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {"session": session, "sent": bool(args.text), "enter": args.enter}


def attach(args: argparse.Namespace) -> None:
    _, _, session, _ = session_from_args(args)
    if not tmux_alive(session):
        raise SystemExit(f"Session is not running: {session}")
    os.execvp(require_tmux(), ["tmux", "attach-session", "-t", session])


def stop(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive_before = tmux_alive(session)
    if alive_before:
        run_tmux(["kill-session", "-t", session])
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "status": "stopped",
            "resource_state": "stopped",
            "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {"session": session, "alive_before": alive_before, "alive_after": tmux_alive(session)}


def release(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive_before = tmux_alive(session)
    if alive_before and not args.keep_session:
        run_tmux(["kill-session", "-t", session])
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "status": "released",
            "resource_state": "released",
            "released_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_note": args.note,
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {
        "session": session,
        "alive_before": alive_before,
        "alive_after": tmux_alive(session),
        "resource_state": "released",
    }


def list_sessions(_: argparse.Namespace) -> list[dict[str, Any]]:
    root = project_root()
    items: list[dict[str, Any]] = []
    for path in sorted(state_root(root).glob("*.json")):
        state = load_state(path)
        session = state.get("session") or path.stem
        items.append({"state_file": str(path), "session": session, "alive": tmux_alive(session), **state})
    return items


def watch(args: argparse.Namespace) -> int:
    try:
        while True:
            os.system("clear")
            print(capture(args))
            print()
            print(f"[watching {session_from_args(args)[2]} every {args.interval:g}s; Ctrl-C to stop]")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def print_result(value: Any) -> None:
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start", help="Start a detached Codex goal session")
    start_parser.add_argument("goal_dir")
    start_parser.add_argument("--name")
    start_parser.add_argument("--worktree", help="Autovideo worktree where the goal runs")
    start_parser.add_argument("--force", action="store_true")
    start_parser.add_argument("--rows", type=int, default=40)
    start_parser.add_argument("--cols", type=int, default=120)
    start_parser.add_argument("--startup-delay", type=float, default=2.0)
    start_parser.set_defaults(func=start)

    for command, help_text, func in (
        ("status", "Show session status", status),
        ("capture", "Capture recent terminal output", capture),
        ("send", "Send text or enter to the session", send),
        ("attach", "Attach to the interactive session", attach),
        ("stop", "Stop the session", stop),
        ("release", "Release session resources and mark state released", release),
        ("watch", "Continuously capture the session", watch),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("goal_dir")
        p.add_argument("--name")
        p.add_argument("--worktree", help="Autovideo worktree that owns the goal")
        if command in {"capture", "watch"}:
            p.add_argument("--lines", type=int, default=80)
        if command == "send":
            p.add_argument("--text", default="")
            p.add_argument("--enter", action="store_true")
        if command == "release":
            p.add_argument("--keep-session", action="store_true")
            p.add_argument("--note", default="")
        if command == "watch":
            p.add_argument("--interval", type=float, default=3.0)
        p.set_defaults(func=func)

    list_parser = sub.add_parser("list", help="List known sessions")
    list_parser.set_defaults(func=list_sessions)

    args = parser.parse_args()
    result = args.func(args)
    if result is not None:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
