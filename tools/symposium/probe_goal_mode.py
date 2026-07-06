#!/usr/bin/env python3
"""Probe Symposium and Codex interactive goal-mode readiness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SKILLS = (
    "wonder",
    "reflect",
    "refine",
    "restate",
    "socrates",
    "evolve-step",
    "ontology",
    "interview-harness",
)


def goal_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    local_bins = [
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / "bin"),
        str(Path.home() / ".codex" / "bin"),
    ]
    env["PATH"] = os.pathsep.join(local_bins + [env.get("PATH", "")])
    env_path = root / ".symposium" / "goal-mode.env"
    if env_path.exists():
        # Keep this parser deliberately small: it supports the KEY=value and
        # export KEY=value lines used by the project-local launcher config.
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            value = value.replace("$HOME", str(Path.home()))
            value = value.replace("$PATH", env.get("PATH", ""))
            env[key.strip()] = value
    return env


def which(command: str, env: dict[str, str]) -> str | None:
    return shutil.which(command, path=env.get("PATH"))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendor_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "Symposium"


def vendor_metadata() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "VENDOR.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def command_help_contains(command: str, needle: str, env: dict[str, str]) -> bool:
    exe = which(command, env)
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return needle.lower() in proc.stdout.lower()


def resolve_codex_autorun(env: dict[str, str]) -> str | None:
    candidates = []
    if env.get("CODEX_AUTORUN"):
        candidates.append(Path(env["CODEX_AUTORUN"]).expanduser())
    candidates.extend(
        [
            Path.home() / "codex_auto_run.py",
            Path.home() / "code/codex_exec/codex_auto_run.py",
        ]
    )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def executable_help_contains(executable: str | None, needle: str, env: dict[str, str]) -> bool:
    if not executable:
        return False
    try:
        proc = subprocess.run(
            [executable, "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return needle.lower() in proc.stdout.lower()


def skill_status(root: Path, target: str) -> dict[str, Any]:
    base = root / f".{target}" / "skills"
    missing = [name for name in SKILLS if not (base / name / "SKILL.md").exists()]
    return {
        "path": str(base),
        "installed": not missing,
        "missing": missing,
    }


def probe() -> dict[str, Any]:
    root = project_root()
    env = goal_env(root)
    vendor = vendor_root()
    codex_command = env.get("CODEX_GOAL_COMMAND") or which("codex", env)
    codex_autorun = resolve_codex_autorun(env)
    claude_command = env.get("CLAUDE_GOAL_COMMAND") or which("claude", env)
    vendor_meta = vendor_metadata()
    return {
        "project_root": str(root),
        "symposium": {
            "vendor_path": str(vendor),
            "present": vendor.exists(),
            "source_url": vendor_meta.get("source_url"),
            "source_commit": vendor_meta.get("source_commit"),
            "skills_present": all((vendor / "skills" / name / "SKILL.md").exists() for name in SKILLS),
        },
        "skills": {
            "codex": skill_status(root, "codex"),
            "claude": skill_status(root, "claude"),
        },
        "commands": {
            "codex": codex_command,
            "codex_help_mentions_goal": command_help_contains("codex", "goal", env),
            "codex_autorun": codex_autorun,
            "codex_autorun_supports_workspace_write": executable_help_contains(
                codex_autorun, "workspace-write", env
            ),
            "claude": claude_command,
        },
        "interactive": {
            "stdin_tty": sys.stdin.isatty(),
            "stdout_tty": sys.stdout.isatty(),
        },
        "can_start_codex_goal_here": bool(codex_command) and bool(codex_autorun),
        "can_attach_codex_goal_here": sys.stdin.isatty() and sys.stdout.isatty(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()
    result = probe()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        return 0
    if not result["symposium"]["present"]:
        return 2
    if not result["skills"]["codex"]["installed"]:
        return 3
    if not result["can_start_codex_goal_here"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
