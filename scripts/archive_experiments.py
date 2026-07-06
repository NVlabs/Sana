#!/usr/bin/env python3
"""Move prior experiments into a timestamped archive without deleting them."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def replace_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old) else value
    if isinstance(value, list):
        return [replace_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_prefix(item, old, new) for key, item in value.items()}
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def move_git_worktree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "worktree", "move", str(source), str(destination)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or f"failed to move {source}")


def archive_one(source: Path, destination: Path) -> dict[str, Any]:
    old_prefix = str(source.resolve())
    new_prefix = str(destination.resolve())
    worktree = source / "worktree"
    is_git_worktree = (worktree / ".git").is_file()
    if destination.exists():
        raise SystemExit(f"Archive destination exists: {destination}")
    if is_git_worktree:
        destination.mkdir(parents=True)
        move_git_worktree(worktree, destination / "worktree")
        for child in list(source.iterdir()):
            shutil.move(str(child), str(destination / child.name))
        source.rmdir()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    metadata_path = destination / "experiment.json"
    metadata = replace_prefix(read_json(metadata_path), old_prefix, new_prefix)
    if metadata:
        metadata["status_before_archive"] = metadata.get("status")
        metadata["status"] = "archived"
        metadata["archived_at_utc"] = utc_now()
        metadata["archive_source"] = old_prefix
        write_json(metadata_path, metadata)
    return {
        "experiment_id": destination.name,
        "source": old_prefix,
        "destination": new_prefix,
        "git_worktree": is_git_worktree,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--archive-root", default="output/archive")
    parser.add_argument("--stamp", default="")
    args = parser.parse_args()
    experiments = (ROOT / args.experiments_root).resolve()
    stamp = args.stamp or utc_stamp()
    archive = (ROOT / args.archive_root / stamp / "experiments").resolve()
    items: list[dict[str, Any]] = []
    if experiments.is_dir():
        for source in sorted(path for path in experiments.iterdir() if path.is_dir()):
            if not (source / "experiment.json").is_file():
                continue
            items.append(archive_one(source, archive / source.name))
    experiments.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "archived_at_utc": utc_now(),
        "source_root": str(experiments),
        "archive_root": str(archive),
        "experiments": items,
    }
    write_json(archive.parent / "INDEX.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
