#!/usr/bin/env python3
"""Install vendored Symposium skills into the autovideo project root."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendor_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "Symposium"


def install(target: str) -> Path:
    root = project_root()
    source = vendor_root() / "skills"
    if not source.exists():
        raise SystemExit(f"Missing Symposium skills source: {source}")

    if target == "codex":
        dest = root / ".codex" / "skills"
    elif target == "claude":
        dest = root / ".claude" / "skills"
    else:
        raise SystemExit("--target must be codex or claude")

    dest.mkdir(parents=True, exist_ok=True)
    for skill in sorted(source.iterdir()):
        if not skill.is_dir():
            continue
        target_dir = dest / skill.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(skill, target_dir)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("codex", "claude"), default="codex")
    args = parser.parse_args()
    dest = install(args.target)
    print(f"Installed Symposium skills for {args.target} to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
