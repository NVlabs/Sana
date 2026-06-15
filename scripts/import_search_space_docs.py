#!/usr/bin/env python3
"""Import the external search_space_docs source into the local search_space/ contract."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        for child in dest.iterdir():
            if child.name == "SOURCE.json":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        dest.mkdir(parents=True)
    for child in src.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def write_source(dest: Path, payload: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fail(dest: Path, args: argparse.Namespace, reason: str) -> int:
    write_source(
        dest,
        {
            "status": "blocked",
            "status_reason": reason,
            "source_url": args.source,
            "source_ref": args.ref,
            "source_path": args.path,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"search_space_docs import blocked: {reason}")
    return 1


def import_from_local(source: Path, dest: Path, args: argparse.Namespace) -> int:
    src = (source / args.path).resolve() if (source / args.path).exists() else source.resolve()
    if not src.exists() or not src.is_dir():
        return fail(dest, args, f"local source directory not found: {src}")
    copy_tree(src, dest)
    write_source(
        dest,
        {
            "status": "imported",
            "source_url": str(source),
            "source_ref": args.ref,
            "source_path": args.path,
            "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(dest)
    return 0


def import_from_git(dest: Path, args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="search-space-docs-") as tmp:
        tmp_path = Path(tmp)
        proc = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                args.ref,
                args.source,
                str(tmp_path / "repo"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            return fail(dest, args, (proc.stderr or proc.stdout).strip() or "git clone failed")
        repo = tmp_path / "repo"
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", args.path], check=False)
        src = repo / args.path
        if not src.exists():
            return fail(dest, args, f"source path missing after clone: {args.path}")
        copy_tree(src, dest)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    write_source(
        dest,
        {
            "status": "imported",
            "source_url": args.source,
            "source_ref": args.ref,
            "source_commit": commit,
            "source_path": args.path,
            "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(dest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="https://github.com/Efficient-Large-Model/Sol-LTX-Infer")
    parser.add_argument("--ref", default="cosmos_exp")
    parser.add_argument("--path", default="search_space_docs")
    parser.add_argument("--dest", default="search_space")
    args = parser.parse_args()

    root = repo_root()
    dest = (root / args.dest).resolve()
    source_path = Path(args.source).expanduser()
    if source_path.exists():
        return import_from_local(source_path, dest, args)
    return import_from_git(dest, args)


if __name__ == "__main__":
    raise SystemExit(main())
