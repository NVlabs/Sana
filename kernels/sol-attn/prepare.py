#!/usr/bin/env python3
"""Stage the maintained Sol-Engine package for Hugging Face kernel-builder."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PACKAGE_FILES = (
    "build.toml",
    "CARD.md",
    "flake.lock",
    "flake.nix",
    "README.md",
)


def _source_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    repository = package_root.parents[1]
    source = repository / "techniques" / "sparse_backends" / "sol_attn"
    output = args.output.resolve()

    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for name in PACKAGE_FILES:
        shutil.copy2(package_root / name, output / name)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "._*")
    shutil.copytree(
        package_root / "tests",
        output / "tests",
        ignore=ignored,
    )
    shutil.copytree(
        source,
        output / "torch-ext" / "sol_attn",
        ignore=ignored,
    )

    source_commit = args.source_commit or _source_commit(repository)
    (output / "SOURCE_COMMIT").write_text(source_commit + "\n")


if __name__ == "__main__":
    main()
