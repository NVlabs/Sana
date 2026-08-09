#!/usr/bin/env python3
"""Focused tests for indexed experiment isolation helpers."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "tools/symposium/experiment.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def init_repo(root: Path) -> str:
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "unit@example.com")
    git(root, "config", "user.name", "Unit Test")
    (root / "README.md").write_text("baseline\n")
    (root / "tools/symposium").mkdir(parents=True)
    (root / "tools/symposium/prepare_goal.py").write_text("# placeholder\n")
    (root / "tools/symposium/codex_goal_session.py").write_text("# placeholder\n")
    (root / "transfeat").mkdir()
    (root / "transfeat/hunyuan_diffusers_baseline.toml").write_text("kind = 'baseline'\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "baseline")
    return git(root, "rev-parse", "HEAD")


def create_args(**overrides):
    values = {
        "experiment_id": "15001",
        "experiments_root": "output/experiments",
        "base_ref": "HEAD",
        "branch": None,
        "detached": False,
        "goal_id": "kwl-fusion",
        "dimension": "kwl_fusion",
        "role": "implementation",
        "model_id": "hunyuan_diffusers",
        "transfeat": None,
        "objective": None,
        "session_name": None,
        "skip_goal": True,
        "overwrite_goal": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_create_experiment_uses_fresh_base_worktree() -> None:
    mod = load_module(EXPERIMENT, "experiment_create_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        base_sha = init_repo(root)
        (root / "README.md").write_text("dirty coordinator change\n")

        meta = mod.create(create_args(), root=root)
        worktree = Path(meta["worktree"])

        assert meta["experiment_id"] == "15001"
        assert meta["base_sha"] == base_sha
        assert worktree.exists()
        assert (worktree / "README.md").read_text() == "baseline\n"
        assert (worktree / "runs").is_dir()
        assert (worktree / "state").is_dir()
        assert (worktree / "caches/triton").is_dir()
        assert meta["coordinator_dirty_status_at_create"]


def test_create_refuses_existing_experiment_id() -> None:
    mod = load_module(EXPERIMENT, "experiment_existing_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        init_repo(root)
        mod.create(create_args(), root=root)
        try:
            mod.create(create_args(branch="exp/15001/other"), root=root)
        except SystemExit as exc:
            assert "Experiment already exists" in str(exc)
        else:
            raise AssertionError("expected duplicate experiment id refusal")


def test_local_state_requires_explicit_resume() -> None:
    mod = load_module(EXPERIMENT, "experiment_state_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        init_repo(root)
        meta = mod.create(create_args(), root=root)
        worktree = Path(meta["worktree"])
        assert mod.local_state_paths(worktree) == []

        (worktree / "runs/README.md").write_text("tracked placeholder\n")
        assert mod.local_state_paths(worktree) == []

        (worktree / "runs/transfeat-1").mkdir()
        paths = [path.name for path in mod.local_state_paths(worktree)]
        assert paths == ["runs"]
        (worktree / "runs/transfeat-1").rmdir()

        (worktree / "AGENT-STATUS.json").write_text("{}\n")
        paths = [path.name for path in mod.local_state_paths(worktree)]
        assert paths == ["AGENT-STATUS.json"]


if __name__ == "__main__":
    test_create_experiment_uses_fresh_base_worktree()
    test_create_refuses_existing_experiment_id()
    test_local_state_requires_explicit_resume()
    print("PASS experiment isolation tests")
