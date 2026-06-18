#!/usr/bin/env python3
"""Self-contained tests for launch_candidate single-flight guards."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/launch_candidate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("launch_candidate", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def args() -> SimpleNamespace:
    return SimpleNamespace(mode="sbatch", confirm_submit=True, run_root="runs")


def expect_block(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as exc:
        text = str(exc)
        assert needle in text, text
        return
    raise AssertionError("Expected SystemExit")


def test_scored_candidate_blocks_on_unrecorded_scored_run(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"candidates": []})

        run_dir = root / "runs/c1"
        write_json(
            run_dir / "metadata.json",
            {
                "candidate_id": "c1",
                "kind": "methodology",
                "status": "completed",
                "run_dir": str(run_dir),
                "slurm_job_id": "123",
            },
        )

        expect_block(
            lambda: module.enforce_single_flight_or_exit(
                args(), {"id": "c2", "kind": "methodology"}
            ),
            "runs/c1",
        )


def test_duplicate_control_blocks(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"candidates": []})

        run_dir = root / "runs/warm"
        write_json(
            run_dir / "metadata.json",
            {
                "candidate_id": "warm-control",
                "kind": "env_only",
                "status": "submitted",
                "run_dir": str(run_dir),
                "slurm_job_id": "456",
            },
        )

        expect_block(
            lambda: module.enforce_single_flight_or_exit(
                args(),
                {
                    "id": "warm-control",
                    "kind": "env_only",
                    "slurm": {"job_name": "warm-ctrl"},
                },
            ),
            "runs/warm",
        )


def test_baseline_does_not_block_on_scored_run(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"candidates": []})

        run_dir = root / "runs/c1"
        write_json(
            run_dir / "metadata.json",
            {
                "candidate_id": "c1",
                "kind": "methodology",
                "status": "submitted",
                "run_dir": str(run_dir),
                "slurm_job_id": "789",
            },
        )

        module.enforce_single_flight_or_exit(
            args(), {"id": "baseline", "kind": "baseline"}
        )


def test_profile_does_not_block_scored_candidate(module) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module.repo_root = lambda: root
        write_json(root / "AGENT-STATUS.json", {"candidates": []})

        run_dir = root / "runs/profile"
        write_json(
            run_dir / "metadata.json",
            {
                "candidate_id": "cosmos3_kwl_profile",
                "kind": "patch",
                "status": "completed",
                "run_dir": str(run_dir),
                "slurm_job_id": "999",
            },
        )

        module.enforce_single_flight_or_exit(
            args(), {"id": "kwl-next-candidate", "kind": "patch"}
        )


def main() -> int:
    module = load_module()
    tests = [
        test_scored_candidate_blocks_on_unrecorded_scored_run,
        test_duplicate_control_blocks,
        test_baseline_does_not_block_on_scored_run,
        test_profile_does_not_block_scored_candidate,
    ]
    for test in tests:
        test(module)
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
