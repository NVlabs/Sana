#!/usr/bin/env python3
"""Focused tests for managed goal stop-hook lifecycle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "tools/symposium/stop_hook.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_goal(root: Path, goal_id: str = "kwl-fusion", role: str = "implementation") -> Path:
    goal_dir = root / "goals" / goal_id
    goal_dir.mkdir(parents=True)
    (goal_dir / "goal.md").write_text("# Goal\n")
    (goal_dir / "context.json").write_text(
        json.dumps(
            {
                "goal_id": goal_id,
                "role": role,
                "dimension": "kwl_fusion",
                "model_id": "hunyuan_diffusers",
                "candidate_manifest": "candidates/hunyuan_diffusers_baseline.toml",
                "loop_contract": {
                    "canonical_baseline_frames": str(root / "runs/baseline/outputs/frames"),
                    "authoritative_python": "python3",
                },
            }
        )
        + "\n"
    )
    return goal_dir


def write_smooth_gate(root: Path, run_id: str = "candidate") -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    gate = run_dir / "assess_verdict.json"
    gate.write_text(
        json.dumps(
            {
                "baseline_total_s": 100.0,
                "candidate_total_s": 99.0,
                "speedup": 1.0101,
                "quality_status": "available",
                "quality_blockers": [],
                "collector_quality_blockers": [],
            }
        )
        + "\n"
    )
    return gate


def args(goal_dir: Path, session: str = "exp-unit-kwl-fusion", dry_run: bool = True):
    return argparse.Namespace(
        goal_dir=str(goal_dir),
        codex_exit_code=0,
        session_name=session,
        dry_run=dry_run,
    )


def test_executor_without_smooth_eval_requests_resume() -> None:
    mod = load_module(HOOK, "stop_hook_resume_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mod.project_root = lambda: root
        goal_dir = write_goal(root)

        code, payload = mod.after_executor(args(goal_dir), root, goal_dir)

        assert code == 10
        assert payload["action"] == "resume_current_agent"
        assert (goal_dir / "STOP_HOOK_RESUME.md").exists()
        assert "smooth full evaluation" in (goal_dir / "STOP_HOOK_RESUME.md").read_text()


def test_executor_with_smooth_eval_starts_reviewer() -> None:
    mod = load_module(HOOK, "stop_hook_reviewer_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mod.project_root = lambda: root
        goal_dir = write_goal(root)
        review_dir = root / "goals/kwl-fusion-reviewer"
        review_dir.mkdir(parents=True)
        (review_dir / "context.json").write_text(json.dumps({"role": "gate"}) + "\n")
        write_smooth_gate(root)
        mod.create_reviewer_goal = lambda _root, _goal_dir: review_dir

        code, payload = mod.after_executor(args(goal_dir), root, goal_dir)

        assert code == 0
        assert payload["action"] == "start_reviewer"
        assert payload["evaluation"]["smooth"] is True
        assert payload["start"]["reason"] == "dry_run"
        assert payload["reviewer_session"] == "exp-unit-kwl-fusion-reviewer"


def test_reviewer_accepts_lifecycle() -> None:
    mod = load_module(HOOK, "stop_hook_accept_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mod.project_root = lambda: root
        goal_dir = write_goal(root, "kwl-fusion-reviewer", role="gate")
        (root / "REVIEWER-STATUS.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "accepted",
                    "decision": "accept",
                    "reason": "unit accept",
                    "required_followups": [],
                }
            )
            + "\n"
        )

        code, payload = mod.after_reviewer(args(goal_dir, session="exp-unit-kwl-fusion-reviewer"), root, goal_dir)

        assert code == 0
        assert payload["action"] == "accepted"
        assert json.loads((root / "STOP-HOOK-STATUS.json").read_text())["action"] == "accepted"


def test_reviewer_can_wake_executor() -> None:
    mod = load_module(HOOK, "stop_hook_wake_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mod.project_root = lambda: root
        executor_goal = write_goal(root, "kwl-fusion", role="implementation")
        reviewer_goal = write_goal(root, "kwl-fusion-reviewer", role="gate")
        (root / "REVIEWER-STATUS.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "target_goal_id": "kwl-fusion",
                    "status": "needs_executor_resume",
                    "decision": "resume_executor",
                    "reason": "more split-proj variants remain",
                    "required_followups": ["test split projection without extra allocation"],
                }
            )
            + "\n"
        )

        code, payload = mod.after_reviewer(
            args(reviewer_goal, session="exp-unit-kwl-fusion-reviewer"),
            root,
            reviewer_goal,
        )

        assert code == 0
        assert payload["action"] == "start_executor"
        assert payload["start"]["reason"] == "dry_run"
        text = (executor_goal / "STOP_HOOK_RESUME.md").read_text()
        assert "test split projection" in text


def main() -> None:
    test_executor_without_smooth_eval_requests_resume()
    test_executor_with_smooth_eval_starts_reviewer()
    test_reviewer_accepts_lifecycle()
    test_reviewer_can_wake_executor()
    print("stop hook tests passed")


if __name__ == "__main__":
    main()
