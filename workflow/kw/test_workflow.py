#!/usr/bin/env python3
"""Focused tests for the KWL workflow loop."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflow/kw/workflow.py"


def load_workflow():
    spec = importlib.util.spec_from_file_location("kw_workflow_test", WORKFLOW)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def make_ctx(tmp: Path, *, dry_run: bool = True):
    wf = load_workflow()
    worktree = tmp / "worktree"
    goal_dir = worktree / "goals/kwl-fusion"
    goal_dir.mkdir(parents=True)
    (goal_dir / "goal.md").write_text("# Goal\n")
    write_json(goal_dir / "context.json", {"role": "implementation", "dimension": "kwl_fusion"})
    state = {
        "workflow_uid": "kw",
        "experiment_uid": "hunyuan-kw-0001",
        "goal_dir": str(goal_dir),
        "reviewer_goal_dir": str(worktree / "goals/kwl-fusion-reviewer"),
        "resume_target": "",
        "resume_reason": "",
        "resume_followups": [],
    }
    return wf.NodeContext(
        root=tmp,
        workflow_dir=tmp / "workflow/kw",
        worktree=worktree,
        goal_dir=goal_dir,
        state_path=worktree / "state/workflow-kw-state.json",
        event_log=worktree / "state/workflow-kw-events.jsonl",
        state=state,
        config={"model_id": "hunyuan_diffusers", "baseline_frames": ""},
        env={},
        dry_run=dry_run,
    )


def test_transition_missing_eval_resumes_executor() -> None:
    wf = load_workflow()
    state = {"cycles": 0, "artifacts": []}
    result = wf.NodeResult("missing", updates={"eval_reason": "baseline_frames_missing"}, message="missing")
    wf.apply_result(state, "check_eval", result)
    next_phase = wf.transition(state, "check_eval", result)
    assert next_phase == "write_resume"
    assert state["resume_target"] == "executor"
    assert "smooth full evaluation" in " ".join(state["resume_followups"])


def test_resume_prompt_writes_executor_file() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        ctx.state["resume_target"] = "executor"
        ctx.state["resume_reason"] = "baseline_frames_missing"
        ctx.state["resume_followups"] = ["run plan_eval.py --assess"]
        result = wf.run_resume_prompt(ctx)
        path = ctx.goal_dir / "STOP_HOOK_RESUME.md"
        assert result.outcome == "written"
        assert path.exists()
        assert "baseline_frames_missing" in path.read_text()


def test_eval_gate_accepts_smooth_assess() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        gate = ctx.worktree / "runs/candidate/assess_verdict.json"
        write_json(
            gate,
            {
                "baseline_total_s": 100.0,
                "candidate_total_s": 90.0,
                "speedup": 1.111,
                "quality_blockers": [],
                "collector_quality_blockers": [],
            },
        )
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "smooth"
        assert result.updates["eval_gate"]["path"] == "runs/candidate/assess_verdict.json"


def test_eval_gate_infra_blocks_without_baseline_frames() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        run_dir = ctx.worktree / "runs/candidate/outputs"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "benchmark.json", {"total_s": 1.0})
        (run_dir / "frames").mkdir()
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "infra_blocked"
        assert result.updates["assess_attempt"]["reason"] == "baseline_frames_missing"


def test_reviewer_accepts_existing_status() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        write_json(
            ctx.worktree / "REVIEWER-STATUS.json",
            {
                "schema_version": 1,
                "target_goal_id": "kwl-fusion",
                "status": "accepted",
                "decision": "accept",
                "reason": "unit accept",
                "required_followups": [],
            },
        )
        result = wf.run_reviewer(ctx)
        assert result.outcome == "accepted"
        assert result.updates["reviewer_reason"] == "unit accept"


def test_run_loop_once_dry_run_initializes_state() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        ctx = make_ctx(tmp)
        meta = tmp / "experiment/hunyuan-kw-0001/experiment.json"
        write_json(
            meta,
            {
                "experiment_id": "hunyuan-kw-0001",
                "worktree": str(ctx.worktree),
                "goal_dir": str(ctx.goal_dir),
            },
        )
        args = Namespace(
            command="run",
            experiment_json=str(meta),
            experiment_uid="hunyuan-kw-0001",
            workflow_uid="kw",
            allow_legacy_experiment_id=False,
            worktree=None,
            goal_dir=None,
            state_file=None,
            event_log=None,
            max_cycles=3,
            reset=False,
            once=True,
            dry_run=True,
            baseline_frames="",
            model_id="hunyuan_diffusers",
            assess_timeout_sec=1,
        )
        result = wf.run_loop(args)
        state = result["state"]
        assert state["workflow_uid"] == "kw"
        assert state["experiment_uid"] == "hunyuan-kw-0001"
        assert state["phase"] == "check_eval"


if __name__ == "__main__":
    test_transition_missing_eval_resumes_executor()
    test_resume_prompt_writes_executor_file()
    test_eval_gate_accepts_smooth_assess()
    test_eval_gate_infra_blocks_without_baseline_frames()
    test_reviewer_accepts_existing_status()
    test_run_loop_once_dry_run_initializes_state()
    print("kw workflow tests passed")
