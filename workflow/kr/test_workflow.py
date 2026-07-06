#!/usr/bin/env python3
"""Focused tests for the KR workflow loop."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflow/kr/workflow.py"


def load_workflow():
    spec = importlib.util.spec_from_file_location("kr_workflow_test", WORKFLOW)
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
    goal_dir = worktree / "goals/kwl-retention"
    goal_dir.mkdir(parents=True)
    (goal_dir / "goal.md").write_text("# Goal\n")
    write_json(goal_dir / "context.json", {"role": "implementation", "dimension": "kwl_fusion"})
    state = {
        "workflow_uid": "kr",
        "experiment_uid": "hunyuan-kr-0001",
        "goal_dir": str(goal_dir),
        "reviewer_goal_dir": str(worktree / "goals/kwl-retention-reviewer"),
        "resume_target": "",
        "resume_reason": "",
        "resume_followups": [],
    }
    return wf.NodeContext(
        root=tmp,
        workflow_dir=tmp / "workflow/kr",
        worktree=worktree,
        goal_dir=goal_dir,
        state_path=worktree / "state/workflow-kr-state.json",
        event_log=worktree / "state/workflow-kr-events.jsonl",
        state=state,
        config={"model_id": "hunyuan_diffusers", "baseline_frames": ""},
        env={},
        dry_run=dry_run,
    )


def test_transition_missing_eval_resumes_executor() -> None:
    wf = load_workflow()
    state = {"cycles": 0, "artifacts": []}
    result = wf.NodeResult("missing", updates={"eval_reason": "no_smooth_dit_gate"}, message="missing")
    wf.apply_result(state, "check_eval", result)
    next_phase = wf.transition(state, "check_eval", result)
    assert next_phase == "write_resume"
    assert state["resume_target"] == "executor"
    assert "single-DiT" in " ".join(state["resume_followups"])


def test_resume_prompt_writes_executor_file() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        ctx.state["resume_target"] = "executor"
        ctx.state["resume_reason"] = "no_smooth_dit_gate"
        ctx.state["resume_followups"] = ["run a single-DiT gate"]
        result = wf.run_resume_prompt(ctx)
        path = ctx.goal_dir / "STOP_HOOK_RESUME.md"
        assert result.outcome == "written"
        assert path.exists()
        assert "no_smooth_dit_gate" in path.read_text()


def test_eval_gate_accepts_smooth_dit_gate() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        gate = ctx.worktree / "runs/candidate/gate_assess.json"
        write_json(
            gate,
            {
                "status": "passed",
                "decision": "needs_reviewer_judgment",
                "summary": {"median_speedup_median": 1.111},
            },
        )
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "smooth"
        assert result.updates["eval_gate"]["path"] == "runs/candidate/gate_assess.json"


def test_eval_gate_does_not_assess_full_run() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        run_dir = ctx.worktree / "runs/candidate/outputs"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "benchmark.json", {"total_s": 1.0})
        (run_dir / "frames").mkdir()
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "missing"
        assert result.updates["eval_reason"] == "no_smooth_dit_gate"
        assert "assess_attempt" not in result.updates


def test_eval_gate_blocks_executor_discard_records() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        write_json(
            ctx.worktree / "AGENT-STATUS.json",
            {
                "discarded_candidates": [
                    {
                        "candidate_id": "candidate_a",
                        "decision": "discarded_regression",
                        "reason": "executor tried to discard",
                    }
                ]
            },
        )
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "policy_violation"
        assert result.updates["eval_reason"] == "executor_discard_decision_forbidden"


def test_reviewer_accepts_existing_status() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        write_json(
            ctx.worktree / "REVIEWER-STATUS.json",
            {
                "schema_version": 1,
                "target_goal_id": "kwl-retention",
                "status": "accepted",
                "decision": "accept",
                "reason": "unit accept",
                "required_followups": [],
            },
        )
        result = wf.run_reviewer(ctx)
        assert result.outcome == "accepted"
        assert result.updates["reviewer_reason"] == "unit accept"


def test_reviewer_accepts_discarded_status() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        write_json(
            ctx.worktree / "REVIEWER-STATUS.json",
            {
                "schema_version": 1,
                "target_goal_id": "kwl-retention",
                "status": "discarded",
                "decision": "discard",
                "reason": "unit discard",
                "required_followups": [],
                "discard_checks": {
                    "smooth_dit_eval": True,
                    "no_speed_memory_quality_gain": True,
                    "not_infra_or_collection_failure": True,
                    "no_remaining_operator_refinement": True,
                },
            },
        )
        result = wf.run_reviewer(ctx)
        assert result.outcome == "discarded"
        state = {"cycles": 0, "artifacts": []}
        wf.apply_result(state, "reviewer", result)
        next_phase = wf.transition(state, "reviewer", result)
        assert next_phase == "final_full_eval"
        assert state["requested_final_decision"] == "discarded_by_reviewer"


def test_final_full_eval_passes_gemini_assess() -> None:
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
                "gemini_overall": "pass",
                "quality_blockers": [],
                "collector_quality_blockers": [],
            },
        )
        result = wf.run_final_full_eval(ctx)
        assert result.outcome == "passed"
        state = {"cycles": 0, "artifacts": [], "requested_final_decision": "accepted"}
        wf.apply_result(state, "final_full_eval", result)
        next_phase = wf.transition(state, "final_full_eval", result)
        assert next_phase == "done"
        assert state["final_decision"] == "accepted"


def test_final_full_eval_failure_routes_reviewer_and_archives_status() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw), dry_run=False)
        write_json(
            ctx.worktree / "REVIEWER-STATUS.json",
            {
                "schema_version": 1,
                "target_goal_id": "kwl-retention",
                "status": "accepted",
                "decision": "accept",
                "reason": "unit accept",
                "required_followups": [],
            },
        )
        gate = ctx.worktree / "runs/candidate/assess_verdict.json"
        write_json(
            gate,
            {
                "baseline_total_s": 100.0,
                "candidate_total_s": 90.0,
                "speedup": 1.111,
                "gemini_overall": "fail",
                "quality_blockers": [],
                "collector_quality_blockers": [],
            },
        )
        result = wf.run_final_full_eval(ctx)
        assert result.outcome == "quality_failed"
        assert not (ctx.worktree / "REVIEWER-STATUS.json").exists()
        assert (ctx.worktree / "REVIEWER-STATUS.final-full-eval-returned.json").exists()
        state = {"cycles": 0, "artifacts": []}
        wf.apply_result(state, "final_full_eval", result)
        next_phase = wf.transition(state, "final_full_eval", result)
        assert next_phase == "write_resume"
        assert state["resume_target"] == "reviewer"


def test_run_loop_once_dry_run_initializes_state() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        ctx = make_ctx(tmp)
        meta = tmp / "experiment/hunyuan-kr-0001/experiment.json"
        write_json(
            meta,
            {
                "experiment_id": "hunyuan-kr-0001",
                "worktree": str(ctx.worktree),
                "goal_dir": str(ctx.goal_dir),
            },
        )
        args = Namespace(
            command="run",
            experiment_json=str(meta),
            experiment_uid="hunyuan-kr-0001",
            workflow_uid="kr",
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
        assert state["workflow_uid"] == "kr"
        assert state["experiment_uid"] == "hunyuan-kr-0001"
        assert state["phase"] == "check_eval"


if __name__ == "__main__":
    test_transition_missing_eval_resumes_executor()
    test_resume_prompt_writes_executor_file()
    test_eval_gate_accepts_smooth_dit_gate()
    test_eval_gate_does_not_assess_full_run()
    test_eval_gate_blocks_executor_discard_records()
    test_reviewer_accepts_existing_status()
    test_reviewer_accepts_discarded_status()
    test_final_full_eval_passes_gemini_assess()
    test_final_full_eval_failure_routes_reviewer_and_archives_status()
    test_run_loop_once_dry_run_initializes_state()
    print("kr workflow tests passed")
