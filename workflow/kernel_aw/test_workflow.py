#!/usr/bin/env python3
"""Focused tests for the KR workflow loop."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflow/kernel_aw/workflow.py"


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
    goal_dir = worktree / "goals/kernel_aw"
    goal_dir.mkdir(parents=True)
    (goal_dir / "goal.md").write_text("# Goal\n")
    write_json(goal_dir / "context.json", {"role": "implementation", "dimension": "kernel_fusion"})
    state = {
        "workflow_uid": "kernel_aw",
        "experiment_uid": "sana-kernel_aw-0001",
        "goal_dir": str(goal_dir),
        "reviewer_goal_dir": str(worktree / "goals/kernel_aw-reviewer"),
        "resume_target": "",
        "resume_reason": "",
        "resume_followups": [],
    }
    return wf.NodeContext(
        root=tmp,
        workflow_dir=tmp / "workflow/kernel_aw",
        worktree=worktree,
        goal_dir=goal_dir,
        state_path=worktree / "state/workflow-kernel_aw-state.json",
        event_log=worktree / "state/workflow-kernel_aw-events.jsonl",
        state=state,
        config={"model_id": "sana_video", "baseline_frames": ""},
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
    assert "current candidate" in " ".join(state["resume_followups"])
    assert "full-DiT" in " ".join(state["resume_followups"])


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


def test_baseline_gate_rejects_bundled_process_wall_timing() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        run_dir = ctx.worktree / "runs/baseline"
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True)
        write_json(ctx.worktree / "state/baseline-run.json", {"run_dir": str(run_dir)})
        write_json(
            outputs / "benchmark.json",
            {
                "total_s": 490.0,
                "timing_scope": "bundled_run_infer_wall_time",
                "timing_contract": {"warm_steady_state": False},
                "config": {
                    "num_frames": 193,
                    "fps": 24,
                    "image_size": 720,
                    "steps": 50,
                    "cfg_scale": 8,
                    "flow_shift": 12,
                    "motion_score": 20,
                    "sample_nums": 5,
                },
            },
        )
        write_json(outputs / "run_config.json", {"sample_nums": 5})
        for index in range(5):
            (outputs / f"{index:03d}.mp4").write_bytes(f"video-{index}".encode())

        result = wf.run_baseline_gate(ctx)
        assert result.outcome == "invalid"
        assert "baseline_timing_scope_not_hot" in result.updates["baseline_issues"]


def test_baseline_run_imports_canonical_run_without_resubmitting() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        ctx = make_ctx(root)
        source = root / "canonical-source"
        outputs = source / "outputs"
        outputs.mkdir(parents=True)
        write_json(outputs / "benchmark.json", {"total_s": 1.0})
        write_json(outputs / "run_config.json", {"sample_nums": 5})
        for index in range(5):
            (outputs / f"source-{index:03d}.mp4").write_bytes(f"video-{index}".encode())
        ctx.env["CANONICAL_BASELINE_RUN"] = str(source)

        result = wf.run_baseline(ctx)

        assert result.outcome == "completed"
        imported = ctx.worktree / "runs/canonical-baseline-import/outputs"
        assert len(list((imported / "videos").glob("*.mp4"))) == 5
        state = json.loads((ctx.worktree / "state/baseline-run.json").read_text())
        assert state["imported_from"] == str(source)


def test_baseline_run_retries_retryable_infra_terminal() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        ctx = make_ctx(root, dry_run=False)
        run_dir = ctx.worktree / "runs/attempt1"
        run_dir.mkdir(parents=True)
        write_json(
            ctx.worktree / "state/baseline-run.json",
            {
                "schema_version": 1,
                "status": "submitted",
                "created_at_utc": "2026-07-01T00:00:00+00:00",
                "submitted_at_utc": "2026-07-01T00:00:00+00:00",
                "run_dir": str(run_dir),
                "slurm_job_id": "123",
                "attempt": 1,
                "attempts": [],
            },
        )
        globals_ = wf.run_baseline.__globals__
        old_slurm_state = globals_["slurm_state"]
        old_launch = globals_["launch"]
        captured = {}

        def fake_launch(_ctx, _state_path, previous=None, retry_reason=""):
            captured["previous"] = previous
            captured["reason"] = retry_reason
            return wf.NodeResult("waiting", message="baseline_resubmitted")

        try:
            globals_["slurm_state"] = lambda _job_id: "TIMEOUT"
            globals_["launch"] = fake_launch
            result = wf.run_baseline(ctx)
        finally:
            globals_["slurm_state"] = old_slurm_state
            globals_["launch"] = old_launch

        assert result.outcome == "waiting"
        assert captured["previous"]["attempt"] == 1
        assert captured["reason"].startswith("baseline_retryable_terminal:TIMEOUT")


def test_baseline_gate_accepts_hot_text_encoder_through_vae_timing() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        run_dir = ctx.worktree / "runs/baseline"
        outputs = run_dir / "outputs"
        outputs.mkdir(parents=True)
        write_json(ctx.worktree / "state/baseline-run.json", {"run_dir": str(run_dir)})
        samples = [
            {
                "sample_index": index,
                "text_encoder_s": 1.0,
                "denoise_s": 50.0,
                "vae_decode_s": 10.0,
                "total_s": 61.0,
            }
            for index in range(5)
        ]
        contract = {
            "scope": "warm_single_sample_text_encoder_through_vae_decode",
            "warm_steady_state": True,
            "warmup_samples": 1,
            "warmup_same_shape": True,
            "stage_isolated": True,
            "includes_process_startup": False,
            "includes_model_and_text_encoder_load": False,
            "includes_text_encoder_inference": True,
            "includes_denoise": True,
            "includes_vae_decode": True,
            "includes_cpu_postprocess": False,
            "includes_video_write": False,
        }
        write_json(
            outputs / "benchmark.json",
            {
                "schema_version": 2,
                "total_s": 305.0,
                "timing_scope": contract["scope"],
                "timing_contract": contract,
                "aggregate": {
                    "sample_count": 5,
                    "sample_total_s": 305.0,
                    "sample_mean_s": 61.0,
                    "text_encoder_s": 5.0,
                    "denoise_s": 250.0,
                    "vae_decode_s": 50.0,
                },
                "samples": samples,
                "config": {
                    "num_frames": 193,
                    "fps": 24,
                    "image_size": 720,
                    "steps": 50,
                    "cfg_scale": 8,
                    "flow_shift": 12,
                    "motion_score": 20,
                    "sample_nums": 5,
                },
            },
        )
        write_json(outputs / "run_config.json", {"sample_nums": 5})
        for index in range(5):
            (outputs / f"{index:03d}.mp4").write_bytes(f"video-{index}".encode())

        result = wf.run_baseline_gate(ctx)
        assert result.outcome == "ready"
        lock = json.loads((ctx.worktree / "BASELINE-LOCK.json").read_text())
        assert lock["baseline_mean_s"] == 61.0
        assert lock["timing_scope"] == contract["scope"]


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


def test_eval_gate_uses_active_candidate_instead_of_older_smooth_gate() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        old_gate = ctx.worktree / "runs/old/gate_assess.json"
        new_gate = ctx.worktree / "runs/new/gate_assess.json"
        write_json(old_gate, {"candidate_id": "old", "status": "passed", "speedup": 2.0})
        write_json(new_gate, {"candidate_id": "new", "status": "failed", "speedup": 0.9})
        write_json(
            ctx.worktree / "AGENT-STATUS.json",
            {
                "active_candidate_id": "new",
                "active_gate": "runs/new/gate_assess.json",
                "candidates": [
                    {"candidate_id": "old", "evidence": ["runs/old/gate_assess.json"]},
                    {"candidate_id": "new", "evidence": ["runs/new/gate_assess.json"]},
                ],
            },
        )
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "missing"
        assert result.updates["eval_all_gates"][0]["path"] == "runs/new/gate_assess.json"
        assert all(item["path"] != "runs/old/gate_assess.json" for item in result.updates["eval_all_gates"])


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
                "target_goal_id": "kernel_aw",
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
                "target_goal_id": "kernel_aw",
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


def test_reviewer_archives_status_for_older_executor_invocation() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        ctx.state["executor_completion"] = {"invocation_id": "executor-new"}
        write_json(
            ctx.worktree / "REVIEWER-STATUS.json",
            {
                "schema_version": 1,
                "target_goal_id": "kernel_aw",
                "reviewed_executor_invocation_id": "executor-old",
                "status": "needs_executor_resume",
                "decision": "resume_executor",
                "reason": "stale review",
                "required_followups": [],
            },
        )
        result = wf.run_reviewer(ctx)
        assert result.outcome == "invalid_status"
        assert not (ctx.worktree / "REVIEWER-STATUS.json").exists()
        assert list(ctx.worktree.glob("REVIEWER-STATUS.stale-executor-new*.json"))


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
                "target_goal_id": "kernel_aw",
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
        meta = tmp / "experiment/sana-kernel_aw-0001/experiment.json"
        write_json(
            meta,
            {
                "experiment_id": "sana-kernel_aw-0001",
                "worktree": str(ctx.worktree),
                "goal_dir": str(ctx.goal_dir),
            },
        )
        args = Namespace(
            command="run",
            experiment_json=str(meta),
            experiment_uid="sana-kernel_aw-0001",
            workflow_uid="kernel_aw",
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
            model_id="sana_video",
            assess_timeout_sec=1,
        )
        result = wf.run_loop(args)
        state = result["state"]
        assert state["workflow_uid"] == "kernel_aw"
        assert state["experiment_uid"] == "sana-kernel_aw-0001"
        assert state["phase"] == "check_eval"


def test_executor_prompt_includes_current_scope_and_timing_contract() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        build_prompt = wf.run_executor.__globals__["build_prompt"]
        prompt = build_prompt(ctx)
        assert "Transformer Optimization Objective" in prompt
        assert "This workflow does maintain a technique denylist" in prompt
        assert "Reducing 32-bit arithmetic" in prompt
        assert "to 16-bit arithmetic is allowed" in prompt
        assert "use speed as the evaluation criterion" in prompt
        assert "Execution Order And Frontier Policy" in prompt
        assert "Every timing claim must use a matching denominator" in prompt
        assert "retained_parked" in prompt
        assert "dit_profile" in prompt


def test_reviewer_prompt_loads_system_policy_and_current_target() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as raw:
        ctx = make_ctx(Path(raw))
        ctx.state["executor_completion"] = {"invocation_id": "executor-current"}
        reviewer_dir = ctx.worktree / "goals/kernel_aw-reviewer"
        reviewer_dir.mkdir(parents=True)
        (reviewer_dir / "goal.md").write_text("# Review Assignment\n")
        build_prompt = wf.run_reviewer.__globals__["build_prompt"]
        prompt = build_prompt(reviewer_dir, ctx)
        assert prompt.startswith("# Kernel AW Reviewer System Policy")
        assert "Enforce the executor's current technique denylist" in prompt
        assert "Reducing 32-bit arithmetic to 16-bit arithmetic is allowed" in prompt
        assert "Speed is the ordinary evaluation criterion" in prompt
        assert "official baseline was run before candidate changes" in prompt
        assert "first two warmup rounds are excluded" in prompt
        assert "accumulated acceleration stack" in prompt
        assert "Separate the disposition of a concrete candidate implementation" in prompt
        assert "target executor goal id: `kernel_aw`" in prompt
        assert "executor invocation id: `executor-current`" in prompt
        assert "# Review Assignment" in prompt


if __name__ == "__main__":
    test_transition_missing_eval_resumes_executor()
    test_resume_prompt_writes_executor_file()
    test_baseline_gate_rejects_bundled_process_wall_timing()
    test_baseline_run_imports_canonical_run_without_resubmitting()
    test_baseline_run_retries_retryable_infra_terminal()
    test_baseline_gate_accepts_hot_text_encoder_through_vae_timing()
    test_eval_gate_accepts_smooth_dit_gate()
    test_eval_gate_uses_active_candidate_instead_of_older_smooth_gate()
    test_eval_gate_does_not_assess_full_run()
    test_eval_gate_blocks_executor_discard_records()
    test_reviewer_accepts_existing_status()
    test_reviewer_accepts_discarded_status()
    test_reviewer_archives_status_for_older_executor_invocation()
    test_final_full_eval_passes_gemini_assess()
    test_final_full_eval_failure_routes_reviewer_and_archives_status()
    test_run_loop_once_dry_run_initializes_state()
    test_executor_prompt_includes_current_scope_and_timing_contract()
    test_reviewer_prompt_loads_system_policy_and_current_target()
    print("kernel_aw workflow tests passed")
