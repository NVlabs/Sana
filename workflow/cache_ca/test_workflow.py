#!/usr/bin/env python3
"""Focused tests for the Cache CA workflow loop."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflow/cache_ca/workflow.py"


def load_workflow():
    for name in list(sys.modules):
        if name == "workflow_types" or name == "nodes" or name.startswith("nodes."):
            del sys.modules[name]
    workflow_dir = str(WORKFLOW.parent)
    if workflow_dir in sys.path:
        sys.path.remove(workflow_dir)
    sys.path.insert(0, workflow_dir)
    spec = importlib.util.spec_from_file_location("cache_ca_workflow", WORKFLOW)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_ctx(wf, tmp: Path):
    worktree = tmp / "worktree"
    goal_dir = worktree / "goals/cache_ca"
    goal_dir.mkdir(parents=True)
    write_json(goal_dir / "context.json", {"role": "implementation", "dimension": "cache"})
    state = {
        "workflow_uid": "cache_ca",
        "experiment_uid": "sana-cache_ca-0001",
        "goal_dir": str(goal_dir),
    }
    return wf.NodeContext(
        root=ROOT,
        workflow_dir=ROOT / "workflow/cache_ca",
        worktree=worktree,
        goal_dir=goal_dir,
        state_path=worktree / "state/workflow-cache_ca-state.json",
        event_log=worktree / "state/workflow-cache_ca-events.jsonl",
        state=state,
        config={"workflow_uid": "cache_ca", "experiment_uid": "sana-cache_ca-0001", "model_id": "sana_video"},
        env={},
        dry_run=False,
    )


def write_full_assess_run(worktree: Path, *, lpips: bool = True, image_size: int = 720) -> Path:
    run_dir = worktree / "runs/cache_candidate"
    (run_dir / "outputs/frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs/frames/f_00001.png").write_text("frame")
    config = {
        "num_frames": 193,
        "fps": 24,
        "image_size": image_size,
        "steps": 50,
        "cfg_scale": 8.0,
        "flow_shift": 12.0,
        "motion_score": 20,
        "prompts_path": "/tmp/models/sana_video/prompts/dpo_holdout_qwen35_val64_concrete40_first5.txt",
    }
    write_json(run_dir / "outputs/benchmark.json", {"total_s": 100.0, "config": config})
    write_json(run_dir / "outputs/run_config.json", config)
    write_json(
        run_dir / "codex_visual_verdict.json",
        {
            "schema_version": 1,
            "provider": "codex",
            "status": "complete",
            "review_id": "cache-review",
            "overall": "pass",
            "max_artifact_severity": "none",
            "new_artifacts": [],
        },
    )
    assess = {
        "run_dir": str(run_dir),
        "baseline_total_s": 200.0,
        "candidate_total_s": 100.0,
        "speedup": 2.0,
        "visual_provider": "codex",
        "codex_visual_overall": "pass",
        "codex_visual_verdict": "runs/cache_candidate/codex_visual_verdict.json",
        "max_artifact_severity": "none",
        "quality_blockers": [],
        "collector_quality_blockers": [],
    }
    if lpips:
        assess["lpips_max"] = 0.05
    write_json(run_dir / "assess_verdict.json", assess)
    return run_dir


def test_eval_gate_accepts_full_lpips_codex_visual_assess() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        write_full_assess_run(ctx.worktree)
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "smooth"
        assert result.updates["eval_gate"]["lpips_max"] == 0.05
        assert result.updates["eval_gate"]["codex_visual_overall"] == "pass"


def test_eval_gate_rejects_missing_lpips_or_config_mismatch() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        write_full_assess_run(ctx.worktree, lpips=False)
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "infra_blocked"
        assert "lpips" in result.message

    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        write_full_assess_run(ctx.worktree, image_size=512)
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "quality_failed"
        assert "full_run_contract_mismatch" in result.message


def test_check_eval_missing_routes_executor_resume() -> None:
    wf = load_workflow()
    state = {"phase": "check_eval", "cycles": 0, "artifacts": []}
    result = wf.NodeResult("missing", updates={"eval_reason": "no_completed_full_run"}, message="no_completed_full_run")
    wf.apply_result(state, "check_eval", result)
    next_phase = wf.transition(state, "check_eval", result)
    assert next_phase == "write_resume"
    assert state["resume_target"] == "executor"
    assert "full Sana Video diffusion" in " ".join(state["resume_followups"])


def test_executor_routes_through_visual_reviewer() -> None:
    wf = load_workflow()
    state = {"phase": "executor", "cycles": 0, "artifacts": []}
    exited = wf.NodeResult("exited")
    assert wf.transition(state, "executor", exited) == "visual_review"

    reviewed = wf.NodeResult("reviewed")
    assert wf.transition(state, "visual_review", reviewed) == "check_eval"

    blocked = wf.NodeResult("infra_blocked", message="codex_visual_launch_failed")
    assert wf.transition(state, "visual_review", blocked) == "write_resume"
    assert "Do not call Gemini" in " ".join(state["resume_followups"])


def test_smooth_gate_resumes_single_executor_until_complete() -> None:
    wf = load_workflow()
    state = {"phase": "check_eval", "cycles": 0, "artifacts": []}
    smooth = wf.NodeResult("smooth", updates={"executor_status": "running"})
    wf.apply_result(state, "check_eval", smooth)
    assert wf.transition(state, "check_eval", smooth) == "write_resume"
    assert state["resume_target"] == "executor"
    followups = " ".join(state["resume_followups"])
    assert "candidate scope closed to TeaCache, EasyCache, and TaylorSeer" in followups
    assert "within 2% relative" in followups

    state = {"phase": "check_eval", "cycles": 0, "artifacts": []}
    complete = wf.NodeResult("smooth", updates={"executor_status": "complete"})
    wf.apply_result(state, "check_eval", complete)
    assert wf.transition(state, "check_eval", complete) == "done"
    assert state["final_decision"] == "executor_complete"


def test_initial_state_uses_cache_uid() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        meta = tmp / "experiment/sana-cache_ca-0001/experiment.json"
        worktree = tmp / "experiment/sana-cache_ca-0001/worktree"
        write_json(
            meta,
            {
                "experiment_id": "sana-cache_ca-0001",
                "worktree": str(worktree),
                "goal_dir": str(worktree / "goals/cache_ca"),
            },
        )
        (worktree / "goals/cache_ca").mkdir(parents=True, exist_ok=True)
        write_json(worktree / "goals/cache_ca/context.json", {"dimension": "cache"})
        args = Namespace(
            command="run",
            experiment_json=str(meta),
            experiment_uid="sana-cache_ca-0001",
            workflow_uid="cache_ca",
            allow_legacy_experiment_id=False,
            worktree=None,
            goal_dir=None,
            state_file=None,
            event_log=None,
            max_cycles=1,
            reset=True,
            once=True,
            dry_run=True,
            assess_timeout_sec=1,
            baseline_frames="",
            model_id="sana_video",
        )
        result = wf.run_loop(args)
        state = result["state"]
        assert state["workflow_uid"] == "cache_ca"
        assert state["experiment_uid"] == "sana-cache_ca-0001"
        assert state["phase"] == "visual_review"
        assert "reviewer_goal_dir" not in state


def test_executor_prompt_requires_adaptive_anchor_families() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        build_prompt = wf.run_executor.__globals__["build_prompt"]
        prompt = build_prompt(ctx)
        assert "TeaCache" in prompt
        assert "EasyCache" in prompt
        assert "TaylorSeer" in prompt
        assert "candidate set for this workflow is closed" in prompt
        assert "contains exactly three cache" in prompt
        assert "Cross-family hybrids are outside this workflow" in prompt
        assert "time_ratio = candidate_total_s / baseline_total_s" in prompt
        assert "differ by at most 2% relative" in prompt
        assert "Never name an overall winner" in prompt
        assert "CACHE-SEARCH-STATE.json" in prompt
        assert "evidence-driven child refinement" in prompt
        assert "Do not label fixed alternating stale-output reuse" in prompt
        assert "This workflow has one decision-making Codex agent: the executor" in prompt
        assert "blind Codex visual reviewer is an evidence-only graph node" in prompt
        assert "Only the reviewer" not in prompt


def test_blind_verdict_decodes_candidate_side() -> None:
    wf = load_workflow()
    decode = wf.run_visual_reviewer.__globals__["decode_blind_verdict"]
    raw = {
        "degraded_side": "right",
        "max_severity": "medium",
        "differences": [{"category": "ghosting", "severity": "medium"}],
    }
    failed = decode(raw, "right")
    assert failed["overall"] == "fail"
    assert failed["candidate_relation"] == "material_loss"
    passed = decode(raw, "left")
    assert passed["overall"] == "pass"
    assert passed["candidate_relation"] == "better"


def test_visual_reviewer_command_attaches_blinded_images() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        module = wf.run_visual_reviewer.__globals__
        images = [f"/tmp/blind/pair_{index:02d}.png" for index in range(20)]
        command = module["build_autorun_command"](
            Path("/home/test/codex_auto_run.py"),
            ctx,
            Path("/tmp/blind/prompt.md"),
            Path("/tmp/blind/runtime"),
            "cache-visual",
            images,
        )
        assert command.count("--image") == 20
        assert "--detach" in command
        assert "workspace-write" in command
        assert "--bypass" not in command

        neutral = ctx.worktree / "state/codex_visual_reviews/review-hash"
        control = {
            "review_id": "review-hash",
            "review_dir": str(neutral),
            "candidate_side": "left",
            "pairs": [
                {"prompt_index": index // 4, "sample_index": index % 4, "timestamp_s": 1.5}
                for index in range(20)
            ],
        }
        prompt = module["build_reviewer_prompt"](
            ctx,
            control,
            ctx.worktree / "runs/sana_teacache_secret_candidate",
        )
        assert "sana_teacache_secret_candidate" not in prompt
        assert "candidate_side" not in prompt


if __name__ == "__main__":
    test_eval_gate_accepts_full_lpips_codex_visual_assess()
    test_eval_gate_rejects_missing_lpips_or_config_mismatch()
    test_check_eval_missing_routes_executor_resume()
    test_executor_routes_through_visual_reviewer()
    test_smooth_gate_resumes_single_executor_until_complete()
    test_initial_state_uses_cache_uid()
    test_executor_prompt_requires_adaptive_anchor_families()
    test_blind_verdict_decodes_candidate_side()
    test_visual_reviewer_command_attaches_blinded_images()
    print("cache_ca workflow tests passed")
