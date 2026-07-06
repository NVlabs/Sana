#!/usr/bin/env python3
"""Focused tests for the Attention PA workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "workflow/attention_pa/workflow.py"


def load_workflow():
    for name in list(sys.modules):
        if name == "workflow_types" or name == "nodes" or name.startswith("nodes."):
            del sys.modules[name]
    workflow_dir = str(WORKFLOW.parent)
    if workflow_dir in sys.path:
        sys.path.remove(workflow_dir)
    sys.path.insert(0, workflow_dir)
    spec = importlib.util.spec_from_file_location("attention_pa_workflow", WORKFLOW)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_ctx(wf, tmp: Path):
    worktree = tmp / "worktree"
    goal_dir = worktree / "goals/attention_pa"
    goal_dir.mkdir(parents=True)
    write_json(goal_dir / "context.json", {"role": "implementation", "dimension": "attention"})
    state = {
        "workflow_uid": "attention_pa",
        "experiment_uid": "sana-attention_pa-0001",
        "goal_dir": str(goal_dir),
    }
    return wf.NodeContext(
        root=ROOT,
        workflow_dir=ROOT / "workflow/attention_pa",
        worktree=worktree,
        goal_dir=goal_dir,
        state_path=worktree / "state/workflow-attention_pa-state.json",
        event_log=worktree / "state/workflow-attention_pa-events.jsonl",
        state=state,
        config={"workflow_uid": "attention_pa", "experiment_uid": "sana-attention_pa-0001", "model_id": "sana_video"},
        env={},
        dry_run=False,
    )


def write_full_assess_run(
    worktree: Path,
    *,
    run_name: str = "pisa_candidate",
    visual: str = "pass",
    lpips: bool = True,
    image_size: int = 720,
) -> Path:
    run_dir = worktree / "runs" / run_name
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
    severity = "none" if visual == "pass" else "medium"
    write_json(
        run_dir / "codex_visual_verdict.json",
        {
            "schema_version": 1,
            "provider": "codex",
            "status": "complete",
            "review_id": run_name + "-review",
            "overall": visual,
            "max_artifact_severity": severity,
            "new_artifacts": [] if visual == "pass" else [{"category": "blur_detail_loss", "severity": severity}],
        },
    )
    assess = {
        "run_dir": str(run_dir),
        "baseline_total_s": 200.0,
        "candidate_total_s": 100.0,
        "speedup": 2.0,
        "visual_provider": "codex",
        "codex_visual_overall": visual,
        "codex_visual_verdict": f"runs/{run_name}/codex_visual_verdict.json",
        "max_artifact_severity": severity,
        "quality_blockers": [] if visual == "pass" else ["codex_visual:fail:medium"],
        "collector_quality_blockers": [],
    }
    if lpips:
        assess["lpips_max"] = 0.05
    write_json(run_dir / "assess_verdict.json", assess)
    return run_dir


def write_recipes(worktree: Path, run_dir: Path) -> None:
    attention_map = worktree / "runs/pisa_preflight/attention_map.json"
    write_json(attention_map, {"schema_version": 1, "layers": 36, "steps": 50})
    write_json(
        worktree / "PISA-SEARCH-STATE.json",
        {
            "schema_version": 1,
            "attention_map": str(attention_map.relative_to(worktree)),
            "trials": [{"candidate_id": "pisa_density_050"}],
        },
    )
    acceptable_run = write_full_assess_run(worktree, run_name="pisa_acceptable")
    aggressive_run = write_full_assess_run(worktree, run_name="pisa_aggressive", visual="fail")

    def recipe(candidate_id: str, measured_run: Path, density: float, visual: str) -> dict:
        return {
            "status": "measured",
            "candidate_id": candidate_id,
            "source_hash": "a" * 64,
            "backend": "piecewise_attn",
            "block_size": [64, 64],
            "route_mode": "score",
            "route_bias": False,
            "only_video_self_attention": True,
            "dense_fallback": "dense_sdpa",
            "run_dir": str(measured_run.relative_to(worktree)),
            "assess_verdict": str((measured_run / "assess_verdict.json").relative_to(worktree)),
            "density": density,
            "sparsity": 1.0 - density,
            "layer_policy": {"default": "pisa"},
            "step_policy": {"default": "pisa"},
            "attention_types": {"video_self": "pisa", "cross": "dense"},
            "dispatch": {"pisa": 24, "fallback": 0},
            "speedup": 2.0,
            "full_e2e_total_s": 100.0,
            "lpips_max": 0.05,
            "codex_visual_overall": visual,
            "max_artifact_severity": "none" if visual == "pass" else "medium",
            "artifacts": [] if visual == "pass" else [{"category": "blur_detail_loss", "severity": "medium"}],
        }
    write_json(
        worktree / "PISA-RECIPES.json",
        {
            "schema_version": 1,
            "model_id": "sana_video",
            "workflow_uid": "attention_pa",
            "recipes": {
                "visually_indistinguishable": recipe("pisa_density_075", run_dir, 0.75, "pass"),
                "acceptable_loss": recipe("pisa_density_050", acceptable_run, 0.5, "pass"),
                "aggressive": recipe("pisa_density_025", aggressive_run, 0.25, "fail"),
            },
        },
    )


def test_eval_gate_accepts_complete_quality_pass_and_failure() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        write_full_assess_run(ctx.worktree, visual="pass")
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "smooth"
        assert result.updates["eval_gate"]["quality_pass"] is True

    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        write_full_assess_run(ctx.worktree, visual="fail")
        result = wf.run_eval_gate(ctx)
        assert result.outcome == "smooth"
        assert result.updates["eval_gate"]["quality_pass"] is False


def test_eval_gate_rejects_incomplete_or_mismatched_assessment() -> None:
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


def test_recipe_contract_requires_all_three_measured_tiers() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        run_dir = write_full_assess_run(ctx.worktree)
        ok, issues = wf.run_eval_gate.__globals__["recipe_contract"](ctx.worktree)
        assert ok is False
        assert "missing:PISA-SEARCH-STATE.json" in issues
        assert "missing:PISA-RECIPES.json" in issues
        write_recipes(ctx.worktree, run_dir)
        ok, issues = wf.run_eval_gate.__globals__["recipe_contract"](ctx.worktree)
        assert ok is True
        assert issues == []


def test_workflow_only_finishes_with_complete_status_and_recipes() -> None:
    wf = load_workflow()
    state = {"phase": "check_eval", "cycles": 0, "artifacts": []}
    result = wf.NodeResult(
        "smooth",
        updates={"executor_status": "complete", "recipe_contract_ok": False},
    )
    wf.apply_result(state, "check_eval", result)
    assert wf.transition(state, "check_eval", result) == "write_resume"

    state = {"phase": "check_eval", "cycles": 0, "artifacts": []}
    result = wf.NodeResult(
        "smooth",
        updates={"executor_status": "complete", "recipe_contract_ok": True},
    )
    wf.apply_result(state, "check_eval", result)
    assert wf.transition(state, "check_eval", result) == "done"
    assert state["final_decision"] == "executor_complete"


def test_executor_routes_through_visual_reviewer() -> None:
    wf = load_workflow()
    state = {"phase": "executor", "cycles": 0, "artifacts": []}
    assert wf.transition(state, "executor", wf.NodeResult("exited")) == "visual_review"
    assert wf.transition(state, "visual_review", wf.NodeResult("reviewed")) == "check_eval"
    blocked = wf.NodeResult("infra_blocked", message="codex_visual_launch_failed")
    assert wf.transition(state, "visual_review", blocked) == "write_resume"
    assert "Do not call Gemini" in " ".join(state["resume_followups"])


def test_initial_state_and_prompt_use_attention_uid() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        meta = tmp / "experiment/sana-attention_pa-0001/experiment.json"
        worktree = tmp / "experiment/sana-attention_pa-0001/worktree"
        write_json(
            meta,
            {
                "experiment_id": "sana-attention_pa-0001",
                "worktree": str(worktree),
                "goal_dir": str(worktree / "goals/attention_pa"),
            },
        )
        (worktree / "goals/attention_pa").mkdir(parents=True, exist_ok=True)
        write_json(worktree / "goals/attention_pa/context.json", {"dimension": "attention"})
        (worktree / "goals/attention_pa/goal.md").write_text("Optimize Sana attention.\n")
        args = Namespace(
            command="run",
            experiment_json=str(meta),
            experiment_uid="sana-attention_pa-0001",
            workflow_uid="attention_pa",
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
        state = wf.run_loop(args)["state"]
        assert state["workflow_uid"] == "attention_pa"
        assert state["experiment_uid"] == "sana-attention_pa-0001"
        assert state["phase"] == "visual_review"

        ctx = make_ctx(wf, tmp / "prompt")
        prompt = wf.run_executor.__globals__["build_prompt"](ctx)
        assert "Authoritative Local PISA Implementation" in prompt
        assert (
            "/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/"
            "Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/"
            "backends/piecewise_attn.py"
        ) in prompt
        assert "7546a4bd1d382923ef4876945172655a84d23686" in prompt
        assert "github.com/xie-lab-ml/piecewise-sparse-attention" not in prompt
        assert "arxiv.org/abs/2602.01077" not in prompt
        assert "density = 1 - sparsity" in prompt
        assert "visually_indistinguishable" in prompt
        assert "acceptable_loss" in prompt
        assert "aggressive" in prompt
        assert "This workflow has one decision-making Codex agent: the executor" in prompt
        assert "blind Codex visual reviewer is an evidence-only graph node" in prompt


def test_blind_verdict_decodes_candidate_side() -> None:
    wf = load_workflow()
    decode = wf.run_visual_reviewer.__globals__["decode_blind_verdict"]
    raw = {
        "degraded_side": "left",
        "max_severity": "low",
        "differences": [{"category": "detail_loss", "severity": "low"}],
    }
    minor = decode(raw, "left")
    assert minor["overall"] == "pass"
    assert minor["candidate_relation"] == "minor_loss"
    better = decode(raw, "right")
    assert better["overall"] == "pass"
    assert better["candidate_relation"] == "better"


def test_visual_reviewer_uses_independent_image_session() -> None:
    wf = load_workflow()
    with tempfile.TemporaryDirectory() as td:
        ctx = make_ctx(wf, Path(td))
        builder = wf.run_visual_reviewer.__globals__["build_autorun_command"]
        images = [f"/tmp/blind/pair_{index:02d}.png" for index in range(20)]
        command = builder(
            Path("/home/test/codex_auto_run.py"),
            ctx,
            Path("/tmp/blind/prompt.md"),
            Path("/tmp/blind/runtime"),
            "attention-visual",
            images,
        )
        assert command.count("--image") == 20
        assert "--detach" in command
        assert "--bypass" not in command


if __name__ == "__main__":
    test_eval_gate_accepts_complete_quality_pass_and_failure()
    test_eval_gate_rejects_incomplete_or_mismatched_assessment()
    test_recipe_contract_requires_all_three_measured_tiers()
    test_workflow_only_finishes_with_complete_status_and_recipes()
    test_executor_routes_through_visual_reviewer()
    test_initial_state_and_prompt_use_attention_uid()
    test_blind_verdict_decodes_candidate_side()
    test_visual_reviewer_uses_independent_image_session()
    print("attention_pa workflow tests passed")
