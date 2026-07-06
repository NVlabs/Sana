#!/usr/bin/env python3
"""Self-contained tests for goal bundle and native Codex goal-session contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "tools/symposium/prepare_goal.py"
SESSION = ROOT / "tools/symposium/codex_goal_session.py"
IMPORT_SEARCH_SPACE = ROOT / "scripts/import_search_space_docs.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_goal_embeds_search_space_and_acceptance() -> None:
    goals_root = ROOT / ".symposium/scratch/test-goals"
    goal_id = "unit-cache-goal"
    goal_dir = goals_root / goal_id
    if goal_dir.exists():
        shutil.rmtree(goal_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--goal-id",
            goal_id,
            "--candidate",
            "candidates/baseline.toml",
            "--objective",
            "Explore caching as an open-ended goal.",
            "--dimension",
            "step_cache",
            "--role",
            "implementation",
            "--goals-root",
            str(goals_root.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    text = (goal_dir / "goal.md").read_text()
    assert "## Search Space Start" in text
    assert "## Required Artifacts" in text
    assert "AGENT-STATUS.json" in text
    assert "## Method Families" in text
    assert "Relevant search doc: `search_space/01_cache.md`" in text
    assert "`02_token_pruning.md`" not in text
    assert "Original source:" not in text
    assert "Imported source:" not in text
    assert "Commit:" not in text
    assert "Cosmos3 inference code" not in text
    assert "target-model inference code" in text
    assert "## Fan-Out Loop Contract" in text
    assert "## Method Baseline Catalog" in text
    assert "`scheduled_step_reuse` [wired/compose_ready]" in text
    assert "`attention_broadcast` [runtime_patch/not_wired]" in text
    assert "bounded per-dimension search loop" in text
    assert "quality improved or speed improved" in text
    assert "retained frontier candidates" in text
    assert "`quality.json` is telemetry" in text
    assert "final speed-target selection" in text
    assert "`max_iters`: 40" in text
    assert "`early_stop_patience`: 0" in text
    assert "fixed_budget_frontier" in text
    assert "terminal_pending_review" in text
    assert "speed targets: `low=1.5x`, `medium=2.0x`, `high=3.0x`" in text
    assert "LPIPS and Gemini are both considered" in text
    assert "terminate the budget with `structured_negative`" in text
    assert "## Historical Record Policy" in text
    assert "clean-start current-experiment loop" in text
    assert "removes stale" in text
    assert "output/fanout_loop_" in text
    assert "output/orchestrator-prompt.txt" in text
    assert "TeaCache-style" in text
    assert "EasyCache-style" in text
    assert "PAB-style" in text
    assert "identify at least five caching mechanisms" in text
    assert "tools/symposium/loop_control.py init" in text
    assert "## Codex Autorun Lifecycle" in text
    assert "gpt-5.6-sol" in text
    assert "workspace-write" in text
    assert "STOP_HOOK_RESUME.md" in text
    assert "REVIEWER-STATUS.json" in text
    assert "speed/quality evidence status" in text
    assert "reference/search_space_docs" not in text
    context = json.loads((goal_dir / "context.json").read_text())
    assert context["role"] == "implementation"
    assert context["dimension"] == "step_cache"
    assert context["model_id"] == "cosmos3"
    assert context["model_profile"] == "models/cosmos3.toml"
    assert context["search_space_root"] == "search_space"
    assert context["search_space_doc"] == "search_space/01_cache.md"
    assert any(item["tier"] == "wired" for item in context["method_baselines"])
    assert any(item["tier"] == "runtime_patch" for item in context["method_baselines"])
    assert context["history_policy"]["mode"] == "clean_start_current_experiment_only"
    assert (
        context["history_policy"]["startup_enforcement"]
        == "clean_stale_records_outside_active_run_id_then_check"
    )
    assert "`output/fanout_loop_*/`" in context["history_policy"]["ignore_paths"]
    assert "`output/orchestrator-prompt.txt`" in context["history_policy"]["ignore_paths"]
    assert context["acceptance_criteria"]
    assert context["loop_contract"]["max_iters"] == 40
    assert context["loop_contract"]["early_stop_patience"] == 0
    assert context["loop_contract"]["loop_mode"] == "fixed_budget_frontier"
    assert context["loop_contract"]["failed_candidate_action"] == "discard_or_reject_log_and_loop"
    assert context["loop_contract"]["successful_candidate_action"] == "retain_frontier_candidate_and_loop"
    assert context["loop_contract"]["candidate_retention"] == "retain_if_quality_improves_or_speed_or_memory_improves_discard_if_neither_improves"
    assert context["loop_contract"]["early_stop_exit_status"] == "terminal_pending_review"
    assert context["loop_contract"]["speed_targets"] == {"low": 1.5, "medium": 2.0, "high": 3.0}
    assert "aligned_pairwise_gemini_max_artifact_severity" in context["loop_contract"]["quality_ranking"]
    assert "aligned_lpips_max" in context["loop_contract"]["quality_ranking"]
    assert context["loop_contract"]["hard_quality_thresholds"].startswith("disabled_by_default")
    assert "select_tiers_for_integration" in context["loop_contract"]["main_agent_review_actions"]
    assert "restart_with_new_direction" in context["loop_contract"]["main_agent_review_actions"]
    assert "aligned_lpips" in context["loop_contract"]["quality_source_of_truth"]
    assert context["loop_contract"]["global_done_requires_integration"] is True
    assert context["loop_contract"]["stop_hook_lifecycle"]["enabled"] is False
    assert context["loop_contract"]["autorun_lifecycle"]["enabled"] is True
    assert context["loop_contract"]["autorun_lifecycle"]["model"] == "gpt-5.6-sol"
    assert context["loop_contract"]["autorun_lifecycle"]["sandbox"] == "workspace-write"
    assert context["loop_contract"]["autorun_lifecycle"]["reviewer_acceptance_required"] is True


def test_start_script_rejects_legacy_orchestrator_records() -> None:
    text = (ROOT / "tools/symposium/start_codex_goal.sh").read_text()
    assert "output/launch_orchestrator.sh" in text
    assert "output/orchestrator-prompt.txt" in text
    assert "output/orchestrator.log" in text
    assert "output/wtest_*.txt" in text
    assert "output/fanout_runs/*" in text
    assert "CURRENT_RUN_ID" in text
    assert "SYMPOSIUM_PRESERVE_HISTORY_RECORDS" in text
    assert "--check-stale-records" in text
    assert "--clean-stale-records" in text
    assert "SYMPOSIUM_CLEAN_HISTORY_RECORDS" in text
    assert "Refusing to start goal because stale optimization records are visible" in text
    assert "codex_auto_run.py" in text
    assert 'AUTORUN_MODEL="${CODEX_AUTORUN_MODEL:-gpt-5.6-sol}"' in text
    assert 'AUTORUN_SANDBOX="${CODEX_AUTORUN_SANDBOX:-workspace-write}"' in text
    assert "--model" in text
    assert "tmux load-buffer" in text
    assert "tmux paste-buffer" in text
    assert 'PANE_CAPTURE" == *"OpenAI Codex"*' in text
    assert "--prompt-file" not in text
    assert "--dangerously-bypass-approvals-and-sandbox" not in text
    assert "CODEX_EXEC_FLAGS" not in text


def test_prepare_goal_stale_check_exempts_active_run() -> None:
    prepare_mod = load_module(PREPARE, "prepare_goal_stale_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "output/fanout_runs/current").mkdir(parents=True)
        (root / "output/fanout_runs/old").mkdir(parents=True)
        (root / "evals/verdicts").mkdir(parents=True)
        (root / "evals/verdicts/gate__old.json").write_text("{}")
        (root / "runs/20260601-tokenprune-old").mkdir(parents=True)

        records = prepare_mod.find_stale_optimization_records(root, "current")

    assert "output/fanout_runs/current" not in records
    assert "output/fanout_runs/old" in records
    assert "evals/verdicts/gate__old.json" in records
    assert "runs/20260601-tokenprune-old" in records


def test_prepare_goal_infers_active_run_from_environment() -> None:
    prepare_mod = load_module(PREPARE, "prepare_goal_env_run_test")
    old_env = {name: os.environ.get(name) for name in prepare_mod.RUN_ID_ENV_VARS}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "output/fanout_runs/current").mkdir(parents=True)
        (root / "output/fanout_runs/old").mkdir(parents=True)
        try:
            for name in prepare_mod.RUN_ID_ENV_VARS:
                os.environ.pop(name, None)
            os.environ["SYMPOSIUM_CURRENT_RUN_ID"] = "current"

            run_id = prepare_mod.infer_run_id(root)
            records = prepare_mod.find_stale_optimization_records(root, run_id)
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    assert run_id == "current"
    assert "output/fanout_runs/current" not in records
    assert "output/fanout_runs/old" in records


def test_prepare_goal_preserves_current_isolated_worktree_records() -> None:
    prepare_mod = load_module(PREPARE, "prepare_goal_isolated_worktree_test")
    with tempfile.TemporaryDirectory() as tmp:
        active = Path(tmp) / "output/fanout_runs/current/step_cache"
        (active / "candidates").mkdir(parents=True)
        (active / "candidates/cosmos3_current.toml").write_text("[candidate]\n")
        (active / "evals/verdicts").mkdir(parents=True)
        (active / "evals/verdicts/cosmos3__current.json").write_text("{}")
        (active / "runs/20260617-stepcache-current").mkdir(parents=True)

        records = prepare_mod.find_stale_optimization_records(active, "current")

    assert records == []


def test_prepare_goal_can_clean_stale_records() -> None:
    prepare_mod = load_module(PREPARE, "prepare_goal_clean_stale_test")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "output/fanout_runs/current").mkdir(parents=True)
        (root / "output/fanout_runs/old").mkdir(parents=True)
        (root / "candidates").mkdir()
        (root / "candidates/cosmos3_old.toml").write_text("[optimization]\n")
        (root / "evals/verdicts").mkdir(parents=True)
        (root / "evals/verdicts/gate__old.json").write_text("{}")

        removed = prepare_mod.remove_stale_optimization_records(root, "current")

        assert "output/fanout_runs/current" not in removed
        assert (root / "output/fanout_runs/current").exists()
        assert not (root / "output/fanout_runs/old").exists()
        assert not (root / "candidates/cosmos3_old.toml").exists()
        assert not (root / "evals/verdicts/gate__old.json").exists()


def test_prepare_goal_can_create_integration_goal() -> None:
    goals_root = ROOT / ".symposium/scratch/test-goals"
    goal_id = "unit-integration-goal"
    goal_dir = goals_root / goal_id
    if goal_dir.exists():
        shutil.rmtree(goal_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--goal-id",
            goal_id,
            "--candidate",
            "candidates/baseline.toml",
            "--objective",
            "Integrate fan-out winners into composed low, medium, and high profiles.",
            "--dimension",
            "integration",
            "--role",
            "integration",
            "--goals-root",
            str(goals_root.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    text = (goal_dir / "goal.md").read_text()
    assert "## Fan-In Integration Contract" in text
    assert "fan-in integration loop" in text
    assert "no_eligible_profile" in text
    assert "INTEGRATION-STATUS.json" in text
    assert "failed composition: record an interaction failure signature" in text
    assert "finish only when every 1.5x/2.0x/3.0x target" in text
    assert "LPIPS and Gemini are both considered" in text
    assert "`max_iters`: 40" in text
    context = json.loads((goal_dir / "context.json").read_text())
    assert context["role"] == "integration"
    assert context["dimension"] == "integration"
    assert context["loop_contract"]["kind"] == "fan_in_integration_loop"
    assert context["loop_contract"]["max_iters"] == 40
    assert context["loop_contract"]["early_stop_patience"] == 0
    assert context["loop_contract"]["failed_candidate_action"] == "record_interaction_failure_and_loop"
    assert context["loop_contract"]["successful_candidate_action"] == "keep_composed_tier_incumbent_and_loop"
    assert context["loop_contract"]["speed_targets"] == {"low": 1.5, "medium": 2.0, "high": 3.0}
    assert "integration/" in context["write_scope"]


def test_prepare_goal_embeds_token_prune_search_space_only() -> None:
    goals_root = ROOT / ".symposium/scratch/test-goals"
    goal_id = "unit-token-prune-goal"
    goal_dir = goals_root / goal_id
    if goal_dir.exists():
        shutil.rmtree(goal_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--goal-id",
            goal_id,
            "--candidate",
            "candidates/baseline.toml",
            "--objective",
            "Explore token pruning as an open-ended goal.",
            "--dimension",
            "token_prune",
            "--role",
            "implementation",
            "--goals-root",
            str(goals_root.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    text = (goal_dir / "goal.md").read_text()
    assert "Relevant search doc: `search_space/02_token_pruning.md`" in text
    assert "`01_cache.md`" not in text
    assert "ToMe-style token merging" in text
    assert "Dynamic token-density control" in text or "dynamic token-density" in text
    assert "Attention-guided token reduction" in text
    assert "identify at least five token-reduction mechanisms" in text
    context = json.loads((goal_dir / "context.json").read_text())
    assert context["dimension"] == "token_prune"
    assert context["search_space_doc"] == "search_space/02_token_pruning.md"


def test_prepare_goal_embeds_sparse_attention_method_baselines() -> None:
    goals_root = ROOT / ".symposium/scratch/test-goals"
    goal_id = "unit-sparse-baselines-goal"
    goal_dir = goals_root / goal_id
    if goal_dir.exists():
        shutil.rmtree(goal_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--goal-id",
            goal_id,
            "--candidate",
            "candidates/baseline.toml",
            "--objective",
            "Explore sparse attention baselines beyond the wired helper.",
            "--dimension",
            "sparse_attention",
            "--role",
            "implementation",
            "--goals-root",
            str(goals_root.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    text = (goal_dir / "goal.md").read_text()
    assert "`piecewise_pisa_env` [wired/compose_ready]" in text
    assert "`online_mask_search_reuse` [runtime_patch/not_wired]" in text
    assert "`dynamic_pattern_probe` [upper_bound_probe/probe_only]" in text
    context = json.loads((goal_dir / "context.json").read_text())
    baselines = context["method_baselines"]
    assert len(baselines) >= 9
    assert any(item["tier"] == "wired" for item in baselines)
    assert any(item["tier"] == "runtime_patch" for item in baselines)
    assert any(item["tier"] == "upper_bound_probe" for item in baselines)
    assert any(item["family"] == "adaspa_mask_search_reuse" for item in baselines)
    assert any(item["family"] == "haste_headwise_budgets" for item in baselines)


def test_prepare_goal_embeds_kwl_quality_gated_rules() -> None:
    goals_root = ROOT / ".symposium/scratch/test-goals"
    goal_id = "unit-kwl-goal"
    goal_dir = goals_root / goal_id
    if goal_dir.exists():
        shutil.rmtree(goal_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--goal-id",
            goal_id,
            "--candidate",
            "candidates/baseline.toml",
            "--objective",
            "Explore quality-gated KWL optimization as an open-ended goal.",
            "--dimension",
            "kwl_fusion",
            "--role",
            "implementation",
            "--goals-root",
            str(goals_root.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    text = (goal_dir / "goal.md").read_text()
    assert "Relevant search doc: `search_space/05_kernel_fusion.md`" in text
    assert "## KWL Quality-Gated Frontier" in text
    assert "run the full fixed-budget frontier loop" in text
    assert "ON bit-exactness is not required" in text
    assert "identify at least seven KWL method families" in text
    assert "GEMM epilogues" in text
    assert "module-level or DiT-block-level microbench" in text
    assert "warm paired DiT/module" in text
    assert "expected full contribution" in text
    assert "not the primary speed authority" in text
    assert "backend-selection" in text
    assert "layout/copy elimination" in text
    context = json.loads((goal_dir / "context.json").read_text())
    assert context["dimension"] == "kwl_fusion"
    assert context["search_space_doc"] == "search_space/05_kernel_fusion.md"
    assert "retain_kwl_candidate" in context["loop_contract"]["candidate_retention"]
    assert "declared_numeric_tolerance" in context["loop_contract"]["quality_source_of_truth"]
    assert "module_level_tensor_diff_when_available" in context["loop_contract"]["quality_source_of_truth"]
    assert (
        context["loop_contract"]["speed_evaluation_policy"]
        == "kwl_primary_speed_evidence_is_warm_paired_dit_or_module_off_on_median_with_expected_full_contribution"
    )


def test_session_start_delegates_to_autorun() -> None:
    session_mod = load_module(SESSION, "codex_goal_session_test")
    launcher_calls: list[dict] = []
    actual_session = "autovideo-sample-20260627-010203-1234-abcd"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        goal_dir = root / "goals/sample"
        goal_dir.mkdir(parents=True)
        (goal_dir / "goal.md").write_text("# Goal\n")
        (goal_dir / "context.json").write_text(
            json.dumps(
                {
                    "role": "implementation",
                    "dimension": "step_cache",
                    "root_branch": "codex/sample",
                    "submodule_branch": "codex/sample-sol",
                }
            )
        )
        launcher = root / "tools/symposium/start_codex_goal.sh"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/usr/bin/env bash\n")

        def fake_launcher(launcher_path: Path, goal_arg: str, worktree: Path, env: dict[str, str]):
            launcher_calls.append(
                {
                    "launcher": launcher_path,
                    "goal_arg": goal_arg,
                    "worktree": worktree,
                    "env": env,
                }
            )
            return subprocess.CompletedProcess(
                [str(launcher_path), goal_arg],
                0,
                stdout=f"Codex running in tmux session: {actual_session}\n",
                stderr="",
            )

        def fake_alive(session: str) -> bool:
            return session == actual_session

        session_mod.project_root = lambda: root
        session_mod.run_goal_launcher = fake_launcher
        session_mod.tmux_alive = fake_alive
        session_mod.time.sleep = lambda _seconds: None

        result = session_mod.start(
            argparse.Namespace(
                goal_dir="goals/sample",
                name=None,
                worktree=None,
                force=False,
                rows=24,
                cols=80,
                startup_delay=0.0,
            )
        )

    assert len(launcher_calls) == 1
    launch = launcher_calls[0]
    assert launch["goal_arg"] == "goals/sample"
    assert launch["worktree"] == root
    assert launch["env"]["SYMPOSIUM_AUTORUN_DETACH"] == "1"
    assert launch["env"]["SYMPOSIUM_AUTORUN_SESSION_PREFIX"] == "autovideo-sample"
    assert launch["env"]["SYMPOSIUM_EXECUTOR_SESSION_NAME"] == "autovideo-sample"
    assert launch["env"]["CODEX_AUTORUN_MODEL"] == "gpt-5.6-sol"
    assert launch["env"]["CODEX_AUTORUN_SANDBOX"] == "workspace-write"
    assert result["session"] == actual_session
    assert result["executor"] == "codex-autorun"
    assert result["model"] == "gpt-5.6-sol"
    assert result["sandbox"] == "workspace-write"
    assert "codex-autorun TUI" in result["goal_follow_command"]


def test_session_start_can_run_in_isolated_worktree() -> None:
    session_mod = load_module(SESSION, "codex_goal_session_worktree_test")
    launcher_calls: list[dict] = []
    actual_session = "sample-20260627-010203-1234-abcd"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "main"
        worktree = root / "output/fanout_runs/fanout_unit/sparse_attention"
        goal_dir = worktree / "goals/sample"
        goal_dir.mkdir(parents=True)
        (goal_dir / "goal.md").write_text("# Goal\n")
        (goal_dir / "context.json").write_text(json.dumps({"dimension": "sparse_attention"}))
        launcher = worktree / "tools/symposium/start_codex_goal.sh"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/usr/bin/env bash\n")

        def fake_launcher(launcher_path: Path, goal_arg: str, cwd: Path, env: dict[str, str]):
            launcher_calls.append(
                {
                    "launcher": launcher_path,
                    "goal_arg": goal_arg,
                    "cwd": cwd,
                    "env": env,
                }
            )
            return subprocess.CompletedProcess(
                [str(launcher_path), goal_arg],
                0,
                stdout=f"Codex running in tmux session: {actual_session}\n",
                stderr="",
            )

        def fake_alive(session: str) -> bool:
            return session == actual_session

        session_mod.project_root = lambda: root
        session_mod.run_goal_launcher = fake_launcher
        session_mod.tmux_alive = fake_alive
        session_mod.time.sleep = lambda _seconds: None

        result = session_mod.start(
            argparse.Namespace(
                goal_dir="goals/sample",
                name="sample",
                worktree=str(worktree),
                force=False,
                rows=24,
                cols=80,
                startup_delay=0.0,
            )
        )

    assert len(launcher_calls) == 1
    launch = launcher_calls[0]
    assert launch["launcher"] == launcher
    assert launch["goal_arg"] == "goals/sample"
    assert launch["cwd"] == worktree
    assert launch["env"]["SYMPOSIUM_CURRENT_RUN_ID"] == "fanout_unit"
    assert launch["env"]["AUTO_VIDEO_RUN_ID"] == "fanout_unit"
    assert launch["env"]["RUN_ID"] == "fanout_unit"
    assert launch["env"]["SYMPOSIUM_AUTORUN_SESSION_PREFIX"] == "sample"
    assert result["worktree"] == str(worktree)
    assert result["run_id"] == "fanout_unit"
    assert result["session"] == actual_session
    assert "codex-autorun TUI" in result["goal_follow_command"]


def test_search_space_import_records_source_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        source = temp / "source"
        docs = source / "search_space_docs"
        docs.mkdir(parents=True)
        (docs / "cache.md").write_text("# Cache directions\n")
        dest = ROOT / ".symposium/scratch/test-search-space-import"
        if dest.exists():
            shutil.rmtree(dest)
        proc = subprocess.run(
            [
                sys.executable,
                str(IMPORT_SEARCH_SPACE),
                "--source",
                str(source),
                "--dest",
                str(dest.relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 0, proc.stderr
        assert (dest / "cache.md").exists()
        source_meta = json.loads((dest / "SOURCE.json").read_text())
        assert source_meta["status"] == "imported"
        assert source_meta["source_path"] == "search_space_docs"


def main() -> None:
    test_prepare_goal_embeds_search_space_and_acceptance()
    test_prepare_goal_stale_check_exempts_active_run()
    test_prepare_goal_infers_active_run_from_environment()
    test_prepare_goal_preserves_current_isolated_worktree_records()
    test_prepare_goal_can_clean_stale_records()
    test_prepare_goal_can_create_integration_goal()
    test_prepare_goal_embeds_token_prune_search_space_only()
    test_prepare_goal_embeds_sparse_attention_method_baselines()
    test_prepare_goal_embeds_kwl_quality_gated_rules()
    test_session_start_delegates_to_autorun()
    test_session_start_can_run_in_isolated_worktree()
    test_search_space_import_records_source_metadata()
    print("goal mode tests passed")


if __name__ == "__main__":
    main()
