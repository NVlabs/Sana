#!/usr/bin/env python3
"""Self-contained tests for goal bundle and native Codex goal-session contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    assert "## Fan-Out Loop Contract" in text
    assert "bounded per-dimension search loop" in text
    assert "rejected candidate: record a failure signature" in text
    assert "Collector `quality.json`" in text
    assert "not promotion authority" in text
    assert "identify at least three caching mechanisms" in text
    assert "quality-gate status" in text
    assert "reference/search_space_docs" not in text
    context = json.loads((goal_dir / "context.json").read_text())
    assert context["role"] == "implementation"
    assert context["dimension"] == "step_cache"
    assert context["search_space_root"] == "search_space"
    assert context["acceptance_criteria"]
    assert context["loop_contract"]["failed_candidate_action"] == "reject_log_and_loop"
    assert context["loop_contract"]["successful_candidate_action"] == "keep_best_per_tier_and_loop"
    assert "aligned_lpips" in context["loop_contract"]["promotion_authority"]
    assert context["loop_contract"]["global_done_requires_integration"] is True


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
    assert "finish only when every low/medium/high tier" in text
    context = json.loads((goal_dir / "context.json").read_text())
    assert context["role"] == "integration"
    assert context["dimension"] == "integration"
    assert context["loop_contract"]["kind"] == "fan_in_integration_loop"
    assert context["loop_contract"]["failed_candidate_action"] == "record_interaction_failure_and_loop"
    assert context["loop_contract"]["successful_candidate_action"] == "keep_composed_tier_incumbent_and_loop"
    assert "integration/" in context["write_scope"]


def test_session_start_sends_native_goal_follow() -> None:
    session_mod = load_module(SESSION, "codex_goal_session_test")
    calls: list[list[str]] = []

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

        def fake_run_tmux(args: list[str], check: bool = True):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def fake_alive(_session: str) -> bool:
            return any(call and call[0] == "new-session" for call in calls)

        session_mod.project_root = lambda: root
        session_mod.run_tmux = fake_run_tmux
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

    send_text = [call for call in calls if call[:3] == ["send-keys", "-t", "autovideo-sample"]]
    assert any("/goal follow goals/sample/goal.md" in call for call in send_text)
    assert result["goal_follow_command"] == "/goal follow goals/sample/goal.md"
    new_session = next(call for call in calls if call and call[0] == "new-session")
    assert "Read AGENT-GOAL.md" not in " ".join(new_session)


def test_session_start_can_run_in_isolated_worktree() -> None:
    session_mod = load_module(SESSION, "codex_goal_session_worktree_test")
    calls: list[list[str]] = []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "main"
        worktree = Path(tmp) / "worktree"
        goal_dir = worktree / "goals/sample"
        goal_dir.mkdir(parents=True)
        (goal_dir / "goal.md").write_text("# Goal\n")
        (goal_dir / "context.json").write_text(json.dumps({"dimension": "sparse_attention"}))
        launcher = worktree / "tools/symposium/start_codex_goal.sh"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/usr/bin/env bash\n")

        def fake_run_tmux(args: list[str], check: bool = True):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def fake_alive(_session: str) -> bool:
            return any(call and call[0] == "new-session" for call in calls)

        session_mod.project_root = lambda: root
        session_mod.run_tmux = fake_run_tmux
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

    new_session = next(call for call in calls if call and call[0] == "new-session")
    assert str(worktree) in new_session
    assert str(launcher) in " ".join(new_session)
    assert result["worktree"] == str(worktree)
    assert result["goal_follow_command"] == "/goal follow goals/sample/goal.md"


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
    test_prepare_goal_can_create_integration_goal()
    test_session_start_sends_native_goal_follow()
    test_session_start_can_run_in_isolated_worktree()
    test_search_space_import_records_source_metadata()
    print("goal mode tests passed")


if __name__ == "__main__":
    main()
