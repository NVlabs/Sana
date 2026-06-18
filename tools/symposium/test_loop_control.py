#!/usr/bin/env python3
"""Self-contained tests for runtime fan-out loop control."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "tools/symposium/loop_control.py"
PY = sys.executable


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def load_status(root: Path) -> dict:
    return json.loads((root / "AGENT-STATUS.json").read_text())


def write_gate_artifact(root: Path, run_dir: str, name: str = "assess_verdict.json") -> str:
    path = root / run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": True, "source": "unit-test"}) + "\n")
    return str(Path(run_dir) / name)


def make_terminal_frontier_dimension(fanout_root: Path, dimension: str) -> Path:
    root = fanout_root / dimension
    root.mkdir(parents=True)
    run(
        [
            PY,
            str(LOOP),
            "init",
            "--dimension",
            dimension,
            "--max-iters",
            "1",
            "--early-stop-patience",
            "0",
        ],
        root,
    )
    evidence = write_gate_artifact(root, "runs/winner")
    run(
        [
            PY,
            str(LOOP),
            "record-candidate",
            "--candidate-id",
            "winner",
            "--decision",
            "speed_improved",
            "--improvement-axis",
            "speed",
            "--tier",
            "high",
            "--run-dir",
            "runs/winner",
            "--evidence",
            evidence,
            "--reason",
            "unit winner",
            "--speedup",
            "1.2",
        ],
        root,
    )
    return root


def test_fixed_budget_discards_do_not_early_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "step_cache",
                "--goal-id",
                "unit-step-cache",
                "--max-iters",
                "3",
                "--early-stop-patience",
                "0",
            ],
            root,
        )
        assert json.loads(out.stdout)["decision"] == "continue"
        status = load_status(root)
        assert status["status"] == "running"
        assert status["iters_used"] == 0

        out = run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "c1",
                "--decision",
                "discarded_regression",
                "--reason",
                "quality_flat_and_speed_regressed",
                "--evidence",
                "runs/c1/quality.json",
            ],
            root,
        )
        assert json.loads(out.stdout)["decision"] == "continue"
        status = load_status(root)
        assert status["iters_used"] == 1
        assert status["no_improve_count"] == 1
        assert status["discarded_candidates"][0]["candidate_id"] == "c1"

        out = run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "c2",
                "--decision",
                "discarded_regression",
                "--reason",
                "no_quality_gain_no_speed_gain",
            ],
            root,
        )
        assert json.loads(out.stdout)["decision"] == "continue"
        status = load_status(root)
        assert status["status"] == "running"
        assert status["no_improve_count"] == 2

        out = run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "c3",
                "--decision",
                "rejected",
                "--reason",
                "missing_quality_artifact",
            ],
            root,
        )
        assert json.loads(out.stdout)["decision"] == "terminal_pending_review"
        status = load_status(root)
        assert status["status"] == "terminal_pending_review"
        assert status["terminal_reason"] == "max_iters_reached"
        assert status["agent_recommendation"] == "drop_dimension"


def test_frontier_retention_and_review_selects_tiers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "step_cache",
                "--max-iters",
                "2",
                "--early-stop-patience",
                "0",
            ],
            root,
        )
        evidence = write_gate_artifact(root, "runs/clean-cache")
        run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "clean-cache",
                "--decision",
                "speed_improved",
                "--improvement-axis",
                "speed",
                "--tier",
                "low",
                "--run-dir",
                "runs/clean-cache",
                "--evidence",
                evidence,
                "--reason",
                "clean_speedup",
                "--speedup",
                "1.12",
            ],
            root,
        )
        status = load_status(root)
        assert status["no_improve_count"] == 0
        assert status["frontier_candidates"][0]["candidate_id"] == "clean-cache"
        assert status["frontier_candidates"][0]["improvement_axis"] == "speed"

        out = run([PY, str(LOOP), "decide-next"], root)
        assert json.loads(out.stdout)["decision"] == "continue"

        run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "later-reject",
                "--decision",
                "discarded_regression",
                "--reason",
                "quality_flat_speed_regressed",
            ],
            root,
        )
        status = load_status(root)
        assert status["status"] == "terminal_pending_review"
        assert status["agent_recommendation"] == "select_tiers_for_integration"

        out = run(
            [
                PY,
                str(LOOP),
                "review-dimensions",
                "--status-file",
                "AGENT-STATUS.json",
            ],
            root,
        )
        review = json.loads(out.stdout)
        assert review["global_decision"] == "tier_selection_pending"
        assert review["dimensions"][0]["action"] == "select_tiers_for_integration"
        assert review["dimensions"][0]["frontier_count"] == 1


def test_structured_negative_does_not_stop_fixed_budget_loop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "step_cache",
                "--max-iters",
                "40",
                "--early-stop-patience",
                "0",
            ],
            root,
        )
        out = run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "premature-negative",
                "--decision",
                "structured_negative",
                "--reason",
                "agent thinks mechanism space is covered",
                "--remaining-hypothesis",
                "try a different cache family before budget closes",
            ],
            root,
        )
        decision = json.loads(out.stdout)
        status = load_status(root)
        assert decision["decision"] == "continue"
        assert status["status"] == "running"
        assert status["iters_used"] == 1
        assert status["terminal_reason"] == ""
        assert status["failure_signatures"][0]["decision"] == "structured_negative_proposal"
        assert "try a different cache family" in status["remaining_hypotheses"][0]


def test_record_rejects_run_dir_without_authoritative_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "token_prune",
            ],
            root,
        )
        (root / "runs/no-gate/outputs").mkdir(parents=True)
        (root / "runs/no-gate/outputs/quality.json").write_text(json.dumps({"ok": True}) + "\n")
        proc = subprocess.run(
            [
                PY,
                str(LOOP),
                "record-candidate",
                "--candidate-id",
                "no-gate",
                "--decision",
                "speed_improved",
                "--improvement-axis",
                "speed",
                "--run-dir",
                "runs/no-gate",
                "--evidence",
                "runs/no-gate/outputs/quality.json",
                "--reason",
                "quality_json_only",
                "--speedup",
                "1.05",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 1
        assert "authoritative gate artifact" in proc.stderr
        assert load_status(root)["iters_used"] == 0


def test_add_evidence_backfills_existing_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "token_prune",
            ],
            root,
        )
        status = load_status(root)
        record = {
            "iter": 1,
            "candidate_id": "needs-backfill",
            "decision": "speed_improved",
            "reason": "recorded before strict gate validation",
            "tier": "low",
            "run_dir": "runs/needs-backfill",
            "manifest": "",
            "evidence": ["runs/needs-backfill/outputs/quality.json"],
            "speedup": 1.08,
            "quality": "",
            "improvement_axis": "speed",
            "recorded_at_utc": "2026-06-17T00:00:00+00:00",
        }
        status["iters_used"] = 1
        status["candidates"].append(record.copy())
        status["frontier_candidates"].append(record.copy())
        (root / "AGENT-STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        gate = write_gate_artifact(root, "runs/needs-backfill")

        out = run(
            [
                PY,
                str(LOOP),
                "add-evidence",
                "--candidate-id",
                "needs-backfill",
                "--evidence",
                gate,
                "--reason",
                "unit backfill",
            ],
            root,
        )
        result = json.loads(out.stdout)
        assert result["records_touched"] == 2

        out = run([PY, str(LOOP), "validate-status"], root)
        assert json.loads(out.stdout)["ok"] is True


def test_add_evidence_allows_incremental_backfill() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "token_prune",
            ],
            root,
        )
        status = load_status(root)
        for idx in (1, 2):
            record = {
                "iter": idx,
                "candidate_id": f"needs-backfill-{idx}",
                "decision": "speed_improved",
                "reason": "recorded before strict gate validation",
                "tier": "",
                "run_dir": f"runs/needs-backfill-{idx}",
                "manifest": "",
                "evidence": [f"runs/needs-backfill-{idx}/outputs/quality.json"],
                "speedup": 1.01 + idx / 100,
                "quality": "",
                "improvement_axis": "speed",
                "recorded_at_utc": "2026-06-17T00:00:00+00:00",
            }
            status["candidates"].append(record.copy())
            status["frontier_candidates"].append(record.copy())
        status["iters_used"] = 2
        (root / "AGENT-STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

        gate = write_gate_artifact(root, "runs/needs-backfill-1")
        out = run(
            [
                PY,
                str(LOOP),
                "add-evidence",
                "--candidate-id",
                "needs-backfill-1",
                "--evidence",
                gate,
            ],
            root,
        )
        result = json.loads(out.stdout)
        assert result["status_ok"] is False
        assert result["records_touched"] == 2
        assert any("needs-backfill" not in err for err in result["remaining_errors"])
        status = load_status(root)
        assert gate in status["candidates"][0]["evidence"]


def test_validate_rejects_bad_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "AGENT-STATUS.json").write_text(json.dumps({"status": "running"}) + "\n")
        proc = subprocess.run(
            [PY, str(LOOP), "validate-status"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 1
        assert "missing field" in proc.stdout


def test_validate_rejects_speed_frontier_without_speedup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "token_prune",
            ],
            root,
        )
        status = load_status(root)
        bad_record = {
            "candidate_id": "fast-but-missing-speedup",
            "decision": "speed_improved",
            "improvement_axis": "speed",
            "reason": "speed improved but omitted numeric field",
            "run_dir": "runs/c1",
            "speedup": None,
        }
        status["iters_used"] = 1
        status["candidates"].append(bad_record)
        status["frontier_candidates"].append(bad_record)
        (root / "AGENT-STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
        proc = subprocess.run(
            [PY, str(LOOP), "validate-status"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 1
        assert "speedup must be numeric" in proc.stdout


def test_validate_rejects_frontier_candidate_with_discarded_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(
            [
                PY,
                str(LOOP),
                "init",
                "--dimension",
                "sparse_attention",
            ],
            root,
        )
        gate = write_gate_artifact(root, "runs/frontier-drift")
        status = load_status(root)
        discarded_source = {
            "iter": 1,
            "candidate_id": "frontier-drift",
            "decision": "discarded_regression",
            "reason": "manually drifted source record",
            "tier": "",
            "run_dir": "runs/frontier-drift",
            "manifest": "",
            "evidence": [gate],
            "speedup": 1.1,
            "quality": "quality improved",
            "improvement_axis": "none",
            "recorded_at_utc": "2026-06-17T00:00:00+00:00",
        }
        retained = discarded_source.copy()
        retained["decision"] = "quality_improved"
        retained["improvement_axis"] = "quality"
        status["iters_used"] = 1
        status["candidates"].append(discarded_source)
        status["frontier_candidates"].append(retained)
        (root / "AGENT-STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

        proc = subprocess.run(
            [PY, str(LOOP), "validate-status"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 1
        assert "source candidate decision must be a keep decision" in proc.stdout


def test_ensure_integration_dry_run_starts_after_tier_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fanout_root = root / "output/fanout_runs/unit"
        make_terminal_frontier_dimension(fanout_root, "step_cache")

        out = run(
            [
                PY,
                str(LOOP),
                "ensure-integration",
                "--fanout-root",
                str(fanout_root),
                "--base",
                "main",
                "--dry-run",
            ],
            root,
        )
        result = json.loads(out.stdout)
        assert result["decision"] == "start_integration"
        assert result["reason"] == "tier_selection_pending"
        assert result["plan"]["run_id"] == "unit"
        assert result["plan"]["integration_dir"].endswith("output/fanout_runs/unit/integration")
        assert any("--role integration" in command for command in result["plan"]["commands"])
        assert any("codex_goal_session.py start" in command for command in result["plan"]["commands"])


def test_ensure_integration_noops_while_dimension_running() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fanout_root = root / "output/fanout_runs/unit"
        dim_root = fanout_root / "token_prune"
        dim_root.mkdir(parents=True)
        run([PY, str(LOOP), "init", "--dimension", "token_prune"], dim_root)

        out = run(
            [
                PY,
                str(LOOP),
                "ensure-integration",
                "--fanout-root",
                str(fanout_root),
                "--dry-run",
            ],
            root,
        )
        result = json.loads(out.stdout)
        assert result["decision"] == "no_op"
        assert result["reason"] == "continue_monitoring"


def test_ensure_integration_noops_when_integration_complete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fanout_root = root / "output/fanout_runs/unit"
        make_terminal_frontier_dimension(fanout_root, "step_cache")
        integration = fanout_root / "integration"
        integration.mkdir(parents=True)
        (integration / "INTEGRATION-STATUS.json").write_text(
            json.dumps({"status": "complete"}) + "\n"
        )

        out = run(
            [
                PY,
                str(LOOP),
                "ensure-integration",
                "--fanout-root",
                str(fanout_root),
                "--dry-run",
            ],
            root,
        )
        result = json.loads(out.stdout)
        assert result["decision"] == "already_complete"


def main() -> None:
    test_fixed_budget_discards_do_not_early_stop()
    test_frontier_retention_and_review_selects_tiers()
    test_structured_negative_does_not_stop_fixed_budget_loop()
    test_record_rejects_run_dir_without_authoritative_gate()
    test_add_evidence_backfills_existing_record()
    test_add_evidence_allows_incremental_backfill()
    test_validate_rejects_bad_schema()
    test_validate_rejects_speed_frontier_without_speedup()
    test_validate_rejects_frontier_candidate_with_discarded_source()
    test_ensure_integration_dry_run_starts_after_tier_selection()
    test_ensure_integration_noops_while_dimension_running()
    test_ensure_integration_noops_when_integration_complete()
    print("loop control tests passed")


if __name__ == "__main__":
    main()
