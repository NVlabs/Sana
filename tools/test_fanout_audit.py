#!/usr/bin/env python3
"""Self-contained tests for tools/fanout_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tools/fanout_audit.py"
PY = sys.executable


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: dict) -> None:
    write(path, json.dumps(payload, indent=2) + "\n")


def make_run(root: Path, *, pending: bool = False) -> Path:
    fanout = root / "output/fanout_runs/unit"
    integration = fanout / "integration"
    run_dir = integration / "runs/c1"
    source_run = integration / "step_cache/runs/source"
    source_run.mkdir(parents=True)
    write_json(run_dir / "assess_verdict.json", {"speedup": 1.7})
    write_json(
        integration / "INTEGRATION-STATUS.json",
        {
            "status": "terminal_pending_review",
            "iters_used": 1,
            "max_iters": 4,
            "agent_recommendation": "stop",
            "frontier_transfeat": [],
            "transfeat": [
                {
                    "transfeat_id": "c1",
                    "decision": "speed_improved",
                    "purpose": "delivery",
                    "run_dir": "runs/c1",
                    "evidence": ["runs/c1/assess_verdict.json"],
                    "speedup": 1.7,
                    "tier": "low",
                }
            ],
            "best_per_tier": {
                "low": {
                    "transfeat_id": "c1",
                    "decision": "speed_improved",
                    "purpose": "delivery",
                    "run_dir": "runs/c1",
                    "speedup": 1.7,
                }
            },
        },
    )
    write(integration / "INTEGRATION-JOURNAL.md", "# journal\n")
    write(fanout / "RELEASE.md", "# release\n")
    status = "pending" if pending else "gated delivery profile"
    write(
        integration / "integration/release_matrix.md",
        "| tier | status |\n|---|---|\n| low | " + status + " |\n",
    )
    write(
        integration / "transfeat/c1.toml",
        'id = "c1"\n[composition]\nsource_runs = ["../step_cache/runs/source"]\n',
    )
    write_json(
        run_dir / "metadata.json",
        {
            "status": "completed",
            "status_history": [{"status": "prepared"}, {"status": "completed"}],
        },
    )
    return fanout


def run_audit(fanout: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            PY,
            str(AUDIT),
            "--run",
            str(fanout),
            "--json",
            "--no-slurm",
            "--no-process-check",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_audit_passes_terminal_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fanout = make_run(Path(tmp))
        proc = run_audit(fanout)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        result = json.loads(proc.stdout)
        assert result["ok"] is True


def test_audit_rejects_pending_release_row() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fanout = make_run(Path(tmp), pending=True)
        proc = run_audit(fanout)
        assert proc.returncode == 1
        result = json.loads(proc.stdout)
        assert any("pending row" in err for err in result["errors"])


def main() -> int:
    test_audit_passes_terminal_run()
    print("PASS test_audit_passes_terminal_run")
    test_audit_rejects_pending_release_row()
    print("PASS test_audit_rejects_pending_release_row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
