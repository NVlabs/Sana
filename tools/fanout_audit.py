#!/usr/bin/env python3
"""Final control-plane audit for one fan-out run.

The audit is intentionally conservative and dependency-light. It verifies the
machine state that previously required a human monitor: terminal integration
status, durable evidence artifacts, release-matrix consistency, source-run
references, and absence of live Slurm/local workflow processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - Python < 3.11
    raise SystemExit("Python 3.11+ is required for tomllib TOML support") from exc


TERMINAL_STATUS = {"terminal_pending_review", "blocked", "complete"}
NONTERMINAL_RUN_STATUS = {"submitted", "running"}
LOCAL_PROCESS_PATTERNS = (
    "search/plan_eval.py",
    "scripts/collect_run.py",
    "scripts/launch_candidate.py",
    "tools/vision/lpips_judge.py",
    "tools/vision/nvidia_gemini_judge.py",
    "sbatch ",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_json_error": str(exc)}


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        return {"_toml_error": str(exc)}


def resolve_run(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    return (project_root() / "output" / "fanout_runs" / value).resolve()


def add(result: dict[str, list[str]], kind: str, message: str) -> None:
    result[kind].append(message)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def walk_source_runs(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_runs" and isinstance(item, list):
                found.extend(str(entry) for entry in item)
            else:
                found.extend(walk_source_runs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_source_runs(item))
    return found


def check_status(run_root: Path, result: dict[str, list[str]]) -> dict[str, Any]:
    integration_dir = run_root / "integration"
    status_path = integration_dir / "INTEGRATION-STATUS.json"
    payload = load_json(status_path)
    if not payload:
        add(result, "errors", f"missing integration status: {rel(status_path, run_root)}")
        return {}
    if payload.get("_json_error"):
        add(result, "errors", f"invalid JSON in {rel(status_path, run_root)}: {payload['_json_error']}")
        return payload

    status = str(payload.get("status") or "")
    if status not in TERMINAL_STATUS:
        add(result, "errors", f"integration status is not terminal: {status}")
    if payload.get("agent_recommendation") not in {"", None, "stop", "select_tiers_for_integration", "accept_frontier_for_integration", "mark_blocked", "drop_dimension", "request_validation", "restart_with_new_direction"}:
        add(result, "warnings", f"unknown agent_recommendation: {payload.get('agent_recommendation')}")

    for idx, record in enumerate(payload.get("candidates", [])):
        if not isinstance(record, dict):
            add(result, "errors", f"candidates[{idx}] is not an object")
            continue
        run_dir = str(record.get("run_dir") or "")
        if run_dir:
            candidate_run = Path(run_dir)
            if not candidate_run.is_absolute():
                candidate_run = integration_dir / candidate_run
            if not candidate_run.exists():
                add(result, "errors", f"candidates[{idx}] run_dir missing: {run_dir}")
        for evidence in record.get("evidence", []) or []:
            ev_path = Path(str(evidence))
            if not ev_path.is_absolute():
                ev_path = integration_dir / ev_path
            if not ev_path.exists():
                add(result, "errors", f"candidates[{idx}] evidence missing: {evidence}")
                continue
            if ev_path.suffix == ".json" and load_json(ev_path).get("_json_error"):
                add(result, "errors", f"candidates[{idx}] evidence invalid JSON: {evidence}")

    for tier, record in (payload.get("best_per_tier") or {}).items():
        if not isinstance(record, dict):
            add(result, "errors", f"best_per_tier[{tier}] is not an object")
            continue
        if record.get("purpose", "delivery") != "delivery":
            add(result, "errors", f"best_per_tier[{tier}] is not a delivery record")

    return payload


def check_release_artifacts(run_root: Path, status: dict[str, Any], result: dict[str, list[str]]) -> None:
    integration_dir = run_root / "integration"
    release_paths = [
        run_root / "RELEASE.md",
        integration_dir / "INTEGRATION-JOURNAL.md",
        integration_dir / "integration" / "release_matrix.md",
    ]
    for path in release_paths:
        if not path.exists():
            add(result, "warnings", f"release artifact missing: {rel(path, run_root)}")

    matrix = integration_dir / "integration" / "release_matrix.md"
    if matrix.exists() and status.get("status") in TERMINAL_STATUS:
        for lineno, line in enumerate(matrix.read_text().splitlines(), start=1):
            if line.lstrip().startswith("|") and " pending " in f" {line.lower()} ":
                add(result, "errors", f"release matrix still has pending row at {rel(matrix, run_root)}:{lineno}")

    high_blocker = integration_dir / "integration" / "high_blocker.json"
    matrix_text = matrix.read_text().lower() if matrix.exists() else ""
    if "blocked" in matrix_text and not high_blocker.exists():
        add(result, "warnings", f"blocked target mentioned but blocker artifact missing: {rel(high_blocker, run_root)}")


def check_manifests(run_root: Path, result: dict[str, list[str]]) -> None:
    integration_dir = run_root / "integration"
    candidates_dir = integration_dir / "candidates"
    if not candidates_dir.exists():
        add(result, "warnings", f"integration candidates dir missing: {rel(candidates_dir, run_root)}")
        return
    for manifest in sorted(candidates_dir.glob("*.toml")):
        data = load_toml(manifest)
        if data.get("_toml_error"):
            add(result, "errors", f"invalid TOML in {rel(manifest, run_root)}: {data['_toml_error']}")
            continue
        for source in walk_source_runs(data):
            source_path = Path(source)
            if not source_path.is_absolute():
                candidates = [manifest.parent / source_path, manifest.parent.parent / source_path]
            else:
                candidates = [source_path]
            if not any(candidate.exists() for candidate in candidates):
                add(result, "errors", f"{rel(manifest, run_root)} source_run missing: {source}")


def collect_metadata(run_root: Path) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("**/runs/*/metadata.json")):
        payload = load_json(path)
        if payload:
            payload["_metadata_path"] = str(path)
            metadata.append(payload)
    return metadata


def check_metadata(metadata: list[dict[str, Any]], run_root: Path, result: dict[str, list[str]]) -> set[str]:
    job_ids: set[str] = set()
    for payload in metadata:
        path = Path(str(payload.get("_metadata_path")))
        status = str(payload.get("status") or "")
        job_id = str(payload.get("slurm_job_id") or "")
        if job_id:
            job_ids.add(job_id)
        if status in NONTERMINAL_RUN_STATUS:
            add(result, "errors", f"nonterminal run metadata: {rel(path, run_root)} status={status} job={job_id}")
        if payload.get("status_history") is None:
            add(result, "warnings", f"run metadata lacks status_history: {rel(path, run_root)}")
    return job_ids


def check_slurm(job_ids: set[str], result: dict[str, list[str]]) -> None:
    if not job_ids or not shutil.which("squeue"):
        return
    proc = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T %M %R"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        add(result, "warnings", f"squeue failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return
    for line in proc.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] in job_ids:
            add(result, "errors", f"recorded Slurm job still active: {line}")


def check_local_processes(run_root: Path, result: dict[str, list[str]]) -> None:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,args"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        add(result, "warnings", f"ps failed: {proc.stderr.strip()}")
        return
    root_text = str(run_root)
    self_pid = os.getpid()
    for line in proc.stdout.splitlines():
        if str(self_pid) in line:
            continue
        if root_text not in line and not any(pattern in line for pattern in LOCAL_PROCESS_PATTERNS):
            continue
        if any(pattern in line for pattern in LOCAL_PROCESS_PATTERNS):
            add(result, "errors", f"workflow-like local process still active: {line.strip()}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_root = resolve_run(args.run)
    result: dict[str, Any] = {
        "ok": False,
        "run_root": str(run_root),
        "errors": [],
        "warnings": [],
    }
    if not run_root.exists():
        add(result, "errors", f"fanout run root does not exist: {run_root}")
        return result

    status = check_status(run_root, result)
    check_release_artifacts(run_root, status, result)
    check_manifests(run_root, result)
    metadata = collect_metadata(run_root)
    job_ids = check_metadata(metadata, run_root, result)
    if not args.no_slurm:
        check_slurm(job_ids, result)
    if not args.no_process_check:
        check_local_processes(run_root, result)
    result["metadata_count"] = len(metadata)
    result["slurm_job_ids"] = sorted(job_ids)
    result["ok"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="fanout run id or path")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--no-slurm", action="store_true", help="skip squeue checks")
    parser.add_argument("--no-process-check", action="store_true", help="skip local process scan")
    args = parser.parse_args()

    result = audit(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"fanout audit: {'ok' if result['ok'] else 'failed'}")
        print(f"run_root: {result['run_root']}")
        for key in ("errors", "warnings"):
            for item in result[key]:
                print(f"{key[:-1]}: {item}")
        print(f"metadata_count: {result['metadata_count']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
