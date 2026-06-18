#!/usr/bin/env python3
"""Prepare and optionally launch an autovideo candidate.

The script is intentionally small and dependency-free. It reads a TOML
candidate manifest, creates a run bundle under runs/, writes the exact launch
shell script, and either stops at dry-run, executes locally, or submits Slurm.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - Python < 3.11
    raise SystemExit("Python 3.11+ is required for tomllib TOML support") from exc


VALID_ID = re.compile(r"[^A-Za-z0-9_.-]+")
CANONICAL_ARTIFACT_DEFAULTS = {
    "log": "run.log",
    "video": "out.mp4",
    "benchmark": "benchmark.json",
    "quality": "quality.json",
    "risk_notes": "risk_notes.md",
    "collection": "collection.json",
    "patch_summary": "patch_summary.md",
    "frames_dir": "frames",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def sanitize_id(value: str) -> str:
    cleaned = VALID_ID.sub("-", value.strip())
    return cleaned.strip("-") or "candidate"


def shell_export(key: str, value: Any) -> str:
    return f"export {key}={shlex.quote(str(value))}"


def run_git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        top = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if Path(top.strip()).resolve() != path.resolve() and not (path / ".git").exists():
            return None
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.strip()


def toml_string(value: Any) -> str:
    return json.dumps(str(value))


def write_resolved_manifest(
    candidate_path: Path,
    run_dir: Path,
    data: dict[str, Any],
    resolved: dict[str, Any],
) -> None:
    original = candidate_path.read_text()
    lines = [original.rstrip(), "", "[resolved]"]
    for key, value in resolved.items():
        lines.append(f"{key} = {toml_string(value)}")
    (run_dir / "manifest.resolved.toml").write_text("\n".join(lines) + "\n")


def write_launch_script(
    run_dir: Path,
    sol_root: Path,
    run_script: Path,
    env: dict[str, Any],
    output_dir: Path,
) -> Path:
    launch_path = run_dir / "launch.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(sol_root))}",
        f"mkdir -p {shlex.quote(str(output_dir))}",
    ]
    for key in sorted(env):
        lines.append(shell_export(key, env[key]))
    lines.append(shell_export("OUT_DIR", output_dir))
    lines.append(f"bash {shlex.quote(str(run_script))}")
    launch_path.write_text("\n".join(lines) + "\n")
    launch_path.chmod(0o755)
    return launch_path


def sbatch_line(flag: str, value: Any | None = None) -> str:
    if value is None:
        return f"#SBATCH {flag}"
    if flag.startswith("--"):
        return f"#SBATCH {flag}={value}"
    return f"#SBATCH {flag} {value}"


def write_sbatch_script(run_dir: Path, launch_script: Path, slurm: dict[str, Any]) -> Path:
    job_path = run_dir / "job.sbatch"
    out_path = run_dir / "slurm-%j.out"
    err_path = run_dir / "slurm-%j.err"

    lines = ["#!/usr/bin/env bash"]
    if slurm.get("account"):
        lines.append(sbatch_line("-A", slurm["account"]))
    if slurm.get("partition"):
        lines.append(sbatch_line("-p", slurm["partition"]))
    if slurm.get("nodes"):
        lines.append(sbatch_line("-N", slurm["nodes"]))
    if slurm.get("gpus_per_node"):
        lines.append(sbatch_line("--gpus-per-node", slurm["gpus_per_node"]))
    if slurm.get("exclusive"):
        lines.append(sbatch_line("--exclusive"))
    if slurm.get("cpus_per_task"):
        lines.append(sbatch_line("--cpus-per-task", slurm["cpus_per_task"]))
    if slurm.get("mem") is not None:
        lines.append(sbatch_line("--mem", slurm["mem"]))
    if slurm.get("time"):
        lines.append(sbatch_line("-t", slurm["time"]))
    if slurm.get("job_name"):
        lines.append(sbatch_line("-J", slurm["job_name"]))
    lines.append(sbatch_line("-o", out_path))
    lines.append(sbatch_line("-e", err_path))
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"bash {shlex.quote(str(launch_script))}",
        ]
    )
    job_path.write_text("\n".join(lines) + "\n")
    job_path.chmod(0o755)
    return job_path


def write_metadata(
    run_dir: Path,
    candidate_path: Path,
    data: dict[str, Any],
    mode: str,
    source_root: Path,
    runtime_root: Path,
    source_commit: str | None,
    runtime_commit: str | None,
    current_commit: str | None,
    launch_script: Path,
    job_script: Path,
    artifact_paths: dict[str, str],
) -> None:
    metadata = {
        "candidate_id": data["id"],
        "kind": data.get("kind"),
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": str(candidate_path),
        "run_dir": str(run_dir),
        "submodule": str(source_root),
        "runtime_root": str(runtime_root),
        "base_commit": data.get("base_commit"),
        "source_commit": source_commit,
        "runtime_commit": runtime_commit,
        "current_commit": current_commit,
        "launch_script": str(launch_script),
        "job_script": str(job_script),
        "artifact_contract": artifact_paths,
        "status": "prepared",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def update_metadata(run_dir: Path, updates: dict[str, Any]) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(updates)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def parse_sbatch_job_id(stdout: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\S+)", stdout)
    if not match:
        return None
    return match.group(1)


def is_scored_candidate(data: dict[str, Any]) -> bool:
    kind = str(data.get("kind", "")).lower()
    if kind == "baseline":
        return False

    slurm = data.get("slurm", {})
    job_name = slurm.get("job_name", "") if isinstance(slurm, dict) else ""
    label = f"{data.get('id', '')} {job_name}".lower()
    non_scored_markers = (
        "baseline_off",
        "trace_off",
        "off_identity",
        "warm_ctrl",
        "warm-ctrl",
        "control",
        "ctrl",
        "profile",
        "profiler",
    )
    return not any(marker in label for marker in non_scored_markers)


def recorded_run_dirs(root: Path, status_path: Path) -> set[Path]:
    try:
        status = json.loads(status_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

    recorded: set[Path] = set()
    for collection in (
        "candidates",
        "frontier_candidates",
        "discarded_candidates",
        "rejected_candidates",
    ):
        for record in status.get(collection, []):
            if not isinstance(record, dict) or not record.get("run_dir"):
                continue
            path = Path(str(record["run_dir"]))
            recorded.add((path if path.is_absolute() else root / path).resolve())
    return recorded


def load_manifest_for_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    manifest = metadata.get("candidate_manifest")
    if manifest:
        manifest_path = Path(str(manifest))
        if manifest_path.exists():
            return load_toml(manifest_path)
    return {"id": metadata.get("candidate_id", ""), "kind": metadata.get("kind", "")}


def enforce_single_flight_or_exit(args: argparse.Namespace, data: dict[str, Any]) -> None:
    if os.environ.get("AUTO_VIDEO_DISABLE_SINGLE_FLIGHT_GUARD") == "1":
        return
    if args.mode != "sbatch" or not args.confirm_submit:
        return

    root = repo_root()
    status_path = root / "AGENT-STATUS.json"
    if not status_path.exists():
        return

    runs_root = (root / args.run_root).resolve()
    if not runs_root.exists():
        return

    recorded = recorded_run_dirs(root, status_path)
    nonblocking_statuses = {
        "prepared",
        "failed",
        "submission_failed",
        "canceled",
        "canceled_by_watchdog",
    }
    candidate_id = str(data.get("id", ""))
    current_is_scored = is_scored_candidate(data)
    blockers: list[str] = []
    for metadata_path in sorted(runs_root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        run_dir = Path(str(metadata.get("run_dir") or metadata_path.parent))
        run_dir = (run_dir if run_dir.is_absolute() else root / run_dir).resolve()
        if run_dir in recorded:
            continue
        if metadata.get("status") in nonblocking_statuses:
            continue
        existing_data = load_manifest_for_metadata(metadata)
        if str(existing_data.get("id", metadata.get("candidate_id", ""))) == candidate_id:
            blockers.append(
                f"{run_dir} status={metadata.get('status')} job={metadata.get('slurm_job_id')}"
            )
            continue
        if current_is_scored and is_scored_candidate(existing_data):
            blockers.append(
                f"{run_dir} status={metadata.get('status')} job={metadata.get('slurm_job_id')}"
            )

    if blockers:
        raise SystemExit(
            "Refusing to submit because this fanout dimension has active or "
            "unrecorded run(s) that would violate single-flight launch control. "
            "Gate and record scored runs with "
            "tools/symposium/loop_control.py before launching another scored "
            "candidate; do not duplicate controls. Set "
            "AUTO_VIDEO_DISABLE_SINGLE_FLIGHT_GUARD=1 only for an explicit "
            "orchestrator-approved override.\n- "
            + "\n- ".join(blockers)
        )


def prepare_run(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = repo_root()
    candidate_path = Path(args.candidate).resolve()
    data = load_toml(candidate_path)

    for field in ("id", "kind", "submodule", "run_script"):
        if field not in data:
            raise SystemExit(f"Missing required field: {field}")

    candidate_id = sanitize_id(str(data["id"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{sanitize_id(args.name_suffix)}" if args.name_suffix else ""
    run_dir = (root / args.run_root / f"{stamp}-{candidate_id}{suffix}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    source_root = (root / str(data["submodule"])).resolve()
    allow_missing_runtime = args.mode == "dry-run"
    if not source_root.exists() and not allow_missing_runtime:
        raise SystemExit(f"Submodule path does not exist: {source_root}")

    runtime_config = data.get("runtime", {})
    runtime_root_raw = args.runtime_root or runtime_config.get("root")
    if runtime_root_raw:
        runtime_root_path = Path(str(runtime_root_raw)).expanduser()
        runtime_root = (
            runtime_root_path
            if runtime_root_path.is_absolute()
            else root / runtime_root_path
        ).resolve()
    else:
        runtime_root = source_root
    if not runtime_root.exists() and not allow_missing_runtime:
        raise SystemExit(f"Runtime root does not exist: {runtime_root}")

    run_script = (runtime_root / str(data["run_script"])).resolve()
    if not run_script.exists() and not allow_missing_runtime:
        raise SystemExit(f"Run script does not exist: {run_script}")

    artifacts = data.get("artifacts", {})
    output_dir = (run_dir / artifacts.get("output_dir", "outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        key: str(output_dir / artifacts.get(key, default))
        for key, default in CANONICAL_ARTIFACT_DEFAULTS.items()
    }

    env = dict(data.get("env", {}))
    for item in args.env or []:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value

    source_commit = run_git_commit(source_root)
    runtime_commit = run_git_commit(runtime_root)
    current_commit = runtime_commit or source_commit
    expected_commit = data.get("base_commit")
    if expected_commit and current_commit and expected_commit != current_commit:
        message = (
            f"Warning: runtime/source is at {current_commit}, "
            f"manifest expects {expected_commit}"
        )
        if args.strict_commit:
            raise SystemExit(message)
        print(message, file=sys.stderr)

    resolved = {
        "run_id": run_dir.name,
        "repo_root": root,
        "run_dir": run_dir,
        "submodule_root": source_root,
        "runtime_root": runtime_root,
        "run_script": run_script,
        "output_dir": output_dir,
        "mode": args.mode,
        "source_commit": source_commit or "",
        "runtime_commit": runtime_commit or "",
        "current_commit": current_commit or "",
    }
    write_resolved_manifest(candidate_path, run_dir, data, resolved)
    launch_script = write_launch_script(run_dir, runtime_root, run_script, env, output_dir)
    job_script = write_sbatch_script(run_dir, launch_script, data.get("slurm", {}))
    write_metadata(
        run_dir,
        candidate_path,
        data,
        args.mode,
        source_root,
        runtime_root,
        source_commit,
        runtime_commit,
        current_commit,
        launch_script,
        job_script,
        artifact_paths,
    )
    return run_dir, launch_script, job_script, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="Path to a candidate TOML manifest")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "local", "sbatch"),
        default="dry-run",
        help="Prepare only, execute locally, or submit with sbatch",
    )
    parser.add_argument("--run-root", default="runs", help="Run bundle root")
    parser.add_argument("--name-suffix", default="", help="Optional run ID suffix")
    parser.add_argument(
        "--runtime-root",
        help="Optional execution checkout root. Defaults to the manifest submodule path.",
    )
    parser.add_argument(
        "--env",
        action="append",
        help="Override or add an exported environment variable, KEY=VALUE",
    )
    parser.add_argument(
        "--strict-commit",
        action="store_true",
        help="Fail if the submodule commit differs from the manifest base_commit",
    )
    parser.add_argument(
        "--confirm-submit",
        action="store_true",
        help="Actually submit when --mode sbatch is selected. Without this flag, sbatch mode renders only.",
    )
    args = parser.parse_args()

    candidate_data = load_toml(Path(args.candidate).resolve())
    enforce_single_flight_or_exit(args, candidate_data)

    run_dir, launch_script, job_script, data = prepare_run(args)
    print(f"candidate: {data['id']}")
    print(f"run_dir: {run_dir}")
    print(f"launch_script: {launch_script}")
    print(f"job_script: {job_script}")

    if args.mode == "dry-run":
        print("status: prepared (dry-run; no GPU work submitted)")
        return 0
    if args.mode == "local":
        return subprocess.call([str(launch_script)])
    if args.mode == "sbatch":
        if not args.confirm_submit:
            print("status: prepared (sbatch script rendered; no job submitted)")
            print("hint: pass --confirm-submit to submit the job")
            return 0
        proc = subprocess.run(
            ["sbatch", str(job_script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout:
            print(proc.stdout.strip())
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
        if proc.returncode == 0:
            job_id = parse_sbatch_job_id(proc.stdout)
            update_metadata(
                run_dir,
                {
                    "status": "submitted",
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "slurm_job_id": job_id,
                    "sbatch_stdout": proc.stdout.strip(),
                    "sbatch_stderr": proc.stderr.strip(),
                },
            )
        else:
            update_metadata(
                run_dir,
                {
                    "status": "failed",
                    "submission_failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sbatch_stdout": proc.stdout.strip(),
                    "sbatch_stderr": proc.stderr.strip(),
                    "sbatch_returncode": proc.returncode,
                },
            )
        return proc.returncode
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
