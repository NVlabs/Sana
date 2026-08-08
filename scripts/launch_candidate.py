#!/usr/bin/env python3
"""Prepare and optionally launch an autovideo candidate.

The script is intentionally small and dependency-free. It reads a TOML
candidate manifest, creates a run bundle under runs/, writes the exact launch
shell script, and either stops at dry-run, executes locally, or submits Slurm.
"""

from __future__ import annotations

import argparse
import hashlib
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

REPO_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_FOR_IMPORT))

from techniques.candidate_manifest import (  # noqa: E402
    dry_run_manifest,
    load_model_profile as load_efficiency_model_profile,
    manifest_id,
    model_spec_key,
    write_dry_run,
)


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
NONBLOCKING_RUN_STATUSES = {
    "prepared",
    "failed",
    "submission_failed",
    "canceled",
    "cancelled",
    "canceled_by_watchdog",
    "cancelled_by_watchdog",
    "canceled_by_orchestrator_release",
    "cancelled_by_orchestrator_release",
}
COSMOS3_UNSUPPORTED_GPU_REASONS = {
    "te_recipe_variant": (
        "Cosmos3 already has an online ModelOpt FP4 linear consumer, but this "
        "TE recipe row still has no Cosmos3-equivalent bias-free SwiGLU fused "
        "epilogue adapter. The LTX2 GELU/bias-gate fused flags are disabled in "
        "the manifest; only run a TE fused-epilogue diagnostic after adding and "
        "validating a Cosmos3 adapter"
    ),
    "env_flag_kwl_bundle": (
        "Cosmos3 already uses several pure KWL-style kernels as baseline, while "
        "LTX2-only KWL flags such as audio/VAE/guidance sharing do not map to "
        "Cosmos3; this bundle does not create an isolated candidate delta"
    ),
    "gemm_epilogue_fusion": (
        "Cosmos3 already fuses gate/up projection and SiluAndMul in its MLP, "
        "while the LTX2 bias+GELU and residual-gate epilogue flags do not map "
        "to Cosmos3's SwiGLU/no-bias MLP and fused add+RMSNorm residual path; "
        "those replay flags are disabled in the manifest"
    ),
    "norm_modulation_residual_fusion": (
        "Cosmos3 GEN already uses fused add+RMSNorm and does not have LTX2-style "
        "AdaLN/modulation semantics for these flags; those replay flags are "
        "disabled in the manifest"
    ),
    "layout_copy_elimination": (
        "Cosmos3 already keeps some layout-friendly RoPE/MLP dataflow, and this "
        "row's old LTX2 sharing flags do not identify an extra Cosmos3 change; "
        "those replay flags are disabled in the manifest"
    ),
}
COSMOS3_DYNAMIC_GPU_READINESS_CANDIDATES = {
    "semantic_permutation",
}
SEMANTIC_PERMUTATION_REQUIRED_MODULES = ("svg", "flashinfer", "cuvs")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def candidate_id(data: dict[str, Any]) -> str:
    return manifest_id(data) or "candidate"


def sanitize_id(value: str) -> str:
    cleaned = VALID_ID.sub("-", value.strip())
    return cleaned.strip("-") or "candidate"


def default_slurm(data: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    official = data.get("official_config") or profile.get("official_config") or {}
    return {
        "account": "nvr_elm_llm",
        "partition": "batch",
        "nodes": 1,
        "gpus_per_node": official.get("num_gpus", 4),
        "cpus_per_task": 64,
        "mem": "0",
        "time": "04:00:00",
        "job_name": f"autovideo-{sanitize_id(candidate_id(data))}",
        "exclusive": True,
    }


def merge_model_profile(data: dict[str, Any]) -> dict[str, Any]:
    profile = load_efficiency_model_profile(data, repo_root())
    if not profile:
        return data

    merged = dict(data)
    for key in ("submodule", "base_commit", "run_script", "eval_profile", "official_config"):
        if key not in merged and key in profile:
            merged[key] = profile[key]
    merged["env"] = {**profile.get("env", {}), **data.get("env", {})}
    merged["slurm"] = {
        **default_slurm(merged, profile),
        **profile.get("slurm", {}),
        **data.get("slurm", {}),
    }
    return merged


def is_cosmos3_candidate(data: dict[str, Any]) -> bool:
    try:
        spec = model_spec_key(data, repo_root())
    except Exception:
        spec = ""
    if spec.lower() == "cosmos3":
        return True

    env = data.get("env", {})
    model_repo = env.get("MODEL_REPO", "") if isinstance(env, dict) else ""
    run_script = str(data.get("run_script", ""))
    return "cosmos3" in f"{model_repo} {run_script}".lower()


def cosmos3_gpu_readiness_blocker(data: dict[str, Any], mode: str) -> str | None:
    if mode == "dry-run":
        return None
    if not is_cosmos3_candidate(data):
        return None

    reason = COSMOS3_UNSUPPORTED_GPU_REASONS.get(candidate_id(data))
    if reason:
        return (
            f"Refusing to run unsupported Cosmos3 GPU candidate {candidate_id(data)!r}: "
            f"{reason}. The candidate remains valid for manifest/dry-run/public-alignment "
            "audits, but a Cosmos3 GPU job would not prove the intended public-reference "
            "technique. Pass --allow-unsupported-gpu only for an explicit diagnostic "
            "env/export run."
        )

    if candidate_id(data) == "semantic_permutation":
        env = data.get("env", {}) if isinstance(data.get("env"), dict) else {}
        runtime_python = resolve_runtime_python(data, env)
        missing = missing_python_modules(
            runtime_python,
            SEMANTIC_PERMUTATION_REQUIRED_MODULES,
            str(env.get("SGLANG_HQ_EXTRA_PYTHONPATH", "") or ""),
        )
        if missing:
            return (
                "Refusing to run Cosmos3 GPU candidate 'semantic_permutation': "
                f"the Sparse VideoGen2/SAP consumer is wired, but runtime python "
                f"{runtime_python!r} is missing required module(s): "
                f"{', '.join(missing)}. Install Sparse-VideoGen plus its "
                "FlashInfer/cuVS runtime dependencies, then rerun without "
                "--allow-unsupported-gpu."
            )
        return None

    return None


def cosmos3_blocked_candidate_ids() -> set[str]:
    blocked = set(COSMOS3_UNSUPPORTED_GPU_REASONS)
    candidates_root = repo_root() / "candidates"
    try:
        paths = sorted(candidates_root.glob("*/*.toml"))
        for path in paths:
            data = merge_model_profile(load_toml(path))
            cid = candidate_id(data)
            if cid not in COSMOS3_DYNAMIC_GPU_READINESS_CANDIDATES:
                continue
            if cosmos3_gpu_readiness_blocker(data, "local"):
                blocked.add(cid)
    except Exception:
        blocked.update(COSMOS3_DYNAMIC_GPU_READINESS_CANDIDATES)
    return blocked


def python_module_available(
    python: str, module: str, extra_pythonpath: str = ""
) -> bool:
    env = None
    if extra_pythonpath:
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{extra_pythonpath}:{env.get('PYTHONPATH', '')}"
            if env.get("PYTHONPATH")
            else extra_pythonpath
        )
    try:
        proc = subprocess.run(
            [
                python,
                "-c",
                (
                    "import importlib.util, sys; "
                    f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
                ),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def missing_python_modules(
    python: str, modules: tuple[str, ...], extra_pythonpath: str = ""
) -> list[str]:
    return [
        module
        for module in modules
        if not python_module_available(python, module, extra_pythonpath)
    ]


def enforce_gpu_readiness_or_exit(args: argparse.Namespace, data: dict[str, Any]) -> None:
    if getattr(args, "allow_unsupported_gpu", False):
        return
    blocker = cosmos3_gpu_readiness_blocker(data, args.mode)
    if blocker:
        raise SystemExit(blocker)


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


def vendored_snapshot_identity(path: Path) -> str | None:
    snapshot_path = path / "SOURCE_SNAPSHOT.json"
    if not snapshot_path.is_file():
        return None
    try:
        snapshot = json.loads(snapshot_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid vendored source snapshot: {snapshot_path}: {exc}") from exc
    core_hashes = snapshot.get("core_sha256", {})
    if not isinstance(core_hashes, dict) or not core_hashes:
        raise SystemExit(f"Vendored source snapshot has no core_sha256 entries: {snapshot_path}")
    source_root = (path / str(snapshot.get("source_root", "lingbot_src"))).resolve()
    try:
        source_root.relative_to(path.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"Vendored source_root escapes runtime root: {source_root}"
        ) from exc
    for relative, expected in core_hashes.items():
        source_path = source_root / str(relative)
        if not source_path.is_file():
            raise SystemExit(f"Vendored source file is missing: {source_path}")
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual != str(expected):
            raise SystemExit(
                f"Vendored source integrity mismatch for {source_path}: "
                f"expected {expected}, got {actual}"
            )
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    return f"snapshot:{digest}"


def toml_string(value: Any) -> str:
    return json.dumps(str(value))


def toml_literal(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        items = [toml_literal(item) for item in value]
        if any(item is None for item in items):
            return None
        return "[" + ", ".join(item for item in items if item is not None) + "]"
    return None


def append_toml_table(lines: list[str], name: str, values: dict[str, Any]) -> None:
    serializable = []
    for key, value in values.items():
        literal = toml_literal(value)
        if literal is not None:
            serializable.append((str(key), literal))
    if not serializable:
        return
    lines.extend(["", f"[{name}]"])
    for key, literal in serializable:
        lines.append(f"{key} = {literal}")


def redact_resolved_env(env: dict[str, Any]) -> dict[str, Any]:
    sensitive = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {
        key: "<redacted>" if any(marker in key.upper() for marker in sensitive) else value
        for key, value in env.items()
    }


def write_resolved_manifest(
    candidate_path: Path,
    run_dir: Path,
    data: dict[str, Any],
    resolved: dict[str, Any],
    effective_env: dict[str, Any],
) -> None:
    original = candidate_path.read_text()
    lines = [original.rstrip(), "", "[resolved]"]
    for key, value in resolved.items():
        lines.append(f"{key} = {toml_string(value)}")
    resolved_profile = {
        key: data[key]
        for key in ("model_profile", "submodule", "base_commit", "run_script", "eval_profile")
        if key in data
    }
    append_toml_table(lines, "resolved_profile", resolved_profile)
    append_toml_table(
        lines,
        "resolved_profile.official_config",
        data.get("official_config", {}) if isinstance(data.get("official_config"), dict) else {},
    )
    append_toml_table(lines, "resolved_profile.env", redact_resolved_env(effective_env))
    append_toml_table(
        lines,
        "resolved_profile.slurm",
        data.get("slurm", {}) if isinstance(data.get("slurm"), dict) else {},
    )
    (run_dir / "manifest.resolved.toml").write_text("\n".join(lines) + "\n")


def write_launch_script(
    run_dir: Path,
    sol_root: Path,
    run_script: Path,
    env: dict[str, Any],
    output_dir: Path,
) -> Path:
    launch_path = run_dir / "launch.sh"
    started_path = run_dir / "job-started.json"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        (
            "printf '{\"schema_version\":1,\"job_id\":\"%s\",\"started_at_utc\":\"%s\","
            "\"hostname\":\"%s\"}\\n' "
            '"${SLURM_JOB_ID:-local}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" '
            f"> {shlex.quote(str(started_path))}"
        ),
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
    lines.append(sbatch_line("--export", "ALL"))
    lines.append(sbatch_line("-o", out_path))
    lines.append(sbatch_line("-e", err_path))
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "export SHELL=/usr/bin/env bash",
            f"exec /usr/bin/env bash {shlex.quote(str(launch_script))}",
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
    runtime_python: str,
    candidate_dry_run: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "candidate_id": candidate_id(data),
        "kind": data.get("kind"),
        "purpose": candidate_purpose(data),
        "mode": mode,
        "created_at_utc": now,
        "candidate_manifest": str(candidate_path),
        "run_dir": str(run_dir),
        "submodule": str(source_root),
        "runtime_root": str(runtime_root),
        "runtime_python": runtime_python,
        "base_commit": data.get("base_commit"),
        "eval_profile": data.get("eval_profile"),
        "official_config": data.get("official_config", {}),
        "source_commit": source_commit,
        "runtime_commit": runtime_commit,
        "current_commit": current_commit,
        "launch_script": str(launch_script),
        "job_script": str(job_script),
        "artifact_contract": artifact_paths,
        "status": "prepared",
        "status_history": [{"status": "prepared", "at_utc": now, "reason": "bundle_created"}],
    }
    if candidate_dry_run is not None:
        metadata["candidate_dry_run"] = candidate_dry_run
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def update_metadata(run_dir: Path, updates: dict[str, Any]) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    old_status = metadata.get("status")
    metadata.update(updates)
    new_status = metadata.get("status")
    if new_status and new_status != old_status:
        history = metadata.setdefault("status_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "status": new_status,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": updates.get("status_reason", ""),
                }
            )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def parse_sbatch_job_id(stdout: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\S+)", stdout)
    if not match:
        return None
    return match.group(1)


def is_scored_candidate(data: dict[str, Any]) -> bool:
    purpose = str(data.get("purpose", "")).lower()
    if purpose in {"control", "evidence", "blocker_probe", "unsafe_probe"}:
        return False
    kind = str(data.get("kind", "")).lower()
    if kind == "baseline":
        return False

    slurm = data.get("slurm", {})
    job_name = slurm.get("job_name", "") if isinstance(slurm, dict) else ""
    label = f"{candidate_id(data)} {job_name}".lower()
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
    return {
        "id": metadata.get("candidate_id", ""),
        "kind": metadata.get("kind", ""),
        "purpose": metadata.get("purpose", ""),
    }


def candidate_purpose(data: dict[str, Any]) -> str:
    raw = str(data.get("purpose") or "").strip().lower()
    if raw:
        return raw
    kind = str(data.get("kind", "")).lower()
    if kind == "baseline":
        return "control"
    label = f"{candidate_id(data)} {data.get('description', '')}".lower()
    if "upper-bound" in label or "upper_bound" in label:
        return "blocker_probe"
    return "delivery"


def resolve_runtime_python(data: dict[str, Any], env: dict[str, Any]) -> str:
    runtime = data.get("runtime", {})
    if isinstance(runtime, dict) and runtime.get("python"):
        return str(runtime["python"])
    for key in ("PYTHON_BIN", "SGLANG_HQ_RUNTIME_PYTHON", "RUNTIME_PYTHON"):
        if env.get(key):
            return str(env[key])
    return sys.executable


def validate_runtime_python(value: str, mode: str) -> None:
    if not value:
        return
    path = Path(value).expanduser()
    if path.is_absolute() or "/" in value:
        if not path.exists():
            if mode == "dry-run":
                print(f"Warning: runtime python does not exist yet: {path}", file=sys.stderr)
                return
            raise SystemExit(f"Runtime python does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise SystemExit(f"Runtime python is not executable: {path}")


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
    current_candidate_id = candidate_id(data)
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
        if metadata.get("status") in NONBLOCKING_RUN_STATUSES:
            continue
        existing_data = load_manifest_for_metadata(metadata)
        existing_candidate_id = candidate_id(existing_data) or str(
            metadata.get("candidate_id", "")
        )
        if existing_candidate_id == current_candidate_id:
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
    data = merge_model_profile(load_toml(candidate_path))

    for field in ("id", "kind", "submodule", "run_script"):
        if field not in data:
            raise SystemExit(f"Missing required field: {field}")

    safe_candidate_id = sanitize_id(candidate_id(data))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{sanitize_id(args.name_suffix)}" if args.name_suffix else ""
    run_dir = (root / args.run_root / f"{stamp}-{safe_candidate_id}{suffix}").resolve()
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

    candidate_dry_run = None
    try:
        candidate_dry_run = dry_run_manifest(data, root)
    except Exception as exc:
        raise SystemExit(f"Candidate dry-run validation failed: {exc}") from exc

    env = {}
    if candidate_dry_run:
        env.update(candidate_dry_run.get("env_preview", {}))
    env.update(data.get("env", {}))
    env.setdefault("AUTOVIDEO_REPO_ROOT", str(root))
    env.setdefault("AUTOVIDEO_RUNTIME_ROOT", str(runtime_root))
    for item in args.env or []:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    # Immutable run identity: candidate-specific wrappers may use this to reject
    # CLI env overrides that would otherwise execute a different topology under
    # the same candidate label.
    env["AUTOVIDEO_CANDIDATE_ID"] = candidate_id(data)
    runtime_python = resolve_runtime_python(data, env)
    validate_runtime_python(runtime_python, args.mode)

    source_commit = run_git_commit(source_root) or vendored_snapshot_identity(source_root)
    runtime_commit = run_git_commit(runtime_root) or vendored_snapshot_identity(runtime_root)
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
    write_resolved_manifest(candidate_path, run_dir, data, resolved, env)
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
        runtime_python,
        candidate_dry_run,
    )
    write_dry_run(run_dir / "candidate_dry_run.json", candidate_dry_run)
    return run_dir, launch_script, job_script, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="Path to a candidate TOML manifest")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "local", "sbatch", "slurm"),
        default="dry-run",
        help="Prepare only, execute locally, or submit with sbatch/slurm",
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
    parser.add_argument(
        "--allow-unsupported-gpu",
        action="store_true",
        help=(
            "Allow local/sbatch launch for a Cosmos3 candidate whose current runtime "
            "does not consume the advertised optimization. Use only for explicit "
            "diagnostic env/export checks."
        ),
    )
    args = parser.parse_args()
    if args.mode == "slurm":
        args.mode = "sbatch"

    candidate_data = merge_model_profile(load_toml(Path(args.candidate).resolve()))
    enforce_gpu_readiness_or_exit(args, candidate_data)
    enforce_single_flight_or_exit(args, candidate_data)

    run_dir, launch_script, job_script, data = prepare_run(args)
    print(f"candidate: {candidate_id(data)}")
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
