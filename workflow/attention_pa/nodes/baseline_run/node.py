#!/usr/bin/env python3
"""Workflow-local launcher/monitor for one canonical baseline run."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


STATE_NAME = "baseline-run.json"
LOCK_NAME = "BASELINE-LOCK.json"
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"}
STARTUP_WATCH_STATES = {"RUNNING", "CONFIGURING"}
RETRYABLE_TERMINAL_STATES = {"TIMEOUT", "CANCELLED", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "REVOKED"}
BASELINE_ENV = {
    "SANA_VIDEO_SAMPLE_NUMS": "5",
    "FORWARD_CACHE_METHOD": "none",
    "SANA_INTEGRATED_KERNEL": "0",
    "SANA_INTEGRATED_PISA": "0",
    "SANA_INTEGRATED_CACHE": "0",
    "SANA_PISA_ENABLED": "0",
    "SANA_OPT_ROPE_FP32": "0",
    "SANA_OPT_ROPE_FLASHATTN": "0",
    "SANA_OPT_RMSNORM_LIGER": "0",
    "SANA_OPT_RMSNORM_NATIVE": "0",
    "SANA_OPT_LINEAR_ATTN": "0",
    "SANA_OPT_LINEAR_ATTN_COMPILE": "0",
    "SANA_OPT_SWIGLU_FUSED": "0",
    "SANA_OPT_SWIGLU_PACKED_LINEAR": "0",
    "SANA_OPT_SOFTMAX_ATTN_BF16_CORE": "0",
    "SANA_OPT_LINEAR_QK_NORM_ROPE_FUSED": "0",
    "SANA_OPT_SOFTMAX_QK_NORM_ROPE_FUSED": "0",
    "SANA_OPT_LINEAR_O_NORM_GATE_FUSED": "0",
    "SANA_OPT_ATTNRES_ATTEND_FUSED": "0",
    "SANA_OPT_ATTNRES_VALUE_KEY_COMMIT_FUSED": "0",
    "SANA_OPT_FFN_DOWN_GATE_RESIDUAL_FUSED": "0",
    "SANA_OPT_FFN_NORM2_MODULATION_FUSED": "0",
    "SANA_OPT_CROSS_QK_RMSNORM_FUSED": "0",
    "SANA_OPT_CROSS_ATTN_MASK_BROADCAST": "0",
    "SANA_OPT_LINEAR_BETA_OUTPUT_GATE_ALIGNED_PACKED": "0",
    "SANA_OPT_LINEAR_ATTN_TF32X3_NATIVE": "0",
    "SANA_OPT_LINEAR_ATTN_TF32X3_NORMALIZER_OUT_FUSED": "0",
    "SANA_OPT_FFN_DOWN_CUTLASS_GATE_DUAL_RESIDUAL_EPILOGUE": "0",
    "SANA_OPT_FFN_DOWN_NATIVE_TRITON_DUAL_RESIDUAL_EPILOGUE": "0",
    "SANA_ATTNRES_CONTEXT_CACHE": "0",
    "SANA_ATTNRES_REUSE_BUFFERS": "0"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def age_seconds(raw: str) -> float:
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - value).total_seconds(), 0.0)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve(ctx: NodeContext, raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ctx.worktree / path).resolve()


def video_paths(run_dir: Path) -> list[Path]:
    return [path for path in (run_dir / "outputs").rglob("*.mp4") if path.is_file()]


def outputs_ready(run_dir: Path) -> bool:
    return (
        (run_dir / "outputs" / "benchmark.json").is_file()
        and (run_dir / "outputs" / "run_config.json").is_file()
        and len(video_paths(run_dir)) >= 5
    )


def canonical_baseline_source(ctx: NodeContext) -> Path | None:
    raw = str((ctx.env or {}).get("CANONICAL_BASELINE_RUN") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ctx.root / path).resolve()


def import_canonical_baseline(ctx: NodeContext, state_path: Path, source: Path) -> NodeResult:
    if not outputs_ready(source):
        reason = f"canonical_baseline_source_incomplete:{source}"
        return NodeResult("infra_blocked", updates={"baseline_reason": reason}, message=reason)
    destination = ctx.worktree / "runs" / "canonical-baseline-import"
    outputs = destination / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    for name in ("benchmark.json", "run_config.json"):
        shutil.copy2(source / "outputs" / name, outputs / name)
    source_videos = [
        path
        for path in video_paths(source)
        if path.resolve() != (source / "outputs" / "out.mp4").resolve()
    ]
    if len(source_videos) < 5:
        reason = f"canonical_baseline_unique_source_videos_incomplete:{len(source_videos)}"
        return NodeResult("infra_blocked", updates={"baseline_reason": reason}, message=reason)
    video_dir = outputs / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(sorted(source_videos)[:5]):
        shutil.copy2(path, video_dir / f"{index:03d}.mp4")
    shutil.copy2(video_dir / "000.mp4", outputs / "out.mp4")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        "run_dir": str(destination.resolve()),
        "slurm_job_id": "",
        "attempt": 0,
        "attempts": [],
        "imported_from": str(source),
        "effective_off_env": BASELINE_ENV,
    }
    write_json(state_path, payload)
    return NodeResult(
        "completed",
        updates={"baseline_run": ctx.rel_to_worktree(destination), "baseline_imported_from": str(source)},
        artifacts=[ctx.rel_to_worktree(state_path)],
        message="canonical_baseline_imported",
    )


def job_started(run_dir: Path) -> bool:
    marker = run_dir / "job-started.json"
    return marker.is_file() and marker.stat().st_size > 0


def mark_run_status(run_dir: Path, status: str, reason: str) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if not metadata:
        return
    metadata.update({"status": status, "status_reason": reason, "status_updated_at_utc": utc_now()})
    write_json(metadata_path, metadata)


def cancel_job(job_id: str) -> bool:
    scancel = shutil.which("scancel")
    if not scancel or not job_id:
        return False
    proc = subprocess.run(
        [scancel, job_id],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def slurm_state(job_id: str) -> str:
    if not job_id:
        return "UNKNOWN"
    sacct = shutil.which("sacct")
    if sacct:
        proc = subprocess.run(
            [sacct, "-j", job_id, "--format=State", "-n", "-P"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        states = [line.split("|", 1)[0].strip().split("+", 1)[0] for line in proc.stdout.splitlines()]
        states = [state for state in states if state]
        if states:
            return states[0]
    squeue = shutil.which("squeue")
    if squeue:
        proc = subprocess.run(
            [squeue, "-h", "-j", job_id, "-o", "%T"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        state = proc.stdout.strip().splitlines()
        if state:
            return state[0].strip().upper()
    return "UNKNOWN"


def launch(
    ctx: NodeContext,
    state_path: Path,
    previous: dict[str, Any] | None = None,
    retry_reason: str = "",
) -> NodeResult:
    manifest = resolve(ctx, str(ctx.config.get("baseline_manifest") or "candidates/sana_video_baseline.toml"))
    launcher = ctx.worktree / "scripts" / "launch_candidate.py"
    if not manifest.is_file() or not launcher.is_file():
        reason = "baseline_manifest_or_launcher_missing"
        return NodeResult("infra_blocked", message=reason, updates={"baseline_reason": reason})
    mode = "dry-run" if ctx.dry_run else "sbatch"
    attempt = int((previous or {}).get("attempt") or 0) + 1
    cmd = [
        sys.executable,
        str(launcher),
        str(manifest),
        "--mode",
        mode,
        "--run-root",
        "runs",
        "--name-suffix",
        f"canonical-{ctx.state.get('experiment_uid', 'experiment')}-attempt{attempt}",
    ]
    if not ctx.dry_run:
        cmd.append("--confirm-submit")
    for key, value in sorted(BASELINE_ENV.items()):
        cmd.extend(["--env", f"{key}={value}"])
    proc = subprocess.run(
        cmd,
        cwd=ctx.worktree,
        env=ctx.env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    match = re.search(r"^run_dir:\s*(.+)$", proc.stdout, flags=re.MULTILINE)
    run_dir = Path(match.group(1)).resolve() if match else Path()
    metadata = read_json(run_dir / "metadata.json") if match else {}
    attempts = list((previous or {}).get("attempts") or [])
    if previous:
        attempts.append(
            {
                "attempt": previous.get("attempt"),
                "run_dir": previous.get("run_dir"),
                "slurm_job_id": previous.get("slurm_job_id"),
                "observed_slurm_state": previous.get("observed_slurm_state"),
                "retry_reason": retry_reason,
                "closed_at_utc": utc_now(),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "prepared" if ctx.dry_run else "submitted",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir) if match else "",
        "slurm_job_id": str(metadata.get("slurm_job_id") or ""),
        "submitted_at_utc": str(metadata.get("submitted_at_utc") or utc_now()),
        "attempt": attempt,
        "attempts": attempts,
        "last_retry_reason": retry_reason,
        "command": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "returncode": proc.returncode,
        "effective_off_env": BASELINE_ENV,
    }
    write_json(state_path, payload)
    if proc.returncode != 0 or not match:
        reason = "baseline_submission_failed"
        return NodeResult("infra_blocked", updates={"baseline_reason": reason}, artifacts=[ctx.rel_to_worktree(state_path)], message=reason)
    return NodeResult(
        "waiting",
        updates={"baseline_run": ctx.rel_to_worktree(run_dir), "baseline_job_id": payload["slurm_job_id"]},
        artifacts=[ctx.rel_to_worktree(state_path)],
        message=("baseline_prepared" if ctx.dry_run else ("baseline_resubmitted" if previous else "baseline_submitted")),
    )


def run(ctx: NodeContext) -> NodeResult:
    lock = ctx.worktree / LOCK_NAME
    if lock.is_file():
        return NodeResult("completed", updates={"baseline_lock": ctx.rel_to_worktree(lock)}, artifacts=[ctx.rel_to_worktree(lock)], message="baseline_already_locked")
    state_path = ctx.worktree / "state" / STATE_NAME
    state = read_json(state_path)
    if not state:
        canonical = canonical_baseline_source(ctx)
        if canonical is not None:
            return import_canonical_baseline(ctx, state_path, canonical)
        return launch(ctx, state_path)
    run_raw = str(state.get("run_dir") or "")
    run_dir = Path(run_raw).resolve() if run_raw else Path()
    if run_raw and outputs_ready(run_dir):
        state["status"] = "completed"
        state["completed_at_utc"] = utc_now()
        write_json(state_path, state)
        return NodeResult("completed", updates={"baseline_run": ctx.rel_to_worktree(run_dir)}, artifacts=[ctx.rel_to_worktree(state_path)], message="baseline_outputs_complete")
    if ctx.dry_run:
        return NodeResult("waiting", updates={"baseline_run": run_raw}, artifacts=[ctx.rel_to_worktree(state_path)], message="baseline_dry_run_prepared")
    job_id = str(state.get("slurm_job_id") or "")
    observed = slurm_state(job_id)
    state["observed_slurm_state"] = observed
    state["checked_at_utc"] = utc_now()
    write_json(state_path, state)
    if observed in ACTIVE_STATES or observed == "UNKNOWN":
        startup_timeout = float(ctx.config.get("baseline_startup_timeout_sec") or 900.0)
        max_retries = int(ctx.config.get("baseline_max_infra_retries") or 2)
        retries_used = max(int(state.get("attempt") or 1) - 1, 0)
        submitted_at = str(state.get("submitted_at_utc") or state.get("created_at_utc") or "")
        if (
            observed in STARTUP_WATCH_STATES
            and run_raw
            and not job_started(run_dir)
            and age_seconds(submitted_at) >= startup_timeout
            and retries_used < max_retries
        ):
            reason = f"baseline_startup_sentinel_timeout:{observed}:{int(startup_timeout)}s"
            if cancel_job(job_id):
                mark_run_status(run_dir, "canceled_by_watchdog", reason)
                state["observed_slurm_state"] = observed
                write_json(state_path, state)
                return launch(ctx, state_path, state, reason)
        time.sleep(max(float(ctx.config.get("baseline_poll_sec") or 30.0), 1.0))
        return NodeResult("waiting", updates={"baseline_run": run_raw, "baseline_job_id": job_id}, artifacts=[ctx.rel_to_worktree(state_path)], message=f"baseline_{observed.lower()}")
    max_retries = int(ctx.config.get("baseline_max_infra_retries") or 2)
    retries_used = max(int(state.get("attempt") or 1) - 1, 0)
    if observed in RETRYABLE_TERMINAL_STATES and retries_used < max_retries:
        reason = f"baseline_retryable_terminal:{observed}:started={job_started(run_dir)}"
        mark_run_status(run_dir, "failed", reason)
        return launch(ctx, state_path, state, reason)
    reason = f"baseline_job_terminal_without_outputs:{observed}"
    return NodeResult("infra_blocked", updates={"baseline_reason": reason, "baseline_job_id": job_id}, artifacts=[ctx.rel_to_worktree(state_path)], message=reason)
