#!/usr/bin/env python3
"""Workflow-local immutable baseline certification gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


LOCK_NAME = "BASELINE-LOCK.json"
STATE_NAME = "baseline-run.json"
EXPECTED_CONFIG = {
    "num_frames": 193,
    "fps": 24,
    "image_size": 720,
    "steps": 50,
    "cfg_scale": 8,
    "flow_shift": 12,
    "motion_score": 20,
    "sample_nums": 5,
}
HOT_TIMING_SCOPE = "warm_single_sample_text_encoder_through_vae_decode"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(ctx: NodeContext, path: Path) -> str:
    return ctx.rel_to_worktree(path.resolve())


def resolve(ctx: NodeContext, raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else ctx.worktree / path).resolve()


def artifact(ctx: NodeContext, path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": relative(ctx, path), "sha256": sha256(path), "size": path.stat().st_size}


def benchmark_total(benchmark: dict[str, Any]) -> tuple[float | None, str]:
    aggregate = benchmark.get("aggregate") if isinstance(benchmark.get("aggregate"), dict) else {}
    if isinstance(aggregate.get("sample_total_s"), (int, float)):
        timing = benchmark.get("timing") if isinstance(benchmark.get("timing"), dict) else {}
        contract = benchmark.get("timing_contract") if isinstance(benchmark.get("timing_contract"), dict) else {}
        return float(aggregate["sample_total_s"]), str(timing.get("scope") or contract.get("scope") or benchmark.get("timing_scope") or "")
    return None, ""


def hot_timing_issues(benchmark: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    contract = benchmark.get("timing_contract") if isinstance(benchmark.get("timing_contract"), dict) else {}
    aggregate = benchmark.get("aggregate") if isinstance(benchmark.get("aggregate"), dict) else {}
    samples = benchmark.get("samples") if isinstance(benchmark.get("samples"), list) else []
    if str(benchmark.get("timing_scope") or contract.get("scope") or "") != HOT_TIMING_SCOPE:
        issues.append("baseline_timing_scope_not_hot")
    required_true = (
        "warm_steady_state",
        "warmup_same_shape",
        "stage_isolated",
        "includes_text_encoder_inference",
        "includes_denoise",
        "includes_vae_decode",
    )
    required_false = (
        "includes_process_startup",
        "includes_model_and_text_encoder_load",
        "includes_cpu_postprocess",
        "includes_video_write",
    )
    for key in required_true:
        if contract.get(key) is not True:
            issues.append(f"baseline_timing_contract_requires_true:{key}")
    for key in required_false:
        if contract.get(key) is not False:
            issues.append(f"baseline_timing_contract_requires_false:{key}")
    if not isinstance(contract.get("warmup_samples"), int) or int(contract["warmup_samples"]) < 1:
        issues.append("baseline_hot_warmup_missing")
    if aggregate.get("sample_count") != EXPECTED_CONFIG["sample_nums"] or len(samples) != EXPECTED_CONFIG["sample_nums"]:
        issues.append("baseline_hot_sample_count_invalid")
    for key in ("sample_total_s", "sample_mean_s", "text_encoder_s", "denoise_s", "vae_decode_s"):
        if not isinstance(aggregate.get(key), (int, float)) or float(aggregate[key]) <= 0:
            issues.append(f"baseline_hot_aggregate_invalid:{key}")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            issues.append(f"baseline_hot_sample_invalid:{index}")
            continue
        stages = [sample.get(key) for key in ("text_encoder_s", "denoise_s", "vae_decode_s", "total_s")]
        if any(not isinstance(value, (int, float)) or float(value) <= 0 for value in stages):
            issues.append(f"baseline_hot_sample_timing_invalid:{index}")
            continue
        stage_sum = sum(float(value) for value in stages[:3])
        if not math.isclose(stage_sum, float(stages[3]), rel_tol=1e-5, abs_tol=1e-4):
            issues.append(f"baseline_hot_sample_stage_sum_mismatch:{index}")
    return issues


def effective_config(benchmark: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    config = benchmark.get("config") if isinstance(benchmark.get("config"), dict) else {}
    return {**config, **run_config}


def unique_videos(outputs: Path) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in sorted(outputs.rglob("*.mp4")):
        if not path.is_file():
            continue
        digest = sha256(path)
        if digest not in seen:
            seen.add(digest)
            result.append(path)
    return result


def source_digest(ctx: NodeContext) -> tuple[str, list[dict[str, Any]]]:
    roots = [
        ctx.worktree / "runtime" / "sana_video_baseline",
        ctx.worktree / "models" / "sana_video",
    ]
    files = [ctx.worktree / "models" / "sana_video.toml", ctx.worktree / "candidates" / "sana_video_baseline.toml"]
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        item = artifact(ctx, path, "baseline_source")
        records.append(item)
        digest.update(item["path"].encode())
        digest.update(b"\0")
        digest.update(item["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest(), records


def validate_lock(ctx: NodeContext, lock: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if lock.get("schema_version") != 1 or lock.get("status") != "locked":
        issues.append("baseline_lock_header_invalid")
    source_digest = hashlib.sha256()
    source_items = lock.get("source_files") or []
    for item in [*(lock.get("artifacts") or []), *source_items]:
        if not isinstance(item, dict):
            issues.append("baseline_lock_artifact_invalid")
            continue
        path = resolve(ctx, str(item.get("path") or ""))
        if not path.is_file():
            issues.append(f"baseline_artifact_missing:{item.get('path')}")
        elif sha256(path) != item.get("sha256"):
            issues.append(f"baseline_artifact_hash_mismatch:{item.get('path')}")
    for item in source_items:
        if not isinstance(item, dict):
            continue
        source_digest.update(str(item.get("path") or "").encode())
        source_digest.update(b"\0")
        source_digest.update(str(item.get("sha256") or "").encode())
        source_digest.update(b"\n")
    if source_digest.hexdigest() != lock.get("source_tree_sha256"):
        issues.append("baseline_source_tree_hash_mismatch")
    return issues


def run(ctx: NodeContext) -> NodeResult:
    lock_path = ctx.worktree / LOCK_NAME
    existing = read_json(lock_path)
    if existing:
        issues = validate_lock(ctx, existing)
        outcome = "invalid" if issues else "ready"
        return NodeResult(outcome, updates={"baseline_issues": issues, "baseline_lock": relative(ctx, lock_path)}, artifacts=[relative(ctx, lock_path)], message=";".join(issues) if issues else "baseline_lock_valid")

    state = read_json(ctx.worktree / "state" / STATE_NAME)
    run_raw = str(state.get("run_dir") or "")
    run_dir = Path(run_raw).resolve() if run_raw else Path()
    outputs = run_dir / "outputs"
    benchmark_path = outputs / "benchmark.json"
    config_path = outputs / "run_config.json"
    benchmark = read_json(benchmark_path)
    run_config = read_json(config_path)
    issues: list[str] = []
    if not run_raw or not run_dir.is_dir():
        issues.append("baseline_run_missing")
    if not benchmark:
        issues.append("baseline_benchmark_missing")
    if not run_config:
        issues.append("baseline_run_config_missing")
    config = effective_config(benchmark, run_config)
    for key, wanted in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if key == "sample_nums" and actual is None:
            actual = config.get("prompt_count")
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), float(wanted), rel_tol=0, abs_tol=1e-6):
            issues.append(f"baseline_config_mismatch:{key}")
    videos = unique_videos(outputs) if outputs.is_dir() else []
    if len(videos) < 5:
        issues.append(f"baseline_unique_videos_incomplete:{len(videos)}")
    total, timing_scope = benchmark_total(benchmark)
    if total is None or total <= 0 or not timing_scope:
        issues.append("baseline_timing_invalid")
    issues.extend(hot_timing_issues(benchmark))
    if issues:
        return NodeResult("invalid", updates={"baseline_issues": issues}, message=";".join(issues))

    source_tree_sha256, source_files = source_digest(ctx)
    prompt = ctx.worktree / "models" / "sana_video" / "prompts" / "dpo_holdout_qwen35_val64_concrete40_first5.txt"
    artifacts = [
        artifact(ctx, benchmark_path, "benchmark"),
        artifact(ctx, config_path, "run_config"),
        *[artifact(ctx, path, f"gold_video_{index:03d}") for index, path in enumerate(videos[:5])],
    ]
    if prompt.is_file():
        artifacts.append(artifact(ctx, prompt, "prompt_set"))
    workload_material = json.dumps({"model_id": ctx.config.get("model_id"), "config": EXPECTED_CONFIG, "prompt_sha256": sha256(prompt) if prompt.is_file() else ""}, sort_keys=True).encode()
    workload_id = "sana-video-720p193-" + hashlib.sha256(workload_material).hexdigest()[:16]
    lock = {
        "schema_version": 1,
        "status": "locked",
        "workflow_uid": ctx.state.get("workflow_uid"),
        "experiment_uid": ctx.state.get("experiment_uid"),
        "model_id": ctx.config.get("model_id"),
        "created_at_utc": utc_now(),
        "successful_baseline_runs": 1,
        "run_dir": relative(ctx, run_dir),
        "workload_id": workload_id,
        "timing_scope": timing_scope,
        "baseline_total_s": total,
        "baseline_mean_s": benchmark["aggregate"]["sample_mean_s"],
        "timing_contract": benchmark["timing_contract"],
        "source_tree_sha256": source_tree_sha256,
        "source_files": source_files,
        "effective_off_env": state.get("effective_off_env") or {},
        "config": config,
        "artifacts": artifacts,
        "immutability": "hash_guarded_read_only_files",
    }
    write_json(lock_path, lock)
    if not ctx.dry_run:
        for item in [*artifacts, *source_files]:
            path = resolve(ctx, item["path"])
            try:
                os.chmod(path, 0o444)
            except OSError:
                pass
        try:
            os.chmod(lock_path, 0o444)
        except OSError:
            pass
    return NodeResult("ready", updates={"baseline_lock": relative(ctx, lock_path), "baseline_run": relative(ctx, run_dir)}, artifacts=[relative(ctx, lock_path)], message="baseline_locked")
