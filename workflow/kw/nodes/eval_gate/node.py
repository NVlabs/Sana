#!/usr/bin/env python3
"""Workflow-local full-evaluation gate node."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


AUTHORITATIVE_GATE_NAMES = {"assess_verdict.json", "gate_assess.json", "verdict.json"}
INFRA_BLOCKER_HINTS = (
    "baseline_frame_missing",
    "candidate_frame_missing",
    "baseline_frames_missing",
    "candidate_frames_missing",
    "ffmpeg_missing",
    "api_key_missing",
    "missing_api_key",
    "missing_frame",
    "missing_video",
    "missing_benchmark",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_gate_path(root: Path, raw: str, run_dir: str | None = None) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [root / path]
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = root / run_path
        candidates.append(run_path / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def gate_paths_from_status(root: Path) -> list[Path]:
    status = read_json(root / "AGENT-STATUS.json")
    paths: list[Path] = []
    for collection in ("candidates", "frontier_candidates", "discarded_candidates", "rejected_candidates"):
        for record in status.get(collection, []) or []:
            if not isinstance(record, dict):
                continue
            run_dir = record.get("run_dir") if isinstance(record.get("run_dir"), str) else None
            for raw in record.get("evidence", []) or []:
                if isinstance(raw, str) and Path(raw).name in AUTHORITATIVE_GATE_NAMES:
                    paths.append(resolve_gate_path(root, raw, run_dir))
    return paths


def discover_gate_paths(root: Path) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in [*gate_paths_from_status(root), *root.glob("runs/*/assess_verdict.json")]:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def blocker_is_infra(blocker: Any) -> bool:
    text = json.dumps(blocker, sort_keys=True) if not isinstance(blocker, str) else blocker
    lowered = text.lower()
    return any(hint in lowered for hint in INFRA_BLOCKER_HINTS)


def smooth_gate(path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "gate_missing", {}
    if path.stat().st_size == 0:
        return False, "gate_empty", {}
    data = read_json(path)
    if not data:
        return False, "gate_invalid_json", {}
    missing = [
        key for key in ("baseline_total_s", "candidate_total_s", "speedup")
        if not isinstance(data.get(key), (int, float))
    ]
    if missing:
        return False, "gate_missing_numeric_fields:" + ",".join(missing), data
    blockers = list(data.get("quality_blockers") or [])
    collector_blockers = list(data.get("collector_quality_blockers") or [])
    infra_blockers = [item for item in [*blockers, *collector_blockers] if blocker_is_infra(item)]
    if infra_blockers:
        return False, "gate_has_infrastructure_blockers:" + ",".join(map(str, infra_blockers)), data
    return True, "smooth_gate", data


def latest_runnable_run(root: Path) -> Path | None:
    candidates = []
    for run_dir in sorted((root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        if (run_dir / "assess_verdict.json").exists():
            continue
        if (run_dir / "outputs" / "benchmark.json").exists() and (
            (run_dir / "outputs" / "frames").exists() or (run_dir / "outputs" / "out.mp4").exists()
        ):
            candidates.append(run_dir)
    return candidates[-1] if candidates else None


def read_toml_like_baseline_run(path: Path) -> str:
    if not path.exists():
        return ""
    in_baseline = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "[baseline]":
            in_baseline = True
            continue
        if stripped.startswith("[") and in_baseline:
            return ""
        if in_baseline and stripped.startswith("run_id"):
            return stripped.split("=", 1)[1].strip().strip('"')
    return ""


def baseline_frames(ctx: NodeContext, context: dict[str, Any]) -> str:
    configured = str(ctx.config.get("baseline_frames") or "")
    if configured and Path(configured).exists():
        return configured
    loop = context.get("loop_contract") if isinstance(context.get("loop_contract"), dict) else {}
    raw = str(loop.get("canonical_baseline_frames") or "")
    if raw and Path(raw).exists():
        return raw
    model_id = str(ctx.config.get("model_id") or context.get("model_id") or "hunyuan_diffusers")
    profile = read_toml_like_baseline_run(ctx.worktree / "models" / f"{model_id}.toml")
    if profile:
        local = ctx.worktree / "runs" / profile / "outputs" / "frames"
        if local.exists():
            return str(local)
        coordinator = ctx.root / "runs" / profile / "outputs" / "frames"
        if coordinator.exists():
            return str(coordinator)
    return raw or configured


def maybe_run_assess(ctx: NodeContext) -> dict[str, Any]:
    context = read_json(ctx.goal_dir / "context.json")
    loop = context.get("loop_contract") if isinstance(context.get("loop_contract"), dict) else {}
    run_dir = latest_runnable_run(ctx.worktree)
    if not run_dir:
        return {"attempted": False, "reason": "no_completed_run_without_assess"}
    frames = baseline_frames(ctx, context)
    if not frames or not Path(frames).exists():
        return {
            "attempted": False,
            "reason": "baseline_frames_missing",
            "run_dir": rel_to(ctx.worktree, run_dir),
            "baseline_frames": frames,
        }
    out = run_dir / "assess_verdict.json"
    model_id = str(ctx.config.get("model_id") or context.get("model_id") or "hunyuan_diffusers")
    pybin = str(loop.get("authoritative_python") or sys.executable)
    cmd = [
        pybin,
        "search/plan_eval.py",
        "--assess",
        rel_to(ctx.worktree, run_dir),
        "--baseline-frames",
        frames,
        "--model",
        model_id,
        "--out",
        rel_to(ctx.worktree, out),
    ]
    if ctx.dry_run:
        return {"attempted": False, "reason": "dry_run", "command": cmd, "run_dir": rel_to(ctx.worktree, run_dir)}
    proc = subprocess.run(
        cmd,
        cwd=ctx.worktree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(ctx.config.get("assess_timeout_sec") or 1800),
    )
    return {
        "attempted": True,
        "returncode": proc.returncode,
        "run_dir": rel_to(ctx.worktree, run_dir),
        "out": rel_to(ctx.worktree, out),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run(ctx: NodeContext) -> NodeResult:
    gate_results = []
    for path in discover_gate_paths(ctx.worktree):
        ok, reason, data = smooth_gate(path)
        gate = {"path": rel_to(ctx.worktree, path), "ok": ok, "reason": reason, "speedup": data.get("speedup")}
        gate_results.append(gate)
        if ok:
            return NodeResult(
                "smooth",
                updates={"eval_gate": gate, "eval_reason": "smooth_gate", "eval_all_gates": gate_results},
                artifacts=[gate["path"]],
                message="smooth_gate",
            )

    assess_attempt = maybe_run_assess(ctx)
    if assess_attempt.get("attempted"):
        for path in discover_gate_paths(ctx.worktree):
            ok, reason, data = smooth_gate(path)
            gate = {"path": rel_to(ctx.worktree, path), "ok": ok, "reason": reason, "speedup": data.get("speedup")}
            gate_results.append(gate)
            if ok:
                return NodeResult(
                    "smooth",
                    updates={
                        "eval_gate": gate,
                        "eval_reason": "smooth_gate_after_assess",
                        "eval_all_gates": gate_results,
                        "assess_attempt": assess_attempt,
                    },
                    artifacts=[gate["path"]],
                    message="smooth_gate_after_assess",
                )

    reason = gate_results[-1]["reason"] if gate_results else str(assess_attempt.get("reason") or "no_authoritative_gate")
    outcome = "infra_blocked" if blocker_is_infra(reason) or "baseline_frames_missing" in reason else "missing"
    return NodeResult(
        outcome,
        updates={"eval_reason": reason, "eval_all_gates": gate_results, "assess_attempt": assess_attempt},
        message=reason,
    )
