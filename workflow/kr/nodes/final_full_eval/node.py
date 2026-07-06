#!/usr/bin/env python3
"""Workflow-local terminal full diffusion/Gemini gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from workflow_types import NodeContext, NodeResult


ASSESS_NAME = "assess_verdict.json"
REVIEWER_STATUS = "REVIEWER-STATUS.json"
INFRA_BLOCKER_HINTS = (
    "api_key_missing",
    "baseline_frame_missing",
    "candidate_frame_missing",
    "baseline_frames_missing",
    "candidate_frames_missing",
    "ffmpeg_missing",
    "missing_frame",
    "missing_video",
    "missing_benchmark",
    "no_output",
    "no-output",
    "slurm",
    "cancelled",
    "allocation",
    "quota",
    "filesystem",
    "stdout",
    "stderr",
    "heartbeat",
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


def blocker_is_infra(blocker: Any) -> bool:
    text = json.dumps(blocker, sort_keys=True) if not isinstance(blocker, str) else blocker
    lowered = text.lower()
    return any(hint in lowered for hint in INFRA_BLOCKER_HINTS)


def archive_reviewer_status(ctx: NodeContext) -> str:
    status = ctx.worktree / REVIEWER_STATUS
    if not status.exists() or ctx.dry_run:
        return ""
    archived = ctx.worktree / "REVIEWER-STATUS.final-full-eval-returned.json"
    status.replace(archived)
    return rel_to(ctx.worktree, archived)


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


def assess_paths_from_status(root: Path) -> list[Path]:
    status = read_json(root / REVIEWER_STATUS)
    paths: list[Path] = []
    for raw in status.get("evidence", []) or []:
        if isinstance(raw, str) and Path(raw).name == ASSESS_NAME:
            path = Path(raw)
            paths.append(path if path.is_absolute() else root / path)
    return paths


def discover_assess_paths(root: Path) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in [*assess_paths_from_status(root), *root.glob(f"runs/*/{ASSESS_NAME}")]:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def latest_completed_full_run(root: Path) -> Path | None:
    candidates = []
    for run_dir in sorted((root / "runs").glob("*")):
        if not run_dir.is_dir() or (run_dir / ASSESS_NAME).exists():
            continue
        if (run_dir / "outputs" / "benchmark.json").exists() and (
            (run_dir / "outputs" / "frames").exists() or (run_dir / "outputs" / "out.mp4").exists()
        ):
            candidates.append(run_dir)
    return candidates[-1] if candidates else None


def run_assess(ctx: NodeContext) -> dict[str, Any]:
    context = read_json(ctx.goal_dir / "context.json")
    loop = context.get("loop_contract") if isinstance(context.get("loop_contract"), dict) else {}
    run_dir = latest_completed_full_run(ctx.worktree)
    if not run_dir:
        return {"attempted": False, "reason": "no_completed_final_full_run"}
    frames = baseline_frames(ctx, context)
    if not frames or not Path(frames).exists():
        return {
            "attempted": False,
            "reason": "baseline_frames_missing",
            "run_dir": rel_to(ctx.worktree, run_dir),
            "baseline_frames": frames,
        }
    out = run_dir / ASSESS_NAME
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
        env=ctx.env,
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


def final_gate(path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, "final_assess_missing", {}
    if path.stat().st_size == 0:
        return False, "final_assess_empty", {}
    data = read_json(path)
    if not data:
        return False, "final_assess_invalid_json", {}
    missing = [
        key for key in ("baseline_total_s", "candidate_total_s", "speedup")
        if not isinstance(data.get(key), (int, float))
    ]
    if missing:
        return False, "final_assess_missing_numeric_fields:" + ",".join(missing), data
    blockers = [*list(data.get("quality_blockers") or []), *list(data.get("collector_quality_blockers") or [])]
    infra_blockers = [item for item in blockers if blocker_is_infra(item)]
    if infra_blockers:
        return False, "final_assess_has_infrastructure_blockers:" + ",".join(map(str, infra_blockers)), data
    if blockers:
        return False, "final_assess_quality_blockers:" + ",".join(map(str, blockers)), data
    gemini = str(data.get("gemini_overall") or "").lower()
    if gemini != "pass":
        return False, f"final_gemini_not_pass:{gemini or 'missing'}", data
    return True, "final_full_eval_passed", data


def result_for_failure(ctx: NodeContext, reason: str, gate_results: list[dict[str, Any]], assess_attempt: dict[str, Any]) -> NodeResult:
    archived = archive_reviewer_status(ctx)
    outcome = "infra_blocked" if blocker_is_infra(reason) or "baseline_frames_missing" in reason else "quality_failed"
    if reason == "no_completed_final_full_run":
        outcome = "missing"
    updates = {
        "final_eval_reason": reason,
        "final_eval_gates": gate_results,
        "final_assess_attempt": assess_attempt,
    }
    if archived:
        updates["archived_reviewer_status"] = archived
    return NodeResult(outcome, updates=updates, artifacts=[archived] if archived else [], message=reason)


def run(ctx: NodeContext) -> NodeResult:
    gate_results = []
    for path in discover_assess_paths(ctx.worktree):
        ok, reason, data = final_gate(path)
        gate = {
            "path": rel_to(ctx.worktree, path),
            "ok": ok,
            "reason": reason,
            "speedup": data.get("speedup"),
            "gemini_overall": data.get("gemini_overall"),
        }
        gate_results.append(gate)
        if ok:
            return NodeResult(
                "passed",
                updates={"final_eval_gate": gate, "final_eval_reason": reason, "final_eval_gates": gate_results},
                artifacts=[gate["path"]],
                message=reason,
            )

    assess_attempt = run_assess(ctx)
    if assess_attempt.get("attempted"):
        for path in discover_assess_paths(ctx.worktree):
            ok, reason, data = final_gate(path)
            gate = {
                "path": rel_to(ctx.worktree, path),
                "ok": ok,
                "reason": reason,
                "speedup": data.get("speedup"),
                "gemini_overall": data.get("gemini_overall"),
            }
            gate_results.append(gate)
            if ok:
                return NodeResult(
                    "passed",
                    updates={
                        "final_eval_gate": gate,
                        "final_eval_reason": reason,
                        "final_eval_gates": gate_results,
                        "final_assess_attempt": assess_attempt,
                    },
                    artifacts=[gate["path"]],
                    message=reason,
                )

    reason = gate_results[-1]["reason"] if gate_results else str(assess_attempt.get("reason") or "no_final_assess")
    return result_for_failure(ctx, reason, gate_results, assess_attempt)
