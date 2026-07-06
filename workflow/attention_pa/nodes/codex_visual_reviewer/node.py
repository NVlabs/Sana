#!/usr/bin/env python3
"""Workflow-local blind Codex image reviewer for PISA candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

from workflow_types import NodeContext, NodeResult


WORKFLOW_UID = "attention_pa"
ASSESS_NAME = "assess_verdict.json"
NUMERIC_ASSESS_NAME = "numeric_assess.json"
VISUAL_VERDICT_NAME = "codex_visual_verdict.json"
RAW_VERDICT_NAME = "raw_verdict.json"
MARKER_NAME = "NODE-COMPLETE.json"
REVIEW_DIR_NAME = "codex_visual_review"
PROMPT_COUNT = 5
FRAME_TIMES = (1.5, 3.958, 4.0, 6.5)
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
AUTORUN_SESSION_RE = re.compile(r"^Codex running in tmux session:\s*(\S+)\s*$", re.MULTILINE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_model_profile(ctx: NodeContext) -> dict[str, Any]:
    path = ctx.worktree / "models" / "sana_video.toml"
    if not path.exists():
        path = ctx.root / "models" / "sana_video.toml"
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def completed_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    for run_dir in (root / "runs").glob("*"):
        if not run_dir.is_dir() or "baseline" in run_dir.name.lower():
            continue
        if not (run_dir / "outputs" / "benchmark.json").exists():
            continue
        outputs = run_dir / "outputs"
        if any(outputs.rglob("*.png")) or any(outputs.rglob("*.mp4")):
            runs.append(run_dir)
    return sorted(runs, key=lambda path: (path / "outputs" / "benchmark.json").stat().st_mtime)


def baseline_run(ctx: NodeContext, profile: dict[str, Any]) -> Path | None:
    lock = read_json(ctx.worktree / "BASELINE-LOCK.json")
    locked_run = str(lock.get("run_dir") or "")
    if locked_run:
        path = Path(locked_run)
        path = path if path.is_absolute() else ctx.worktree / path
        if path.is_dir():
            return path.resolve()
    configured = str(ctx.config.get("baseline_frames") or "")
    if configured:
        frames = Path(configured).expanduser()
        if frames.exists():
            if frames.name == "frames" and frames.parent.name == "outputs":
                return frames.parent.parent
            if frames.name == "frames" and frames.parent.name == "canonical_baseline":
                return frames.parent
    baseline = profile.get("baseline") if isinstance(profile.get("baseline"), dict) else {}
    run_id = str(baseline.get("run_id") or "")
    for base in (ctx.worktree, ctx.root):
        candidate = base / "runs" / run_id
        if run_id and candidate.exists():
            return candidate
    local = ctx.worktree / "runs" / "canonical_baseline"
    return local if local.exists() else None


def benchmark_total(benchmark: dict[str, Any]) -> tuple[float | None, str]:
    aggregate = benchmark.get("aggregate") if isinstance(benchmark.get("aggregate"), dict) else {}
    timing = benchmark.get("timing") if isinstance(benchmark.get("timing"), dict) else {}
    contract = benchmark.get("timing_contract") if isinstance(benchmark.get("timing_contract"), dict) else {}
    if isinstance(aggregate.get("sample_total_s"), (int, float)):
        return float(aggregate["sample_total_s"]), str(
            timing.get("scope") or contract.get("scope") or benchmark.get("timing_scope") or ""
        )
    if isinstance(benchmark.get("total_s"), (int, float)):
        return float(benchmark["total_s"]), str(benchmark.get("timing_scope") or "")
    return None, ""


def prompt_index(path: Path) -> int | None:
    for pattern in (r"prompt[_ .-]?(\d{3})", r"^(\d{3})[. _-]"):
        match = re.search(pattern, path.name.lower())
        if match:
            return int(match.group(1))
    for part in reversed(path.parts):
        match = re.fullmatch(r"prompt[_-]?(\d{3})", part.lower())
        if match:
            return int(match.group(1))
    return None


def prompt_videos(run_dir: Path) -> dict[int, Path]:
    outputs = run_dir / "outputs" if (run_dir / "outputs").exists() else run_dir
    candidates = sorted(outputs.rglob("*.mp4"))
    result: dict[int, Path] = {}
    for path in candidates:
        lowered = path.name.lower()
        if lowered in {"out.mp4", "side_by_side.mp4"} or "side-by-side" in lowered:
            continue
        index = prompt_index(path)
        if index is not None and 0 <= index < PROMPT_COUNT and index not in result:
            result[index] = path
    return result


def prompt_frame_groups(run_dir: Path) -> dict[int, list[Path]]:
    outputs = run_dir / "outputs" if (run_dir / "outputs").exists() else run_dir
    roots = [outputs / "frames_by_prompt", outputs / "frames", run_dir / "frames"]
    groups: dict[int, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.png")):
            index = prompt_index(path)
            if index is not None and 0 <= index < PROMPT_COUNT:
                groups.setdefault(index, []).append(path)
        if len(groups) == PROMPT_COUNT:
            return groups
    return groups


def select_group_frames(paths: list[Path]) -> list[Path]:
    if len(paths) < len(FRAME_TIMES):
        return []
    last = len(paths) - 1
    fractions = (0.2, 0.495, 0.5, 0.8)
    return [paths[round(last * fraction)] for fraction in fractions]


def resolve_ffmpeg(ctx: NodeContext) -> str | None:
    candidates: list[Path] = []
    if ctx.env.get("FFMPEG"):
        candidates.append(Path(ctx.env["FFMPEG"]).expanduser())
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(Path(on_path))
    profile = load_model_profile(ctx)
    env_cfg = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    for key in ("PYTHON_BIN", "SANA_VIDEO_INFER_PYTHON"):
        pybin = str(env_cfg.get(key) or "")
        if pybin:
            candidates.append(Path(pybin).expanduser().parent / "ffmpeg")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def render_pair(
    ffmpeg: str,
    left: Path,
    right: Path,
    out: Path,
    *,
    left_time: float | None = None,
    right_time: float | None = None,
) -> tuple[bool, str]:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if left_time is not None:
        cmd.extend(["-ss", f"{left_time:.3f}"])
    cmd.extend(["-i", str(left)])
    if right_time is not None:
        cmd.extend(["-ss", f"{right_time:.3f}"])
    cmd.extend(
        [
            "-i",
            str(right),
            "-filter_complex",
            "[0:v]scale=w=min(960\\,iw):h=-2[l];[1:v]scale=w=min(960\\,iw):h=-2[r];[l][r]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-frames:v",
            "1",
            str(out),
        ]
    )
    try:
        proc = run_command(cmd, cwd=out.parent)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"ffmpeg_pair_render_failed:{exc}"
    if proc.returncode != 0 or not out.exists():
        return False, proc.stderr[-2000:] or "ffmpeg_pair_render_failed"
    return True, ""


def extract_frame(ffmpeg: str, source: Path, out: Path, timestamp: float) -> tuple[bool, str]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-vf",
        "scale=w=min(960\\,iw):h=-2",
        "-frames:v",
        "1",
        str(out),
    ]
    try:
        proc = run_command(cmd, cwd=out.parent)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"ffmpeg_frame_extract_failed:{exc}"
    if proc.returncode != 0 or not out.exists():
        return False, proc.stderr[-2000:] or "ffmpeg_frame_extract_failed"
    return True, ""


def build_review_bundle(
    ctx: NodeContext,
    run_dir: Path,
    baseline: Path,
) -> tuple[dict[str, Any], str]:
    ffmpeg = resolve_ffmpeg(ctx)
    if not ffmpeg:
        return {}, "ffmpeg_missing"
    digest = hashlib.sha256(f"{ctx.state.get('experiment_uid')}:{run_dir.name}".encode()).hexdigest()
    review_id = f"{WORKFLOW_UID}-{digest[:16]}"
    review_dir = ctx.worktree / "state" / "codex_visual_reviews" / review_id
    pair_dir = review_dir / "pairs"
    source_dir = review_dir / "aligned_sources"
    pair_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    candidate_side = "left" if int(digest[:2], 16) % 2 == 0 else "right"
    baseline_side = "right" if candidate_side == "left" else "left"
    candidate_videos = prompt_videos(run_dir)
    baseline_videos = prompt_videos(baseline)
    candidate_groups = prompt_frame_groups(run_dir)
    baseline_groups = prompt_frame_groups(baseline)

    use_videos = len(candidate_videos) == PROMPT_COUNT and len(baseline_videos) == PROMPT_COUNT
    use_frames = len(candidate_groups) == PROMPT_COUNT and len(baseline_groups) == PROMPT_COUNT
    if not use_videos and not use_frames:
        return {}, "five_prompt_aligned_visual_sources_missing"

    images: list[str] = []
    lpips_pairs: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    for prompt in range(PROMPT_COUNT):
        if use_frames:
            baseline_selected = select_group_frames(baseline_groups[prompt])
            candidate_selected = select_group_frames(candidate_groups[prompt])
            if len(baseline_selected) != len(FRAME_TIMES) or len(candidate_selected) != len(FRAME_TIMES):
                return {}, f"prompt_{prompt:03d}_frames_insufficient"
        for offset, timestamp in enumerate(FRAME_TIMES):
            stem = f"prompt_{prompt:03d}_sample_{offset:02d}"
            baseline_frame = source_dir / f"{stem}_baseline.png"
            candidate_frame = source_dir / f"{stem}_candidate.png"
            if use_videos:
                ok, reason = extract_frame(ffmpeg, baseline_videos[prompt], baseline_frame, timestamp)
                if not ok:
                    return {}, reason
                ok, reason = extract_frame(ffmpeg, candidate_videos[prompt], candidate_frame, timestamp)
                if not ok:
                    return {}, reason
            else:
                shutil.copy2(baseline_selected[offset], baseline_frame)
                shutil.copy2(candidate_selected[offset], candidate_frame)

            left = candidate_frame if candidate_side == "left" else baseline_frame
            right = baseline_frame if candidate_side == "left" else candidate_frame
            pair_path = pair_dir / f"{stem}.png"
            ok, reason = render_pair(ffmpeg, left, right, pair_path)
            if not ok:
                return {}, reason
            images.append(str(pair_path.resolve()))
            lpips_pairs.append((str(baseline_frame.resolve()), str(candidate_frame.resolve())))
            rows.append(
                {
                    "prompt_index": prompt,
                    "sample_index": offset,
                    "timestamp_s": timestamp,
                    "image": rel_to(ctx.worktree, pair_path),
                }
            )

    public_manifest = {
        "schema_version": 1,
        "review_id": review_id,
        "pair_count": len(rows),
        "pair_layout": "left_vs_right",
        "identity_blinded": True,
        "pairs": rows,
    }
    write_json(review_dir / "image_manifest.json", public_manifest)
    control = {
        **public_manifest,
        "candidate_side": candidate_side,
        "baseline_side": baseline_side,
        "run_dir": rel_to(ctx.worktree, run_dir),
        "images": images,
        "lpips_pairs": lpips_pairs,
    }
    control["review_dir"] = str(review_dir.resolve())
    write_json(run_dir / "codex_visual_review_control.json", control)
    return control, ""


def existing_lpips(run_dir: Path) -> dict[str, Any]:
    quality = read_json(run_dir / "outputs" / "quality.json")
    result = (((quality.get("judges") or {}).get("lpips") or {}).get("result") or {})
    if isinstance(result.get("max"), (int, float)):
        return result
    old_assess = read_json(run_dir / ASSESS_NAME)
    if isinstance(old_assess.get("lpips_max"), (int, float)):
        return {
            "status": "ok",
            "mean": old_assess.get("lpips_mean"),
            "max": old_assess.get("lpips_max"),
            "source": "prior_assess",
        }
    return {}


def run_lpips(ctx: NodeContext, run_dir: Path, profile: dict[str, Any], pairs: list[tuple[str, str]]) -> tuple[dict[str, Any], str]:
    prior = existing_lpips(run_dir)
    if prior:
        return prior, ""
    tool = ctx.worktree / "tools" / "vision" / "lpips_judge.py"
    if not tool.exists():
        return {}, "lpips_tool_missing"
    env_cfg = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    pybin = str(env_cfg.get("PYTHON_BIN") or env_cfg.get("SANA_VIDEO_INFER_PYTHON") or sys.executable)
    out = run_dir / REVIEW_DIR_NAME / "lpips.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [pybin, str(tool), "--out", str(out)]
    for baseline_frame, candidate_frame in pairs:
        cmd.extend(["--baseline-frame", baseline_frame, "--candidate-frame", candidate_frame])
    env = ctx.env.copy()
    python_deps = ctx.worktree / "caches" / "python_deps"
    if python_deps.exists():
        env["PYTHONPATH"] = str(python_deps) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = run_command(
            cmd,
            cwd=ctx.worktree,
            env=env,
            timeout=int(ctx.config.get("assess_timeout_sec") or 1800),
        )
    except subprocess.TimeoutExpired:
        return {}, "lpips_timeout"
    result = read_json(out)
    if proc.returncode != 0 or result.get("status") != "ok" or not isinstance(result.get("max"), (int, float)):
        return {}, "lpips_failed:" + (proc.stderr[-1000:] or str(result.get("reason") or "unknown"))
    return result, ""


def numeric_assess(ctx: NodeContext, run_dir: Path, profile: dict[str, Any], lpips: dict[str, Any]) -> tuple[dict[str, Any], str]:
    benchmark = read_json(run_dir / "outputs" / "benchmark.json")
    candidate_total, candidate_scope = benchmark_total(benchmark)
    lock = read_json(ctx.worktree / "BASELINE-LOCK.json")
    baseline_cfg = profile.get("baseline") if isinstance(profile.get("baseline"), dict) else {}
    baseline_total = lock.get("baseline_total_s", baseline_cfg.get("total_s"))
    baseline_scope = str(lock.get("timing_scope") or benchmark.get("timing_scope") or candidate_scope)
    if not isinstance(candidate_total, (int, float)) or not isinstance(baseline_total, (int, float)):
        return {}, "numeric_timing_missing"
    if not candidate_scope or candidate_scope != baseline_scope:
        return {}, f"numeric_timing_scope_mismatch:{candidate_scope or 'missing'}:{baseline_scope or 'missing'}"
    speedup = float(baseline_total) / float(candidate_total) if float(candidate_total) else None
    data = {
        "schema_version": 1,
        "run_dir": rel_to(ctx.worktree, run_dir),
        "baseline_total_s": float(baseline_total),
        "candidate_total_s": float(candidate_total),
        "speedup": round(speedup, 4) if speedup else None,
        "timing_scope": candidate_scope,
        "baseline_lock": "BASELINE-LOCK.json" if lock else "",
        "workload_id": lock.get("workload_id"),
        "lpips_mean": lpips.get("mean"),
        "lpips_max": lpips.get("max"),
        "lpips_pair_count": lpips.get("count") or lpips.get("pairs"),
        "quality_blockers": [],
        "collector_quality_blockers": [],
        "visual_provider": "codex",
        "visual_status": "pending",
        "created_at_utc": utc_now(),
    }
    write_json(run_dir / NUMERIC_ASSESS_NAME, data)
    return data, ""


def resolve_autorun(env: dict[str, str]) -> Path | None:
    candidates = []
    if env.get("CODEX_AUTORUN"):
        candidates.append(Path(env["CODEX_AUTORUN"]).expanduser())
    candidates.extend(
        [
            Path.home() / "codex_auto_run.py",
            Path.home() / "code" / "codex_exec" / "codex_auto_run.py",
        ]
    )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def tmux_alive(session: str) -> bool:
    if not session or not shutil.which("tmux"):
        return False
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def stop_and_capture(session: str, transcript: Path) -> None:
    if not session or not shutil.which("tmux"):
        return
    try:
        capture = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-t", session, "-S", "-240"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        transcript.write_text(f"capture failed: {exc}\n")
        capture = None
    if capture is not None:
        transcript.write_text(capture.stdout or capture.stderr)
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def build_reviewer_prompt(ctx: NodeContext, control: dict[str, Any], run_dir: Path) -> str:
    static = (Path(__file__).with_name("prompt.md")).read_text()
    review_dir = Path(str(control["review_dir"]))
    raw = review_dir / RAW_VERDICT_NAME
    marker = review_dir / MARKER_NAME
    rows = "\n".join(
        f"- prompt {row['prompt_index']}, sample {row['sample_index']}, t={row['timestamp_s']}s"
        for row in control["pairs"]
    )
    return f"""{static}

## Attached Evidence

Review id: `{control['review_id']}`

The {len(control['pairs'])} attached images appear in this order. The two middle
samples for each prompt are adjacent frames used to expose flicker or popping:

{rows}

Write `{rel_to(ctx.worktree, raw)}` with exactly this schema:

```json
{{
  "schema_version": 1,
  "review_id": "{control['review_id']}",
  "status": "complete",
  "degraded_side": "left | right | neither | both | unclear",
  "max_severity": "none | low | medium | high | critical",
  "confidence": "low | medium | high",
  "summary": "brief comparison without guessing method identity",
  "differences": [
    {{"category": "artifact category", "severity": "low", "prompt_indices": [0], "sample_indices": [1], "description": "visible evidence"}}
  ],
  "per_prompt": [
    {{"prompt_index": 0, "degraded_side": "left | right | neither | both | unclear", "max_severity": "none | low | medium | high | critical", "description": "brief evidence"}}
  ]
}}
```

Do not include Markdown in the JSON file. Do not guess which side is the
candidate. After the verdict is durable, make this your final filesystem action:
write `{rel_to(ctx.worktree, marker)}` as:

```json
{{"schema_version": 1, "review_id": "{control['review_id']}", "status": "complete", "completed_at_utc": "<ISO-8601 UTC>"}}
```
"""


def raw_verdict_valid(raw: dict[str, Any], review_id: str) -> tuple[bool, str]:
    if raw.get("review_id") != review_id or raw.get("status") != "complete":
        return False, "codex_visual_identity_or_status_invalid"
    if raw.get("degraded_side") not in {"left", "right", "neither", "both", "unclear"}:
        return False, "codex_visual_degraded_side_invalid"
    if raw.get("max_severity") not in SEVERITY_ORDER:
        return False, "codex_visual_severity_invalid"
    if raw.get("confidence") not in {"low", "medium", "high"}:
        return False, "codex_visual_confidence_invalid"
    if not isinstance(raw.get("differences"), list) or not isinstance(raw.get("per_prompt"), list):
        return False, "codex_visual_details_missing"
    return True, ""


def build_autorun_command(
    autorun: Path,
    ctx: NodeContext,
    prompt_path: Path,
    runtime_dir: Path,
    prefix: str,
    images: list[str],
) -> list[str]:
    cmd = [
        str(autorun),
        "--detach",
        "-C",
        str(ctx.worktree),
        "--sandbox",
        "workspace-write",
        "--auto-trust-directory",
        "--session-prefix",
        prefix,
        "--runtime-dir",
        str(runtime_dir),
        "--prompt-file",
        str(prompt_path),
        "--",
        "--model",
        str(ctx.config.get("autorun_model") or "gpt-5.6-sol"),
    ]
    for image in images:
        cmd.extend(["--image", image])
    return cmd


def decode_blind_verdict(raw: dict[str, Any], candidate_side: str) -> dict[str, Any]:
    degraded = str(raw.get("degraded_side"))
    observed_severity = str(raw.get("max_severity"))
    if degraded == candidate_side:
        candidate_severity = observed_severity
        relation = "minor_loss" if SEVERITY_ORDER[candidate_severity] <= 1 else "material_loss"
        overall = "pass" if SEVERITY_ORDER[candidate_severity] <= 1 else "fail"
        artifacts = list(raw.get("differences") or [])
    elif degraded in {"left", "right"}:
        candidate_severity = "none"
        relation = "better"
        overall = "pass"
        artifacts = []
    elif degraded == "neither":
        candidate_severity = "none"
        relation = "equivalent"
        overall = "pass"
        artifacts = []
    else:
        candidate_severity = observed_severity
        relation = "inconclusive"
        overall = "inconclusive"
        artifacts = list(raw.get("differences") or [])
    return {
        "overall": overall,
        "candidate_relation": relation,
        "max_artifact_severity": candidate_severity,
        "observed_pair_severity": observed_severity,
        "new_artifacts": artifacts,
    }


def launch_and_wait(ctx: NodeContext, control: dict[str, Any], run_dir: Path) -> tuple[dict[str, Any], str, str]:
    review_dir = Path(str(control["review_dir"]))
    raw_path = review_dir / RAW_VERDICT_NAME
    marker_path = review_dir / MARKER_NAME
    session_state_path = review_dir / "session.json"
    transcript_path = review_dir / "codex_transcript.txt"
    session_state = read_json(session_state_path)
    session = str(session_state.get("session") or "")
    review_id = str(control["review_id"])

    valid, _ = raw_verdict_valid(read_json(raw_path), review_id)
    marker = read_json(marker_path)
    if valid and marker.get("review_id") == review_id and marker.get("status") == "complete":
        return read_json(raw_path), session, ""

    if not tmux_alive(session):
        for path in (raw_path, marker_path):
            if path.exists():
                path.unlink()
        autorun = resolve_autorun(ctx.env)
        if not autorun:
            return {}, "", "codex_autorun_missing"
        prompt_path = review_dir / "review_prompt.md"
        prompt_path.write_text(build_reviewer_prompt(ctx, control, run_dir))
        runtime_dir = review_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{ctx.state.get('experiment_uid')}-visual")[:48]
        cmd = build_autorun_command(
            autorun,
            ctx,
            prompt_path,
            runtime_dir,
            prefix,
            list(control["images"]),
        )
        env = ctx.env.copy()
        env["TERM"] = "xterm-256color"
        try:
            proc = run_command(cmd, cwd=ctx.worktree, env=env, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {}, "", f"codex_visual_launch_failed:{exc}"
        if proc.returncode != 0:
            return {}, "", "codex_visual_launch_failed:" + (proc.stderr[-2000:] or proc.stdout[-2000:])
        match = AUTORUN_SESSION_RE.search(proc.stdout)
        if not match:
            return {}, "", "codex_visual_session_missing"
        session = match.group(1)
        write_json(
            session_state_path,
            {
                "schema_version": 1,
                "review_id": review_id,
                "session": session,
                "started_at_utc": utc_now(),
                "model": str(ctx.config.get("autorun_model") or "gpt-5.6-sol"),
            },
        )

    timeout = int(ctx.config.get("visual_review_timeout_sec") or 1800)
    poll = max(float(ctx.config.get("autorun_poll_sec") or 5.0), 1.0)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = read_json(raw_path)
        marker = read_json(marker_path)
        valid, reason = raw_verdict_valid(raw, review_id)
        if valid and marker.get("review_id") == review_id and marker.get("status") == "complete":
            stop_and_capture(session, transcript_path)
            return raw, session, ""
        if not tmux_alive(session):
            return {}, session, reason if raw else "codex_visual_session_exited_without_verdict"
        time.sleep(poll)
    stop_and_capture(session, transcript_path)
    return {}, session, "codex_visual_review_timeout"


def merge_assess(
    ctx: NodeContext,
    run_dir: Path,
    numeric: dict[str, Any],
    raw: dict[str, Any],
    control: dict[str, Any],
    session: str,
) -> dict[str, Any]:
    decoded = decode_blind_verdict(raw, str(control["candidate_side"]))
    visual_path = run_dir / VISUAL_VERDICT_NAME
    visual = {
        "schema_version": 1,
        "provider": "codex",
        "review_id": control["review_id"],
        "status": "complete",
        **decoded,
        "confidence": raw.get("confidence"),
        "summary": raw.get("summary"),
        "per_prompt": raw.get("per_prompt"),
        "blind_assignment": {
            "candidate_side": control["candidate_side"],
            "baseline_side": control["baseline_side"],
        },
        "image_manifest": rel_to(ctx.worktree, Path(str(control["review_dir"])) / "image_manifest.json"),
        "raw_verdict": rel_to(ctx.worktree, Path(str(control["review_dir"])) / RAW_VERDICT_NAME),
        "codex_session": session,
        "completed_at_utc": utc_now(),
    }
    write_json(visual_path, visual)
    blockers: list[str] = []
    if decoded["overall"] == "fail":
        blockers.append(f"codex_visual:fail:{decoded['max_artifact_severity']}")
    elif decoded["overall"] == "inconclusive":
        blockers.append("codex_visual:inconclusive")
    merged = {
        **numeric,
        "visual_status": "complete",
        "visual_provider": "codex",
        "codex_visual_overall": decoded["overall"],
        "candidate_visual_relation": decoded["candidate_relation"],
        "max_artifact_severity": decoded["max_artifact_severity"],
        "new_artifacts": decoded["new_artifacts"],
        "codex_visual_verdict": rel_to(ctx.worktree, visual_path),
        "quality_status": "available" if not blockers else "blocked_quality",
        "quality_blockers": blockers,
        "collector_quality_blockers": [],
        "updated_at_utc": utc_now(),
    }
    for legacy in ("gemini_overall", "gemini_verdict"):
        merged.pop(legacy, None)
    write_json(run_dir / ASSESS_NAME, merged)
    return merged


def existing_complete_assess(run_dir: Path) -> dict[str, Any]:
    data = read_json(run_dir / ASSESS_NAME)
    if (
        data.get("visual_provider") == "codex"
        and data.get("codex_visual_overall") in {"pass", "fail"}
        and isinstance(data.get("lpips_max"), (int, float))
    ):
        return data
    return {}


def run(ctx: NodeContext) -> NodeResult:
    runs = completed_runs(ctx.worktree)
    if not runs:
        return NodeResult("missing", updates={"visual_review_reason": "no_completed_full_run"}, message="no_completed_full_run")
    run_dir = runs[-1]
    existing = existing_complete_assess(run_dir)
    if existing:
        path = run_dir / ASSESS_NAME
        return NodeResult(
            "reviewed",
            updates={
                "visual_review_reason": "existing_codex_visual_assess",
                "visual_review_run": rel_to(ctx.worktree, run_dir),
                "codex_visual_overall": existing.get("codex_visual_overall"),
            },
            artifacts=[rel_to(ctx.worktree, path)],
            message="existing_codex_visual_assess",
        )
    profile = load_model_profile(ctx)
    baseline = baseline_run(ctx, profile)
    if not baseline:
        return NodeResult("infra_blocked", updates={"visual_review_reason": "baseline_run_missing"}, message="baseline_run_missing")
    control, reason = build_review_bundle(ctx, run_dir, baseline)
    if reason:
        return NodeResult("infra_blocked", updates={"visual_review_reason": reason}, message=reason)
    lpips, reason = run_lpips(ctx, run_dir, profile, control["lpips_pairs"])
    if reason:
        return NodeResult("infra_blocked", updates={"visual_review_reason": reason}, message=reason)
    numeric, reason = numeric_assess(ctx, run_dir, profile, lpips)
    if reason:
        return NodeResult("infra_blocked", updates={"visual_review_reason": reason}, message=reason)
    if ctx.dry_run:
        return NodeResult(
            "infra_blocked",
            updates={"visual_review_reason": "dry_run", "visual_review_request": control},
            artifacts=[rel_to(ctx.worktree, run_dir / NUMERIC_ASSESS_NAME)],
            message="dry_run",
        )
    raw, session, reason = launch_and_wait(ctx, control, run_dir)
    if reason:
        return NodeResult(
            "infra_blocked",
            updates={
                "visual_review_reason": reason,
                "visual_review_run": rel_to(ctx.worktree, run_dir),
                "codex_visual_session": session,
            },
            message=reason,
        )
    merged = merge_assess(ctx, run_dir, numeric, raw, control, session)
    artifacts = [
        rel_to(ctx.worktree, run_dir / NUMERIC_ASSESS_NAME),
        rel_to(ctx.worktree, run_dir / VISUAL_VERDICT_NAME),
        rel_to(ctx.worktree, run_dir / ASSESS_NAME),
    ]
    return NodeResult(
        "reviewed",
        updates={
            "visual_review_reason": "codex_visual_complete",
            "visual_review_run": rel_to(ctx.worktree, run_dir),
            "codex_visual_overall": merged.get("codex_visual_overall"),
            "codex_visual_session": session,
        },
        artifacts=artifacts,
        message="codex_visual_complete",
    )
