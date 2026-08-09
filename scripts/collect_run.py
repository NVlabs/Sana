#!/usr/bin/env python3
"""Collect artifacts for one autovideo run bundle.

This completes the control-plane loop: inspect a generated run directory,
classify its status, optionally extract video frames, and write the canonical
outputs/ artifacts:

- patch_summary.md
- benchmark.json
- quality.json
- risk_notes.md
- collection.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ERROR_PATTERNS = (
    "traceback (most recent call last)",
    "runtimeerror:",
    "cuda out of memory",
    "outofmemoryerror",
    "error: repository not found",
    "error:",
    "fatal:",
    "slurmstepd: error",
    "command not found",
    "no such file or directory",
)

TIMING_FIELDS = ("total_s", "denoise_s", "decode_s")
PRESERVED_RUNNER_BENCHMARK_FIELDS = (
    "schema_version",
    "timing_scope",
    "timing_note",
    "warm_steady_state",
    "warmup_requests",
    "includes_model_load",
    "load_excluded_request_s",
    "source_phase_subset_s",
    "wall_total_s",
    "max_device_memory_used_mib",
    "config",
    "aggregate",
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOG_TIMING_PATTERNS = {
    "denoise_s": re.compile(
        r"\[Cosmos3DenoisingStage\]\s+finished in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds",
        re.IGNORECASE,
    ),
    "decode_s": re.compile(
        r"\[Cosmos3DecodingStage\]\s+finished in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds",
        re.IGNORECASE,
    ),
    "total_s": re.compile(
        r"Pixel data generated successfully in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds",
        re.IGNORECASE,
    ),
}
NVIDIA_KEY_ENVS = ("NVIDIA_API_KEY", "NVIDIA_VISION_API_KEY", "API_KEY", "NGC_API_KEY")
STRICT_QUALITY_JUDGES = ("lpips", "nvidia_gemini")
DEFAULT_FRAME_COUNT = 189
LPIPS_MAX_PAIRS = 48
LPIPS_STRATIFIED_PAIRS = 32
LPIPS_WORST_CASE_PAIRS = 16
GEMINI_MAX_FRAME_PAIRS = 32
GEMINI_STRATIFIED_PAIRS = 24
GEMINI_WORST_CASE_PAIRS = 8
GEMINI_VIDEO_MAX_FRAMES = 32
GEMINI_VIDEO_FRAME_INTERVAL = 0.5
PATCH_BOUNDARY_SIZES = (8, 16, 32)
GEMINI_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def max_gemini_severity(result: dict[str, Any] | None) -> str:
    artifacts = (result or {}).get("new_artifacts") or []
    if not artifacts:
        return "none"
    return max(
        (str(artifact.get("severity", "low")) for artifact in artifacts),
        key=lambda severity: GEMINI_SEVERITY_RANK.get(severity, 1),
    )


def gemini_quality_blocker(judge: dict[str, Any]) -> str | None:
    if judge.get("status") != "complete":
        return None
    result = judge.get("result") or {}
    overall = result.get("overall")
    severity = max_gemini_severity(result)
    if overall == "fail":
        return f"nvidia_gemini:fail:{severity}"
    if GEMINI_SEVERITY_RANK.get(severity, 0) >= GEMINI_SEVERITY_RANK["medium"]:
        return f"nvidia_gemini:artifact:{severity}"
    return None


def append_status_history(metadata: dict[str, Any], status: str, reason: str = "") -> None:
    previous = metadata.get("status")
    if previous == status:
        return
    history = metadata.setdefault("status_history", [])
    if not isinstance(history, list):
        history = []
        metadata["status_history"] = history
    history.append(
        {
            "status": status,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
    )


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def parse_run_log_timing(log_path: Path) -> dict[str, float | None]:
    timing: dict[str, float | None] = {field: None for field in TIMING_FIELDS}
    if not log_path.exists():
        return timing

    text = strip_ansi(log_path.read_text(errors="replace"))
    for field, pattern in LOG_TIMING_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            timing[field] = float(matches[-1].group(1))
    return timing


def parse_existing_benchmark(path: Path) -> dict[str, Any]:
    data = load_json(path)
    timing: dict[str, Any] = {field: None for field in TIMING_FIELDS}
    timing["stage_seconds"] = {}
    if not data:
        return timing

    for field in TIMING_FIELDS:
        value = data.get(field)
        if isinstance(value, (int, float)):
            timing[field] = float(value)

    # Diffusers-style runners (e.g. HunyuanVideo gpu_infer.py) write timings
    # NESTED under "timings" and memory under "memory", reporting a single
    # end-to-end generation call ("generate_s") plus a wall-clock "total_s" that
    # INCLUDES one-time model load/placement. The framework's speedup convention
    # (see Cosmos3 models/cosmos3.toml [baseline]: total_s == denoise_s + decode_s,
    # model load excluded) measures GENERATION time, so map generate_s ->
    # total_s/denoise_s and preserve the raw nested timings + memory as evidence
    # rather than overwriting them or adopting the load-inclusive wall total.
    nested = data.get("timings")
    if isinstance(nested, dict):
        generate_s = nested.get("generate_s")
        if isinstance(generate_s, (int, float)):
            if timing["total_s"] is None:
                timing["total_s"] = float(generate_s)
            if timing["denoise_s"] is None:
                timing["denoise_s"] = float(generate_s)
        timing["timings"] = {
            str(key): float(value)
            for key, value in nested.items()
            if isinstance(value, (int, float))
        }
    memory = data.get("memory")
    if isinstance(memory, dict):
        timing["memory"] = {
            str(key): float(value)
            for key, value in memory.items()
            if isinstance(value, (int, float))
        }

    for field in PRESERVED_RUNNER_BENCHMARK_FIELDS:
        if field in data:
            timing[field] = data[field]

    stage_seconds = data.get("stage_seconds")
    if isinstance(stage_seconds, dict):
        timing["stage_seconds"] = {
            str(key): float(value)
            for key, value in stage_seconds.items()
            if isinstance(value, (int, float))
        }

    total_ms = data.get("total_duration_ms")
    if timing["total_s"] is None and isinstance(total_ms, (int, float)):
        timing["total_s"] = float(total_ms) / 1000.0

    stages = data.get("stages") or data.get("steps") or []
    if isinstance(stages, list):
        for item in stages:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("stage") or item.get("step") or "")
            duration = (
                item.get("duration_ms")
                or item.get("execution_time_ms")
                or item.get("elapsed_ms")
            )
            if name and isinstance(duration, (int, float)):
                timing["stage_seconds"][name] = float(duration) / 1000.0

    if timing["denoise_s"] is None:
        denoise_s = sum(
            value
            for name, value in timing["stage_seconds"].items()
            if "denois" in name.lower()
        )
        timing["denoise_s"] = denoise_s or None
    if timing["decode_s"] is None:
        decode_s = sum(
            value
            for name, value in timing["stage_seconds"].items()
            if "decod" in name.lower() or "vae" in name.lower()
        )
        timing["decode_s"] = decode_s or None
    return timing


def build_benchmark(benchmark_path: Path, log_path: Path, model_hint: str = "") -> dict[str, Any]:
    existing = parse_existing_benchmark(benchmark_path)
    log_timing = parse_run_log_timing(log_path)
    stage_seconds = dict(existing.get("stage_seconds") or {})

    benchmark: dict[str, Any] = {}
    sources: dict[str, str | None] = {}
    for field in TIMING_FIELDS:
        if log_timing.get(field) is not None:
            benchmark[field] = log_timing[field]
            sources[field] = "run.log"
        else:
            benchmark[field] = existing.get(field)
            sources[field] = "benchmark.json" if existing.get(field) is not None else None

    # Stage label must match the model so reports do not drift. The Cosmos3
    # reference audits key off "Cosmos3DenoisingStage"/"Cosmos3DecodingStage";
    # other models (e.g. HunyuanVideo, whose diffusers pipe() is a single fused
    # generation call covering denoise + VAE decode) use a generic key.
    is_cosmos = "cosmos" in (model_hint or "").lower()
    denoise_label = "Cosmos3DenoisingStage" if is_cosmos else "generation"
    decode_label = "Cosmos3DecodingStage" if is_cosmos else "decode"
    if benchmark["denoise_s"] is not None:
        stage_seconds.setdefault(denoise_label, benchmark["denoise_s"])
    if benchmark["decode_s"] is not None:
        stage_seconds.setdefault(decode_label, benchmark["decode_s"])

    benchmark["stage_seconds"] = stage_seconds
    benchmark["sources"] = sources
    # Preserve the runner's raw nested timings + memory as durable evidence so a
    # collect pass never discards wall-clock/load/peak-memory data (the prior
    # failure: collecting the Hunyuan baseline nulled its runner timings).
    if existing.get("timings"):
        benchmark["timings"] = existing["timings"]
    if existing.get("memory"):
        benchmark["memory"] = existing["memory"]
    for field in PRESERVED_RUNNER_BENCHMARK_FIELDS:
        if field in existing:
            benchmark[field] = existing[field]
    benchmark["collected_at_utc"] = datetime.now(timezone.utc).isoformat()
    return benchmark


def detect_log_errors(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    lowered = text.lower()
    hits = [pattern for pattern in ERROR_PATTERNS if pattern in lowered]
    return hits


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "bytes": stat.st_size,
        "path": str(path),
    }


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def executable_path(value: str) -> str | None:
    expanded = Path(value).expanduser()
    has_separator = "/" in value or os.sep in value
    if expanded.is_absolute() or has_separator:
        if expanded.exists() and os.access(expanded, os.X_OK):
            return str(expanded)
        return None
    return shutil.which(value)


def resolve_ffmpeg(override: str | None) -> dict[str, Any]:
    config: list[tuple[str, str]] = []
    if override:
        config.append(("cli", override))
    if os.environ.get("FFMPEG_BIN"):
        config.append(("env", os.environ["FFMPEG_BIN"]))
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        config.append(("path", path_ffmpeg))
    config.append(("lustre", str(Path.home() / "lustre/bin/ffmpeg")))

    checked: list[dict[str, str]] = []
    for source, config in config:
        checked.append({"source": source, "config": config})
        resolved = executable_path(config)
        if resolved:
            return {"path": resolved, "source": source, "checked": checked}
    return {"path": None, "source": None, "checked": checked}


def resolve_ffprobe(ffmpeg_path: str) -> str | None:
    sibling = Path(ffmpeg_path).with_name("ffprobe")
    if sibling.exists() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("ffprobe")


def probe_video_duration(video_path: Path, ffmpeg_path: str) -> float | None:
    ffprobe = resolve_ffprobe(ffmpeg_path)
    if not ffprobe:
        return None
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    try:
        duration = float(proc.stdout.strip())
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration


def sample_timestamps(duration_s: float, count: int) -> list[float]:
    count = max(1, count)
    return [duration_s * (index + 0.5) / count for index in range(count)]


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    fps: float,
    frame_count: int,
    overwrite: bool,
    ffmpeg_override: str | None,
) -> dict[str, Any]:
    if not video_path.exists():
        return {"status": "skipped", "reason": "video_missing", "count": 0}

    ffmpeg = resolve_ffmpeg(ffmpeg_override)
    ffmpeg_path = ffmpeg.get("path")
    if not ffmpeg_path:
        return {
            "status": "skipped",
            "reason": "ffmpeg_missing",
            "count": 0,
            "checked": ffmpeg["checked"],
        }

    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("f_*.png"))
    if existing and not overwrite:
        return {
            "status": "existing",
            "count": len(existing),
            "ffmpeg": ffmpeg_path,
            "ffmpeg_source": ffmpeg["source"],
        }

    for old in existing:
        old.unlink()

    duration_s = probe_video_duration(video_path, ffmpeg_path)
    # Single-pass passthrough decode (-vsync 0) yields exactly the video's native
    # frames in chronological order (frame-accurate). This avoids the trailing
    # frame that per-timestamp input seeking drops near end-of-stream, so a
    # 129-frame HunyuanVideo output produces a 129-frame set (not 128). Baseline
    # and config runs use the identical policy, so aligned LPIPS/Gemini frame
    # pairs stay index-matched. frame_count caps below native via even subsample.
    tmp_glob = "f_*.png"
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        str(frames_dir / "f_%05d.png"),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    produced = sorted(frames_dir.glob(tmp_glob))
    if proc.returncode != 0 or not produced:
        return {
            "status": "failed",
            "count": len(produced),
            "ffmpeg": ffmpeg_path,
            "ffmpeg_source": ffmpeg["source"],
            "stderr": proc.stderr.strip(),
        }

    target = max(1, frame_count)
    if len(produced) > target:
        if target == 1:
            keep_indices = {0}
        else:
            keep_indices = {
                round(i * (len(produced) - 1) / (target - 1)) for i in range(target)
            }
        keep = {produced[i] for i in keep_indices}
        for frame_path in produced:
            if frame_path not in keep:
                frame_path.unlink()
        produced = sorted(frames_dir.glob(tmp_glob))

    # Renumber to contiguous zero-padded f_001.. (sortable, matches f_*.png glob
    # used by the LPIPS/Gemini pairing in plan_eval and collect_run).
    for new_index, frame_path in enumerate(sorted(produced), start=1):
        dest = frames_dir / f"f_{new_index:03d}.png"
        if frame_path != dest:
            frame_path.rename(dest)
    final = sorted(frames_dir.glob("f_*.png"))
    return {
        "status": "created",
        "count": len(final),
        "ffmpeg": ffmpeg_path,
        "ffmpeg_source": ffmpeg["source"],
        "duration_s": duration_s,
    }


def determine_status(
    metadata: dict[str, Any],
    log_path: Path,
    video_path: Path,
    log_errors: list[str],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    previous = str(metadata.get("status") or "")

    if log_errors:
        notes.append("log contains error patterns: " + ", ".join(log_errors))
        return "failed", notes

    if nonempty(video_path) and nonempty(log_path):
        return "completed", notes

    if previous in {"prepared", "submitted", "running"} and not log_path.exists():
        notes.append("no run.log yet")
        return previous or "prepared", notes

    if log_path.exists() and not nonempty(video_path):
        notes.append("run.log exists but out.mp4 is missing or empty")
        return "failed", notes

    if video_path.exists() and not nonempty(log_path):
        notes.append("out.mp4 exists but run.log is missing or empty")
        return "failed", notes

    notes.append("required artifacts are missing")
    return "blocked", notes


def deferred(reason: str) -> dict[str, str]:
    return {"status": "deferred", "reason": reason}


def blocked(reason: str) -> dict[str, str]:
    return {"status": "blocked", "reason": reason}


def have_nvidia_key() -> bool:
    return any(os.environ.get(name) for name in NVIDIA_KEY_ENVS)


def nvidia_helper_path() -> Path:
    default = Path.home() / ".codex/skills/nvidia-vision-api/scripts/nvidia_multimodal_chat.py"
    return Path(os.environ.get("NVIDIA_VISION_HELPER", default)).expanduser()


def resolve_baseline_frames(args: argparse.Namespace) -> list[str]:
    frames = [str(Path(path).expanduser()) for path in (args.baseline_frame or [])]
    baseline_run_dir = getattr(args, "baseline_run_dir", None)
    if baseline_run_dir:
        baseline_dir = Path(baseline_run_dir).expanduser()
        frames_dir = baseline_dir / "outputs" / "frames"
        frames.extend(str(path) for path in sorted(frames_dir.glob("f_*.png")))
    # Keep order stable and remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        if frame not in seen:
            deduped.append(frame)
            seen.add(frame)
    return deduped


def paired_frames(frame_paths: list[Path], baseline_frames: list[str]) -> list[tuple[Path, Path]]:
    baseline_paths = [Path(path).expanduser() for path in baseline_frames]
    return list(zip(baseline_paths, frame_paths))


def select_pairs(
    pairs: list[tuple[Path, Path]],
    limit: int,
) -> list[tuple[Path, Path]]:
    if len(pairs) <= limit:
        return pairs
    if limit <= 1:
        return [pairs[0]]
    selected = []
    last = len(pairs) - 1
    for i in range(limit):
        selected.append(pairs[round(i * last / (limit - 1))])
    return selected


def select_stratified_and_worst_pairs(
    pairs: list[tuple[Path, Path]],
    stratified_limit: int,
    worst_case_limit: int,
    total_limit: int,
) -> list[tuple[Path, Path]]:
    """Select chronological coverage plus frames with largest pixel drift."""
    selected: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()

    def add(pair: tuple[Path, Path]) -> None:
        key = (str(pair[0]), str(pair[1]))
        if key not in seen and len(selected) < total_limit:
            selected.append(pair)
            seen.add(key)

    for pair in select_pairs(pairs, min(stratified_limit, total_limit)):
        add(pair)

    if worst_case_limit <= 0 or len(selected) >= total_limit:
        return selected

    try:
        import numpy as np  # type: ignore

        scored: list[tuple[float, tuple[Path, Path]]] = []
        for baseline, config in pairs:
            ba = image_array(baseline)
            ca = image_array(config)
            if ba.shape != ca.shape:
                continue
            scored.append((float(np.abs(ca - ba).mean()), (baseline, config)))
        for _score, pair in sorted(scored, key=lambda item: item[0], reverse=True)[:worst_case_limit]:
            add(pair)
    except Exception:
        # Pixel dependencies are best-effort here; LPIPS/Gemini still receive the
        # stratified chronological pairs.
        return selected
    return selected


def run_lpips_judge(frame_paths: list[Path], baseline_frames: list[str], skip: bool) -> dict[str, Any]:
    if skip:
        return deferred("disabled")
    if not frame_paths:
        return blocked("frames_missing")
    if not baseline_frames:
        return blocked("baseline_frame_missing")

    tool = project_root() / "tools/vision/lpips_judge.py"
    if not tool.exists():
        return blocked("tool_missing")
    missing = [
        module
        for module in ("torch", "lpips")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return {"status": "blocked", "reason": "dependencies_missing", "missing": missing}

    with tempfile.TemporaryDirectory(prefix="autovideo-lpips-") as tmp:
        out_path = Path(tmp) / "lpips.json"
        pairs = select_stratified_and_worst_pairs(
            paired_frames(frame_paths, baseline_frames),
            stratified_limit=LPIPS_STRATIFIED_PAIRS,
            worst_case_limit=LPIPS_WORST_CASE_PAIRS,
            total_limit=LPIPS_MAX_PAIRS,
        )
        cmd = [sys.executable, str(tool), "--out", str(out_path)]
        for baseline, config in pairs:
            cmd.extend(["--baseline-frame", str(baseline), "--config-frame", str(config)])
        proc = subprocess.run(
            cmd,
            cwd=project_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = load_json(out_path)
        if proc.returncode != 0:
            return {
                "status": "blocked",
                "reason": "lpips_judge_failed",
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
                "result": payload,
            }
        status = payload.get("status")
        if status != "ok":
            return {"status": "blocked", "reason": status or "lpips_not_ok", "result": payload}
        return {"status": "complete", "result": payload, "pairs_scored": len(pairs)}


def run_nvidia_gemini_judge(
    frame_paths: list[Path],
    baseline_frames: list[str],
    config_id: str,
    skip: bool,
    config_video: Path | None = None,
    baseline_video: Path | None = None,
    side_by_side_video: Path | None = None,
) -> dict[str, Any]:
    if skip:
        return deferred("disabled")
    if not frame_paths:
        return blocked("frames_missing")
    if not baseline_frames:
        return blocked("baseline_frame_missing")

    tool = project_root() / "tools/vision/nvidia_gemini_judge.py"
    if not tool.exists():
        return blocked("tool_missing")
    if not have_nvidia_key():
        return blocked("api_key_missing")
    helper = nvidia_helper_path()
    if not helper.exists():
        return blocked("helper_missing")

    with tempfile.TemporaryDirectory(prefix="autovideo-gemini-") as tmp:
        out_path = Path(tmp) / "nvidia_gemini.json"
        pairs = select_stratified_and_worst_pairs(
            paired_frames(frame_paths, baseline_frames),
            stratified_limit=GEMINI_STRATIFIED_PAIRS,
            worst_case_limit=GEMINI_WORST_CASE_PAIRS,
            total_limit=GEMINI_MAX_FRAME_PAIRS,
        )
        cmd = [
            sys.executable,
            str(tool),
            "--out",
            str(out_path),
            "--video-max-frames",
            str(GEMINI_VIDEO_MAX_FRAMES),
            "--video-frame-interval",
            str(GEMINI_VIDEO_FRAME_INTERVAL),
            "--context",
            (
                f"Autovideo config run: {config_id}. Images are provided "
                "as matched baseline/config pairs in chronological order. "
                "Videos, when present, are provided as baseline video first, "
                "config video second, and side-by-side video third. Prioritize "
                "temporal flicker/popping, patch-boundary instability, patch-level "
                "texture mismatch, broken motion coherence, blur/detail loss, "
                "ghosting/smearing, snow/static, and severe perceptual degradation."
            ),
        ]
        for baseline, config in pairs:
            cmd.extend(["--baseline-frame", str(baseline)])
            cmd.extend(["--config-frame", str(config)])
        if baseline_video and baseline_video.exists():
            cmd.extend(["--video", str(baseline_video)])
        if config_video and config_video.exists():
            cmd.extend(["--video", str(config_video)])
        if side_by_side_video and side_by_side_video.exists():
            cmd.extend(["--video", str(side_by_side_video)])

        proc = subprocess.run(
            cmd,
            cwd=project_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = load_json(out_path)
        if proc.returncode != 0:
            return {
                "status": "blocked",
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
                "result": payload,
            }
        if payload.get("overall") in (None, "inconclusive"):
            return {"status": "blocked", "reason": "gemini_inconclusive", "result": payload}
        return {"status": "complete", "result": payload, "pairs_scored": len(pairs)}


def load_image_arrays():
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"image metric dependencies missing: {exc}") from exc
    return np, Image


def image_array(path: Path):
    np, Image = load_image_arrays()
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def mean_abs_gradient(arr) -> float:
    import numpy as np  # type: ignore

    dx = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean() if arr.shape[1] > 1 else 0.0
    dy = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean() if arr.shape[0] > 1 else 0.0
    return float(dx + dy)


def patch_boundary_score(arr, patch: int = 16) -> float:
    import numpy as np  # type: ignore

    scores = []
    for x in range(patch, arr.shape[1], patch):
        scores.append(np.abs(arr[:, x, :] - arr[:, x - 1, :]).mean())
    for y in range(patch, arr.shape[0], patch):
        scores.append(np.abs(arr[y, :, :] - arr[y - 1, :, :]).mean())
    return float(np.mean(scores)) if scores else 0.0


def patch_boundary_scores(arr) -> dict[int, float]:
    return {patch: patch_boundary_score(arr, patch=patch) for patch in PATCH_BOUNDARY_SIZES}


def build_off_identity(frame_paths: list[Path], baseline_frames: list[str]) -> dict[str, Any]:
    pairs = paired_frames(frame_paths, baseline_frames)
    if not frame_paths:
        return blocked("frames_missing")
    if not baseline_frames:
        return blocked("baseline_frame_missing")
    if not pairs:
        return blocked("no_frame_pairs")
    try:
        np, _Image = load_image_arrays()
        max_abs = 0.0
        nonidentical = 0
        for baseline, config in pairs:
            ba = image_array(baseline)
            ca = image_array(config)
            if ba.shape != ca.shape:
                return {"status": "failed", "reason": "shape_mismatch", "baseline": str(baseline), "config": str(config)}
            diff = np.abs(ca - ba)
            frame_max = float(diff.max())
            max_abs = max(max_abs, frame_max)
            if frame_max != 0.0:
                nonidentical += 1
        return {
            "status": "ok" if nonidentical == 0 else "different",
            "pairs": len(pairs),
            "nonidentical_frames": nonidentical,
            "max_abs_diff_uint8": max_abs,
        }
    except Exception as exc:
        return {"status": "blocked", "reason": str(exc)}


def build_pixel_metrics(frame_paths: list[Path], baseline_frames: list[str]) -> dict[str, Any]:
    pairs = paired_frames(frame_paths, baseline_frames)
    if not frame_paths:
        return blocked("frames_missing")
    if not baseline_frames:
        return blocked("baseline_frame_missing")
    if not pairs:
        return blocked("no_frame_pairs")
    try:
        np, _Image = load_image_arrays()
        mse_values: list[float] = []
        mae_values: list[float] = []
        psnr_values: list[float] = []
        sharpness_ratios: list[float] = []
        patch_ratios: list[float] = []
        patch_ratios_by_size: dict[int, list[float]] = {patch: [] for patch in PATCH_BOUNDARY_SIZES}
        baseline_prev = None
        config_prev = None
        temporal_delta_errors: list[float] = []
        temporal_jitter_ratios: list[float] = []
        for baseline, config in pairs:
            ba = image_array(baseline)
            ca = image_array(config)
            if ba.shape != ca.shape:
                return {"status": "blocked", "reason": "shape_mismatch", "baseline": str(baseline), "config": str(config)}
            diff = ca - ba
            mse = float(np.square(diff).mean())
            mae = float(np.abs(diff).mean())
            psnr = math.inf if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            base_sharp = mean_abs_gradient(ba)
            cand_sharp = mean_abs_gradient(ca)
            base_patch_scores = patch_boundary_scores(ba)
            cand_patch_scores = patch_boundary_scores(ca)
            per_size_patch_ratios = []
            for patch in PATCH_BOUNDARY_SIZES:
                ratio = cand_patch_scores[patch] / max(base_patch_scores[patch], 1e-8)
                patch_ratios_by_size[patch].append(ratio)
                per_size_patch_ratios.append(ratio)
            mse_values.append(mse)
            mae_values.append(mae)
            psnr_values.append(psnr)
            sharpness_ratios.append(cand_sharp / max(base_sharp, 1e-8))
            patch_ratios.append(max(per_size_patch_ratios))
            if baseline_prev is not None and config_prev is not None:
                base_delta = ba - baseline_prev
                cand_delta = ca - config_prev
                temporal_delta_errors.append(float(np.abs(cand_delta - base_delta).mean()))
                base_delta_mag = float(np.abs(base_delta).mean())
                cand_delta_mag = float(np.abs(cand_delta).mean())
                temporal_jitter_ratios.append(cand_delta_mag / max(base_delta_mag, 1e-8))
            baseline_prev = ba
            config_prev = ca
        finite_psnr = [value for value in psnr_values if math.isfinite(value)]
        return {
            "status": "ok",
            "pairs": len(pairs),
            "mse_mean": fmean(mse_values),
            "mse_max": max(mse_values),
            "mean_abs_pixel_diff": fmean(mae_values),
            "psnr_mean": fmean(finite_psnr) if finite_psnr else None,
            "psnr_min": min(finite_psnr) if finite_psnr else None,
            "sharpness_ratio_mean": fmean(sharpness_ratios),
            "patch_boundary_ratio_mean": fmean(patch_ratios),
            "patch_boundary_ratio_max": max(patch_ratios),
            "patch_boundary_ratio_by_size_mean": {
                str(patch): fmean(values) if values else 0.0
                for patch, values in patch_ratios_by_size.items()
            },
            "patch_boundary_ratio_by_size_max": {
                str(patch): max(values) if values else 0.0
                for patch, values in patch_ratios_by_size.items()
            },
            "temporal_delta_error_mean": fmean(temporal_delta_errors) if temporal_delta_errors else 0.0,
            "temporal_delta_error_max": max(temporal_delta_errors) if temporal_delta_errors else 0.0,
            "temporal_jitter_ratio_mean": fmean(temporal_jitter_ratios) if temporal_jitter_ratios else 1.0,
            "temporal_jitter_ratio_min": min(temporal_jitter_ratios) if temporal_jitter_ratios else 1.0,
            "temporal_jitter_ratio_max": max(temporal_jitter_ratios) if temporal_jitter_ratios else 1.0,
        }
    except Exception as exc:
        return {"status": "blocked", "reason": str(exc)}


def build_quality(
    run_dir: Path,
    metadata: dict[str, Any],
    frames_dir: Path,
    frames: dict[str, Any],
    baseline_frames: list[str],
    skip_judges: bool,
    config_video: Path | None = None,
    baseline_video: Path | None = None,
    side_by_side_video: Path | None = None,
) -> dict[str, Any]:
    frame_paths = sorted(frames_dir.glob("f_*.png")) if frames_dir.exists() else []
    frame_metrics: dict[str, Any]
    if frame_paths:
        frame_metrics = {
            "status": "available",
            "frame_count": len(frame_paths),
            "frames": [rel(path, run_dir) for path in frame_paths],
        }
    else:
        frame_metrics = {
            "status": "deferred",
            "reason": "frames_missing",
            "frame_count": 0,
        }

    config_id = str(metadata.get("config_id", run_dir.name))
    off_identity = build_off_identity(frame_paths, baseline_frames)
    pixel_metrics = build_pixel_metrics(frame_paths, baseline_frames)
    judges = {
        "lpips": run_lpips_judge(frame_paths, baseline_frames, skip_judges),
        "nvidia_gemini": run_nvidia_gemini_judge(
            frame_paths,
            baseline_frames,
            config_id,
            skip_judges,
            config_video=config_video,
            baseline_video=baseline_video,
            side_by_side_video=side_by_side_video,
        ),
    }
    promotion_blockers: list[str] = []
    if not baseline_frames and not skip_judges:
        promotion_blockers.append("baseline_frames_missing")
    for name, result in judges.items():
        if name in STRICT_QUALITY_JUDGES and result.get("status") != "complete":
            promotion_blockers.append(f"{name}:{result.get('reason', result.get('status', 'missing'))}")
    gemini_blocker = gemini_quality_blocker(judges["nvidia_gemini"])
    if gemini_blocker:
        promotion_blockers.append(gemini_blocker)
    if pixel_metrics.get("status") != "ok" and not skip_judges:
        promotion_blockers.append(f"pixel_metrics:{pixel_metrics.get('reason', pixel_metrics.get('status'))}")
    return {
        "status": "blocked_quality" if promotion_blockers else ("available" if frame_paths else "deferred"),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_extraction": frames,
        "frame_metrics": frame_metrics,
        "off_identity": off_identity,
        "pixel_metrics": pixel_metrics,
        "judges": judges,
        "promotion_blockers": promotion_blockers,
    }


def resolve_baseline_video(args: argparse.Namespace) -> Path | None:
    baseline_run_dir = getattr(args, "baseline_run_dir", None)
    if not baseline_run_dir:
        return None
    config = Path(baseline_run_dir).expanduser() / "outputs" / "out.mp4"
    return config if config.exists() else None


def render_risk_notes(metadata: dict[str, Any]) -> str:
    if metadata.get("kind") == "baseline":
        return "no risk; baseline reference run\n"
    return "risk notes pending for non-baseline config run\n"


def render_patch_summary(
    run_dir: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    status: str,
    notes: list[str],
    paths: dict[str, Path],
    benchmark: dict[str, Any],
    frames: dict[str, Any],
    quality: dict[str, Any],
    log_errors: list[str],
) -> str:
    resolved_profile = manifest.get("resolved_profile", {})
    resolved_official = (
        resolved_profile.get("official_config", {})
        if isinstance(resolved_profile, dict)
        else {}
    )
    official = manifest.get("official_config", {}) or resolved_official
    artifacts = manifest.get("artifacts", {})
    lines = [
        f"# Config Report: {metadata.get('config_id', run_dir.name)}",
        "",
        f"Status: `{status}`",
        f"Run: `{run_dir.name}`",
        f"Collected: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Config",
        "",
    ]
    if official:
        for key in sorted(official):
            lines.append(f"- `{key}`: `{official[key]}`")
    else:
        lines.append("- official config: unavailable")

    lines.extend(
        [
            "",
            "## Timing",
            "",
            f"- total: `{format_seconds(benchmark.get('total_s'))}`",
            f"- denoise: `{format_seconds(benchmark.get('denoise_s'))}`",
            f"- decode: `{format_seconds(benchmark.get('decode_s'))}`",
        ]
    )
    stage_seconds = benchmark.get("stage_seconds") or {}
    if stage_seconds:
        lines.append("- stages:")
        for name, seconds in sorted(stage_seconds.items()):
            lines.append(f"  - `{name}`: `{format_seconds(seconds)}`")

    lines.extend(["", "## Artifacts", ""])
    for label, path in paths.items():
        if label == "patch_summary":
            continue
        info = file_info(path)
        if info["exists"]:
            lines.append(f"- {label}: `{rel(path, run_dir)}` ({info['bytes']} bytes)")
        else:
            lines.append(f"- {label}: missing (`{rel(path, run_dir)}`)")
    lines.append(f"- frames: `{frames.get('status')}` count=`{frames.get('count', 0)}`")

    lines.extend(["", "## Quality", ""])
    frame_metrics = quality.get("frame_metrics", {})
    lines.append(
        "- frame metrics: "
        f"`{frame_metrics.get('status', 'deferred')}` "
        f"count=`{frame_metrics.get('frame_count', 0)}`"
    )
    for name, result in sorted((quality.get("judges") or {}).items()):
        lines.append(f"- {name}: `{result.get('status', 'deferred')}`")
    blockers = quality.get("promotion_blockers") or []
    if blockers:
        lines.append("- promotion blockers: `" + "`, `".join(blockers) + "`")

    lines.extend(["", "## Notes", ""])
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- no collector notes")
    if log_errors:
        lines.append("- log error patterns: `" + "`, `".join(log_errors) + "`")
    if artifacts:
        lines.append("- artifact contract loaded from manifest")
    return "\n".join(lines) + "\n"


def collection_payload(
    run_dir: Path,
    paths: dict[str, Path],
    status: str,
    frames: dict[str, Any],
    benchmark: dict[str, Any],
    quality: dict[str, Any],
    notes: list[str],
    log_errors: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "run_dir": str(run_dir),
        "artifacts": {key: file_info(path) for key, path in paths.items()},
        "frames": frames,
        "timing": {
            "total_s": benchmark.get("total_s"),
            "denoise_s": benchmark.get("denoise_s"),
            "decode_s": benchmark.get("decode_s"),
            "stage_seconds": benchmark.get("stage_seconds", {}),
        },
        "quality": quality,
        "notes": notes,
        "log_errors": log_errors,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    metadata_path = run_dir / "metadata.json"
    metadata = load_json(metadata_path)
    if not metadata:
        raise SystemExit(f"Missing or invalid metadata.json: {metadata_path}")

    manifest = load_toml(run_dir / "manifest.resolved.toml")
    artifacts = manifest.get("artifacts", {})
    output_dir = run_dir / artifacts.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "log": output_dir / artifacts.get("log", "run.log"),
        "video": output_dir / artifacts.get("video", "out.mp4"),
        "side_by_side_video": output_dir / artifacts.get("side_by_side_video", "side_by_side.mp4"),
        "benchmark": output_dir / artifacts.get("benchmark", "benchmark.json"),
        "quality": output_dir / artifacts.get("quality", "quality.json"),
        "risk_notes": output_dir / artifacts.get("risk_notes", "risk_notes.md"),
        "collection": output_dir / artifacts.get("collection", "collection.json"),
        "patch_summary": output_dir / artifacts.get("patch_summary", "patch_summary.md"),
    }
    frames_dir = output_dir / artifacts.get("frames_dir", "frames")

    # The run self-describes its model and frame count via run_config.json
    # (written by the runner). Frame extraction defaults to the run's actual
    # num_frames so aligned LPIPS/Gemini frame sets match the model contract
    # (HunyuanVideo=129, Cosmos3=189) instead of a single hardcoded count.
    run_config = load_json(output_dir / "run_config.json")
    model_hint = ""
    if isinstance(run_config, dict):
        model_hint = str(run_config.get("model_path") or "")
    if not model_hint:
        model_hint = str(manifest.get("model_profile") or metadata.get("config_id") or "")

    if args.frame_count is not None:
        effective_frame_count = args.frame_count
    else:
        num_frames = run_config.get("num_frames") if isinstance(run_config, dict) else None
        effective_frame_count = (
            int(num_frames)
            if isinstance(num_frames, (int, float)) and num_frames
            else DEFAULT_FRAME_COUNT
        )

    log_errors = detect_log_errors(paths["log"])
    benchmark = build_benchmark(paths["benchmark"], paths["log"], model_hint=model_hint)
    write_json(paths["benchmark"], benchmark)

    status, notes = determine_status(
        metadata,
        paths["log"],
        paths["video"],
        log_errors,
    )

    should_extract = args.extract_frames or (status == "completed" and not args.no_extract_frames)
    if should_extract:
        frames = extract_frames(
            paths["video"],
            frames_dir,
            args.frame_fps,
            effective_frame_count,
            args.overwrite_frames,
            args.ffmpeg,
        )
    else:
        existing = len(list(frames_dir.glob("f_*.png"))) if frames_dir.exists() else 0
        frames = {"status": "skipped", "reason": "disabled", "count": existing}

    quality = build_quality(
        run_dir,
        metadata,
        frames_dir,
        frames,
        resolve_baseline_frames(args),
        args.skip_judges,
        config_video=paths["video"],
        baseline_video=resolve_baseline_video(args),
        side_by_side_video=paths["side_by_side_video"],
    )
    write_json(paths["quality"], quality)
    paths["risk_notes"].write_text(render_risk_notes(metadata))
    write_json(
        paths["collection"],
        collection_payload(run_dir, paths, status, frames, benchmark, quality, notes, log_errors),
    )

    patch_summary = render_patch_summary(
        run_dir,
        metadata,
        manifest,
        status,
        notes,
        paths,
        benchmark,
        frames,
        quality,
        log_errors,
    )
    paths["patch_summary"].write_text(patch_summary)

    append_status_history(metadata, status, "; ".join(notes))
    metadata.update(
        {
            "status": status,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "collector": {
                "patch_summary": str(paths["patch_summary"]),
                "benchmark": str(paths["benchmark"]),
                "quality": str(paths["quality"]),
                "risk_notes": str(paths["risk_notes"]),
                "frames": frames,
                "notes": notes,
                "log_errors": log_errors,
                "timing": {
                    "total_s": benchmark.get("total_s"),
                    "denoise_s": benchmark.get("denoise_s"),
                    "decode_s": benchmark.get("decode_s"),
                },
            },
        }
    )
    write_json(metadata_path, metadata)
    write_json(
        paths["collection"],
        collection_payload(run_dir, paths, status, frames, benchmark, quality, notes, log_errors),
    )
    return {
        "status": status,
        "run_dir": str(run_dir),
        "patch_summary": str(paths["patch_summary"]),
        "benchmark": str(paths["benchmark"]),
        "quality": str(paths["quality"]),
        "risk_notes": str(paths["risk_notes"]),
        "frames": frames,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run bundle directory")
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames even if the run is not marked completed",
    )
    parser.add_argument(
        "--no-extract-frames",
        action="store_true",
        help="Do not auto-extract frames for completed runs",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Regenerate frames if they already exist",
    )
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help=(
            "Frames to extract for all-frame metrics. Default: the run's own "
            "num_frames from run_config.json (model-agnostic; HunyuanVideo=129, "
            "Cosmos3=189), falling back to DEFAULT_FRAME_COUNT when unknown."
        ),
    )
    parser.add_argument("--ffmpeg", help="ffmpeg executable path override")
    parser.add_argument(
        "--baseline-frame",
        action="append",
        help="Baseline frame for optional LPIPS comparison; may be repeated",
    )
    parser.add_argument(
        "--baseline-run-dir",
        help="Baseline run directory whose outputs/frames are paired with config frames",
    )
    parser.add_argument(
        "--skip-judges",
        action="store_true",
        help="Skip optional network/GPU/dependency-backed quality judges",
    )
    args = parser.parse_args()

    result = collect(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    status = result["status"]
    return 1 if status in {"failed", "blocked", "rejected_quality"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
