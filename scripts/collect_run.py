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
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - Python < 3.11
    raise SystemExit("Python 3.11+ is required for tomllib TOML support") from exc


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


def build_benchmark(benchmark_path: Path, log_path: Path) -> dict[str, Any]:
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

    if benchmark["denoise_s"] is not None:
        stage_seconds.setdefault("Cosmos3DenoisingStage", benchmark["denoise_s"])
    if benchmark["decode_s"] is not None:
        stage_seconds.setdefault("Cosmos3DecodingStage", benchmark["decode_s"])

    benchmark["stage_seconds"] = stage_seconds
    benchmark["sources"] = sources
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
    candidates: list[tuple[str, str]] = []
    if override:
        candidates.append(("cli", override))
    if os.environ.get("FFMPEG_BIN"):
        candidates.append(("env", os.environ["FFMPEG_BIN"]))
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        candidates.append(("path", path_ffmpeg))
    candidates.append(("lustre", str(Path.home() / "lustre/bin/ffmpeg")))

    checked: list[dict[str, str]] = []
    for source, candidate in candidates:
        checked.append({"source": source, "candidate": candidate})
        resolved = executable_path(candidate)
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
    errors: list[str] = []
    if duration_s:
        for index, timestamp in enumerate(sample_timestamps(duration_s, frame_count), start=1):
            out_path = frames_dir / f"f_{index:03d}.png"
            proc = subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(out_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                errors.append(proc.stderr.strip())
        created = sorted(frames_dir.glob("f_*.png"))
        if created:
            result: dict[str, Any] = {
                "status": "created",
                "count": len(created),
                "ffmpeg": ffmpeg_path,
                "ffmpeg_source": ffmpeg["source"],
                "duration_s": duration_s,
            }
            if errors:
                result["partial_errors"] = [error for error in errors if error]
            return result

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        "-frames:v",
        str(max(1, frame_count)),
        str(frames_dir / "f_%03d.png"),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return {
            "status": "failed",
            "count": 0,
            "ffmpeg": ffmpeg_path,
            "ffmpeg_source": ffmpeg["source"],
            "stderr": proc.stderr.strip(),
        }
    return {
        "status": "created",
        "count": len(list(frames_dir.glob("f_*.png"))),
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


def have_nvidia_key() -> bool:
    return any(os.environ.get(name) for name in NVIDIA_KEY_ENVS)


def nvidia_helper_path() -> Path:
    default = Path.home() / ".codex/skills/nvidia-vision-api/scripts/nvidia_multimodal_chat.py"
    return Path(os.environ.get("NVIDIA_VISION_HELPER", default)).expanduser()


def run_lpips_judge(frame_paths: list[Path], baseline_frames: list[str], skip: bool) -> dict[str, Any]:
    if skip:
        return deferred("disabled")
    if not frame_paths:
        return deferred("frames_missing")
    if not baseline_frames:
        return deferred("baseline_frame_missing")

    tool = project_root() / "tools/vision/lpips_judge.py"
    if not tool.exists():
        return deferred("tool_missing")
    missing = [
        module
        for module in ("torch", "lpips")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return {"status": "deferred", "reason": "dependencies_missing", "missing": missing}

    with tempfile.TemporaryDirectory(prefix="autovideo-lpips-") as tmp:
        out_path = Path(tmp) / "lpips.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(tool),
                "--baseline-frame",
                str(Path(baseline_frames[0]).expanduser()),
                "--candidate-frame",
                str(frame_paths[0]),
                "--out",
                str(out_path),
            ],
            cwd=project_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = load_json(out_path)
        if proc.returncode != 0:
            return {
                "status": "failed",
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
                "result": payload,
            }
        return {"status": "complete", "result": payload}


def run_nvidia_gemini_judge(
    frame_paths: list[Path],
    candidate_id: str,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return deferred("disabled")
    if not frame_paths:
        return deferred("frames_missing")

    tool = project_root() / "tools/vision/nvidia_gemini_judge.py"
    if not tool.exists():
        return deferred("tool_missing")
    if not have_nvidia_key():
        return deferred("api_key_missing")
    helper = nvidia_helper_path()
    if not helper.exists():
        return deferred("helper_missing")

    with tempfile.TemporaryDirectory(prefix="autovideo-gemini-") as tmp:
        out_path = Path(tmp) / "nvidia_gemini.json"
        cmd = [
            sys.executable,
            str(tool),
            "--out",
            str(out_path),
            "--context",
            f"Autovideo candidate run: {candidate_id}",
        ]
        for frame in frame_paths[:4]:
            cmd.extend(["--candidate-frame", str(frame)])

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
                "status": "failed",
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
                "result": payload,
            }
        return {"status": "complete", "result": payload}


def build_quality(
    run_dir: Path,
    metadata: dict[str, Any],
    frames_dir: Path,
    frames: dict[str, Any],
    baseline_frames: list[str],
    skip_judges: bool,
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

    candidate_id = str(metadata.get("candidate_id", run_dir.name))
    judges = {
        "lpips": run_lpips_judge(frame_paths, baseline_frames, skip_judges),
        "nvidia_gemini": run_nvidia_gemini_judge(
            frame_paths,
            candidate_id,
            skip_judges,
        ),
    }
    return {
        "status": "available" if frame_paths else "deferred",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_extraction": frames,
        "frame_metrics": frame_metrics,
        "judges": judges,
    }


def render_risk_notes(metadata: dict[str, Any]) -> str:
    if metadata.get("kind") == "baseline":
        return "no risk; baseline reference run\n"
    return "risk notes pending for non-baseline candidate run\n"


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
    official = manifest.get("official_config", {})
    artifacts = manifest.get("artifacts", {})
    lines = [
        f"# Candidate Report: {metadata.get('candidate_id', run_dir.name)}",
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
        "benchmark": output_dir / artifacts.get("benchmark", "benchmark.json"),
        "quality": output_dir / artifacts.get("quality", "quality.json"),
        "risk_notes": output_dir / artifacts.get("risk_notes", "risk_notes.md"),
        "collection": output_dir / artifacts.get("collection", "collection.json"),
        "patch_summary": output_dir / artifacts.get("patch_summary", "patch_summary.md"),
    }
    frames_dir = output_dir / artifacts.get("frames_dir", "frames")

    log_errors = detect_log_errors(paths["log"])
    benchmark = build_benchmark(paths["benchmark"], paths["log"])
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
            args.frame_count,
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
        args.baseline_frame or [],
        args.skip_judges,
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
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--ffmpeg", help="ffmpeg executable path override")
    parser.add_argument(
        "--baseline-frame",
        action="append",
        help="Baseline frame for optional LPIPS comparison; may be repeated",
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
