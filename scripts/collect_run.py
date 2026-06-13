#!/usr/bin/env python3
"""Collect artifacts for one autovideo run bundle.

This completes the M1.5 loop: inspect a generated run directory, classify its
status, optionally extract video frames, and write outputs/report.md plus an
updated metadata.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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


def parse_perf(perf_path: Path) -> dict[str, Any]:
    data = load_json(perf_path)
    if not data:
        return {}

    total_ms = data.get("total_duration_ms")
    stages = data.get("stages") or data.get("steps") or []
    stage_seconds: dict[str, float] = {}
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
            stage_seconds[name] = float(duration) / 1000.0

    denoise_s = sum(v for k, v in stage_seconds.items() if "denois" in k.lower())
    decode_s = sum(
        v
        for k, v in stage_seconds.items()
        if "decod" in k.lower() or "vae" in k.lower()
    )
    return {
        "raw": data,
        "total_s": float(total_ms) / 1000.0 if isinstance(total_ms, (int, float)) else None,
        "denoise_s": denoise_s or None,
        "decode_s": decode_s or None,
        "stage_seconds": stage_seconds,
    }


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


def extract_frames(video_path: Path, frames_dir: Path, fps: float, overwrite: bool) -> dict[str, Any]:
    if not video_path.exists():
        return {"status": "skipped", "reason": "video_missing", "count": 0}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"status": "skipped", "reason": "ffmpeg_missing", "count": 0}
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("f_*.png"))
    if existing and not overwrite:
        return {"status": "existing", "count": len(existing)}

    for old in existing:
        old.unlink()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        str(frames_dir / "f_%03d.png"),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return {
            "status": "failed",
            "count": 0,
            "stderr": proc.stderr.strip(),
        }
    return {"status": "created", "count": len(list(frames_dir.glob("f_*.png")))}


def determine_status(
    metadata: dict[str, Any],
    log_path: Path,
    video_path: Path,
    perf_path: Path,
    log_errors: list[str],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    previous = str(metadata.get("status") or "")

    if log_errors:
        notes.append("log contains error patterns: " + ", ".join(log_errors))
        return "failed", notes

    if nonempty(video_path) and nonempty(log_path):
        if not perf_path.exists():
            notes.append("perf.json missing; marking completed with partial timing")
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


def render_report(
    run_dir: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    status: str,
    notes: list[str],
    paths: dict[str, Path],
    perf: dict[str, Any],
    frames: dict[str, Any],
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
            f"- total: `{format_seconds(perf.get('total_s'))}`",
            f"- denoise: `{format_seconds(perf.get('denoise_s'))}`",
            f"- decode: `{format_seconds(perf.get('decode_s'))}`",
        ]
    )
    stage_seconds = perf.get("stage_seconds") or {}
    if stage_seconds:
        lines.append("- stages:")
        for name, seconds in sorted(stage_seconds.items()):
            lines.append(f"  - `{name}`: `{format_seconds(seconds)}`")

    lines.extend(["", "## Artifacts", ""])
    for label, path in paths.items():
        if label == "report":
            continue
        info = file_info(path)
        if info["exists"]:
            lines.append(f"- {label}: `{rel(path, run_dir)}` ({info['bytes']} bytes)")
        else:
            lines.append(f"- {label}: missing (`{rel(path, run_dir)}`)")
    lines.append(f"- frames: `{frames.get('status')}` count=`{frames.get('count', 0)}`")

    lines.extend(["", "## Quality", ""])
    lines.append("- quantitative gate: `pending`")
    lines.append("- visual gate: `pending`")
    lines.append("- baseline comparison: `pending`")

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
        "perf": output_dir / artifacts.get("perf", "perf.json"),
        "report": output_dir / artifacts.get("report", "report.md"),
    }
    frames_dir = output_dir / artifacts.get("frames_dir", "frames")

    log_errors = detect_log_errors(paths["log"])
    perf = parse_perf(paths["perf"])
    status, notes = determine_status(
        metadata,
        paths["log"],
        paths["video"],
        paths["perf"],
        log_errors,
    )

    should_extract = args.extract_frames or (status == "completed" and not args.no_extract_frames)
    if should_extract:
        frames = extract_frames(paths["video"], frames_dir, args.frame_fps, args.overwrite_frames)
    else:
        existing = len(list(frames_dir.glob("f_*.png"))) if frames_dir.exists() else 0
        frames = {"status": "skipped", "count": existing}

    report = render_report(
        run_dir,
        metadata,
        manifest,
        status,
        notes,
        paths,
        perf,
        frames,
        log_errors,
    )
    paths["report"].write_text(report)

    metadata.update(
        {
            "status": status,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "collector": {
                "report": str(paths["report"]),
                "frames": frames,
                "notes": notes,
                "log_errors": log_errors,
                "timing": {
                    "total_s": perf.get("total_s"),
                    "denoise_s": perf.get("denoise_s"),
                    "decode_s": perf.get("decode_s"),
                },
            },
        }
    )
    write_json(metadata_path, metadata)
    write_json(
        output_dir / "collection.json",
        {
            "status": status,
            "run_dir": str(run_dir),
            "artifacts": {key: file_info(path) for key, path in paths.items()},
            "frames": frames,
            "timing": {
                "total_s": perf.get("total_s"),
                "denoise_s": perf.get("denoise_s"),
                "decode_s": perf.get("decode_s"),
                "stage_seconds": perf.get("stage_seconds", {}),
            },
            "notes": notes,
            "log_errors": log_errors,
        },
    )
    return {
        "status": status,
        "run_dir": str(run_dir),
        "report": str(paths["report"]),
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
    args = parser.parse_args()

    result = collect(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    status = result["status"]
    return 1 if status in {"failed", "blocked", "rejected_quality"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
