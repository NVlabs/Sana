#!/usr/bin/env python3
"""Score candidate frames against baseline frames with LPIPS when available."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable


class MetricUnavailable(RuntimeError):
    """Raised when optional runtime support is not available."""


FramePair = tuple[Path, Path]


def unavailable_payload(reason: str) -> dict[str, object]:
    return {
        "metric": "lpips",
        "status": "unavailable",
        "reason": reason,
        "n": 0,
    }


def success_payload(
    scores: list[float],
    notes: Iterable[str] = (),
) -> dict[str, object]:
    return {
        "metric": "lpips",
        "status": "ok",
        "per_frame": scores,
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "n": len(scores),
        "notes": list(notes),
    }


def load_lpips_modules(
    import_module: Callable[[str], object] = importlib.import_module,
) -> tuple[object, object]:
    try:
        lpips_module = import_module("lpips")
    except Exception as exc:  # pragma: no cover - exact exception varies by install.
        raise MetricUnavailable(f"lpips is not importable: {exc}") from exc
    try:
        torch_module = import_module("torch")
    except Exception as exc:  # pragma: no cover - exact exception varies by install.
        raise MetricUnavailable(f"torch is not importable: {exc}") from exc
    return lpips_module, torch_module


def build_lpips_model(lpips_module: object) -> object:
    lpips_factory = getattr(lpips_module, "LPIPS", None)
    if lpips_factory is None:
        raise MetricUnavailable("lpips.LPIPS is not available")
    try:
        model = lpips_factory(net="alex", verbose=False)
    except TypeError:
        model = lpips_factory(net="alex")
    if hasattr(model, "eval"):
        model.eval()
    return model


def frame_to_tensor(lpips_module: object, frame: Path) -> object:
    load_image = getattr(lpips_module, "load_image", None)
    image_to_tensor = getattr(lpips_module, "im2tensor", None)
    if load_image is None or image_to_tensor is None:
        raise MetricUnavailable("lpips image loading helpers are not available")
    return image_to_tensor(load_image(str(frame)))


def score_frame_pairs(
    frame_pairs: list[FramePair],
    import_module: Callable[[str], object] = importlib.import_module,
) -> list[float]:
    lpips_module, torch_module = load_lpips_modules(import_module)
    try:
        model = build_lpips_model(lpips_module)
        scores: list[float] = []
        no_grad = getattr(torch_module, "no_grad")
        with no_grad():
            for baseline_frame, candidate_frame in frame_pairs:
                baseline_tensor = frame_to_tensor(lpips_module, baseline_frame)
                candidate_tensor = frame_to_tensor(lpips_module, candidate_frame)
                value = model(baseline_tensor, candidate_tensor)
                if hasattr(value, "item"):
                    value = value.item()
                scores.append(float(value))
        return scores
    except MetricUnavailable:
        raise
    except Exception as exc:  # Keep runtime/dependency failures deferred.
        raise MetricUnavailable(f"LPIPS scoring failed: {exc}") from exc


def resolve_ffmpeg() -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return Path(ffmpeg)
    fallback = Path("~/lustre/bin/ffmpeg").expanduser()
    if fallback.exists():
        return fallback
    raise MetricUnavailable("ffmpeg is not available for video frame extraction")


def extract_video_frames(video: Path, out_dir: Path, sample_fps: float, ffmpeg: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={sample_fps}",
        str(pattern),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise MetricUnavailable(f"ffmpeg failed for {video}: {detail}")
    return sorted(out_dir.glob("frame_*.png"))


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--sample-fps must be greater than 0")
    return parsed


def validate_existing_paths(parser: argparse.ArgumentParser, paths: Iterable[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            parser.error(f"input path does not exist: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-frame", action="append", default=[])
    parser.add_argument("--candidate-frame", action="append", default=[])
    parser.add_argument("--baseline-video")
    parser.add_argument("--candidate-video")
    parser.add_argument("--sample-fps", type=positive_float, default=1.0)
    parser.add_argument("--out", help="Write JSON to this path. Defaults to stdout.")
    args = parser.parse_args(argv)

    if len(args.baseline_frame) != len(args.candidate_frame):
        parser.error("--baseline-frame and --candidate-frame counts must match")
    if bool(args.baseline_video) != bool(args.candidate_video):
        parser.error("--baseline-video and --candidate-video must be provided together")
    if not args.baseline_frame and not args.baseline_video:
        parser.error("provide at least one frame pair or one video pair")

    validate_existing_paths(parser, args.baseline_frame + args.candidate_frame)
    validate_existing_paths(
        parser,
        [path for path in (args.baseline_video, args.candidate_video) if path],
    )
    return args


def collect_frame_pairs(
    args: argparse.Namespace,
    temp_dir: Path | None = None,
) -> tuple[list[FramePair], list[str]]:
    notes = ["lower_is_better", "frames_paired_by_order"]
    frame_pairs: list[FramePair] = [
        (Path(baseline).expanduser(), Path(candidate).expanduser())
        for baseline, candidate in zip(args.baseline_frame, args.candidate_frame)
    ]

    if args.baseline_video and args.candidate_video:
        if temp_dir is None:
            raise MetricUnavailable("internal error: video extraction requires a temporary directory")
        ffmpeg = resolve_ffmpeg()
        baseline_frames = extract_video_frames(
            Path(args.baseline_video).expanduser(),
            temp_dir / "baseline",
            args.sample_fps,
            ffmpeg,
        )
        candidate_frames = extract_video_frames(
            Path(args.candidate_video).expanduser(),
            temp_dir / "candidate",
            args.sample_fps,
            ffmpeg,
        )
        if len(baseline_frames) != len(candidate_frames):
            notes.append(
                "video frame counts differed after sampling; paired common prefix only"
            )
        frame_pairs.extend(zip(baseline_frames, candidate_frames))
        notes.append(f"sample_fps={args.sample_fps}")

    return frame_pairs, notes


def evaluate(frame_pairs: list[FramePair], notes: Iterable[str]) -> dict[str, object]:
    if not frame_pairs:
        return unavailable_payload("no frame pairs are available")
    try:
        scores = score_frame_pairs(frame_pairs)
    except MetricUnavailable as exc:
        return unavailable_payload(str(exc))
    if not scores:
        return unavailable_payload("LPIPS produced no scores")
    return success_payload(scores, notes)


def emit_json(payload: dict[str, object], out: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    else:
        print(text, end="")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.baseline_video:
        with tempfile.TemporaryDirectory(prefix="lpips-judge-") as temp_root:
            frame_pairs, notes = collect_frame_pairs(args, Path(temp_root))
            payload = evaluate(frame_pairs, notes)
    else:
        frame_pairs, notes = collect_frame_pairs(args)
        payload = evaluate(frame_pairs, notes)
    emit_json(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
