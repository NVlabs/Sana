#!/usr/bin/env python3
"""Run the visual artifact rubric through the local NVIDIA/Gemini vision skill."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_HELPER = Path.home() / ".codex/skills/nvidia-vision-api/scripts/nvidia_multimodal_chat.py"
DEFAULT_RUBRIC = "evals/rubrics/gemini_visual_artifact_gate.md"
DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_MODEL = "gcp/google/gemini-3.5-flash"
DEFAULT_VIDEO_MAX_FRAMES = 32
DEFAULT_VIDEO_FRAME_INTERVAL = 0.5


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_prompt(rubric: Path, extra: str) -> str:
    prompt = rubric.read_text()
    if extra:
        prompt += "\n\n## Config Context\n\n" + extra.strip() + "\n"
    prompt += (
        "\n\nReturn only valid JSON matching the Required JSON Output schema. "
        "Do not wrap it in Markdown."
    )
    return prompt


def normalize_result(payload: Any, raw_response: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {
        "overall": "inconclusive",
        "raw_response": raw_response,
        "parse_error": f"json_root_not_object:{type(payload).__name__}",
    }


def extract_json(text: str) -> dict[str, Any]:
    try:
        return normalize_result(json.loads(text), text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "overall": "inconclusive",
            "raw_response": text,
            "parse_error": "no_json_object_found",
        }
    try:
        return normalize_result(json.loads(match.group(0)), text)
    except json.JSONDecodeError as exc:
        return {
            "overall": "inconclusive",
            "raw_response": text,
            "parse_error": str(exc),
        }


def build_command(args: argparse.Namespace, prompt: str) -> list[str]:
    helper = Path(os.environ.get("NVIDIA_VISION_HELPER", args.helper)).expanduser()
    cmd = [
        sys.executable,
        str(helper),
        "--prompt",
        prompt,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--max-tokens",
        str(args.max_tokens),
    ]
    for path in args.baseline_frame:
        cmd.extend(["--image", str(Path(path).expanduser())])
    for path in args.config_frame:
        cmd.extend(["--image", str(Path(path).expanduser())])
    for path in args.side_by_side_frame:
        cmd.extend(["--image", str(Path(path).expanduser())])
    for path in args.video:
        cmd.extend(["--video", str(Path(path).expanduser())])
    cmd.extend(["--video-max-frames", str(args.video_max_frames)])
    cmd.extend(["--video-frame-interval", str(args.video_frame_interval)])
    cmd.extend(["--video-frame-width", str(args.video_frame_width)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default=DEFAULT_RUBRIC)
    parser.add_argument("--helper", default=str(DEFAULT_HELPER))
    parser.add_argument("--base-url", default=os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("NVIDIA_VISION_MODEL", DEFAULT_MODEL))
    parser.add_argument("--baseline-frame", action="append", default=[])
    parser.add_argument("--config-frame", action="append", default=[])
    parser.add_argument("--side-by-side-frame", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--context", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--video-max-frames", type=int, default=DEFAULT_VIDEO_MAX_FRAMES)
    parser.add_argument("--video-frame-interval", type=float, default=DEFAULT_VIDEO_FRAME_INTERVAL)
    parser.add_argument("--video-frame-width", type=int, default=960)
    args = parser.parse_args()

    root = project_root()
    rubric = (root / args.rubric).resolve()
    if not rubric.exists():
        raise SystemExit(f"Rubric does not exist: {rubric}")

    media_count = (
        len(args.baseline_frame)
        + len(args.config_frame)
        + len(args.side_by_side_frame)
        + len(args.video)
    )
    if media_count == 0:
        raise SystemExit("Provide at least one frame or video input.")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt_file = out.with_suffix(".prompt.md")
    prompt = read_prompt(rubric, args.context)
    prompt_file.write_text(prompt)
    cmd = build_command(args, prompt)

    if args.dry_run:
        payload = {
            "overall": "inconclusive",
            "dry_run": True,
            "provider": "nvidia_gemini",
            "base_url": args.base_url,
            "model": args.model,
            "helper": str(Path(os.environ.get("NVIDIA_VISION_HELPER", args.helper)).expanduser()),
            "prompt_file": str(prompt_file),
            "command": " ".join(shlex.quote(part) for part in cmd),
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    helper = Path(os.environ.get("NVIDIA_VISION_HELPER", args.helper)).expanduser()
    if not helper.exists():
        raise SystemExit(f"NVIDIA vision helper not found: {helper}")
    if not any(os.environ.get(name) for name in ("NVIDIA_API_KEY", "NVIDIA_VISION_API_KEY", "API_KEY", "NGC_API_KEY")):
        raise SystemExit("Missing NVIDIA_API_KEY or compatible NVIDIA vision API key env var.")

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        out.write_text(
            json.dumps(
                {
                    "overall": "inconclusive",
                    "provider": "nvidia_gemini",
                    "error": proc.stderr.strip(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    result = extract_json(proc.stdout)
    result.setdefault("provider", "nvidia_gemini")
    result.setdefault("model", args.model)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
