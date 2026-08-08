#!/usr/bin/env python3
"""Run one official-profile MiniMax-H3 T2VA request and download its MP4."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.getenv("H3_ROOT", str(RUNTIME_ROOT)))
BASE_URL = os.getenv("H3_SERVER_URL", "http://127.0.0.1:30010")
STEPS = int(os.getenv("H3_NUM_STEPS", "5"))
DURATION_SECONDS = int(os.getenv("H3_DURATION_SECONDS", "5"))
SEED = int(os.getenv("H3_SEED", "1101"))
OUTPUT_DIR = Path(
    os.getenv("H3_OUTPUT_DIR", str(ROOT / "outputs" / f"t2va_{STEPS}step_seed{SEED}"))
)
PROMPT = os.getenv(
    "H3_PROMPT",
    (
        "At night, while their owner sleeps in a bedroom, three cats march in "
        "loudly playing tiny brass instruments, then abruptly file out."
    ),
)


def request_json(method: str, path: str, body: dict[str, Any] | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def wait_for_server(timeout_seconds: int = 7200) -> None:
    deadline = time.monotonic() + timeout_seconds
    latest_error = "server has not answered"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            latest_error = repr(exc)
        time.sleep(10)
    raise TimeoutError(f"server did not become healthy: {latest_error}")


def poll_video(video_id: str, timeout_seconds: int = 7200) -> dict:
    deadline = time.monotonic() + timeout_seconds
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = request_json("GET", f"/v1/videos/{video_id}")
        status = str(latest.get("status", "")).lower()
        print(f"video_id={video_id} status={status}", flush=True)
        if status in {"completed", "succeeded"}:
            return latest
        if status in {"failed", "cancelled", "canceled"}:
            raise RuntimeError(json.dumps(latest, ensure_ascii=False, indent=2))
        time.sleep(5)
    raise TimeoutError(f"video generation timed out; latest={latest!r}")


def download_video(video_id: str, destination: Path) -> None:
    request = urllib.request.Request(
        f"{BASE_URL}/v1/videos/{video_id}/content", method="GET"
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        destination.write_bytes(response.read())


def main() -> None:
    if STEPS <= 0:
        raise ValueError("H3_NUM_STEPS must be positive")
    if not 4 <= DURATION_SECONDS <= 15:
        raise ValueError("H3_DURATION_SECONDS must be in [4, 15]")

    payload = {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": PROMPT,
        "seconds": DURATION_SECONDS,
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": float(DURATION_SECONDS),
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": STEPS,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": SEED,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    wait_for_server()

    started_at = time.time()
    started_monotonic = time.monotonic()
    create_response = request_json("POST", "/v1/videos", payload)
    (OUTPUT_DIR / "create_response.json").write_text(
        json.dumps(create_response, ensure_ascii=False, indent=2) + "\n"
    )
    video_id = str(create_response["id"])
    final_status = poll_video(video_id)

    output_path = OUTPUT_DIR / (
        f"minimax_h3_t2va_1344x768_{DURATION_SECONDS}s_"
        f"{STEPS}step_seed{SEED}.mp4"
    )
    download_video(video_id, output_path)
    metadata = {
        "base_url": BASE_URL,
        "video_id": video_id,
        "request_started_unix": started_at,
        "request_finished_unix": time.time(),
        "request_wall_seconds": time.monotonic() - started_monotonic,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "final_status": final_status,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
