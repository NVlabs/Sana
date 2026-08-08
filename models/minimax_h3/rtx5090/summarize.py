#!/usr/bin/env python3
"""Summarize one RTX 5090 MiniMax-H3 run from official API metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _request_summary(path: Path) -> dict[str, Any]:
    metadata = _load_json(path / "run_metadata.json")
    status = metadata.get("final_status", {})
    return {
        "inference_time_s": status.get("inference_time_s"),
        "peak_memory_mb": status.get("peak_memory_mb"),
        "request_wall_seconds": metadata.get("request_wall_seconds"),
        "video": metadata.get("output_path"),
    }


def _event_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dense_backends": [],
        "route_density_samples": [],
        "teacache_generation_summaries": [],
    }
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event.get("event")
        if kind == "dense_backend":
            result["dense_backends"].append(event.get("backend"))
        elif kind == "route_density":
            result["route_density_samples"].append(event)
        elif kind == "teacache_generation_summary":
            result["teacache_generation_summaries"].append(event)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    report = {
        "run_root": str(args.run_root.resolve()),
        "requests": {
            name: _request_summary(args.run_root / name)
            for name in ("warmup", "measured")
        },
        "events": _event_summary(args.run_root / "sol_events_rank0.jsonl"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
