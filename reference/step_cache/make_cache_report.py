#!/usr/bin/env python3
"""Build a cache report from canonical autovideo artifacts.

Provenance: ported from Sol-LTX-Infer scripts/make_ltx23_cache_report.py @
29d0d9e464000a2472345dcad51054b15aacca8d, adapted to read outputs/benchmark.json
and outputs/quality.json and to write outputs/patch_summary.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


STATS_PATTERNS = {
    "teacache": re.compile(r"(?:LTX2|Cosmos3)?\s*TeaCache stats.*?:\s*(\{.*\})"),
    "step_cache": re.compile(
        r"(?:LTX2|Cosmos3)?.*?(?:StepCache|stage1 cache[- ]core) stats.*?:\s*(\{.*\})",
        re.IGNORECASE,
    ),
}
TIMING_FIELDS = ("total_s", "denoise_s", "decode_s")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _outputs_dir(run_path: Path) -> Path:
    if (run_path / "benchmark.json").exists() or (run_path / "run.log").exists():
        return run_path
    return run_path / "outputs"


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stage_seconds(data: dict[str, Any]) -> dict[str, float]:
    stage_seconds: dict[str, float] = {}
    raw = data.get("stage_seconds")
    if isinstance(raw, dict):
        for key, value in raw.items():
            fvalue = _float_or_none(value)
            if fvalue is not None:
                stage_seconds[str(key)] = fvalue

    for item in data.get("steps", []) or data.get("stages", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("stage") or item.get("step") or "")
        duration_ms = item.get("duration_ms") or item.get("elapsed_ms")
        if name and isinstance(duration_ms, (int, float)):
            stage_seconds[name] = float(duration_ms) / 1000.0
    return stage_seconds


def _benchmark(outputs_dir: Path) -> dict[str, Any]:
    data = _load_json(outputs_dir / "benchmark.json")
    timing = {field: _float_or_none(data.get(field)) for field in TIMING_FIELDS}

    total_ms = data.get("total_duration_ms")
    if timing["total_s"] is None and isinstance(total_ms, (int, float)):
        timing["total_s"] = float(total_ms) / 1000.0

    stage_seconds = _stage_seconds(data)
    if timing["denoise_s"] is None:
        denoise = sum(v for k, v in stage_seconds.items() if "denois" in k.lower())
        timing["denoise_s"] = denoise or None
    if timing["decode_s"] is None:
        decode = sum(
            v
            for k, v in stage_seconds.items()
            if "decod" in k.lower() or "vae" in k.lower()
        )
        timing["decode_s"] = decode or None

    timing["stage_seconds"] = stage_seconds
    return timing


def _parse_stats_literal(text: str) -> dict[str, Any]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        try:
            value = json.loads(text)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _cache_stats(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    stats: dict[str, Any] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        for name, pattern in STATS_PATTERNS.items():
            match = pattern.search(line)
            if match:
                parsed = _parse_stats_literal(match.group(1))
                if parsed:
                    stats[name] = parsed
    return stats


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    skipped: set[int] = set()
    computes = 0
    hits = 0
    calls = 0
    for item in _walk_dicts(stats):
        computes += int(item.get("computes", 0) or 0)
        hits += int(item.get("hits", 0) or 0)
        calls += int(item.get("calls", 0) or 0)
        for step in item.get("skipped_steps", []) or []:
            try:
                skipped.add(int(step))
            except (TypeError, ValueError):
                continue
    return {
        "calls": calls,
        "computes": computes,
        "hits": hits,
        "skipped_steps": sorted(skipped),
    }


def _quality(outputs_dir: Path) -> dict[str, Any]:
    data = _load_json(outputs_dir / "quality.json")
    if not data:
        return {"status": "missing"}
    status = data.get("status") or data.get("gate") or data.get("result") or "present"
    return {
        "status": status,
        "frame_count": data.get("frame_count"),
        "duration_s": data.get("duration_s"),
        "visual_artifact": data.get("visual_artifact") or data.get("visual_artifact_gate"),
        "raw": data,
    }


def _case(run_path: Path) -> dict[str, Any]:
    outputs = _outputs_dir(run_path)
    stats = _cache_stats(outputs / "run.log")
    return {
        "run": str(run_path),
        "outputs": str(outputs),
        "benchmark": _benchmark(outputs),
        "quality": _quality(outputs),
        "cache_stats": stats,
        "cache_stats_summary": _summarize_stats(stats),
        "artifacts": {
            "video": str(outputs / "out.mp4"),
            "log": str(outputs / "run.log"),
            "benchmark": str(outputs / "benchmark.json"),
            "quality": str(outputs / "quality.json"),
            "risk_notes": str(outputs / "risk_notes.md"),
        },
    }


def _speedup(base: float | None, candidate: float | None) -> float | None:
    if not base or not candidate:
        return None
    return base / candidate


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_bench = baseline["benchmark"]
    cand_bench = candidate["benchmark"]
    speedups = {
        field: _speedup(base_bench.get(field), cand_bench.get(field))
        for field in TIMING_FIELDS
    }
    stage_speedups = {}
    base_stages = base_bench.get("stage_seconds") or {}
    cand_stages = cand_bench.get("stage_seconds") or {}
    for stage, base_value in base_stages.items():
        stage_speedups[stage] = _speedup(base_value, cand_stages.get(stage))
    return {
        "baseline": baseline,
        "candidate": candidate,
        "speedups": speedups,
        "stage_speedups": stage_speedups,
    }


def _markdown(summary: dict[str, Any]) -> str:
    base = summary["baseline"]["benchmark"]
    cand = summary["candidate"]["benchmark"]
    speed = summary["speedups"]
    cand_stats = summary["candidate"]["cache_stats_summary"]
    quality = summary["candidate"]["quality"]

    lines = [
        "# Step Cache Candidate Report",
        "",
        "| Metric | Baseline | Candidate | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for field in TIMING_FIELDS:
        lines.append(
            f"| {field} | {_fmt(base.get(field))} | {_fmt(cand.get(field))} | {_fmt(speed.get(field))} |"
        )

    lines.extend(["", "## Stage Timing", "", "| Stage | Baseline s | Candidate s | Speedup |", "|---|---:|---:|---:|"])
    base_stages = base.get("stage_seconds") or {}
    cand_stages = cand.get("stage_seconds") or {}
    for stage in sorted(set(base_stages) | set(cand_stages)):
        lines.append(
            f"| {stage} | {_fmt(base_stages.get(stage))} | {_fmt(cand_stages.get(stage))} | {_fmt(summary['stage_speedups'].get(stage))} |"
        )

    lines.extend(
        [
            "",
            "## Cache Stats",
            "",
            f"- calls: {cand_stats['calls']}",
            f"- computes: {cand_stats['computes']}",
            f"- hits: {cand_stats['hits']}",
            f"- skipped_steps: {cand_stats['skipped_steps']}",
            "",
            "## Quality",
            "",
            f"- status: {quality.get('status')}",
            f"- frame_count: {quality.get('frame_count')}",
            f"- duration_s: {quality.get('duration_s')}",
            f"- visual_artifact: {quality.get('visual_artifact')}",
            "",
            "## Artifacts",
            "",
        ]
    )
    for label, path in summary["candidate"]["artifacts"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, help="Baseline run dir or outputs dir")
    parser.add_argument("--candidate-run", required=True, help="Candidate run dir or outputs dir")
    parser.add_argument(
        "--output",
        help="Markdown output path. Defaults to candidate outputs/patch_summary.md",
    )
    args = parser.parse_args()

    baseline = _case(Path(args.baseline_run))
    candidate = _case(Path(args.candidate_run))
    summary = _compare(baseline, candidate)
    output = Path(args.output) if args.output else _outputs_dir(Path(args.candidate_run)) / "patch_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
