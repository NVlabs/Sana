#!/usr/bin/env python3
"""Self-contained tests for scripts/collect_run.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECT_RUN = ROOT / "scripts/collect_run.py"


def load_collect_run():
    spec = importlib.util.spec_from_file_location("collect_run", COLLECT_RUN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def approx(value: float | None, expected: float, tolerance: float = 0.01) -> None:
    assert value is not None
    assert abs(value - expected) <= tolerance, (value, expected)


def write_fake_run(root: Path) -> Path:
    run_dir = root / "20260613-baseline"
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "candidate_id": "baseline",
                "kind": "baseline",
                "status": "prepared",
            }
        )
        + "\n"
    )
    (output_dir / "run.log").write_text(
        "\x1b[32m[06-12 11:07:19] [Cosmos3DenoisingStage] "
        "finished in 121.4198 seconds\x1b[0m\n"
        "[06-12 11:07:25] [Cosmos3DecodingStage] "
        "finished in 5.8017 seconds\n"
        "[06-12 11:07:28] Pixel data generated successfully "
        "in 130.41 seconds\n"
    )
    (output_dir / "out.mp4").write_bytes(b"not a real mp4, just non-empty")
    return run_dir


def test_collect_run_timing_and_canonical_outputs() -> None:
    collect_run = load_collect_run()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_fake_run(Path(tmp))
        output_dir = run_dir / "outputs"
        timing = collect_run.parse_run_log_timing(output_dir / "run.log")
        approx(timing["denoise_s"], 121.42)
        approx(timing["decode_s"], 5.80)
        approx(timing["total_s"], 130.41)

        result = collect_run.collect(
            argparse.Namespace(
                run_dir=str(run_dir),
                extract_frames=False,
                no_extract_frames=True,
                overwrite_frames=False,
                frame_fps=2.0,
                frame_count=8,
                ffmpeg=None,
                baseline_frame=None,
                skip_judges=True,
            )
        )
        assert result["status"] == "completed"

        benchmark = json.loads((output_dir / "benchmark.json").read_text())
        approx(benchmark["denoise_s"], 121.42)
        approx(benchmark["decode_s"], 5.80)
        approx(benchmark["total_s"], 130.41)

        assert (output_dir / "patch_summary.md").exists()
        assert (output_dir / "quality.json").exists()
        assert (output_dir / "risk_notes.md").exists()
        assert (output_dir / "collection.json").exists()
        assert not (output_dir / "perf.json").exists()
        assert not (output_dir / "report.md").exists()


if __name__ == "__main__":
    test_collect_run_timing_and_canonical_outputs()
    print("tests/test_collect_run.py: ok")
