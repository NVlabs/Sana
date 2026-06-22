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


def test_quality_blocks_promotion_without_baseline_frames() -> None:
    collect_run = load_collect_run()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        frames_dir = run_dir / "outputs" / "frames"
        frames_dir.mkdir(parents=True)
        (frames_dir / "f_001.png").write_bytes(b"placeholder")
        quality = collect_run.build_quality(
            run_dir=run_dir,
            metadata={"candidate_id": "candidate"},
            frames_dir=frames_dir,
            frames={"status": "existing", "count": 1},
            baseline_frames=[],
            skip_judges=False,
        )
        assert quality["status"] == "blocked_quality"
        assert "baseline_frames_missing" in quality["promotion_blockers"]
        assert quality["judges"]["lpips"]["status"] == "blocked"


def test_lpips_missing_baseline_is_blocked() -> None:
    collect_run = load_collect_run()
    result = collect_run.run_lpips_judge(
        frame_paths=[Path("candidate.png")],
        baseline_frames=[],
        skip=False,
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "baseline_frame_missing"


def test_lpips_unavailable_dependency_is_blocked() -> None:
    collect_run = load_collect_run()
    original_find_spec = collect_run.importlib.util.find_spec

    def fake_find_spec(module: str):
        if module == "lpips":
            return None
        return original_find_spec(module)

    collect_run.importlib.util.find_spec = fake_find_spec
    try:
        result = collect_run.run_lpips_judge(
            frame_paths=[Path("candidate.png")],
            baseline_frames=["baseline.png"],
            skip=False,
        )
        assert result["status"] == "blocked"
        assert result["reason"] == "dependencies_missing"
    finally:
        collect_run.importlib.util.find_spec = original_find_spec


def test_targeted_quality_defaults_and_rubric() -> None:
    collect_run = load_collect_run()
    assert collect_run.DEFAULT_FRAME_COUNT == 189
    assert collect_run.GEMINI_MAX_FRAME_PAIRS >= 32
    assert collect_run.LPIPS_WORST_CASE_PAIRS > 0

    rubric = (ROOT / "evals/rubrics/gemini_visual_artifact_gate.md").read_text()
    for phrase in (
        "frame-to-frame flicker",
        "patch-boundary discontinuity",
        "broken temporal movement",
        "temporal_checks",
        "motion_coherence",
        "patch_boundary_stability",
    ):
        assert phrase in rubric


def test_pixel_metrics_include_temporal_and_patch_targets() -> None:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return

    collect_run = load_collect_run()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_dir = root / "baseline"
        cand_dir = root / "candidate"
        base_dir.mkdir()
        cand_dir.mkdir()

        base1 = np.zeros((64, 64, 3), dtype=np.uint8)
        base2 = np.full((64, 64, 3), 24, dtype=np.uint8)
        cand1 = base1.copy()
        cand2 = base2.copy()
        cand2[:, 32:, :] = 80

        for path, arr in (
            (base_dir / "f_001.png", base1),
            (base_dir / "f_002.png", base2),
            (cand_dir / "f_001.png", cand1),
            (cand_dir / "f_002.png", cand2),
        ):
            Image.fromarray(arr).save(path)

        metrics = collect_run.build_pixel_metrics(
            sorted(cand_dir.glob("f_*.png")),
            [str(path) for path in sorted(base_dir.glob("f_*.png"))],
        )
        assert metrics["status"] == "ok"
        assert "psnr_min" in metrics
        assert "mse_max" in metrics
        assert "patch_boundary_ratio_max" in metrics
        assert "patch_boundary_ratio_by_size_max" in metrics
        assert "temporal_delta_error_max" in metrics
        assert "temporal_jitter_ratio_max" in metrics


if __name__ == "__main__":
    test_collect_run_timing_and_canonical_outputs()
    test_quality_blocks_promotion_without_baseline_frames()
    test_lpips_missing_baseline_is_blocked()
    test_lpips_unavailable_dependency_is_blocked()
    test_targeted_quality_defaults_and_rubric()
    test_pixel_metrics_include_temporal_and_patch_targets()
    print("tests/test_collect_run.py: ok")
