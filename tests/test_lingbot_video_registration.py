from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_toml(relative: str) -> dict:
    with (ROOT / relative).open("rb") as handle:
        return tomllib.load(handle)


def load_adapter_module():
    path = ROOT / "runtime/lingbot_video_baseline/gpu_infer.py"
    spec = importlib.util.spec_from_file_location("lingbot_video_gpu_infer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lingbot_profile_and_transfeat_share_fixed_workload() -> None:
    profile = load_toml("models/lingbot_video.toml")
    eval_profile = load_toml("evals/profiles/official_video_t2v_lingbot_video.toml")
    baseline = load_toml("transfeat/lingbot_video/baseline.toml")
    optimized = load_toml("transfeat/lingbot_video_cudnn_optimized.toml")
    optimized_off = load_toml("transfeat/lingbot_video_cudnn_off.toml")
    fsdp_reference = load_toml("transfeat/lingbot_video_fsdp4_reference.toml")

    assert profile["official_config"] == eval_profile["official_config"]
    assert profile["official_config"]["num_gpus"] == 4
    assert baseline["model_profile"] == optimized["model_profile"] == "lingbot_video"
    assert baseline["runtime"]["root"] == "runtime/lingbot_video_baseline"
    assert optimized["runtime"]["root"] == "runtime/lingbot_video_optimized"
    assert baseline["env"]["LINGBOT_ATTN_KERNEL"] == "fa2"
    assert optimized["env"]["LINGBOT_ATTN_KERNEL"] == "cudnn"
    assert optimized_off["runtime"]["root"] == "runtime/lingbot_video_optimized"
    assert optimized_off["env"]["LINGBOT_ATTN_KERNEL"] == "fa2"
    assert profile["official_config"]["context_parallel_degree"] == 4
    assert fsdp_reference["official_config"]["context_parallel_degree"] == 1
    assert fsdp_reference["official_config"]["batch_cfg"] is False


def test_lingbot_baseline_contract_does_not_copy_optimized_runtime() -> None:
    contract = load_toml("models/lingbot_video/model.toml")
    includes = contract["baseline"]["copy"]["include"]
    excludes = contract["baseline"]["copy"]["exclude"]

    assert "runtime/lingbot_video_baseline/**" in includes
    assert not any("lingbot_video_optimized" in item for item in includes)
    assert not any("lingbot_video_optimized" in item for item in excludes)
    assert contract["baseline"]["manifest"] == "transfeat/lingbot_video/baseline.toml"


def test_lingbot_baseline_and_optimized_sources_are_physically_isolated() -> None:
    baseline_transformer = (
        ROOT
        / "runtime/lingbot_video_baseline/lingbot_src/lingbot_video/transformer_lingbot_video.py"
    ).read_text()
    optimized_transformer = (
        ROOT
        / "runtime/lingbot_video_optimized/lingbot_src/lingbot_video/transformer_lingbot_video.py"
    ).read_text()
    baseline_runner = (
        ROOT / "runtime/lingbot_video_baseline/lingbot_src/lingbot_video/runner.py"
    ).read_text()

    assert "_cudnn_varlen_attention" not in baseline_transformer
    assert "LINGBOT_ATTN_KERNEL" not in baseline_transformer
    assert "_cudnn_varlen_attention" in optimized_transformer
    assert "LINGBOT_ATTN_KERNEL" in optimized_transformer
    assert "LINGBOT_PHASE_TIMING" in baseline_runner
    assert "LINGBOT_BCAST_WEIGHTS" not in baseline_runner
    baseline_adapter = (ROOT / "runtime/lingbot_video_baseline/gpu_infer.py").read_text()
    assert "lingbot_video_optimized" not in baseline_adapter
    assert "registered c5" not in baseline_adapter

    for runtime in ("lingbot_video_baseline", "lingbot_video_optimized"):
        runtime_root = ROOT / "runtime" / runtime
        snapshot = json.loads((runtime_root / "SOURCE_SNAPSHOT.json").read_text())
        for relative, expected in snapshot["core_sha256"].items():
            actual = hashlib.sha256((runtime_root / "lingbot_src" / relative).read_bytes()).hexdigest()
            assert actual == expected


def test_lingbot_phase_parser_and_hot_sum_contract() -> None:
    adapter = load_adapter_module()
    phases = adapter.parse_phase_lines(
        [
            "PHASE base_denoise_done dt=89.10 total=325.10\n",
            "noise\n",
            "PHASE refiner_denoise_done dt=123.88 total=500.00\n",
        ]
    )

    assert phases["base_denoise_done"]["dt_s"] == 89.10
    assert phases["refiner_denoise_done"]["dt_s"] == 123.88
    assert round(adapter.sum_if_complete(89.10, 19.30, 123.88, 1.54), 2) == 233.82
    assert adapter.sum_if_complete(89.10, None) is None

    adapter.validate_topology(4, 4, True)
    adapter.validate_topology(4, 1, True)
    for invalid in ((8, 4, True), (4, 2, True), (4, 1, False)):
        try:
            adapter.validate_topology(*invalid)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"invalid topology was accepted: {invalid}")


def test_lingbot_dry_run_persists_merged_profile_and_snapshot_identity() -> None:
    expected = {
        "lingbot_video_baseline.toml": (4, "fa2", "official_video_t2v_lingbot_video.toml"),
        "lingbot_video_fsdp4_reference.toml": (
            1,
            "fa2",
            "official_video_t2v_lingbot_video_fsdp4_reference.toml",
        ),
        "lingbot_video_cudnn_optimized.toml": (
            4,
            "cudnn",
            "official_video_t2v_lingbot_video.toml",
        ),
        "lingbot_video_cudnn_off.toml": (
            4,
            "fa2",
            "official_video_t2v_lingbot_video.toml",
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "runs"
        for filename in expected:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/launch_transfeat.py",
                    f"transfeat/{filename}",
                    "--mode",
                    "dry-run",
                    "--strict-commit",
                    "--run-root",
                    str(run_root),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        for run_dir in run_root.iterdir():
            metadata = json.loads((run_dir / "metadata.json").read_text())
            manifest = tomllib.loads((run_dir / "manifest.resolved.toml").read_text())
            filename = Path(metadata["transfeat_manifest"]).name
            cp_degree, kernel, eval_name = expected[filename]
            resolved_profile = manifest["resolved_profile"]
            assert resolved_profile["official_config"]["context_parallel_degree"] == cp_degree
            assert resolved_profile["env"]["LINGBOT_ATTN_KERNEL"] == kernel
            assert resolved_profile["eval_profile"].endswith(eval_name)
            assert metadata["runtime_commit"].startswith("snapshot:")
