"""Dependency-light contracts for the registered H100/A100 runtime."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import runpy
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "models/minimax_h3/h100_a100"
CANDIDATES = ROOT / "candidates"
PINNED_IMAGE = "docker://lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86"


def _candidate(hardware: str, profile: str) -> dict:
    path = CANDIDATES / f"minimax_h3_{hardware}_{profile}.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_candidates_cover_both_hardware_targets_and_all_profiles() -> None:
    profiles = ("dense", "quality", "balanced", "aggressive", "fullopt_exact")
    for hardware in ("h100", "a100"):
        for profile in profiles:
            candidate = _candidate(hardware, profile)
            assert candidate["env"]["H3_HARDWARE"] == hardware
            assert candidate["env"]["H3_SOL_PROFILE"] == profile
            assert candidate["env"]["H3_CONTAINER_IMAGE"] == PINNED_IMAGE
            assert not candidate["inherit_profile_env"]
            assert candidate["official_config"]["num_gpus"] == 4
            assert candidate["official_config"]["context_parallel_degree"] == 4
            assert candidate["artifacts"] == {
                "output_dir": "outputs",
                "video": "out.mp4",
                "log": "run.log",
                "benchmark": "benchmark.json",
            }


def test_h100_and_a100_share_the_same_algorithm_profiles() -> None:
    from models.minimax_h3.h100_a100 import profiles

    assert profiles.PROFILES["quality"].tau == 0.5
    assert profiles.PROFILES["quality"].dense_steps == 15
    assert profiles.PROFILES["balanced"].cache == "easycache"
    assert profiles.PROFILES["aggressive"].threshold_type == "diag"
    assert profiles.PROFILES["fullopt_exact"].threshold_type == "exact"
    assert profiles.HARDWARE["h100"].sol_backend == "cute_sm90"
    assert profiles.HARDWARE["a100"].sol_backend == "triton"

    with patch.dict(
        os.environ,
        {"H3_HARDWARE": "a100", "H3_SOL_PROFILE": "fullopt_exact"},
        clear=True,
    ):
        hardware, profile = profiles.configure_runtime()
        assert hardware.name == "a100"
        assert profile.name == "fullopt_exact"
        assert os.environ["H3_SOL_SINK_MODE"] == "prefix"
        assert os.environ["H3_REORDER"] == "0"
        assert os.environ["H3_OFFLOAD"] == "0"


def test_locked_profile_rejects_policy_drift() -> None:
    from models.minimax_h3.h100_a100 import profiles

    environment = {
        "H3_HARDWARE": "h100",
        "H3_SOL_PROFILE": "fullopt_exact",
        "H3_SOL_TAU": "0.5",
    }
    with patch.dict(os.environ, environment, clear=True):
        try:
            profiles.configure_runtime()
        except RuntimeError as exc:
            assert "H3_SOL_TAU is locked" in str(exc)
        else:
            raise AssertionError("conflicting profile override was accepted")


def test_candidates_do_not_inherit_the_legacy_diffusers_environment() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/launch_candidate.py"))
    merge = namespace["merge_model_profile"]
    merged = merge(_candidate("h100", "fullopt_exact"))
    assert "H3_CONDA_ROOT" not in merged["env"]
    assert "H3_CONDA_ENV" not in merged["env"]
    assert "PYTHON_BIN" not in merged["env"]
    assert merged["official_config"]["width"] == 1344
    assert merged["official_config"]["revision"] == (
        "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
    )


def test_runtime_registers_locally_and_never_patches_sglang() -> None:
    files = {path.name for path in RUNTIME.iterdir() if path.is_file()}
    assert "model.py" in files
    assert "registration.py" in files
    assert "request.py" not in files
    assert "summarize.py" not in files
    assert not list(RUNTIME.rglob("*.patch"))

    registration = (RUNTIME / "registration.py").read_text(encoding="utf-8")
    runner = (RUNTIME / "scripts/run_minimax_h3_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "ModelRegistry.register_model" in registration
    assert "PINNED_SGLANG_MODEL_SHA256" in registration
    assert "sglang serve" not in runner
    assert "git apply" not in runner
    assert "cp -a" not in runner


def test_model_contains_only_the_required_attention_and_cache_hooks() -> None:
    source = (RUNTIME / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert "_sol_engine_forward_attention" in source
    assert "_sol_engine_easycache_before_blocks" in source
    assert "_sol_engine_easycache_after_blocks" in source
    assert "_sol_engine_cache_after_head" in source
    assert "_sol_engine_cache_after_tail" in source
    assert "models.minimax_h3.portable" not in source


def test_adapter_uses_full_prefix_sink_without_reordering() -> None:
    source = (RUNTIME / "adapter.py").read_text(encoding="utf-8")
    assert "sink_start = 0" in source
    assert "sink_tokens = context.prefix_tokens" in source
    assert "segment[:, :sink_tokens] = _dense_queries" in source
    assert "from sol_attn import get_sol_attn_backend, sol_attn" in source
    assert "reorder" not in source.lower()


def test_offline_runner_uses_official_metrics_and_four_way_ulysses() -> None:
    source = (RUNTIME / "gpu_infer.py").read_text(encoding="utf-8")
    assert "DiffGenerator.from_pretrained" in source
    assert '"total_duration_s"' in source
    assert "result.peak_memory_mb" in source
    assert "num_gpus=4" in source
    assert "ulysses_degree=4" in source
    assert "use_fsdp_inference=True" in source
    assert "layerwise_offload_components=[]" in source
    assert "enable_torch_compile=False" in source
    assert "generator.shutdown()" in source


def test_backend_selection_is_public() -> None:
    interface = (
        ROOT / "techniques/sparse_backends/sol_attn/interface.py"
    ).read_text(encoding="utf-8")
    package = (
        ROOT / "techniques/sparse_backends/sol_attn/__init__.py"
    ).read_text(encoding="utf-8")
    assert "def get_sol_attn_backend" in interface
    assert "get_sol_attn_backend" in package


ContractsTest = type(
    "ContractsTest",
    (unittest.TestCase,),
    {
        name: staticmethod(value)
        for name, value in tuple(globals().items())
        if name.startswith("test_") and callable(value)
    },
)


if __name__ == "__main__":
    unittest.main()
