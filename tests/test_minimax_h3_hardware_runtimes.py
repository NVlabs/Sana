"""Dependency-light contracts for the isolated H100 and A100 runtimes."""

from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = {
    "h100": ROOT / "models/minimax_h3/h100",
    "a100": ROOT / "models/minimax_h3/a100",
}
CANDIDATES = ROOT / "candidates"
PINNED_IMAGE = "docker://lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86"


def _candidate(hardware: str, profile: str) -> dict:
    path = CANDIDATES / f"minimax_h3_{hardware}_{profile}.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _profiles(hardware: str):
    return importlib.import_module(f"models.minimax_h3.{hardware}.profiles")


def test_hardware_runtimes_are_self_contained() -> None:
    assert not (ROOT / "models/minimax_h3" / ("h100" + "_a100")).exists()
    required = {
        "README.md",
        "SOURCE_SNAPSHOT.json",
        "adapter.py",
        "easycache.py",
        "first_block_cache.py",
        "gpu_infer.py",
        "model.py",
        "profiles.py",
        "registration.py",
    }
    for hardware, runtime in RUNTIMES.items():
        assert required <= {path.name for path in runtime.iterdir() if path.is_file()}
        other = "a100" if hardware == "h100" else "h100"
        for path in runtime.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".sh"}:
                source = path.read_text(encoding="utf-8")
                assert f"models.minimax_h3.{other}" not in source


def test_candidates_point_directly_to_one_hardware_runtime() -> None:
    profiles = ("dense", "quality", "balanced", "aggressive", "fullopt_exact")
    for hardware in RUNTIMES:
        expected_root = f"models/minimax_h3/{hardware}"
        for profile in profiles:
            candidate = _candidate(hardware, profile)
            assert candidate["submodule"] == expected_root
            assert candidate["runtime"]["root"] == expected_root
            assert "H3_HARDWARE" not in candidate["env"]
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


def test_profiles_share_algorithm_values_but_lock_hardware_locally() -> None:
    h100 = _profiles("h100")
    a100 = _profiles("a100")
    assert h100.HARDWARE.name == "h100"
    assert h100.HARDWARE.capability == (9, 0)
    assert h100.HARDWARE.sol_backend == "cute_sm90"
    assert a100.HARDWARE.name == "a100"
    assert a100.HARDWARE.capability == (8, 0)
    assert a100.HARDWARE.sol_backend == "triton"
    assert {name: asdict(value) for name, value in h100.PROFILES.items()} == {
        name: asdict(value) for name, value in a100.PROFILES.items()
    }

    for hardware, module in (("h100", h100), ("a100", a100)):
        with patch.dict(os.environ, {"H3_SOL_PROFILE": "fullopt_exact"}, clear=True):
            selected_hardware, profile = module.configure_runtime()
            assert selected_hardware.name == hardware
            assert profile.name == "fullopt_exact"
            assert os.environ["H3_HARDWARE"] == hardware
            assert os.environ["H3_SOL_SINK_MODE"] == "prefix"
            assert os.environ["H3_REORDER"] == "0"
            assert os.environ["H3_OFFLOAD"] == "0"


def test_hardware_and_policy_overrides_fail_closed() -> None:
    for hardware, module in (("h100", _profiles("h100")), ("a100", _profiles("a100"))):
        other = "a100" if hardware == "h100" else "h100"
        with patch.dict(
            os.environ,
            {"H3_SOL_PROFILE": "fullopt_exact", "H3_HARDWARE": other},
            clear=True,
        ):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "H3_HARDWARE is locked"):
                module.configure_runtime()
        with patch.dict(
            os.environ,
            {"H3_SOL_PROFILE": "fullopt_exact", "H3_SOL_TAU": "0.5"},
            clear=True,
        ):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "H3_SOL_TAU is locked"):
                module.configure_runtime()


def test_candidates_do_not_inherit_the_legacy_diffusers_environment() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/launch_candidate.py"))
    merge = namespace["merge_model_profile"]
    for hardware in RUNTIMES:
        merged = merge(_candidate(hardware, "fullopt_exact"))
        assert "H3_CONDA_ROOT" not in merged["env"]
        assert "H3_CONDA_ENV" not in merged["env"]
        assert "PYTHON_BIN" not in merged["env"]
        assert merged["official_config"]["width"] == 1344
        assert merged["official_config"]["revision"] == (
            "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
        )


def test_each_runtime_registers_locally_and_never_patches_sglang() -> None:
    for hardware, runtime in RUNTIMES.items():
        registration = (runtime / "registration.py").read_text(encoding="utf-8")
        runner = (runtime / "run_minimax_h3_gpu.sh").read_text(
            encoding="utf-8"
        )
        assert "ModelRegistry.register_model" in registration
        assert "PINNED_SGLANG_MODEL_SHA256" in registration
        assert "sglang serve" not in runner
        assert "git apply" not in runner
        assert "cp -a" not in runner
        assert not list(runtime.rglob("*.patch"))
        assert f"models/minimax_h3/{hardware}/run_minimax_h3_gpu.sh" in runner


def test_each_source_snapshot_matches_its_runtime() -> None:
    for hardware, runtime in RUNTIMES.items():
        snapshot = json.loads(
            (runtime / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8")
        )
        assert snapshot["variant"] == hardware
        assert list(snapshot["hardware"]) == [hardware]
        for relative, expected in snapshot["core_sha256"].items():
            actual = hashlib.sha256((runtime / relative).read_bytes()).hexdigest()
            assert actual == expected, relative


def test_models_contain_only_the_required_attention_and_cache_hooks() -> None:
    for runtime in RUNTIMES.values():
        source = (runtime / "model.py").read_text(encoding="utf-8")
        assert ast.parse(source) is not None
        assert "_sol_engine_forward_attention" in source
        assert "_sol_engine_easycache_before_blocks" in source
        assert "_sol_engine_easycache_after_blocks" in source
        assert "_sol_engine_cache_after_head" in source
        assert "_sol_engine_cache_after_tail" in source
        assert "models.minimax_h3.portable" not in source


def test_adapters_use_full_prefix_sink_without_reordering() -> None:
    for runtime in RUNTIMES.values():
        source = (runtime / "adapter.py").read_text(encoding="utf-8")
        assert "sink_start = 0" in source
        assert "sink_tokens = context.prefix_tokens" in source
        assert "segment[:, :sink_tokens] = _dense_queries" in source
        assert "from sol_attn import get_sol_attn_backend, sol_attn" in source
        assert "reorder" not in source.lower()


def test_offline_runners_use_official_metrics_and_four_way_ulysses() -> None:
    for hardware, runtime in RUNTIMES.items():
        source = (runtime / "gpu_infer.py").read_text(encoding="utf-8")
        assert f"models.minimax_h3.{hardware}.registration" in source
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
