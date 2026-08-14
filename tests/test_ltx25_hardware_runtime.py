from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "models" / "ltx25" / "GB200"


def _config(name: str) -> dict:
    with (RUNTIME / name).open("rb") as handle:
        return tomllib.load(handle)


def test_ltx25_gb200_configs_are_four_gpu_peer_arms() -> None:
    dense = _config("dense.toml")
    fullopt = _config("fullopt.toml")

    for config in (dense, fullopt):
        assert config["runtime"] == "."
        assert config["entry"] == "run_ltx25_gpu.sh"
        assert config["gpus"] == 4
        assert config["LTX25_WORLD_SIZE"] == "4"
        assert config["LTX_SOL_STAGE1"] == "0"
        assert config["LTX25_PROFILE"] == "default5s"
        assert config["LTX25_STAGE1_STEPS"] == "30"
        assert config["LTX25_SEED"] == "42"
        assert config["LTX25_PROMPT_FILES"] == dense["LTX25_PROMPT_FILES"]
        assert config["LTX25_WARMUP_REQUESTS"] == dense["LTX25_WARMUP_REQUESTS"]
        assert config["LTX25_MEASURE_REQUESTS"] == dense["LTX25_MEASURE_REQUESTS"]
        assert "PYTHON_BIN" not in config
        assert config["LTX25_COMPILE_CACHE_ROOT"].startswith(".cache/")

    assert dense["LTX25_S1_PARALLEL"] == "sp"
    assert dense["LTX25_CACHE"] == "off"
    assert dense["LTX25_COMPILE"] == "0"

    assert fullopt["LTX25_S1_PARALLEL"] == "cfg"
    assert fullopt["LTX25_CACHE"] == "fbcache"
    assert fullopt["LTX25_CACHE_THRESHOLD"] == "0.08"
    assert fullopt["LTX25_COMPILE"] == "1"


def test_ltx25_gb200_runtime_files_and_prompts_exist() -> None:
    for relative in (
        "README.md",
        "SOURCE_SNAPSHOT.json",
        "gpu_infer.py",
        "run_ltx25_gpu.sh",
        "setup_env.sh",
        "validate_env.py",
        "environment/LTX-2/pyproject.toml",
        "environment/LTX-2/uv.lock",
        "environment/LTX-2/packages/ltx-kernels/src/ltx_kernels/__init__.py",
        "ltx_src/ltx_core/opt/step_cache.py",
        "ltx_src/ltx_pipelines/multigpu/cfgp_builder.py",
        "ltx_src/ltx_pipelines/ti2vid_two_stages_mgpu.py",
        "ltx_src/ltx_pipelines/utils/opt_stack.py",
        "ltx_src/ltx_pipelines/utils/opt_timing.py",
    ):
        assert (RUNTIME / relative).is_file(), relative

    for prompt in _config("dense.toml")["LTX25_PROMPT_FILES"].split(":"):
        assert (REPO / prompt).is_file(), prompt

    snapshot = json.loads((RUNTIME / "SOURCE_SNAPSHOT.json").read_text())
    assert snapshot["upstream_commit"].startswith("7954dcb")
    assert snapshot["optimized_commit"].startswith("ccedf84")
    assert snapshot["environment"]["default_python"] == ".venv/bin/python"
    assert "optimized_checkout" not in snapshot
    assert "python_env" not in snapshot


def test_ltx25_gb200_delivery_excludes_sol_experiment() -> None:
    source = RUNTIME / "ltx_src"
    assert not list(source.rglob("sol_attention.py*"))
    for relative in (
        "ltx_pipelines/ti2vid_two_stages_mgpu.py",
        "ltx_pipelines/utils/opt_stack.py",
    ):
        text = (source / relative).read_text()
        assert "ltx_core.opt.sol_attention" not in text

    launcher = (RUNTIME / "run_ltx25_gpu.sh").read_text()
    assert "mode=max-autotune-no-cudagraphs" in launcher
    assert "fullgraph=false" in launcher
    assert "capture=false" in launcher
    assert "export LTX_SOL_STAGE1=0" in launcher


def test_ltx25_stage1_cache_scope_is_explicit() -> None:
    cache = (RUNTIME / "ltx_src/ltx_core/opt/step_cache.py").read_text()
    stack = (RUNTIME / "ltx_src/ltx_pipelines/utils/opt_stack.py").read_text()
    assert "if CTRL.stage != 1" in cache
    assert "stage = 2 if name == \"SimpleDenoiser\" else 1" in stack
    assert "ctrl.begin_step(step_index, stage=stage)" in stack


def test_ltx25_delivery_has_no_sibling_checkout_dependency() -> None:
    forbidden = ("ltx25" + "-opt", "Sol-LTX" + "-Infer", "sol-engine" + "-ltx25")
    for relative in (
        "README.md",
        "SOURCE_SNAPSHOT.json",
        "dense.toml",
        "fullopt.toml",
        "run_ltx25_gpu.sh",
    ):
        text = (RUNTIME / relative).read_text()
        assert not any(value in text for value in forbidden), relative


def test_ltx25_authored_python_is_syntactically_valid() -> None:
    for path in (
        RUNTIME / "gpu_infer.py",
        RUNTIME / "ltx_src/ltx_core/opt/step_cache.py",
        RUNTIME / "ltx_src/ltx_pipelines/multigpu/cfgp_builder.py",
        RUNTIME / "ltx_src/ltx_pipelines/ti2vid_two_stages_mgpu.py",
        RUNTIME / "ltx_src/ltx_pipelines/utils/opt_stack.py",
    ):
        ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    tests = sorted(name for name in globals() if name.startswith("test_"))
    for name in tests:
        globals()[name]()
    print(f"CONTRACT_PASS {len(tests)}")
