"""Static and dependency-light contracts for the released Sol-Attn backend."""

from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "techniques" / "sparse_backends"


def load_backend():
    path = BACKENDS / "sol_attn_backend.py"
    spec = importlib.util.spec_from_file_location("sana_sol_backend_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_api_and_integration_defaults() -> None:
    backend = load_backend()
    assert backend.DEFAULT_TAU == 1.0
    assert backend.DEFAULT_THRESH_TYPE == "diag"
    assert backend.get_sol_attn_stats() == {
        "dispatch_calls": 0,
        "kernel_calls": 0,
        "hunyuan_calls": 0,
        "dense_guard_calls": 0,
    }
    assert backend._parse_layer_ranges("0,2-4,7") == frozenset(
        {0, 2, 3, 4, 7}
    )

    interface = BACKENDS / "sol_attn" / "sol_attn" / "interface.py"
    tree = ast.parse(interface.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sol_attn"
    )
    keyword_names = {arg.arg for arg in function.args.kwonlyargs}
    assert {
        "tau",
        "thresh_type",
        "kv_splits",
        "sink_start",
        "sink_tokens",
    } <= keyword_names


def test_only_one_sol_attn_backend_tree_remains() -> None:
    legacy = (
        "pisa_hyvideo",
        "sol_attn_colmask",
        "sol_attn_hunyuan_v2.py",
        "sol_attn_hunyuan_v3.py",
    )
    assert all(not (BACKENDS / name).exists() for name in legacy)
    assert (BACKENDS / "sol_attn" / "sol_attn" / "sm90").is_dir()
    assert (BACKENDS / "sol_attn" / "sol_attn" / "sm100").is_dir()
    readme = (BACKENDS / "README.md").read_text()
    assert "sink_start" in readme
    assert "text Q is dense" in " ".join(readme.split())


def test_model_callers_use_the_release_configuration() -> None:
    forbidden = (
        "HUNYUAN_SOL_V2",
        "HUNYUAN_SOL_V3",
        "HUNYUAN_SOLV2",
        "HUNYUAN_SOLV3",
        "WAN22_SOL_DENSITY",
        "target_density",
    )
    paths = (
        ROOT / "models" / "hunyuan_video" / "optimized" / "gpu_infer.py",
        ROOT / "models" / "wan21_t2v_14b" / "optimized" / "gpu_infer.py",
        ROOT / "models" / "wan21_t2v_1_3b" / "optimized" / "gpu_infer.py",
    )
    combined = "\n".join(path.read_text() for path in paths)
    assert all(token not in combined for token in forbidden)
    assert "make_hunyuan_sol_attn_dispatch" in paths[0].read_text()

    sol_candidates = []
    for path in sorted((ROOT / "candidates").glob("*.toml")):
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        env = payload.get("env", {})
        if env.get("HUNYUAN_SOL_ATTN") == "1":
            assert env["HUNYUAN_SOL_TAU"] == "1.0"
            assert env["HUNYUAN_SOL_THRESH_TYPE"] == "diag"
            assert env["HUNYUAN_SOL_KV_SPLITS"] == "auto"
            sol_candidates.append(path)
        if env.get("WAN22_SOL_ATTN") == "1":
            assert env["WAN22_SOL_TAU"] == "1.0"
            assert env["WAN22_SOL_THRESH_TYPE"] == "diag"
            assert env["WAN22_SOL_KV_SPLITS"] == "auto"
            sol_candidates.append(path)
    assert sol_candidates
