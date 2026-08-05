"""Static and dependency-light contracts for the released Sol-Attn backend."""

from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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
    assert set(backend.__all__) == {
        "make_sol_attn_dispatch",
        "make_mmdit_sol_attn_dispatch",
    }
    assert hasattr(backend, "_run_sol_attn_bthd")
    assert hasattr(backend, "_run_mmdit_sol_attn_bthd")
    assert not hasattr(backend, "sol_attn_attention")
    assert not hasattr(backend, "sol_attn_hunyuan")
    assert not hasattr(backend, "make_hunyuan_sol_attn_dispatch")

    interface = BACKENDS / "sol_attn" / "interface.py"
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


def test_sol_attn_backend_dispatch_contract() -> None:
    interface = BACKENDS / "sol_attn" / "interface.py"
    source = interface.read_text()
    tree = ast.parse(source)

    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "cutlass.cute" not in top_level_imports
    assert "cuda.bindings.driver" not in top_level_imports

    dispatch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_backend_for_arch"
    )
    namespace = {
        "_CUTE_BACKENDS": {
            (9, 0): "cute_sm90",
            (10, 0): "cute_sm100",
            (12, 0): "cute_sm120",
        },
        "_cute_runtime_available": lambda: True,
    }
    dispatch_module = ast.fix_missing_locations(
        ast.Module(body=[dispatch], type_ignores=[])
    )
    exec(compile(dispatch_module, str(interface), "exec"), namespace)
    backend_for_arch = namespace["_backend_for_arch"]
    assert backend_for_arch((9, 0), cute_available=True) == "cute_sm90"
    assert backend_for_arch((10, 0), cute_available=True) == "cute_sm100"
    assert backend_for_arch((12, 0), cute_available=True) == "cute_sm120"
    assert backend_for_arch((8, 0), cute_available=True) == "triton"
    assert backend_for_arch((8, 9), cute_available=True) == "triton"
    assert backend_for_arch((9, 0), cute_available=False) == "triton"
    assert backend_for_arch((10, 0), cute_available=False) == "triton"
    assert backend_for_arch((12, 0), cute_available=False) == "triton"

    triton_forward = (
        BACKENDS / "sol_attn" / "triton_ref" / "fwd.py"
    ).read_text()
    triton_preprocess = (
        BACKENDS / "sol_attn" / "triton_ref" / "preprocess.py"
    ).read_text()
    assert "_forward_tma" in triton_forward
    assert "_forward_ptr" in triton_forward
    assert "TensorDescriptor" in triton_forward
    assert "TensorDescriptor" not in triton_preprocess
    assert "from ..preprocess import prepare as prepare_tma" in triton_forward
    assert "from .preprocess import prepare as prepare_ptr" in triton_forward
    assert "HAS_SINK: tl.constexpr" in triton_forward
    assert "if HAS_SINK:" in triton_forward
    assert "_pad_to_block" not in triton_forward
    assert "tokens % BLOCK == 0" not in triton_forward

    backend_source = (BACKENDS / "sol_attn_backend.py").read_text()
    assert "arch[0] >= 8" in backend_source

    bf16 = object()
    fake_torch = SimpleNamespace(
        bfloat16=bf16,
        cuda=SimpleNamespace(
            get_device_capability=lambda _device: (8, 9),
        ),
    )
    fake_q = SimpleNamespace(
        is_cuda=True,
        ndim=4,
        shape=(1, 256, 2, 128),
        dtype=bf16,
        device="cuda:0",
    )
    backend = load_backend()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert backend.sol_attn_supported(fake_q)
        fake_torch.cuda.get_device_capability = lambda _device: (12, 0)
        assert backend.sol_attn_supported(fake_q)
        fake_torch.cuda.get_device_capability = lambda _device: (7, 5)
        assert not backend.sol_attn_supported(fake_q)


def test_only_one_sol_attn_backend_tree_remains() -> None:
    legacy = (
        "pisa_hyvideo",
        "sol_attn_colmask",
        "sol_attn_hunyuan_v2.py",
        "sol_attn_hunyuan_v3.py",
    )
    assert all(not (BACKENDS / name).exists() for name in legacy)
    assert (BACKENDS / "pyproject.toml").is_file()
    assert not (BACKENDS / "sol_attn" / "sol_attn").exists()
    assert (BACKENDS / "sol_attn" / "sm90").is_dir()
    assert (BACKENDS / "sol_attn" / "sm100").is_dir()
    assert (BACKENDS / "sol_attn" / "sm120").is_dir()
    readme = (BACKENDS / "README.md").read_text()
    assert "sink_start" in readme
    assert "RTX 4090" in readme and "SM89" in readme
    assert "RTX 5090" in readme and "SM120" in readme
    assert "A100" in readme and "SM80" in readme
    assert "Triton research implementation" in readme
    normalized_readme = " ".join(readme.lower().split())
    assert "text-query rows" in normalized_readme
    assert "dense" in normalized_readme

    root_readme = (ROOT / "README.md").read_text()
    root_readme = " ".join(root_readme.split())
    assert "A100 (SM80)" in root_readme
    assert "RTX 4090 (SM89)" in root_readme
    assert "RTX 5090 (SM120)" in root_readme


def test_sm120_is_eligible_and_keeps_single_kv_split(monkeypatch) -> None:
    backend = load_backend()
    bf16 = object()
    fake_torch = SimpleNamespace(
        bfloat16=bf16,
        cuda=SimpleNamespace(get_device_capability=lambda _device: (12, 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    q = SimpleNamespace(
        is_cuda=True,
        ndim=4,
        shape=(1, 65536, 32, 128),
        dtype=bf16,
        device="cuda:0",
    )

    assert backend.sol_attn_supported(q)
    assert backend._resolve_kv_splits(q, "auto") == 1

    fake_torch.cuda.get_device_capability = lambda _device: (9, 0)
    monkeypatch.setattr(backend, "_cute_runtime_available", lambda: True)
    assert backend._resolve_kv_splits(q, "auto") == 4
    monkeypatch.setattr(backend, "_cute_runtime_available", lambda: False)
    assert backend._resolve_kv_splits(q, "auto") == 1


def test_core_interface_defines_sm120_compile_dispatch() -> None:
    tree = ast.parse((BACKENDS / "sol_attn" / "interface.py").read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    assert "_compile_sm120" in functions
    assert "cute_sm120" in (
        BACKENDS / "sol_attn" / "interface.py"
    ).read_text()
    assert "_compile_sm120" in ast.unparse(functions["_sol_attn_cute"])


def test_sm120_p19_lookahead_is_the_release_default() -> None:
    sm120 = BACKENDS / "sol_attn" / "sm120"
    mainloop_path = sm120 / "mainloop.py"
    kernel_path = sm120 / "kernel.py"

    mainloop_tree = ast.parse(mainloop_path.read_text())
    kernel_tree = ast.parse(kernel_path.read_text())
    mainloop_class = next(
        node
        for node in mainloop_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SolAttnForwardSm120"
    )
    constructor = next(
        node
        for node in mainloop_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    make_kernel = next(
        node
        for node in kernel_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_kernel"
    )

    for function in (constructor, make_kernel):
        defaults = {
            arg.arg: default
            for arg, default in zip(
                function.args.kwonlyargs,
                function.args.kw_defaults,
            )
        }
        for flag in (
            "prefetch_first_exact_k",
            "prefetch_next_route_k",
        ):
            assert isinstance(defaults[flag], ast.Constant)
            assert defaults[flag].value is True

    source = mainloop_path.read_text()
    first_exact_prefetch = source.index(
        "if cutlass.const_expr(self.prefetch_first_exact_k):"
    )
    approximate_v_wait = source.index(
        "v_wait = V_pipeline.consumer_try_wait(V_consumer)",
        first_exact_prefetch,
    )
    assert first_exact_prefetch < approximate_v_wait
    assert "previous_group_exact_count" in source
    assert "tKCgKC[None, next_route_group]" in source


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
    assert "make_mmdit_sol_attn_dispatch" in paths[0].read_text()

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
