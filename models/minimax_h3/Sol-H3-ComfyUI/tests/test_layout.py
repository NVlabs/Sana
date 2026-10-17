"""Exercise imports across the plugin/engine/runtime directory boundaries."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "sol_h3_engine"


class LayoutTest(unittest.TestCase):
    def test_comfy_entry_from_unrelated_working_directory(self):
        code = '''
import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("custom_sol_h3", root / "__init__.py",
                                             submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert set(module.NODE_CLASS_MAPPINGS) == {"SolH3Generate", "SolH3Release"}
assert (root / module.WEB_DIRECTORY / "timing.js").is_file()
from custom_sol_h3.sol_h3_engine import config, bridge
assert config.config_path() == root / "config.json"
assert (bridge.ROOT / "worker.py").is_file()
assert "torch" not in sys.modules
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("SOL_ENGINE_CONFIG", None)
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([sys.executable, "-c", code, str(ROOT)],
                           cwd=temp, env=env, check=True, timeout=15)

    def test_nested_worker_finds_runtime_and_optional_decoder(self):
        code = '''
import os, pathlib, subprocess, sys
from runtime.config import load_recipe
from runtime.pipeline import worker_environment
recipe = load_recipe()
assert recipe["stage1"]["updates"] == int(sys.argv[2]) and "stage2" not in recipe
paths = {key: "/unused" for key in ("fa4_root", "fastvideo_root", "ltx_root")}
env = worker_environment("stage1", paths)
probe = "import pathlib,sys,runtime.config,fast_vae_adapter; assert pathlib.Path(fast_vae_adapter.__file__).parent == pathlib.Path(sys.argv[1])"
subprocess.run([sys.executable, "-c", probe, sys.argv[1]], env=env,
               check=True, timeout=15)
'''
        with tempfile.TemporaryDirectory() as temp:
            for precision, updates in (("fp8", 4), ("bf16", 4), ("baseline", 50)):
                with self.subTest(precision=precision):
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                               PYTHONPATH=str(ENGINE / "runtimes" / ("h3_" + precision)))
                    env.pop("SOL_H3_FAST_VAE", None)
                    subprocess.run([sys.executable, "-c", code, str(ENGINE), str(updates)],
                                   cwd=temp, env=env, check=True, timeout=30)


class SyntaxTest(unittest.TestCase):
    def test_every_engine_file_compiles(self):
        # Runtime modules are only imported inside the GPU worker; catch syntax errors on CPU.
        for path in sorted(ENGINE.rglob("*.py")):
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_text(), str(path), "exec")


class HyperFlowHookTest(unittest.TestCase):
    def test_step_hook_returns_none(self):
        # A non-None forward pre-hook return replaces the DiT inputs (seen as a duplicate
        # `hidden_states` error on GPU); keep the counter hook side-effect only.
        import ast
        tree = ast.parse((ENGINE / "hyperflow_worker.py").read_text())
        hooks = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "register_forward_pre_hook"]
        self.assertTrue(hooks)
        for call in hooks:
            self.assertNotIsInstance(call.args[0], ast.Lambda)
