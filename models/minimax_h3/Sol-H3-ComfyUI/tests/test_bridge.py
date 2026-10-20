"""CPU protocol tests with an explicit fake pipeline, never GPU evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sol_engine_test", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from sol_engine_test.sol_h3_engine import bridge

FAKE = '''
import json,time,os
from pathlib import Path
class Pipeline:
 def __init__(self,paths,root,offload_mode):
  self.root=Path(root); self.root.mkdir(); self.report={"startup_and_warmup_s":1.25}
 def start(self,case): pass
 def generate(self,case):
  assert os.environ.get("PYTORCH_ALLOC_CONF")=="backend:native"
  assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
  if case["prompt"]=="FAIL": raise ValueError("deliberate pipeline failure")
  if case["prompt"]=="WAIT": time.sleep(30)
  output=self.root/(case["case_id"]+".mp4"); output.write_bytes(b"protocol-test-only")
  return dict(case,status="PASS",output=str(output),e2e_s=0.5)
 def finish(self): pass
 def close(self):
  time.sleep(0.2)  # Caller sees FAIL while teardown is still in progress.
  (self.root.parent/"closed").touch()
'''


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "sol_h3_engine/worker.py", self.root / "worker.py")
        runtime = self.root / "runtimes/h3_fp8/runtime"
        runtime.mkdir(parents=True)
        (runtime / "__init__.py").touch()
        (runtime / "pipeline.py").write_text(FAKE)
        (runtime / "config.py").write_text('def load_paths(p): return {}\ndef load_recipe(): return {"stage1":{"width":1344}}\ndef normalize_case(c): return c\n')
        self.old_root = bridge.ROOT
        bridge.ROOT = self.root
        self.profile = dict(python=sys.executable, precision="fp8", gpu="0", paths="unused", name="fake")

    def tearDown(self):
        bridge.close()
        bridge.ROOT = self.old_root
        self.temp.cleanup()

    def test_resident_process_and_unique_outputs(self):
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF":"backend:cudaMallocAsync",
                                     "PYTORCH_ALLOC_CONF":"backend:cudaMallocAsync"}):
            first = bridge.generate(self.profile, self.root, "你好 ' $() \n second line", 42)
        pid = bridge._engine.process.pid
        second = bridge.generate(self.profile, self.root, "second", 43)
        self.assertEqual(pid, bridge._engine.process.pid)
        self.assertTrue(first["cold_start"])
        self.assertFalse(second["cold_start"])
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(first["prompt"], "你好 ' $() \n second line")
        work = bridge._engine.root
        bridge.close()
        self.assertTrue((work / "closed").is_file())

    def test_error_closes_session(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate pipeline failure"):
            bridge.generate(self.profile, self.root, "FAIL", 42)
        self.assertIsNone(bridge._engine)
        self.assertTrue(list((self.root / "sol_h3").glob("*/closed")))

    def test_cancel_closes_session(self):
        first = bridge.generate(self.profile, self.root, "warmup", 42)
        root = bridge._engine.root
        def cancel():
            raise InterruptedError("cancelled")
        with self.assertRaises(InterruptedError):
            bridge.generate(self.profile, self.root, "WAIT", 42, cancel=cancel)
        self.assertTrue((root / "closed").is_file())

    def test_failed_launch_has_no_cached_engine(self):
        self.profile["python"] = "/nonexistent/runtime/python"
        with self.assertRaises(FileNotFoundError):
            bridge.generate(self.profile, self.root, "prompt", 42)
        self.assertIsNone(bridge._engine)


if __name__ == "__main__":
    unittest.main()
