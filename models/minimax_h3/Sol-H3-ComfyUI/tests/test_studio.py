"""CPU tests for the Studio backend contract: capabilities, request forwarding, progress, library."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_bridge import FAKE, ROOT, bridge
from sol_engine_test.sol_h3_engine import comfy_nodes, comfy_routes, config

# Records each request the fake pipeline sees, and writes a worker progress file.
RECORD = FAKE.replace('''  output=self.root/(case["case_id"]+".mp4")''', '''  (self.root.parent/"request.json").write_text(json.dumps(REQUEST))
  if case["prompt"]=="SLOW":
   Path(REQUEST["progress"]).write_text(json.dumps({"stage":"denoising","step":2,"steps":4}))
   time.sleep(1.5)
  output=self.root/(case["case_id"]+".mp4")''')


class StudioConfig(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "config.json"
        path.write_text(json.dumps({"profiles": {
            "FP8": {"precision": "fp8", "label": "Fast", "python": "x", "paths": "y"},
            "Image": {"precision": "bf16", "steps": 8, "modes": ["text", "first_frame"],
                      "resolutions": ["768p", "480p"], "latency_s": 67.2},
            "Two": {"precision": "two_stage"}}}))
        self.env = patch.dict(os.environ, {"SOL_ENGINE_CONFIG": str(path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_capabilities(self):
        value = json.loads(json.dumps(config.capabilities()))
        self.assertEqual(value["profiles"], [
            {"name": "FP8", "label": "Fast", "steps": 4, "pipeline": "one_stage", "modes": ["text"],
             "resolutions": ["768p"], "latency_s": None, "default": True},
            {"name": "Image", "label": "Image", "steps": 8, "pipeline": "one_stage",
             "modes": ["text", "first_frame"], "resolutions": ["768p", "480p"], "latency_s": 67.2, "default": False},
            {"name": "Two", "label": "Two", "steps": 7, "pipeline": "two_stage", "modes": ["text"],
             "resolutions": ["768p"], "latency_s": None, "default": False}])
        self.assertEqual(value["device"], {"id": "rtx5090", "label": "RTX 5090"})

    def test_node_rejects_unsupported_requests(self):
        node = comfy_nodes.SolH3Generate()
        with self.assertRaisesRegex(ValueError, "does not support first-frame mode"):
            node.generate("FP8", "a scene", 1, mode="first_frame", image="a.png")
        with self.assertRaisesRegex(ValueError, "does not support reference-image mode"):
            node.generate("Image", "a scene", 1, mode="reference", image="a.png")
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            node.generate("FP8", "a scene", 1, backend="remote")
        with self.assertRaisesRegex(ValueError, "renders 768p only"):
            node.generate("FP8", "a scene", 1, resolution="480p")
        with self.assertRaisesRegex(ValueError, "Upload an image"):
            node.generate("Image", "a scene", 1, mode="first_frame", image="")
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            node.generate("Image", "a scene", 1, mode="first_frame", image="a.png,b.png")
        optional = comfy_nodes.SolH3Generate.INPUT_TYPES()["optional"]
        self.assertEqual([optional[k][1]["default"] for k in ("mode", "image", "backend", "resolution")],
                         ["text", "", "local", "768p"])


class StudioBridge(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "sol_h3_engine/worker.py", self.root / "worker.py")
        # The fake pipeline reads the raw request through a module global set by worker.py's loop.
        worker = (self.root / "worker.py").read_text().replace(
            '            request = json.loads(line)\n',
            '            request = json.loads(line)\n            sys.modules["runtime.pipeline"].REQUEST = request\n')
        (self.root / "worker.py").write_text(worker)
        runtime = self.root / "runtimes/h3_fp8/runtime"
        runtime.mkdir(parents=True)
        (runtime / "__init__.py").touch()
        (runtime / "pipeline.py").write_text(RECORD)
        (runtime / "config.py").write_text('def load_paths(p): return {}\ndef load_recipe(): return {"stage1":{"width":1344}}\ndef normalize_case(c): return c\n')
        self.old_root = bridge.ROOT
        bridge.ROOT = self.root
        self.profile = dict(python=sys.executable, precision="fp8", gpu="0", paths="unused", name="fake")

    def tearDown(self):
        bridge.close()
        bridge.ROOT = self.old_root
        self.temp.cleanup()

    def test_forwards_mode_and_images(self):
        image = str(self.root / "frame.png")
        bridge.generate(self.profile, self.root, "scene", 7, mode="first_frame", images=[image])
        request = json.loads(next((self.root / "sol_h3").glob("*/request.json")).read_text())
        self.assertEqual((request["mode"], request["images"], request["resolution"]), ("first_frame", [image], "768p"))
        bridge.generate(self.profile, self.root, "scene", 8, resolution="480p")
        request = json.loads(next((self.root / "sol_h3").glob("*/request.json")).read_text())
        self.assertEqual((request["mode"], request["images"], request["resolution"]), ("text", [], "480p"))

    def test_progress_callback(self):
        updates = []
        bridge.generate(self.profile, self.root, "SLOW", 7, progress=updates.append)
        self.assertTrue(updates)
        self.assertTrue(all(isinstance(u["elapsed_s"], float) for u in updates))
        # Before the worker writes its file the stage is generic ("generating"); afterwards it is the worker's.
        # The first poll can land on either side of that write, so only the stage set is checked.
        self.assertLessEqual({u["stage"] for u in updates}, {"generating", "denoising"})
        self.assertIn({"stage": "denoising", "step": 2, "steps": 4},
                      [{k: u.get(k) for k in ("stage", "step", "steps")} for u in updates])


class StudioLibrary(unittest.TestCase):
    def test_listing_merges_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "SolH3").mkdir()
            now = time.time()
            for index, name in enumerate(("studio_00001_.mp4", "studio_00002_.mp4", "other.mp4")):
                path = root / "SolH3" / name
                path.write_bytes(b"mp4")
                os.utime(path, (now + index * 60, now + index * 60))
            (root / "unrelated.mp4").write_bytes(b"outside library")
            comfy_nodes.write_sidecar(root, "p1", {"output": "/engine/a.mp4", "prompt": "first",
                                                   "profile": "FP8", "e2e_s": 12.5}, "text", 1)
            record = root / "SolH3/library/p1.json"
            data = json.loads(record.read_text()) | {"created": now - 1}
            record.write_text(json.dumps(data))
            (root / "SolH3/library/p2.json").write_text(json.dumps(
                {"id": "p2", "prompt": "second", "mode": "text", "seed": 2, "created": now + 59}))
            items = comfy_routes.list_videos(root, limit=8)["items"]
            self.assertEqual([i["file"] for i in items],
                             ["other.mp4", "studio_00002_.mp4", "studio_00001_.mp4"])
            self.assertEqual([i["prompt"] for i in items], [None, "second", "first"])
            self.assertEqual((items[2]["subfolder"], items[2]["type"], items[2]["e2e_s"]),
                             ("SolH3", "output", 12.5))
            self.assertEqual(len(comfy_routes.list_videos(root, limit=1)["items"]), 1)
            with self.assertRaises(ValueError):
                comfy_routes.video_path(root, "../unrelated.mp4")


class WebSyntax(unittest.TestCase):
    def test_web_modules_parse(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        for path in sorted((ROOT / "web").glob("*.js")):
            with self.subTest(path=path.name):
                # --check needs module syntax: import/export would fail as a script.
                subprocess.run([node, "--input-type=module", "--check"], input=path.read_text(),
                               text=True, check=True, timeout=30, capture_output=True)


class StudioGallery(unittest.TestCase):
    def test_listing_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "b-koi.mp4").write_bytes(b"mp4")
            (root / "b-koi.json").write_text(json.dumps({"prompt": "koi", "image": "b-koi.png", "order": 2}))
            (root / "b-koi.png").write_bytes(b"png")
            (root / "a-eagle.mp4").write_bytes(b"mp4")
            (root / "a-eagle.json").write_text(json.dumps({"prompt": "eagle", "title": "Eagle", "order": 1}))
            (root / "c-orphan.mp4").write_bytes(b"no sidecar")
            (root / "d-bad.mp4").write_bytes(b"mp4")
            (root / "d-bad.json").write_text(json.dumps({"title": "no prompt"}))
            items = comfy_routes.list_gallery(root)["items"]
            self.assertEqual([(i["name"], i["mode"], i["image"], i["title"]) for i in items],
                             [("a-eagle", "text", None, "Eagle"), ("b-koi", "first_frame", "b-koi.png", "b koi")])
            self.assertEqual(comfy_routes.gallery_file(root, "b-koi.png")[1], "image/png")
            for bad in ("../x.mp4", "b-koi.json", ".hidden.mp4", "", "missing.mp4"):
                with self.assertRaises(ValueError):
                    comfy_routes.gallery_file(root, bad)
            self.assertEqual(comfy_routes.list_gallery(None), {"items": []})


if __name__ == "__main__":
    unittest.main()
