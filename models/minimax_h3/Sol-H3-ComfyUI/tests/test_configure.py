"""Regression for the broken checkpoint links encountered during deployment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("configure", Path(__file__).parents[1]/"scripts/configure.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointTest(unittest.TestCase):
    def test_broken_shard_rejected_then_valid_model_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for component in ("transformer", "vae", "audio_vae"):
                folder=root/component
                folder.mkdir()
                (folder/"diffusion_pytorch_model.safetensors").write_bytes(b"file-existence-only")
            folder=root/"transformer"
            (folder/"diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({
                "weight_map":{"parameter":"shard.safetensors"}}))
            shard=folder/"shard.safetensors"
            shard.symlink_to(root/"absent")
            with self.assertRaisesRegex(FileNotFoundError,"shard.safetensors"):
                module.validate_model(root)
            shard.unlink()
            shard.write_bytes(b"file-existence-only")
            module.validate_model(root)


class FastHyperFlowProfileTest(unittest.TestCase):
    def test_fast_profile_is_default_and_carries_the_validated_recipe(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "MiniMax-H3"
            for component in ("transformer", "vae", "audio_vae"):
                (model / component).mkdir(parents=True)
                (model / component / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
            python = root / "python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
            manifest = {key: str(root / key) for key in module.REQUIRED}
            manifest.update(stage1_python=str(python), qwen_python=str(python), h3_model=str(model))
            for key, value in manifest.items():
                if key not in ("stage1_python", "qwen_python", "h3_model"):
                    Path(value).mkdir(exist_ok=True)
            (root / "paths.json").write_text(json.dumps(manifest))
            files = {name: root / name for name in ("lora.safetensors", "hf.safetensors", "table.pt", "fvae.safetensors")}
            for path in files.values():
                path.write_bytes(b"x")
            for name in ("deps", "fp8", "fvae-code"):
                (root / name).mkdir()
            argv = ["configure.py", "--reuse-paths", str(root / "paths.json"), "--dense-lora", str(files["lora.safetensors"]),
                    "--output-dir", str(root / "out"), "--hyperflow-weights", str(files["hf.safetensors"]),
                    "--hyperflow-deps", str(root / "deps"), "--hyperflow-fp8-dir", str(root / "fp8"),
                    "--hyperflow-adaln-table", str(files["table.pt"]),
                    "--hyperflow-fast-vae-weights", str(files["fvae.safetensors"]), "--fast-vae-code", str(root / "fvae-code")]
            with mock.patch.object(sys, "argv", argv):
                module.main()
            profiles = json.loads((root / "out" / "config.json").read_text())["profiles"]
            first, settings = next(iter(profiles.items()))
            self.assertEqual(first, "Sol-H3 one stage — RTX 5090")
            self.assertEqual((settings["modes"], settings["resolutions"]), (["text", "first_frame"], ["768p", "480p"]))
            self.assertEqual(settings["hyperflow"]["workflows"], ["t2va", "fl2va"])
            hf = settings["hyperflow"]
            for key in ("sol_attn", "compile", "pin_unmatched_group", "qwen_resident"):
                self.assertIs(hf[key], True)
            self.assertEqual(hf["dense_attn"], "sage2pp")
            self.assertEqual(hf["fast_vae"]["vae_config"], str(model / "vae/config.json"))
            self.assertIn("Sol-H3 HyperFlow · 8 steps — RTX 5090", profiles)  # original BF16 HyperFlow kept
            self.assertNotIn("H3 FP8 + noncausal VAE — RTX 5090", profiles)  # --fast-vae-code alone does not add it


class TwoStageProfileTest(unittest.TestCase):
    def test_two_stage_profile_points_at_the_package(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "MiniMax-H3"
            for component in ("transformer", "vae", "audio_vae"):
                (model / component).mkdir(parents=True)
                (model / component / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
            python = root / "python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
            manifest = {key: str(root / key) for key in module.REQUIRED}
            manifest.update(stage1_python=str(python), qwen_python=str(python), h3_model=str(model))
            for key, value in manifest.items():
                if key not in ("stage1_python", "qwen_python", "h3_model"):
                    Path(value).mkdir(exist_ok=True)
            (root / "paths.json").write_text(json.dumps(manifest))
            (root / "lora.safetensors").write_bytes(b"x")
            package = root / "Sol-H3-Spark"
            (package / "runtime").mkdir(parents=True)
            (package / "runtime" / "pipeline.py").write_text("")
            (root / "two.json").write_text(json.dumps({"stage1_python": str(python)}))
            (root / "gallery").mkdir()
            argv = ["configure.py", "--reuse-paths", str(root / "paths.json"), "--dense-lora", str(root / "lora.safetensors"),
                    "--output-dir", str(root / "out"), "--two-stage-package", str(package),
                    "--two-stage-paths", str(root / "two.json"), "--device", "spark", "--gallery", str(root / "gallery")]
            with mock.patch.object(sys, "argv", argv):
                module.main()
            config = json.loads((root / "out" / "config.json").read_text())
            two = config["profiles"]["Sol-H3 two stage — DGX Spark"]
            self.assertEqual((two["precision"], two["two_stage"]["offload"], two["resolutions"]), ("two_stage", "none", ["768p"]))
            self.assertEqual(config["gallery"], str((root / "gallery").resolve()))


class OneStageWithoutFastVaeTest(unittest.TestCase):
    def test_one_stage_uses_original_vae_when_fast_vae_is_not_given(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "MiniMax-H3"
            for component in ("transformer", "vae", "audio_vae"):
                (model / component).mkdir(parents=True)
                (model / component / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
            python = root / "python"
            python.write_text("#!/bin/sh\n")
            python.chmod(0o755)
            manifest = {key: str(root / key) for key in module.REQUIRED}
            manifest.update(stage1_python=str(python), qwen_python=str(python), h3_model=str(model))
            for key, value in manifest.items():
                if key not in ("stage1_python", "qwen_python", "h3_model"):
                    Path(value).mkdir(exist_ok=True)
            (root / "paths.json").write_text(json.dumps(manifest))
            for name in ("lora.safetensors", "hf.safetensors", "table.pt"):
                (root / name).write_bytes(b"x")
            for name in ("deps", "fp8"):
                (root / name).mkdir()
            argv = ["configure.py", "--reuse-paths", str(root / "paths.json"), "--dense-lora", str(root / "lora.safetensors"),
                    "--output-dir", str(root / "out"), "--hyperflow-weights", str(root / "hf.safetensors"),
                    "--hyperflow-deps", str(root / "deps"), "--hyperflow-fp8-dir", str(root / "fp8"),
                    "--hyperflow-adaln-table", str(root / "table.pt")]
            with mock.patch.object(sys, "argv", argv):
                module.main()
            profiles = json.loads((root / "out" / "config.json").read_text())["profiles"]
            self.assertNotIn("fast_vae", profiles["Sol-H3 one stage — RTX 5090"]["hyperflow"])
