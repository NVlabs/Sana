#!/usr/bin/env python3

import ast
import hashlib
import tomllib
import unittest
from pathlib import Path


GB200 = Path(__file__).resolve().parents[1]
MODEL_ROOT = GB200.parent


def literal_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                result[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


class RefinerContractTest(unittest.TestCase):
    def test_runner_fixed_workload(self) -> None:
        values = literal_constants(GB200 / "refiner_head_cp.py")
        self.assertEqual(values["WIDTH"], 1920)
        self.assertEqual(values["HEIGHT"], 1088)
        self.assertEqual(values["FRAME_COUNT"], 241)
        self.assertEqual(values["FPS"], 24.0)
        self.assertEqual(values["WORLD_SIZE"], 4)
        self.assertEqual(values["STAGE2_SIGMAS"], (0.909375, 0.725, 0.421875, 0.0))
        self.assertEqual(values["LORA_STRENGTH"], 0.8)
        self.assertEqual(values["EXPECTED_VIDEO_TOKENS"], 63240)

    def test_launch_config_matches_head_context_parallel_contract(self) -> None:
        config = tomllib.loads((GB200 / "refiner.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["gpus"], 4)
        self.assertEqual(config["LTX25_REFINER_PARALLELISM"], "head_context")
        self.assertEqual(config["LTX25_REFINER_PARAMETER_REPLICATION"], "full")
        self.assertEqual(config["LTX25_REFINER_SELF_ATTN_HEAD_SHARDS"], "4")
        self.assertEqual(config["LTX25_REFINER_SELF_ATTN_TOKEN_SCOPE"], "full_sequence")
        self.assertEqual(config["LTX25_REFINER_CACHE"], "0")
        self.assertEqual(config["LTX25_REFINER_COMPILE"], "0")
        self.assertEqual(config["LTX25_REFINER_OFFLOAD"], "0")
        self.assertEqual(config["LTX25_REFINER_QUANTIZATION"], "none")

    def test_compile_arm_changes_only_compile_and_run_identity(self) -> None:
        eager = tomllib.loads((GB200 / "refiner.toml").read_text(encoding="utf-8"))
        compiled = tomllib.loads(
            (GB200 / "refiner_compile.toml").read_text(encoding="utf-8")
        )
        compile_only = {
            "name",
            "LTX25_REFINER_COMPILE",
            "LTX25_REFINER_COMPILE_MODE",
            "LTX25_REFINER_COMPILE_FULLGRAPH",
            "LTX25_REFINER_COMPILE_CAPTURE",
            "LTX25_REFINER_COMPILE_CACHE_ROOT",
        }
        self.assertEqual(
            {key: value for key, value in eager.items() if key not in compile_only},
            {key: value for key, value in compiled.items() if key not in compile_only},
        )
        self.assertEqual(compiled["gpus"], 4)
        self.assertEqual(compiled["LTX25_REFINER_COMPILE"], "1")
        self.assertEqual(
            compiled["LTX25_REFINER_COMPILE_MODE"],
            "max-autotune-no-cudagraphs",
        )
        self.assertEqual(compiled["LTX25_REFINER_COMPILE_FULLGRAPH"], "0")
        self.assertEqual(compiled["LTX25_REFINER_COMPILE_CAPTURE"], "0")
        cache_root = Path(compiled["LTX25_REFINER_COMPILE_CACHE_ROOT"])
        self.assertTrue(cache_root.is_absolute())
        self.assertTrue(
            str(cache_root).startswith("/home/yitongl/code/.cache/sol-engine/")
        )

    def test_launcher_uses_four_distributed_workers_and_shared_venv(self) -> None:
        launcher = (GB200 / "run_refiner_gb200.sh").read_text(encoding="utf-8")
        self.assertIn(
            'readonly PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"',
            launcher,
        )
        self.assertIn("--nproc_per_node=4", launcher)
        self.assertIn("torch.distributed.run", launcher)
        self.assertNotIn("TiledDataParallel", launcher)

    def test_all_compiler_caches_are_persistent_and_exported_pre_import(self) -> None:
        launcher = (GB200 / "run_refiner_gb200.sh").read_text(encoding="utf-8")
        for variable, leaf in (
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("TRITON_CACHE_DIR", "triton"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
        ):
            self.assertIn(
                f'export {variable}="$LTX25_REFINER_COMPILE_CACHE_ROOT/{leaf}"',
                launcher,
            )
        self.assertNotIn("/tmp", launcher)
        self.assertLess(
            launcher.index("export CUTE_DSL_CACHE_DIR="),
            launcher.index('exec "$PYTHON_BIN" -m torch.distributed.run'),
        )

    def test_split_vae_latent_statistics_key_is_supported(self) -> None:
        runner = (GB200 / "refiner_head_cp.py").read_text(encoding="utf-8")
        self.assertIn('"per_channel_statistics."', runner)
        self.assertIn('"vae.per_channel_statistics."', runner)

    def test_vendored_taehv_matches_pinned_source(self) -> None:
        vendor = GB200 / "vendor" / "taehv"
        expected = {
            "taehv.py": "607c2a578bc2684e6cd21e96f8c1d024b1c32912e6f58c724f14131a3d4a2773",
            "LICENSE": "532f9e394518ffddecd294a517d5b41d79d3d3866c3fb95a6cb0e8bcc02370bf",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((vendor / name).read_bytes()).hexdigest(), digest)

    def test_documentation_does_not_advertise_tdp(self) -> None:
        readme = (MODEL_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("TDP", readme)
        self.assertIn("head/context parallel", readme)


if __name__ == "__main__":
    unittest.main()
