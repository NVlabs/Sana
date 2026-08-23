#!/usr/bin/env python3
"""CPU-only contract tests for the fixed MiniMax-H3 super-acceleration arm."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "stage1"
STAGE2 = ROOT / "stage2"


def _load_sol_attention():
    spec = importlib.util.spec_from_file_location(
        "h3_super_sol_attention", STAGE2 / "sol_attention.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Stage-2 Sol policy")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _AttentionWrapper:
    def __init__(self) -> None:
        self.original_attention = lambda *args, **kwargs: (args, kwargs)


class _Block:
    def __init__(self) -> None:
        self.attn1 = types.SimpleNamespace(attention_function=_AttentionWrapper())


class SuperAccelerationContractTest(unittest.TestCase):
    def test_source_snapshot_hashes_match_current_tree(self) -> None:
        snapshot = json.loads((ROOT / "SOURCE_SNAPSHOT.json").read_text())
        for relative, expected in snapshot["source_files_sha256"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_stage2_sol_policy_patches_exactly_layers_1_through_47(self) -> None:
        module = _load_sol_attention()
        blocks = [_Block() for _ in range(48)]
        transformer = types.SimpleNamespace(transformer_blocks=blocks)
        originals = [
            block.attn1.attention_function.original_attention for block in blocks
        ]

        module.Stage2SolAttention(transformer, isolate_sol_from_compile=False)

        self.assertEqual(module.STAGE2_TAUS, (1.0, 1.25, 1.5))
        self.assertEqual(module.STAGE2_TRANSFORMER_LAYERS, 48)
        self.assertIs(
            blocks[0].attn1.attention_function.original_attention, originals[0]
        )
        for index in range(1, 48):
            self.assertIsNot(
                blocks[index].attn1.attention_function.original_attention,
                originals[index],
            )

    def test_stage2_policy_rejects_non_48_layer_transformer(self) -> None:
        module = _load_sol_attention()
        transformer = types.SimpleNamespace(
            transformer_blocks=[_Block() for _ in range(47)]
        )
        with self.assertRaisesRegex(ValueError, "exactly 48"):
            module.Stage2SolAttention(transformer)

    def test_fixed_stage2_profile_is_present(self) -> None:
        base = (STAGE2 / "refiner_encoder_ablation_single_gpu.py").read_text()
        server = (STAGE2 / "stage2_server.py").read_text()
        self.assertIn("WIDTH = 1344", base)
        self.assertIn("HEIGHT = 768", base)
        self.assertIn("SOURCE_WIDTH = 896", base)
        self.assertIn("SOURCE_HEIGHT = 512", base)
        self.assertIn(
            "STAGE2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)", base
        )
        self.assertIn('choices=("default", "full")', server)
        self.assertIn('"updates": 3', server)
        self.assertIn('"layer0_dense_layers1_47_sol_strict"', server)

    def test_first_frame_identity_is_end_to_end_and_pinned(self) -> None:
        stage1 = (STAGE1 / "stage1_producer.py").read_text()
        stage2 = (STAGE2 / "stage2_server.py").read_text()
        compat = (
            STAGE2 / "official_compat_h3_refiner_diagnostic.py"
        ).read_text()
        manifest = json.loads((ROOT / "assets/refiner_manifest.json").read_text())
        record = manifest[0]
        expected = (
            "0f41282b5101d1be9ef51ee2f0bb13d2c599f0a7139b7406d6534b678387f491"
        )
        self.assertEqual(record["first_frame_sha256"], expected)
        self.assertFalse(Path(record["first_frame"]).is_absolute())
        self.assertIn('"first_frame_sha256": first_frame_sha256', stage1)
        self.assertIn('"first_frame_sha256",', stage2)
        self.assertIn("first-frame SHA-256 mismatch", compat)

    def test_launcher_is_standalone_and_uses_two_independent_pairs(self) -> None:
        launcher = (ROOT / "run_gb200.sh").read_text()
        forbidden = (
            "/home/",
            "/lustre/",
            "agent_deploy",
            "minimax-h3-resolution-handoff",
            "E2E_EXPERIMENTS",
        )
        for value in forbidden:
            self.assertNotIn(value, launcher)
        self.assertIn("H3_SUPER_RUNTIME_ROOT", launcher)
        self.assertEqual(launcher.count("--ntasks=2"), 2)
        self.assertEqual(launcher.count("--gpus-per-task=1"), 2)
        self.assertIn("direct_tensor", launcher)
        self.assertIn("mp4", launcher)
        self.assertIn(
            'temporal_tile=${H3_SUPER_INPUT_VAE_TEMPORAL_TILE:-full}', launcher
        )
        self.assertIn(
            "docker://lmsysorg/sglang@sha256:"
            "71145ca99ebc458265e93cebd00b52bb9f419f052e7d0de09a54fa0f72fed888",
            launcher,
        )
        self.assertNotIn(
            "lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86", launcher
        )

    def test_no_personal_absolute_paths_in_executable_sources(self) -> None:
        for path in (
            ROOT / "run_gb200.sh",
            ROOT / "stage1/run_worker.sh",
            ROOT / "stage2/run_worker.sh",
            ROOT / "stage1/stage1_producer.py",
            ROOT / "stage2/stage2_server.py",
        ):
            text = path.read_text()
            self.assertNotIn("/home/", text, str(path))
            self.assertNotIn("/lustre/", text, str(path))


if __name__ == "__main__":
    unittest.main()
