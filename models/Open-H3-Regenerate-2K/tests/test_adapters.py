"""CPU checks: merged-LoRA loading, the rank-concatenation identity and the Sharp curve interpolation.

    python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_h3 import adapters  # noqa: E402


class MergedLoraTest(unittest.TestCase):
    def test_rank_concatenation_equals_sum_of_scaled_deltas(self):
        torch.manual_seed(0)
        parts = [(torch.randn(r, 32, dtype=torch.float64), torch.randn(48, r, dtype=torch.float64), s)
                 for r, s in ((12, 0.5), (4, 8 / 128), (2, 1.0))]
        a = torch.cat([a for a, _, _ in parts])
        b = torch.cat([s * b for _, b, s in parts], dim=1)
        expected = sum(s * b @ a for a, b, s in parts)
        torch.testing.assert_close(b @ a, expected)

    def test_load_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.safetensors"
            save_file({"transformer.blocks.0.to_q.lora_A.weight": torch.ones(3, 8),
                       "transformer.blocks.0.to_q.lora_B.weight": torch.ones(5, 3)}, str(path))
            pairs = adapters.load_lora(path)
        self.assertEqual(list(pairs), ["blocks.0.to_q"])
        self.assertEqual(tuple(pairs["blocks.0.to_q"][0].shape), (3, 8))
        self.assertEqual(tuple(pairs["blocks.0.to_q"][1].shape), (5, 3))

    def test_curve_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = torch.arange(1025 * 8, dtype=torch.float32).reshape(1025, 8)
            path = Path(tmp) / "table.bin"
            path.write_bytes(table.numpy().tobytes())
            context = adapters.CurveContext(path)
            context.pre(None, (), {"timestep": torch.tensor([0.0, 0.5, 1.0])})
        torch.testing.assert_close(context.coords, table[[0, 512, 1024]])

    def test_curve_linear_is_identity_at_zero_strength(self):
        torch.manual_seed(0)
        base = torch.nn.Linear(16, 6, bias=True)
        context = type("Ctx", (), {"strength": 0.0, "coords": None})()
        layer = adapters.CurveLinear(base, torch.randn(2, 8), torch.randn(6, 2), context)
        x = torch.randn(3, 16)
        torch.testing.assert_close(layer(x), base(x))
        context.strength, context.coords = 1.0, torch.randn(3, 8)
        expected = base(x) + (context.coords @ layer.curve_a.T) @ layer.curve_b.T
        torch.testing.assert_close(layer(x), expected)


if __name__ == "__main__":
    unittest.main()
