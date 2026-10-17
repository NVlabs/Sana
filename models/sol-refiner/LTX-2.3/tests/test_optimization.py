from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock
from sol_refiner import fusion
from sol_refiner.optimization import FusedVideoBlock, RefinerAttention, reference_rotary
from sol_refiner import sol_attention
from sol_refiner.sol_attention import _morton3d_perm, _prep_small, solattn_attention


class OptimizationTests(unittest.TestCase):
    def test_video_block_matches_upstream(self):
        torch.manual_seed(13)
        block = LTX2VideoTransformerBlock(
            dim=32,
            num_attention_heads=2,
            attention_head_dim=16,
            cross_attention_dim=32,
            audio_dim=16,
            audio_num_attention_heads=1,
            audio_attention_head_dim=16,
            audio_cross_attention_dim=16,
            video_gated_attn=True,
            video_cross_attn_adaln=True,
            audio_cross_attn_adaln=True,
            rope_type="split",
        ).eval()
        optimized = deepcopy(block)
        optimized.__class__ = FusedVideoBlock
        engine = SimpleNamespace(layers=1, sparse_shape=False)
        optimized.attn1.set_processor(RefinerAttention(engine, 0, False))
        optimized.attn2.set_processor(RefinerAttention(engine, 0, False))
        args = dict(
            hidden_states=torch.randn(1, 8, 32),
            audio_hidden_states=torch.randn(1, 1, 16),
            encoder_hidden_states=torch.randn(1, 4, 32),
            audio_encoder_hidden_states=torch.randn(1, 4, 16),
            temb=torch.randn(1, 8, 9 * 32),
            temb_audio=torch.randn(1, 1, 9 * 16),
            temb_prompt=torch.randn(1, 1, 2 * 32),
            temb_prompt_audio=torch.randn(1, 1, 2 * 16),
            temb_ca_scale_shift=None,
            temb_ca_audio_scale_shift=None,
            temb_ca_gate=None,
            temb_ca_audio_gate=None,
            use_a2v_cross_attention=False,
            use_v2a_cross_attention=False,
        )
        with torch.no_grad():
            ref = block(**args)[0]
            out = optimized(**args)[0]
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_reference_bf16_rotary_rounding(self):
        # Golden scalar case from the native multiply-then-addcmul evaluation.
        # Computing the entire rotation in FP32 instead produces 1.0703125.
        x = torch.tensor([[[1.234375, 0.3984375]]], dtype=torch.bfloat16)
        cos = torch.tensor([[[[0.95703125]]]], dtype=torch.bfloat16)
        sin = torch.tensor([[[[0.287109375]]]], dtype=torch.bfloat16)
        expected = torch.tensor([[[1.0625, 0.734375]]], dtype=torch.bfloat16)
        torch.testing.assert_close(
            reference_rotary(x, (cos, sin)), expected, rtol=0, atol=0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_rotary_preserves_bf16_rounding(self):
        x = torch.tensor(
            [[[1.234375, 0.3984375]]], device="cuda", dtype=torch.bfloat16
        )
        cos = torch.tensor(
            [[[[0.95703125]]]], device="cuda", dtype=torch.bfloat16
        )
        sin = torch.tensor(
            [[[[0.287109375]]]], device="cuda", dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            fusion.rope_split(x, cos, sin),
            reference_rotary(x, (cos, sin)),
            rtol=0,
            atol=0,
        )

    def test_morton_roundtrip(self):
        perm, inv = _morton3d_perm((3, 8, 8), "cpu")
        x = torch.arange(192)
        torch.testing.assert_close(x[perm][inv], x)
        self.assertEqual(perm.unique().numel(), 192)

    def test_sol_attention_does_not_expose_padded_tokens_to_kernel(self):
        seen = {}

        def fake_kernel(q, k, v, **kwargs):
            seen["shape"] = q.shape
            return q

        q = torch.randn(1, 2, 65, 128, dtype=torch.bfloat16)
        with patch.object(sol_attention, "_kernel", return_value=fake_kernel):
            out = solattn_attention(q, q, q)
        self.assertEqual(seen["shape"], (1, 65, 2, 128))
        torch.testing.assert_close(out, q)

    def test_density_calibration_supports_partial_tail_block(self):
        seen = {}

        def fake_kernel(q, k, v, **kwargs):
            seen["shape"] = q.shape
            return q

        q = torch.randn(1, 1, 65, 128, dtype=torch.bfloat16)
        with patch.object(sol_attention, "_kernel", return_value=fake_kernel):
            out = solattn_attention(q, q, q, target_density=0.5)
        self.assertEqual(seen["shape"], (1, 65, 1, 128))
        torch.testing.assert_close(out, q)

        qc, kc, *_ = _prep_small(q, q, 128**-0.5)
        torch.testing.assert_close(qc[:, :, 0], q[:, :, :64].float().mean(dim=2))
        torch.testing.assert_close(qc[:, :, 1], q[:, :, 64:].float().mean(dim=2))
        torch.testing.assert_close(kc, qc.to(kc.dtype))


if __name__ == "__main__":
    unittest.main()
