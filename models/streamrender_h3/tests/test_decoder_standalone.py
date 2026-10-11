"""CPU equivalence checks for the teacher-free decoder constructor."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'render/vendor'), str(ROOT / 'render/decoder/native')]
from causal_vit_decoder import CausalViTDecoder
from dev.yanzuolu.projects.minimax_h3.modeling.video_vae.vae_vit import ViT3DDecoder


class StandaloneDecoderTests(unittest.TestCase):
    def test_checkpoint_and_streaming_output_equivalence(self):
        torch.manual_seed(7)
        kwargs = dict(patch_size=16, patch_size_t=4, t_causal=False,
                      in_channels=24, out_channels=3, num_layers=2,
                      heads=2, dim_head=12, norm_type='rms_norm', norm_affine=True,
                      qk_norm_type='rms_norm', qk_norm_affine=False,
                      ffn_activation_fn='silu', ffn_use_gated=True,
                      rope_dim_ratio=1.0, rope_theta=100)
        source = SimpleNamespace(decoder=ViT3DDecoder(**kwargs),
                                 post_quant_conv=torch.nn.Conv3d(24, 24, 1),
                                 latents_mean=torch.zeros(24), latents_std=torch.ones(24))
        legacy = CausalViTDecoder(source, [0, 1], window=2)
        config = dict(decoder_kwargs=kwargs, embed_dim=24,
                      latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
        standalone = CausalViTDecoder.from_config(config, window=2)
        standalone.load_state_dict(legacy.state_dict(), strict=True, assign=True)
        self.assertFalse(any(t.is_meta for t in list(standalone.parameters()) + list(standalone.buffers())))
        left, right = legacy.init_state(), standalone.init_state()
        with torch.no_grad():
            for _ in range(3):
                z = torch.randn(1, 2, 24, 2, 2)
                a, b = legacy.decode_chunk(z, left), standalone.decode_chunk(z, right)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertEqual(left['t'], right['t'])
                for ka, kb in zip(left['k'], right['k']):
                    torch.testing.assert_close(ka, kb, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
