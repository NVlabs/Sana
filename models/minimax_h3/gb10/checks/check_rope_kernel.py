"""Does the fused rotary kernel reproduce `_apply_rotary_emb` bit for bit?"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch


from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
import h3_fp8

torch.manual_seed(0)
S, H, D, ROT = 38247, 56, 128, 96
x = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
cos = torch.randn(S, ROT, device="cuda", dtype=torch.float32).cos()
sin = torch.randn(S, ROT, device="cuda", dtype=torch.float32).sin()

want = _apply_rotary_emb(x, cos, sin)
got = h3_fp8.fused_apply_rotary_emb(x, cos, sin)
same = torch.equal(got, want)
err = (got.float() - want.float()).abs()
print(f"bit-exact={same}  differing={(got != want).float().mean():.4%}  max|d|={err.max():.3e}")
if not same:
    i = (got != want).nonzero()[0].tolist()
    print(f"  e.g. idx {i}: eager {want[tuple(i)].float():.9f}  fused {got[tuple(i)].float():.9f}")
