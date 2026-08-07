"""Does the fused SwiGLU reproduce `value * silu(gate)` bit for bit?"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch

import h3_fp8

torch.manual_seed(0)
S, D = 15381, 14336
x = torch.randn(1, S, 2 * D, device="cuda", dtype=torch.bfloat16)
value, gate = x.chunk(2, dim=-1)
want = value * torch.nn.functional.silu(gate)
got = h3_fp8.fused_swiglu(x)
same = torch.equal(got, want)
print(f"bit-exact={same}  differing={(got != want).float().mean():.4%}  "
      f"max|d|={(got.float() - want.float()).abs().max():.3e}")
if not same:
    i = (got != want).nonzero()[0].tolist()
    print(f"  gate={gate[tuple(i)].float():.6f} value={value[tuple(i)].float():.6f} -> "
          f"eager {want[tuple(i)].float():.9f}  fused {got[tuple(i)].float():.9f}")
