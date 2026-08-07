"""Do the fused modulation kernels reproduce the eager expressions bit for bit?

The stride bug is fixed and audio recovered, but the output still moves 4.2%. That is the
signature of a rounding mode again: BF16 keeps 8 mantissa bits, so half an ulp is 0.2%, and
200 modulation sites compound it into the observed few percent. `tl.to` was already shown to
need `fp_downcast_rounding` spelled out for FP8; this checks whether BF16 needs it too.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch

import h3_fp8

torch.manual_seed(0)
S, C, T = 38247, 5376, 9
x = torch.randn(S, C, device="cuda", dtype=torch.bfloat16)
residual = torch.randn(S, C, device="cuda", dtype=torch.bfloat16)
scale = torch.randn(T, C, device="cuda", dtype=torch.bfloat16) * 0.1
shift = torch.randn(T, C, device="cuda", dtype=torch.bfloat16) * 0.1
gate = torch.randn(T, C, device="cuda", dtype=torch.bfloat16) * 0.1
idx = torch.randint(0, T, (S,), device="cuda")

eager_mod = x * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
eager_gate = residual + gate.index_select(0, idx) * x

got_mod = h3_fp8.fused_modulate(x.unsqueeze(0), scale, shift, idx).squeeze(0)
got_gate = h3_fp8.fused_gate_add(residual.unsqueeze(0), gate, x.unsqueeze(0), idx).squeeze(0)

for name, got, want in (("modulate", got_mod, eager_mod), ("gate_add", got_gate, eager_gate)):
    same = torch.equal(got, want)
    differing = (got != want).float().mean()
    err = (got.float() - want.float()).abs()
    print(f"{name:10s} bit-exact={str(same):5s}  differing={differing:7.3%}  "
          f"max|d|={err.max():.3e}  mean|d|/mean={err.mean() / want.float().abs().mean():.3e}")
    if not same:
        i = (got != want).nonzero()[0].tolist()
        print(f"    e.g. eager {want[i[0], i[1]].float():.9f}  fused {got[i[0], i[1]].float():.9f}")
