"""Where exactly does eager round, in `x * (1 + scale) + shift`?

The fused kernel is consistently one BF16 ulp off, so it is rounding a different number of
times or at different points than the three aten ops do. Enumerate the plausible orders and
see which one eager actually is.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch

import h3_fp8

torch.manual_seed(0)
S, C, T = 4096, 5376, 9
x = torch.randn(S, C, device="cuda", dtype=torch.bfloat16)
scale = torch.randn(T, C, device="cuda", dtype=torch.bfloat16) * 0.1
shift = torch.randn(T, C, device="cuda", dtype=torch.bfloat16) * 0.1
idx = torch.randint(0, T, (S,), device="cuda")
sg, hg = scale.index_select(0, idx), shift.index_select(0, idx)

eager = x * (1.0 + sg) + hg
fused = h3_fp8.fused_modulate(x.unsqueeze(0), scale, shift, idx).squeeze(0)

def bf(t):
    return t.bfloat16()

variants = {
    "round after every op (what the kernel does)":
        bf(bf(x.float() * bf(1.0 + sg.float()).float()).float() + hg.float()),
    "no intermediate rounding, one at the end":
        bf(x.float() * (1.0 + sg.float()) + hg.float()),
    "round the (1+scale) only":
        bf(x.float() * bf(1.0 + sg.float()).float() + hg.float()),
    "round the product only":
        bf(bf(x.float() * (1.0 + sg.float())).float() + hg.float()),
}

print(f"{'variant':46s} {'== eager':>9s} {'== fused':>9s}")
for name, v in variants.items():
    print(f"{name:46s} {str(torch.equal(v, eager)):>9s} {str(torch.equal(v, fused)):>9s}")
