"""Do the fused quantisers reproduce eager's tie-breaking?

E4M3 has 256 codes, so every exact midpoint between adjacent representable values can be
enumerated. Those are precisely the inputs where a rounding rule shows itself, and precisely
what real activations kept landing on.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch

import h3_fp8

FP8, MAX = torch.float8_e4m3fn, 448.0

codes = torch.arange(256, dtype=torch.uint8, device="cuda").view(FP8).float()
finite = codes[torch.isfinite(codes)].sort().values
midpoints = ((finite[:-1] + finite[1:]) / 2)
midpoints = midpoints[midpoints.abs() <= MAX]
# Feed them through as activations at scale 1 so `x/scale` lands exactly on the midpoint.
x = midpoints.to(torch.bfloat16).repeat(64, 1)
x = x[:, (x.to(torch.float32)[0] == midpoints).nonzero().flatten()]  # keep only exact ones
scale = torch.tensor(1.0, device="cuda")

print(f"exact E4M3 midpoints exercised: {x.shape[1]}")
reference = h3_fp8._quantize_eager(x, scale)
for name, fn in (("compiled (inductor)", h3_fp8._quantize_compiled),
                 ("triton (rtne)", h3_fp8._quantize_triton)):
    if fn is None:
        print(f"{name:22s} unavailable")
        continue
    got = fn(x, scale)
    same = torch.equal(got.view(torch.uint8), reference.view(torch.uint8))
    differing = (got.view(torch.uint8) != reference.view(torch.uint8)).float().mean()
    print(f"{name:22s} bit-exact={same}  differing={differing:.2%}")
    if not same:
        i = (got.view(torch.uint8) != reference.view(torch.uint8)).nonzero()[0].tolist()
        print(f"    e.g. x={x[i[0], i[1]].float():.4f} -> eager {reference[i[0], i[1]].float():.3f}, "
              f"{name} {got[i[0], i[1]].float():.3f}")
