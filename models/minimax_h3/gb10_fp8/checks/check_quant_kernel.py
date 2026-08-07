"""Why does the compiled activation quantiser disagree with the eager one?

Fusing `(x.float() / scale).clamp_(-448, 448).to(float8_e4m3fn)` into one inductor kernel cut
4.0 s off the forward but moved the output by 3.9% mean relative error — far too much for a
reassociation, and suspiciously close to half an E4M3 ulp (3 mantissa bits, so a 12.5%
relative step). That signature points at the FP8 cast's rounding mode rather than at the
arithmetic, which this checks directly on activation-shaped data.
"""

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize(x, scale):
    return (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


torch.manual_seed(0)
# H3's largest quantised activation: the block-stack input, 38,247 rows of 5,376.
x = torch.randn(38247, 5376, device="cuda", dtype=torch.bfloat16)
scale = torch.tensor(0.0353655144572258, device="cuda")  # blocks.0.attn.qkv_proj.input_scale

eager = quantize(x, scale)

variants = {}
variants["compiled (default)"] = torch.compile(quantize, dynamic=False)(x, scale)

torch._inductor.config.emulate_precision_casts = True
torch._dynamo.reset()
variants["compiled (emulate_precision_casts)"] = torch.compile(quantize, dynamic=False)(x, scale)

print(f"{'variant':40s} {'bit-exact':>10s} {'differing':>12s} {'max |d| (fp8 codes)':>21s}")
for name, got in variants.items():
    same = torch.equal(got.view(torch.uint8), eager.view(torch.uint8))
    differing = (got.view(torch.uint8) != eager.view(torch.uint8)).float().mean()
    delta = (got.float() - eager.float()).abs().max()
    print(f"{name:40s} {str(same):>10s} {differing:11.2%} {delta:21.4f}")

# Is the eager result itself the correctly rounded one? Compare both against a float64
# round-to-nearest reference computed the long way.
exact = (x.double() / scale.double()).clamp(-FP8_MAX, FP8_MAX)
for name, got in [("eager", eager), *variants.items()]:
    err = (got.double() - exact).abs()
    # An E4M3 value's own spacing, to express the error in ulps.
    ulp = torch.where(got.double().abs() > 0, got.double().abs() * 2**-3, torch.tensor(2.0**-9, device="cuda"))
    print(f"{name:40s} mean {(err / ulp).mean():.4f} ulp, max {(err / ulp).max():.4f} ulp")

# --- does it stay correct across the 200 different scales and two widths the model uses? ---
print()
compiled = torch.compile(quantize, dynamic=False)
torch.manual_seed(1)
bad = 0
for i in range(8):
    width = 5376 if i % 2 == 0 else 7168
    xs = torch.randn(4096, width, device="cuda", dtype=torch.bfloat16)
    s = torch.tensor(0.005 * (i + 1), device="cuda")
    if not torch.equal(compiled(xs, s).view(torch.uint8), quantize(xs, s).view(torch.uint8)):
        bad += 1
        print(f"  MISMATCH at call {i}: width={width} scale={s.item():.4f}")
print(f"interleaved shapes/scales: {8 - bad}/8 bit-exact")
