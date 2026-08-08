"""Do the two quantisers agree once the clamp actually fires?

The per-layer check used `randn` activations, which after dividing by `input_scale` reach
about +-85 — the +-448 clamp never engaged, so that branch went untested. Real activations do
reach it, which is the one regime left that could explain a model-level 3.9%.
"""
import torch
FP8_DTYPE, FP8_MAX = torch.float8_e4m3fn, 448.0

def quantize(x, scale):
    return (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

compiled = torch.compile(quantize, dynamic=False)
scale = torch.tensor(0.0353655144572258, device="cuda")

torch.manual_seed(0)
cases = {
    "no clamping (|x/s| <= 85)":      torch.randn(4096, 5376, device="cuda", dtype=torch.bfloat16),
    "clamping (|x/s| up to ~2800)":   torch.randn(4096, 5376, device="cuda", dtype=torch.bfloat16) * 33,
    "far past the clamp":             torch.randn(4096, 5376, device="cuda", dtype=torch.bfloat16) * 1000,
    "exactly at the boundary":        torch.full((4096, 5376), FP8_MAX, device="cuda", dtype=torch.bfloat16) * scale.bfloat16(),
}
print(f"{'case':34s} {'max |x/s|':>12s} {'clamped %':>10s} {'bit-exact':>10s} {'differing':>10s}")
for name, x in cases.items():
    ratio = (x.float() / scale).abs()
    a, b = quantize(x, scale), compiled(x, scale)
    same = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    diff = (a.view(torch.uint8) != b.view(torch.uint8)).float().mean()
    print(f"{name:34s} {ratio.max():12.1f} {(ratio > FP8_MAX).float().mean():9.2%} "
          f"{str(same):>10s} {diff:9.2%}")
    if not same:
        m = a.view(torch.uint8) != b.view(torch.uint8)
        i = m.nonzero()[0].tolist()
        print(f"    e.g. x/s={ratio[i[0], i[1]]:.3f} -> eager {a[i[0], i[1]].float():.1f}, "
              f"compiled {b[i[0], i[1]].float():.1f}")
