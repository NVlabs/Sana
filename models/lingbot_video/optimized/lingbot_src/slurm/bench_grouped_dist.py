"""Does grouped_mm time depend ONLY on total tokens, or also on how tokens are
distributed across experts? Same total T, different distributions across E=128
experts. Measures ns/token for each. Run at an under-saturated total (~96K) and a
saturated total (~506K).
"""
import torch
import torch.nn.functional as F

H, I, E = 2048, 768, 128
dev = torch.device("cuda", 0); dt = torch.bfloat16
g = torch.Generator(device="cpu").manual_seed(0)
if not hasattr(torch, "_grouped_mm"):
    print("no _grouped_mm"); raise SystemExit

w1 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
w3 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02


def counts_uniform(T):
    c = torch.full((E,), T // E, dtype=torch.int64); c[: T % E] += 1; return c

def counts_one_hot(T):
    c = torch.zeros(E, dtype=torch.int64); c[0] = T; return c

def counts_k_hot(T, k):
    c = torch.zeros(E, dtype=torch.int64); per = T // k
    c[:k] = per; c[0] += T - per * k; return c

def counts_zipf(T):
    r = torch.arange(1, E + 1, dtype=torch.float64)
    w = 1.0 / r; w /= w.sum()
    c = torch.floor(w * T).to(torch.int64); c[0] += T - int(c.sum()); return c

def counts_random(T):
    # messy: multinomial scatter of T tokens into E bins (very uneven, some zero)
    probs = torch.rand(E, generator=g, dtype=torch.float64); probs /= probs.sum()
    c = torch.multinomial(probs, T, replacement=True, generator=g).bincount(minlength=E).to(torch.int64)
    return c


def bench(counts, iters=40, warmup=10):
    counts = counts.to(dev)
    T = int(counts.sum())
    offs = torch.cumsum(counts, 0, dtype=torch.int32)
    x = torch.randn(T, H, device=dev, dtype=dt) * 0.1
    def one():
        h = F.silu(torch._grouped_mm(x, w1.transpose(-2, -1), offs=offs))
        h = h * torch._grouped_mm(x, w3.transpose(-2, -1), offs=offs)
        torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs)
    for _ in range(warmup): one()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): one()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    nonzero = int((counts > 0).sum())
    return ms, ms * 1e6 / T, nonzero


print(f"GPU: {torch.cuda.get_device_name(0)}  grouped_mm 128 experts — same total T, diff distributions")
for T in (96768, 505920):
    print(f"\n===== total tokens T = {T} ({'under-saturated' if T < 2e5 else 'saturated'}) =====")
    print(f"{'distribution':>14} | {'nonzero exp':>11} | {'ms/call':>8} | {'ns/token':>9}")
    print("-" * 54)
    dists = [
        ("uniform", counts_uniform(T)),
        ("one_hot(1)", counts_one_hot(T)),
        ("8_hot", counts_k_hot(T, 8)),
        ("zipf(skew)", counts_zipf(T)),
        ("random_messy", counts_random(T)),
    ]
    base = None
    for name, c in dists:
        ms, nspt, nz = bench(c)
        if base is None: base = nspt
        print(f"{name:>14} | {nz:>11} | {ms:>8.3f} | {nspt:>8.2f}  ({nspt/base:.2f}x vs uniform)")
