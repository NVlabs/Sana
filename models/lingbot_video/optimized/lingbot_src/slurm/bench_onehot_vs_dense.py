"""Why is grouped_mm(one-hot) slow? Compare, for the SAME single-expert FFN over T
tokens:  grouped_mm with 1 non-empty group  vs  a NATIVE dense matmul.
If native dense is much faster, grouped_mm's slowness on one-hot is its grouped-kernel
overhead/config for the degenerate single-group case — NOT a fundamental GEMM property.
"""
import torch
import torch.nn.functional as F

H, I, E = 2048, 768, 128
dev = torch.device("cuda", 0); dt = torch.bfloat16
torch.manual_seed(0)
T = 505920

# grouped weights (E experts) and a single-expert dense weight
w1 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
w3 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02
w1a, w3a, w2a = w1[0], w3[0], w2[0]   # expert 0 only

x = torch.randn(T, H, device=dev, dtype=dt) * 0.1


def timed(fn, iters=40, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return ms, ms * 1e6 / T


def native_dense():
    h = F.silu(x @ w1a.t()) * (x @ w3a.t())
    return h @ w2a.t()

offs_onehot = torch.tensor([T] + [T] * (E - 1), device=dev, dtype=torch.int32)
def grouped_onehot():
    h = F.silu(torch._grouped_mm(x, w1.transpose(-2, -1), offs=offs_onehot))
    h = h * torch._grouped_mm(x, w3.transpose(-2, -1), offs=offs_onehot)
    return torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs_onehot)

per = T // E
offs_uni = torch.cumsum(torch.full((E,), per, device=dev, dtype=torch.int64), 0).to(torch.int32)
xu = torch.randn(per * E, H, device=dev, dtype=dt) * 0.1
def grouped_uniform():
    h = F.silu(torch._grouped_mm(xu, w1.transpose(-2, -1), offs=offs_uni))
    h = h * torch._grouped_mm(xu, w3.transpose(-2, -1), offs=offs_uni)
    return torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs_uni)


print(f"GPU: {torch.cuda.get_device_name(0)}  T={T} single-expert FFN")
print(f"{'method':>22} | {'ms/call':>8} | {'ns/token':>9}")
print("-" * 48)
for name, fn in [("native dense matmul", native_dense),
                 ("grouped_mm one-hot", grouped_onehot),
                 ("grouped_mm uniform(128)", grouped_uniform)]:
    ms, nspt = timed(fn)
    print(f"{name:>22} | {ms:>8.3f} | {nspt:>8.2f}")
