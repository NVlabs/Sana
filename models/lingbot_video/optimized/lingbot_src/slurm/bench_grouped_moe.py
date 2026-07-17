"""Rigorous version: the ACTUAL code path — torch._grouped_mm over all experts on
a rank — for FSDP (128 experts, M tokens each) vs EP (32 experts, 4M each). Same
total tokens processed; the only difference is group count vs per-group size.
Answers whether the grouped kernel already recovers the skinny-GEMM loss FSDP has.
"""
import torch

H, I = 2048, 768
dev = torch.device("cuda", 0)
dt = torch.bfloat16
torch.manual_seed(0)

if not hasattr(torch, "_grouped_mm"):
    print("torch._grouped_mm unavailable"); raise SystemExit(0)


def make_experts(E):
    w1 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
    w3 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
    w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02
    return w1, w3, w2


def grouped_ffn(x, offs, w1, w3, w2):
    h = torch.nn.functional.silu(torch._grouped_mm(x, w1.transpose(-2, -1), offs=offs))
    h = h * torch._grouped_mm(x, w3.transpose(-2, -1), offs=offs)
    return torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs)


def bench_grouped(E, M_per_expert, iters=30, warmup=8):
    total = E * M_per_expert
    x = torch.randn(total, H, device=dev, dtype=dt) * 0.1
    counts = torch.full((E,), M_per_expert, device=dev, dtype=torch.int64)
    offs = torch.cumsum(counts, 0, dtype=torch.int32)
    w1, w3, w2 = make_experts(E)
    for _ in range(warmup):
        grouped_ffn(x, offs, w1, w3, w2)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        grouped_ffn(x, offs, w1, w3, w2)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    ns_per_tok = ms * 1e6 / total
    return ms, ns_per_tok


print(f"GPU: {torch.cuda.get_device_name(0)}  grouped_mm MoE FFN (real path)  H={H} I={I}")
print(f"{'resolution':>16} | {'FSDP 128exp x M':>18} | {'EP 32exp x 4M':>18} | {'EP faster':>10}")
print("-" * 74)
# tokens/expert/rank under CP4: FSDP = tokens*8/128 ; EP = 4x
for name, m_fsdp in [("base 480p", 1500), ("720p", 3560), ("refiner 1080p", 7900)]:
    ms_f, ns_f = bench_grouped(128, m_fsdp)
    ms_e, ns_e = bench_grouped(32, m_fsdp * 4)
    print(f"{name:>16} | {ms_f:7.3f} ms {ns_f:6.2f} ns/t | {ms_e:7.3f} ms {ns_e:6.2f} ns/t | {ns_f/ns_e:9.2f}x")
