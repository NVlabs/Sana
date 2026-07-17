"""Measure a single MoE expert's SwiGLU FFN throughput vs M (tokens routed to it).

Proves the GEMM-saturation claim: below some M* the per-token cost is high
(memory-bound / skinny GEMM); above M* it flattens (compute-bound). FSDP keeps
each expert at M_fsdp = tokens/expert; EP concentrates to 4x that. Whether EP's
compute is better than FSDP depends purely on where M_fsdp sits on this curve.

Expert shapes (from the model config): hidden H=2048, moe_intermediate I=768.
FFN: h = silu(x@w1^T) * (x@w3^T); out = h@w2^T.  w1,w3:[I,H]  w2:[H,I]
"""
import torch

H, I = 2048, 768
dev = torch.device("cuda", 0)
dt = torch.bfloat16
torch.manual_seed(0)

w1 = torch.randn(I, H, device=dev, dtype=dt) * 0.02
w3 = torch.randn(I, H, device=dev, dtype=dt) * 0.02
w2 = torch.randn(H, I, device=dev, dtype=dt) * 0.02

flops_per_token = 3 * 2 * H * I  # w1 + w3 + w2, each 2*H*I MACs*2


def ffn(x):
    h = torch.nn.functional.silu(x @ w1.t()) * (x @ w3.t())
    return h @ w2.t()


def bench(M, iters=50, warmup=10):
    x = torch.randn(M, H, device=dev, dtype=dt) * 0.1
    for _ in range(warmup):
        ffn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ffn(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    tflops = (M * flops_per_token) / (ms * 1e-3) / 1e12
    ns_per_tok = ms * 1e6 / M
    return ms, tflops, ns_per_tok


print(f"GPU: {torch.cuda.get_device_name(0)}  bf16 expert FFN  H={H} I={I}")
print(f"{'M (tokens/expert)':>18} | {'ms/call':>9} | {'TFLOP/s':>9} | {'ns/token':>9} | {'vs peak':>8}")
print("-" * 72)
Ms = [64, 128, 256, 512, 768, 1024, 1500, 2048, 3072, 3560, 4096, 6000, 7900, 12288, 16384, 24576, 31500]
peak = 0.0
rows = []
for M in Ms:
    ms, tf, nspt = bench(M)
    peak = max(peak, tf)
    rows.append((M, ms, tf, nspt))
for M, ms, tf, nspt in rows:
    print(f"{M:>18} | {ms:>9.3f} | {tf:>9.1f} | {nspt:>9.3f} | {100*tf/peak:>7.1f}%")

print()
print("Interpretation anchors (tokens/expert/rank, CP4):")
for name, m_fsdp in [("base 480p", 1500), ("720p", 3560), ("refiner 1080p", 7900)]:
    m_ep = m_fsdp * 4
    _, tf_f, ns_f = bench(m_fsdp)
    _, tf_e, ns_e = bench(m_ep)
    # same total tokens processed either way; efficiency ratio = ns/token FSDP vs EP
    speedup = ns_f / ns_e
    print(f"  {name:>16}: FSDP M={m_fsdp} ({100*tf_f/peak:.0f}% peak, {ns_f:.3f} ns/tok) | "
          f"EP M={m_ep} ({100*tf_e/peak:.0f}% peak, {ns_e:.3f} ns/tok) | "
          f"EP compute {speedup:.2f}x {'faster' if speedup>1.02 else '≈ same'}")
