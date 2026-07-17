"""Attention share vs resolution: measure per-step attention time and MoE-FFN time
at the actual per-rank shapes (CP4, batch_cfg) for 480p / 720p / 1080p, so we can
report attention as a fraction of (attention + FFN) at each resolution.

Per rank (CP4, Ulysses gathered for attention, sequence-sharded for FFN):
- attention: block-diagonal over 2 batch_cfg segments of S_sample tokens, 4 heads, D=128
- FFN: per-rank tokens = S_total/4 = S_sample/2, grouped_mm over 128 experts
Both x48 layers = per denoise step.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = torch.device("cuda", 0)
dt = torch.bfloat16
torch.manual_seed(0)
LAYERS = 48
H_MODEL, I_MOE, E, TOPK = 2048, 768, 128, 8
HEADS_PER_RANK, D = 4, 128  # 16 heads / CP4

# S_sample = latent_frames(31) * spatial_tokens (patch 1,2,2 -> /16 each dim)
RES = {
    "480p (480x832)":  31 * (480 // 16) * (832 // 16),    # 48360
    "720p (736x1280)": 31 * (736 // 16) * (1280 // 16),   # 114080
    "1080p(1088x1920)":31 * (1088 // 16) * (1920 // 16),  # 252960
}

CUDNN = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def time_ms(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def attn_one_step(S_sample):
    # 2 batch_cfg segments, each [1, heads, S_sample, D]
    q = torch.randn(1, HEADS_PER_RANK, S_sample, D, device=dev, dtype=dt)
    k = torch.randn_like(q); v = torch.randn_like(q)
    def one_layer():
        with sdpa_kernel(CUDNN, set_priority=True):
            for _ in range(2):  # 2 cfg segments (block-diagonal)
                F.scaled_dot_product_attention(q, k, v)
    return time_ms(one_layer, iters=10) * LAYERS / 1000.0  # s/step


def ffn_one_step(tokens_per_rank):
    w1 = torch.randn(E, I_MOE, H_MODEL, device=dev, dtype=dt) * 0.02
    w3 = torch.randn(E, I_MOE, H_MODEL, device=dev, dtype=dt) * 0.02
    w2 = torch.randn(E, H_MODEL, I_MOE, device=dev, dtype=dt) * 0.02
    M = tokens_per_rank * TOPK          # total expert assignments on this rank
    per = M // E
    x = torch.randn(per * E, H_MODEL, device=dev, dtype=dt) * 0.1
    offs = torch.cumsum(torch.full((E,), per, device=dev, dtype=torch.int64), 0).to(torch.int32)
    def one_layer():
        h = F.silu(torch._grouped_mm(x, w1.transpose(-2, -1), offs=offs))
        h = h * torch._grouped_mm(x, w3.transpose(-2, -1), offs=offs)
        torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs)
    return time_ms(one_layer, iters=10) * LAYERS / 1000.0  # s/step


print(f"GPU: {torch.cuda.get_device_name(0)}  per-step (48 layers), CP4, cuDNN attention")
print(f"{'resolution':>18} | {'S_sample':>9} | {'attn s/step':>11} | {'ffn s/step':>10} | {'attn share':>10}")
print("-" * 78)
for name, S in RES.items():
    tokens_rank = S // 2  # S_total/4 = (2*S)/4 = S/2
    a = attn_one_step(S)
    f = ffn_one_step(tokens_rank)
    share = 100 * a / (a + f)
    print(f"{name:>18} | {S:>9} | {a:>11.3f} | {f:>10.3f} | {share:>9.1f}%")
print()
print("Note: 'attn share' = attn/(attn+ffn); real step also has norms/router/reorder/a2a on top")
print("(~5-15%), which dilutes both. Attention is O(S^2), FFN is O(S) -> attention share")
print("rises with resolution.")
