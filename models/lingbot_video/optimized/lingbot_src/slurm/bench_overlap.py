"""Approach 2: cross-request overlap. Run request-A attention (compute-bound, tensor
cores) concurrently with request-B MoE-FFN (58% memory-bound: the restore scatter)
on two CUDA streams. If total wall time < sequential, the FFN hides behind attention.
"""
import os, sys
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "vectorized")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

dev = torch.device("cuda", 0); dt = torch.bfloat16
torch.manual_seed(0)
CUDNN = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
H = 2048

# 1080p per-rank shapes
S_sample = 252960          # attention block (one cfg segment), 4 heads, D=128
TOK = 126480               # MoE per-rank tokens

block = LingBotVideoSparseMoeBlock(H, 6144, 128, 8, 768, "sigmoid", True, 4, 2, 2.5, 1).to(dev).to(dt)
with torch.no_grad():
    for p in block.parameters(): p.data.normal_(0, 0.02)

qA = torch.randn(1, 4, S_sample, 128, device=dev, dtype=dt)
kA = torch.randn_like(qA); vA = torch.randn_like(qA)
xB = torch.randn(1, TOK, H, device=dev, dtype=dt) * 0.1

def attn():
    with sdpa_kernel(CUDNN, set_priority=True):
        return F.scaled_dot_product_attention(qA, kA, vA)

@torch.no_grad()
def ffn():
    return block(xB)

s1 = torch.cuda.Stream(); s2 = torch.cuda.Stream()

def timed(fn, it=20, wu=8):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it

def seq():
    attn(); ffn()

def overlap():
    torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(s1): oa = attn()
    with torch.cuda.stream(s2): ob = ffn()
    s1.synchronize(); s2.synchronize()

with torch.no_grad():
    ta = timed(attn); tf = timed(ffn)
    ts = timed(seq); to = timed(overlap)

print(f"GPU: {torch.cuda.get_device_name(0)}  1080p per-rank shapes")
print(f"  attention alone      : {ta:7.3f} ms")
print(f"  MoE FFN alone        : {tf:7.3f} ms")
print(f"  SEQUENTIAL (attn+ffn): {ts:7.3f} ms")
print(f"  OVERLAPPED (2 streams): {to:7.3f} ms")
print(f"  -> overlap speedup   : {ts/to:.2f}x   (hidden FFN: {100*(ts-to)/tf:.0f}% of FFN absorbed)")
print(f"  ideal (max of two)   : {max(ta,tf):7.3f} ms  (perfect overlap floor)")
