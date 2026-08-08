"""Does the memory layout handed to flash attention matter at H3's shape?

Two earlier measurements of the same call differed by 11%, and the only difference was how
q/k/v were laid out: allocated as (B, S, H, D) and transposed into (B, H, S, D), versus
allocated as (B, H, S, D) directly. If that is real it is free, and it decides whether the
model is already on the good path.
"""
import torch

S, H, D = 38247, 56, 128
FLOPS = 4.0 * S * S * H * D
sdpa = torch.nn.functional.scaled_dot_product_attention
backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION

def bench(fn, warmup=3, iters=8):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(iters):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        xs.append(a.elapsed_time(b))
    return min(xs), sum(xs) / len(xs)

print(f"{'layout of q/k/v passed to SDPA':44s} {'min ms':>8s} {'mean ms':>8s} {'TFLOPS':>8s}")

# (B, S, H, D) allocated, transposed to (B, H, S, D) -- strided, and what the model produces.
qkv_bshd = [torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
strided = [t.transpose(1, 2) for t in qkv_bshd]
with torch.nn.attention.sdpa_kernel(backend):
    mn, mean = bench(lambda: sdpa(*strided))
print(f"{'(B,S,H,D) allocated, .transpose(1,2) [model path]':44s} {mn:8.1f} {mean:8.1f} {FLOPS/(mn*1e9):8.1f}")

# The same values, made contiguous in (B, H, S, D).
contig = [t.contiguous() for t in strided]
with torch.nn.attention.sdpa_kernel(backend):
    mn, mean = bench(lambda: sdpa(*contig))
print(f"{'(B,H,S,D) contiguous':44s} {mn:8.1f} {mean:8.1f} {FLOPS/(mn*1e9):8.1f}")

ref = sdpa(*strided)
got = sdpa(*contig)
print(f"\nsame result either way: max|d|={(ref-got).abs().max():.2e}")
