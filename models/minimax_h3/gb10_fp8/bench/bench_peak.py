#!/usr/bin/env python3
"""What is GB10's actual dense throughput, and how close to it is H3's attention?

"The default backend is the fastest one installed" is a weaker claim than "there is nothing
left". This measures the machine's own ceiling with large square GEMMs in each precision,
then expresses the attention call as a fraction of it. If attention is already at the BF16
ceiling, no backend swap can help and the only remaining levers change the arithmetic.
"""

from __future__ import annotations

import torch

SEQ_LEN, HEADS, HEAD_DIM = 38247, 56, 128


def bench(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return min(samples)


print(f"{torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}\n")

# --- the ceiling: square GEMMs big enough to be compute bound ---------------------------
print(f"{'dense GEMM ceiling':34s} {'ms':>8s} {'TFLOPS':>9s}")
peak = {}
for label, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    for n in (8192, 16384):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        ms = bench(lambda: a @ b)
        tflops = 2 * n**3 / (ms * 1e9)
        peak[label] = max(peak.get(label, 0), tflops)
        print(f"{label + f' {n}x{n}':34s} {ms:8.2f} {tflops:9.1f}")

for n in (8192, 16384):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    s = torch.tensor(1.0, device="cuda")
    ms = bench(lambda: torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.bfloat16))
    tflops = 2 * n**3 / (ms * 1e9)
    peak["fp8"] = max(peak.get("fp8", 0), tflops)
    print(f"{'fp8 e4m3 ' + f'{n}x{n}':34s} {ms:8.2f} {tflops:9.1f}")

# --- H3's attention against that ceiling ------------------------------------------------
print(f"\n{'H3 packed attention':34s} {'ms':>8s} {'TFLOPS':>9s} {'% of bf16 peak':>15s}")
flops = 4.0 * SEQ_LEN * SEQ_LEN * HEADS * HEAD_DIM
for label, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    q, k, v = (torch.randn(1, HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=dtype)
               for _ in range(3))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), iters=5)
    tflops = flops / (ms * 1e9)
    print(f"{'flash ' + label:34s} {ms:8.1f} {tflops:9.1f} {100 * tflops / peak[label]:14.0f}%")
    del q, k, v
    torch.cuda.empty_cache()

print(f"\nfp8 GEMM ceiling is {peak['fp8'] / peak['bf16']:.2f}x the bf16 one, so an FP8 attention "
      f"kernel is\nworth about that much — but it changes the arithmetic, so it is not lossless.")
