#!/usr/bin/env python3
"""One of H3's three FP8 GEMMs falls back to an sm_89 kernel. Which, and can it be avoided?

The 480p profile shows `_scaled_mm` served by three different cuBLAS kernels, and the middle
one is `sm89_xmma_gemm_e4m3bf16_...` — an Ada kernel on a Blackwell part — at 28.4 ms per
call, slower than the *larger* GEMM next to it at 21.8 ms. Kernel selection is not something
the caller sets directly, but it does depend on the shapes and on the output dtype, so both
are worth sweeping.

Also measures the BF16 `fc2` against an FP16 version of the same GEMM: GB10's cuBLAS reaches
96 TFLOPS in FP16 against 73 in BF16, and `fc2`'s weights come from FP8 (3 mantissa bits), so
they are exactly representable either way.
"""

from __future__ import annotations

import torch

SEQ = 15381  # 832x480 packed rows; the official cell is 38247


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


def fp8_gemm(m, k, n, out_dtype):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s = torch.tensor(0.01, device="cuda")
    fn = lambda: torch._scaled_mm(a, w.t(), scale_a=s, scale_b=s, out_dtype=out_dtype)
    ms = bench(fn)
    return ms, 2 * m * k * n / (ms * 1e9)


print(f"H3's quantised GEMMs at seq={SEQ}\n")
print(f"{'layer':16s} {'shape (m,k,n)':26s} {'out':>8s} {'ms':>8s} {'TFLOPS':>8s}")
shapes = {
    "to_qkv": (SEQ, 5376, 21504),
    "to_out.0": (SEQ, 7168, 5376),
    "ff.net.0.proj": (SEQ, 5376, 28672),
}
for name, (m, k, n) in shapes.items():
    for out_dtype in (torch.bfloat16, torch.float16):
        ms, tflops = fp8_gemm(m, k, n, out_dtype)
        print(f"{name:16s} {str((m, k, n)):26s} {str(out_dtype).split('.')[-1]:>8s} "
              f"{ms:8.2f} {tflops:8.1f}")

# Would splitting the fused QKV back into three avoid the bad kernel?
m, k, n = shapes["to_qkv"]
third = n // 3
ms_split = 0.0
for _ in range(3):
    ms, _ = fp8_gemm(m, k, third, torch.bfloat16)
    ms_split += ms
ms_fused, tflops_fused = fp8_gemm(m, k, n, torch.bfloat16)
print(f"\nto_qkv fused {ms_fused:.2f} ms  vs  three separate {ms_split:.2f} ms")

# fc2 is BF16 by the checkpoint's instruction; FP16 uses a faster cuBLAS path on GB10.
print(f"\n{'fc2 (bf16 weights)':26s} {'ms':>8s} {'TFLOPS':>8s}")
m, k, n = SEQ, 14336, 5376
for dtype in (torch.bfloat16, torch.float16):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype)
    ms = bench(lambda: a @ w.t())
    print(f"{str(dtype).split('.')[-1]:26s} {ms:8.2f} {2 * m * k * n / (ms * 1e9):8.1f}")
