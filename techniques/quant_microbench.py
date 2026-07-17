#!/usr/bin/env python3
"""Micro-benchmark the full quant+matmul+bias linear at the video-model linear
shapes, comparing bf16 vs FP8(e4m3, torch._scaled_mm) vs NVFP4(TransformerEngine).

Runs on synthetic tensors (no 30B load). Emits JSON with per-shape speedup so we
can pick which shapes are worth swapping to low precision end-to-end.
"""
import json
import os
import sys
import time

import torch

OUT = os.environ.get("QUANT_BENCH_OUT", "quant_microbench_result.json")
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

# (label, K, N, M-list). M spans base/refiner/expert regimes.
DENSE_M = [4096, 16384, 65536, 131072]
EXPERT_M = [2048, 6144, 16384, 32768]
SHAPES = [
    ("self_qkv", 2048, 6144, DENSE_M),
    ("self_out", 2048, 2048, DENSE_M),
    ("cross_kv", 2560, 4096, DENSE_M),
    ("shared_ffn_gate_up", 2048, 12288, DENSE_M),
    ("shared_ffn_down", 6144, 2048, DENSE_M),
    ("routed_expert_gate_up", 2048, 1536, EXPERT_M),
    ("routed_expert_down", 768, 2048, EXPERT_M),
    # Wan14B dense DiT linears (hidden 5120, ffn 13824, heads 40)
    ("wan_qkv", 5120, 15360, DENSE_M),
    ("wan_out", 5120, 5120, DENSE_M),
    ("wan_ffn_up", 5120, 13824, DENSE_M),
    ("wan_ffn_down", 13824, 5120, DENSE_M),
]

ITERS = 30
WARM = 8


def _bench(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / ITERS * 1e3  # ms


def bench_bf16(M, K, N):
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(N, device=DEV, dtype=torch.bfloat16)
    return _bench(lambda: torch.nn.functional.linear(x, w, b))


def bench_fp8(M, K, N):
    """Full linear: dynamic per-row act quant + per-col weight quant + _scaled_mm + bias."""
    if not hasattr(torch, "_scaled_mm"):
        return None
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(N, device=DEV, dtype=torch.bfloat16)
    e4m3 = torch.float8_e4m3fn

    def run():
        xs = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-4) / 448.0
        ws = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-4) / 448.0
        xq = (x / xs).to(e4m3)
        wq = (w / ws).to(e4m3)
        out = torch._scaled_mm(
            xq, wq.t(), scale_a=xs.float(), scale_b=ws.t().float(),
            bias=b, out_dtype=torch.bfloat16,
        )
        return out

    try:
        return _bench(run)
    except Exception as e:
        return {"error": str(e)[:160]}


def bench_nvfp4(M, K, N):
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except Exception as e:
        return {"error": f"TE import failed: {str(e)[:120]}"}
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    try:
        lin = te.Linear(K, N, bias=True, params_dtype=torch.bfloat16).to(DEV)
        recipe = NVFP4BlockScaling()

        def run():
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                return lin(x)

        return _bench(run)
    except Exception as e:
        return {"error": str(e)[:160]}


def main():
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    print(f"_scaled_mm={hasattr(torch,'_scaled_mm')} e4m3={hasattr(torch,'float8_e4m3fn')}", flush=True)
    results = []
    for label, K, N, Ms in SHAPES:
        for M in Ms:
            row = {"label": label, "M": M, "K": K, "N": N}
            try:
                bf16 = bench_bf16(M, K, N)
                row["bf16_ms"] = round(bf16, 4)
                fp8 = bench_fp8(M, K, N)
                if isinstance(fp8, dict):
                    row["fp8"] = fp8
                elif fp8:
                    row["fp8_ms"] = round(fp8, 4)
                    row["fp8_speedup"] = round(bf16 / fp8, 3)
                nvfp4 = bench_nvfp4(M, K, N)
                if isinstance(nvfp4, dict):
                    row["nvfp4"] = nvfp4
                elif nvfp4:
                    row["nvfp4_ms"] = round(nvfp4, 4)
                    row["nvfp4_speedup"] = round(bf16 / nvfp4, 3)
            except Exception as e:
                row["error"] = str(e)[:160]
            sp8 = row.get("fp8_speedup"); sp4 = row.get("nvfp4_speedup")
            print(f"{label:<24} M={M:>7} K={K:>5} N={N:>6}  bf16={row.get('bf16_ms')}ms  "
                  f"fp8x={sp8}  nvfp4x={sp4}", flush=True)
            results.append(row)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
