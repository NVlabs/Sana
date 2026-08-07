#!/usr/bin/env python3
"""Which attention backend should H3's packed self-attention use on GB10?

The module breakdown puts ~28 s of the DiT's 48 s in the attention call itself — 2097 TFLOP
of work running at roughly 75 TFLOPS, against the ~250 TFLOPS GB10 reaches in BF16. That gap
is a backend choice, not a law, and it is the largest lossless item in the model.

The shape is H3's, at the published cell: one packed document of 38,247 rows, 56 heads of 128,
no mask (37,296 video rows + a 951-row text/audio prefix fill the sequence exactly, so there
is no padding and every fast path stays available).

Running this standalone rather than through the model keeps the iteration honest and cheap:
the tensors are 1.6 GB, so a sweep costs seconds instead of a 25 s model load per variant.
"""

from __future__ import annotations

import paths

paths.setup(need_sol_engine=False)

import argparse
import sys


import torch

SEQ_LEN = 38247
HEADS = 56
HEAD_DIM = 128


def flops(seq_len: int = SEQ_LEN) -> float:
    """QK^T and AV, both `seq_len x seq_len x inner_dim` with a multiply-add each."""
    return 4.0 * seq_len * seq_len * HEADS * HEAD_DIM


def bench(fn, warmup: int = 2, iters: int = 5) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return min(samples), out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    shape = (1, args.seq_len, HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    total_flops = flops(args.seq_len)
    print(f"packed self-attention  seq={args.seq_len}  heads={HEADS}x{HEAD_DIM}  "
          f"bf16  {total_flops / 1e12:.0f} TFLOP\n")

    from diffusers.models.attention_dispatch import dispatch_attention_fn

    # `dispatch_attention_fn` takes (B, S, H, D); torch SDPA takes (B, H, S, D).
    candidates: dict[str, callable] = {}

    def sdpa(kernel):
        def run():
            with torch.nn.attention.sdpa_kernel(kernel):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                ).transpose(1, 2)
        return run

    backends = torch.nn.attention.SDPBackend
    candidates["sdpa: flash"] = sdpa(backends.FLASH_ATTENTION)
    candidates["sdpa: efficient"] = sdpa(backends.EFFICIENT_ATTENTION)
    candidates["sdpa: cudnn"] = sdpa(backends.CUDNN_ATTENTION)
    candidates["sdpa: auto"] = lambda: torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    for name in ("native", "_native_cudnn", "_native_efficient", "_native_flash",
                 "flash", "flash_varlen", "_flash_3", "sage", "xformers"):
        def run(backend=name):
            return dispatch_attention_fn(q, k, v, backend=backend)
        candidates[f"diffusers: {name}"] = run

    reference = None
    print(f"{'backend':28s} {'ms':>9s} {'TFLOPS':>9s} {'vs best':>9s}  check")
    results = []
    for name, fn in candidates.items():
        try:
            ms, out = bench(fn, iters=args.iters)
        except Exception as error:
            print(f"{name:28s} {'-':>9s} {'-':>9s} {'-':>9s}  {type(error).__name__}: "
                  f"{str(error).splitlines()[0][:60]}")
            continue
        if reference is None:
            reference = out.float()
            check = "reference"
        else:
            diff = (out.float() - reference).abs().max()
            check = f"max|d|={diff:.2e}"
        results.append((name, ms, out))
        print(f"{name:28s} {ms:9.1f} {total_flops / (ms * 1e9):9.1f} {'':>9s}  {check}")

    if results:
        best = min(results, key=lambda r: r[1])
        print(f"\nfastest: {best[0]} at {best[1]:.1f} ms "
              f"({total_flops / (best[1] * 1e9):.0f} TFLOPS)")
        print(f"50 blocks would be {50 * best[1] / 1000:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
