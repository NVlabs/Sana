#!/usr/bin/env python3
"""Is Sol-Attn's Triton reference usable on GB10, and what does it cost in quality?

Sol-Engine's released CuTe kernels are compiled per architecture and GB10's SM121 has none, so
`_backend_for_arch` routes it to the Triton reference. That is architecture-independent and
runs here through the public entry point — this measures whether it is worth running.

Both questions have to be asked on *real* q/k/v. Sparse attention exploits structure in the
activations, so random tensors understate the quality (cos 0.695 on `randn`) and misstate the
speed. These are captured from the model mid-trajectory, after `norm_q`/`norm_k` and rotary,
which is exactly what `dispatch_attention_fn` receives.

One caveat on what the reference is: it is a readability reference rather than a tuned kernel.
The 951-row exact KV sink the released policy specifies is available through `sink_tokens` and
is measured separately in `check_sol_policy.py`; this run is the bare threshold.
"""

from __future__ import annotations

import paths

paths.setup(need_sol_engine=True)

import argparse
import sys


import torch


def released_sol_attn():
    """The released entry point, called as-is.

    This function used to patch `interface._validate` to get past an SM90/SM100 gate. The gate
    is gone: `_backend_for_arch` now picks a CuTe backend only when the capability matches
    exactly and returns "triton" for everything else at SM8.0 or above, so GB10's (12, 1)
    arrives at the Triton path with nothing bypassed.
    """
    import sol_attn

    return sol_attn.sol_attn


def capture_qkv(blocks: list[int], cell: str, step: int) -> dict:
    """Record what `dispatch_attention_fn` is handed, for the requested block indices."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    from bench_dit import load_inputs, load_transformer

    model, _ = load_transformer(fuse_qkv=True, quantizer="triton", fuse_adaln=True,
                                fuse_rope=True, fuse_swiglu=True)
    inputs = load_inputs(step, cell=cell)["tensors"]

    original = h3.dispatch_attention_fn
    seen = {"index": 0}
    captured = {}

    def wrapper(query, key, value, **kwargs):
        index = seen["index"]
        seen["index"] += 1
        # The two token-refiner blocks run first and are a different (tiny) shape.
        if index - 2 in blocks:
            captured[index - 2] = tuple(t.detach().clone() for t in (query, key, value))
        return original(query, key, value, **kwargs)

    h3.dispatch_attention_fn = wrapper
    with torch.no_grad():
        model(**inputs, return_dict=False)
    h3.dispatch_attention_fn = original
    del model
    torch.cuda.empty_cache()
    return captured


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="832x480")
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 10, 25, 49])
    parser.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    args = parser.parse_args()

    sol_attn = released_sol_attn()
    captured = capture_qkv(args.blocks, args.cell, args.step)
    print(f"captured q/k/v from blocks {sorted(captured)} at {args.cell} step {args.step}\n")

    sdpa = torch.nn.functional.scaled_dot_product_attention
    flash = torch.nn.attention.SDPBackend.FLASH_ATTENTION

    print(f"{'block':>6s} {'variant':>14s} {'ms':>8s} {'vs dense':>9s} {'cos':>10s} {'max|d|':>10s}")
    for index in sorted(captured):
        q, k, v = captured[index]
        with torch.nn.attention.sdpa_kernel(flash):
            dense_fn = lambda: sdpa(q.transpose(1, 2), k.transpose(1, 2),
                                    v.transpose(1, 2)).transpose(1, 2)
            dense_ms = bench(dense_fn)
            reference = dense_fn().float()
        print(f"{index:6d} {'dense flash':>14s} {dense_ms:8.2f} {'1.000x':>9s} {'':>10s} {'':>10s}")

        for tau in args.taus:
            try:
                out = sol_attn(q, k, v, tau=tau)
                ms = bench(lambda: sol_attn(q, k, v, tau=tau))
            except Exception as error:
                print(f"{index:6d} {f'sol tau={tau}':>14s} {'-':>8s} {'-':>9s}  "
                      f"{type(error).__name__}: {str(error)[:50]}")
                continue
            got = out.float()
            cos = torch.nn.functional.cosine_similarity(
                got.flatten(), reference.flatten(), dim=0
            )
            print(f"{index:6d} {f'sol tau={tau}':>14s} {ms:8.2f} {dense_ms / ms:8.3f}x "
                  f"{cos:10.6f} {(got - reference).abs().max():10.3e}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
