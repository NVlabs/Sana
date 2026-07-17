"""Compile and validate the Wan-facing SM100 colmask adapter on B200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from integrations.wan import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    shape = (args.batch, args.heads, args.tokens, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    started = time.perf_counter()
    output = run(q, k, v, tau=args.tau, block_size=64)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    finite = bool(torch.isfinite(output).all().item())
    result = {
        "schema": "wan-sol-attn-sm100-colmask-smoke-v1",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shape": list(shape),
        "tau": args.tau,
        "seed": args.seed,
        "elapsed_s_including_compile_and_gate": elapsed_s,
        "output_finite": finite,
        "output_abs_mean": float(output.float().abs().mean().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not finite:
        raise RuntimeError("SM100 colmask output contains non-finite values")


if __name__ == "__main__":
    main()
