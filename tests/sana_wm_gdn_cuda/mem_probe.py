#!/usr/bin/env python3
"""Peak transient VRAM: Triton cam_scan_bidi_chunkwise vs CUDA run_cuda_cam.

The Triton entry packs qkv into [B,N,3,H,D] (3 big copies) and produces an fp32
[B,N,H,D] num_out before a permute+contiguous -> several large transients. The
CUDA path reads q/k/v [B,H,D,N] directly, reuses scratch buffers, and writes the
[B,H,D,N] fp32 output directly. This measures the difference.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _harness import make_cam_inputs

from diffusion.model.ops.fused_gdn_chunkwise import cam_scan_bidi_chunkwise
from diffusion.model.ops.fused_gdn_chunkwise_cuda import build, run_cuda_cam


def peak_mb(fn, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - base) / 1e6


def main():
    for B, H, D in [(1, 20, 128), (4, 20, 128), (1, 32, 128)]:
        F, S = 11, 920
        dev = torch.device("cuda")
        q, k, v, beta, decay = make_cam_inputs(B, F, S, H, D, dev, 0)
        build()
        inbytes = sum(t.numel() * t.element_size() for t in (q, k, v)) / 1e6
        t_peak = peak_mb(lambda: cam_scan_bidi_chunkwise(q, k, v, beta, decay))
        c_peak = peak_mb(lambda: run_cuda_cam(q, k, v, beta, decay))
        print(f"# B={B} H={H} D={D} N={F*S}  (q+k+v inputs={inbytes:.0f}MB, " f"output={B*H*D*F*S*4/1e6:.0f}MB)")
        print(f"#   Triton cam transient peak: {t_peak:8.1f} MB")
        print(
            f"#   CUDA   cam transient peak: {c_peak:8.1f} MB   "
            f"(saves {t_peak - c_peak:.1f} MB, {t_peak/max(c_peak,1e-6):.2f}x less)"
        )


if __name__ == "__main__":
    main()
