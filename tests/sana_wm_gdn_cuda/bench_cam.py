#!/usr/bin/env python3
"""Benchmark the LIVE model chunkwise entry `cam_scan_bidi_chunkwise`
(sana_gdn_blocks_triton.py:404). num-only, identity norm/rope, skip_relu,
output transposed to [B,H,D,N]. Profiles the Triton baseline + the final
permute/contiguous cost (the analogue of the divide bottleneck in the full path).
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

torch.backends.cuda.matmul.allow_tf32 = False
from _harness import make_cam_inputs, time_ms

from diffusion.model.ops.fused_gdn_chunkwise import (
    _cam_identity_tables,
    cam_scan_bidi_chunkwise,
    phase_a,
    phase_b_triton,
    phase_c,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--F", type=int, default=11)
    ap.add_argument("--S", type=int, default=920)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    print(f"# cam path  device sm={cap[0]}{cap[1]}  B={args.B} F={args.F} S={args.S} H={args.H} D={args.D}")
    q, k, v, beta, decay = make_cam_inputs(args.B, args.F, args.S, args.H, args.D, dev)
    out = cam_scan_bidi_chunkwise(q, k, v, beta, decay)
    assert torch.isfinite(out).all()
    print(f"# out shape={tuple(out.shape)} dtype={out.dtype} absmean={out.abs().mean():.3e}")
    t = time_ms(lambda: cam_scan_bidi_chunkwise(q, k, v, beta, decay), args.warmup, args.iters)
    print(f"# TRITON cam end2end={t:.4f} ms")

    # profile: phase A/B/C + transpose
    B, H, D, N = q.shape
    F = beta.shape[2]
    S = N // F
    qkv = torch.empty(B, N, 3, H, D, device=dev, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))
    oir, onw, ocos, osin = _cam_identity_tables(B=B, N=N, H=H, D=D, device=dev)

    def pa():
        return phase_a(
            qkv,
            beta,
            oir,
            oir,
            onw,
            onw,
            ocos,
            osin,
            F=F,
            S=S,
            k_scale=1.0,
            dot_precision=0,
            skip_relu=True,
            skip_z=True,
        )

    Ip, A, Iz, Bz = pa()

    def pb():
        return phase_b_triton(
            Ip, A, Iz, Bz, decay, F=F, dot_precision=0, direction=0, combined_history=True, skip_z=True
        )

    Mh, zh, _, _ = pb()

    def pc():
        return phase_c(qkv, oir, onw, ocos, osin, Mh, zh, F=F, S=S, dot_precision=0, skip_relu=True, num_only=True)

    num, _ = pc()

    def transpose():
        return num.permute(0, 2, 3, 1).contiguous().to(torch.float32)

    ta = time_ms(pa, args.warmup, args.iters)
    tb = time_ms(pb, args.warmup, args.iters)
    tc = time_ms(pc, args.warmup, args.iters)
    tt = time_ms(transpose, args.warmup, args.iters)
    print(f"#   phaseA={ta:.4f}  phaseB={tb:.4f}  phaseC={tc:.4f}  transpose={tt:.4f}  sum={ta+tb+tc+tt:.4f}")

    # ── CUDA cam path ──
    from diffusion.model.ops.fused_gdn_chunkwise_cuda import build, run_cuda_cam

    build()
    cout = run_cuda_cam(q, k, v, beta, decay)
    assert torch.isfinite(cout).all(), "CUDA cam NaN/Inf"
    d = (cout.float() - out.float()).abs()
    denom = out.float().abs().clamp_min(1e-3)
    mx, mn = (d / denom).max().item(), (d / denom).mean().item()
    ok = mx <= 3e-2 and mn <= 3e-3
    print(f"# CUDA cam check: max_rel={mx:.3e} mean_rel={mn:.3e} -> {'PASS' if ok else 'FAIL'}")
    tcuda = time_ms(lambda: run_cuda_cam(q, k, v, beta, decay), args.warmup, args.iters)
    print(f"# CUDA cam end2end={tcuda:.4f} ms   speedup vs triton cam={t/tcuda:.3f}x")
    # CUDA sub-kernel breakdown
    import diffusion.model.ops.fused_gdn_chunkwise_cuda as ci

    ext = ci.build()
    BH = B * H
    BD = 128
    IPk = torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16)
    A_ = torch.empty_like(IPk)
    betaf = beta.contiguous().float()
    kc = k.contiguous()
    vc = v.contiguous()
    qc = q.contiguous()
    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    tca = time_ms(lambda: ext.cam_phase_a_kv(kc, vc, betaf, IPk, A_, F, S), args.warmup, args.iters)
    ext.cam_phase_a_kv(kc, vc, betaf, IPk, A_, F, S)
    Mh, _, _, _ = phase_b_triton(
        IPk, A_, dummy, dummy, decay.contiguous(), F=F, dot_precision=0, direction=0, combined_history=True, skip_z=True
    )
    tcb = time_ms(
        lambda: phase_b_triton(
            IPk,
            A_,
            dummy,
            dummy,
            decay.contiguous(),
            F=F,
            dot_precision=0,
            direction=0,
            combined_history=True,
            skip_z=True,
        ),
        args.warmup,
        args.iters,
    )
    Mbf = Mh.to(torch.bfloat16).contiguous()
    oo = torch.empty(B, H, D, N, device=dev, dtype=torch.float32)
    tcc = time_ms(lambda: ext.cam_phase_c(qc, Mbf, oo, F, S), args.warmup, args.iters)
    print(f"#   CUDA cam_phase_a={tca:.4f}  phase_b={tcb:.4f}  cam_phase_c={tcc:.4f}")


if __name__ == "__main__":
    main()
