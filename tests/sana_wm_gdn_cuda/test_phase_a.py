#!/usr/bin/env python3
"""Validate CUDA Phase A (I_P_kv, A, I_P_z, B_z) against Triton phase_a."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

torch.backends.cuda.matmul.allow_tf32 = False
from _harness import make_inputs

from diffusion.model.ops.fused_gdn_chunkwise import phase_a
from diffusion.model.ops.fused_gdn_chunkwise_cuda import build, cuda_phase_a


def cmp(name, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    denom = b.abs().clamp_min(1e-3)
    print(
        f"  {name:8s} shape={tuple(a.shape)} max_abs={d.max():.3e} "
        f"mean_abs={d.mean():.3e} max_rel={(d/denom).max():.3e} "
        f"ref_absmean={b.abs().mean():.3e}"
    )


def main():
    B, F, S, H, D = 1, 11, 920, 20, 128
    dev = torch.device("cuda")
    inp = make_inputs(B, F, S, H, D, dev, torch.bfloat16, 0)
    build()
    IPk_t, A_t, IPz_t, Bz_t = phase_a(
        inp["qkv"],
        inp["beta"],
        inp["q_inv_rms"],
        inp["k_inv_rms"],
        inp["q_norm_w"],
        inp["k_norm_w"],
        inp["rope_cos"],
        inp["rope_sin"],
        F=F,
        S=S,
        dot_precision=0,
    )
    IPk_c, A_c, IPz_c, Bz_c = cuda_phase_a(inp)
    print("# Phase A: CUDA vs Triton")
    cmp("I_P_kv", IPk_c, IPk_t)
    cmp("A", A_c, A_t)
    cmp("I_P_z", IPz_c, IPz_t)
    cmp("B_z", Bz_c, Bz_t)

    # per-kernel timing
    from _harness import time_ms

    import diffusion.model.ops.fused_gdn_chunkwise_cuda as ci

    ext = ci.build()
    qkv = inp["qkv"].contiguous()
    beta = inp["beta"].float()
    kir = inp["k_inv_rms"].float()
    knw = inp["k_norm_w"].float()
    rc = inp["rope_cos"]
    rs = inp["rope_sin"]
    BH = 20
    BD = 128
    dev = qkv.device
    IPk = torch.empty(BH, F, BD, BD, device=dev, dtype=torch.bfloat16)
    A = torch.empty_like(IPk)
    IPz = torch.empty_like(IPk)
    Bz = torch.empty(BH, F, BD, device=dev, dtype=torch.float32)
    t_kv = time_ms(lambda: ext.phase_a_kv(qkv, beta, kir, knw, rc, rs, IPk, A, F, S, 1.0), 20, 50)
    t_z = time_ms(lambda: ext.phase_a_z(qkv, beta, kir, knw, IPz, Bz, F, S, 1.0), 20, 50)
    t_a_tot = time_ms(lambda: cuda_phase_a(inp), 20, 50)
    print(f"# CUDA phase_a_kv={t_kv:.4f}ms  phase_a_z={t_z:.4f}ms  total(cuda_phase_a)={t_a_tot:.4f}ms")
    print(f"# (Triton phase_a ~0.337ms for both)")


if __name__ == "__main__":
    main()
