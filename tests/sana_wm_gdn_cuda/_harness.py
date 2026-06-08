"""Shared helpers for the CUDA chunkwise-GDN validation tests.

Tiny input generators + a CUDA-event timer used by ``test_phase_a.py``,
``bench_cam.py`` and ``mem_probe.py``. Inputs are tuned so the GDN recurrence
stays O(1) over F frames (near-identity ``I - P_kv``, ``decay < 1``).
"""

from __future__ import annotations

import torch


def time_ms(fn, warmup, iters):
    """Median wall time (ms) of ``fn`` over ``iters`` CUDA-event-timed runs."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return t[len(t) // 2]


def make_inputs(B, F, S, H, D, device, dtype, seed=0, scale=0.1, beta_scale=0.02):
    """Valid, finite production-shape inputs for the full BiGDN phase_a/b/c."""
    g = torch.Generator(device=device).manual_seed(seed)
    N = F * S
    BH = B * H

    def rn(*shape, s=1.0):
        return torch.randn(*shape, device=device, generator=g) * s

    qkv = (rn(B, N, 3, H, D, s=scale)).to(dtype)
    q_inv_rms = (0.75 + 0.5 * torch.rand(B, N, device=device, generator=g)).float()
    k_inv_rms = (0.75 + 0.5 * torch.rand(B, N, device=device, generator=g)).float()
    q_norm_w = (1.0 + 0.1 * rn(H * D)).float()
    k_norm_w = (1.0 + 0.1 * rn(H * D)).float()
    theta = torch.rand(N, D, device=device, generator=g) * 6.2831853
    rope_cos = theta.cos().float().contiguous()
    rope_sin = theta.sin().float().contiguous()
    beta = (beta_scale * torch.rand(BH, N, device=device, generator=g)).float()
    decay = (0.90 + 0.09 * torch.rand(BH, F, device=device, generator=g)).float()
    return dict(
        qkv=qkv,
        q_inv_rms=q_inv_rms,
        k_inv_rms=k_inv_rms,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        beta=beta,
        decay=decay,
        F=F,
        S=S,
    )


def make_cam_inputs(B, F, S, H, D, dev, seed=0, scale=0.1, beta_scale=0.02):
    """Cam-branch inputs (q/k/v already prep'd to ``[B, H, D, N]`` fp32)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    N = F * S
    q = (scale * torch.randn(B, H, D, N, device=dev, generator=g)).float().contiguous()
    k = (scale * torch.randn(B, H, D, N, device=dev, generator=g)).float().contiguous()
    v = (scale * torch.randn(B, H, D, N, device=dev, generator=g)).float().contiguous()
    beta = (beta_scale * torch.rand(B, H, F, S, device=dev, generator=g)).float().contiguous()
    decay = (0.90 + 0.09 * torch.rand(B, H, F, device=dev, generator=g)).float().contiguous()
    return q, k, v, beta, decay
