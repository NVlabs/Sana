# Fast H3 VAE decoder (non-causal, 24 layers), from the StreamRender-H3 fast H3 VAE work (NVIDIA); see fast_vae/__init__.py.
# Copied from the delivery code with only the imports changed.
"""Fused Triton kernels shared by the non-causal engine.

_resid_norm_kernel     x[r] += y[src[r]] * scale (optional), then RMSNorm(x[r]) * w -> bf16
_resid_ln_split_kernel x[r] += y[r] * scale, LayerNorm(x[r]) * w + b, written as [hi | lo | hi] bf16 for an fp32-accurate head GEMM
_interleave            reorder SwiGLU rows [gate | up] -> gate0, up0, gate1, up1, ... (quack's gated epilogue layout)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _resid_norm_kernel(x, y, src, scale, w, hout, D: tl.constexpr, EPS: tl.constexpr, ADD: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    xr = x + r.to(tl.int64) * D
    xv = tl.load(xr + c)
    if ADD:
        sr = tl.load(src + r).to(tl.int64)
        yv = tl.load(y + sr * D + c).to(tl.float32)
        xv = xv + yv * tl.load(scale + c)
        tl.store(xr + c, xv)
    rs = tl.rsqrt(tl.sum(xv * xv, 0) / D + EPS)
    tl.store(hout + r.to(tl.int64) * D + c, (xv * rs * tl.load(w + c)).to(tl.bfloat16))


@triton.jit
def _resid_ln_split_kernel(x, y, scale, w, b, aout, D: tl.constexpr, EPS: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    xv = tl.load(x + r.to(tl.int64) * D + c)
    yv = tl.load(y + r.to(tl.int64) * D + c).to(tl.float32)
    xv = xv + yv * tl.load(scale + c)
    mean = tl.sum(xv, 0) / D
    xc = xv - mean
    rs = tl.rsqrt(tl.sum(xc * xc, 0) / D + EPS)
    o = xc * rs * tl.load(w + c) + tl.load(b + c)
    hi = o.to(tl.bfloat16)
    lo = (o - hi.to(tl.float32)).to(tl.bfloat16)
    ar = aout + r.to(tl.int64) * (3 * D)
    tl.store(ar + c, hi)
    tl.store(ar + D + c, lo)
    tl.store(ar + 2 * D + c, hi)


def _interleave(w):
    """rows [gate | up] -> gate0, up0, gate1, up1, ... (quack's gated-epilogue layout)."""
    I = w.shape[0] // 2
    return torch.stack([w[:I], w[I:]], 1).reshape(2 * I, *w.shape[1:]).contiguous()
