"""Hand-written Triton fusions for the LTX DiT block (no torch.compile needed).

Targets the two elementwise chains that the 4K profile showed to be the single
largest GPU bucket (~35% of a SOL-accelerated step):

  1) AdaLN:  rms_norm(x) * (1 + scale) + shift
     eager = F.rms_norm (1r+1w) -> mul (2r+1w) -> add (2r+1w)  ~5 reads + 3 writes
     fused = 2 reads of x + 1 write                             (scale/shift are tiny/broadcast)

  2) gate+residual:  x + y * gate
     eager = mul (2r+1w) -> add (2r+1w)   fused = 2r + 1w

Both are exact elementwise rewrites (same math, fp32 accumulation like torch's).
Only the accelerated processor calls these kernels; unsupported shapes use eager operations.
"""

import torch


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_adaln_kernel(
        X, S, SH, Y, stride_x, stride_s, D, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        xp = X + row * stride_x
        yp = Y + row * stride_x
        sp = S + row * stride_s
        hp = SH + row * stride_s
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, D, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < D
            a = tl.load(xp + cols, mask=m, other=0.0).to(tl.float32)
            acc += a * a
        inv = 1.0 / tl.sqrt(tl.sum(acc) / D + eps)
        for off in range(0, D, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < D
            a = tl.load(xp + cols, mask=m, other=0.0).to(tl.float32)
            s = tl.load(sp + cols, mask=m, other=0.0).to(tl.float32)
            h = tl.load(hp + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(yp + cols, a * inv * (1.0 + s) + h, mask=m)

    @triton.jit
    def _gate_residual_kernel(X, Y, G, O, stride_x, stride_g, D, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        xp = X + row * stride_x
        yp = Y + row * stride_x
        op = O + row * stride_x
        gp = G + row * stride_g
        for off in range(0, D, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < D
            a = tl.load(xp + cols, mask=m, other=0.0).to(tl.float32)
            b = tl.load(yp + cols, mask=m, other=0.0).to(tl.float32)
            g = tl.load(gp + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(op + cols, a + b * g, mask=m)


def _row_stride_for(param, B, T, D):
    """Return per-row stride for an AdaLN param of shape (B, Ts, D); 0 == broadcast."""
    if param.dim() != 3 or param.shape[0] != B or param.shape[2] != D:
        return None
    ts = param.shape[1]
    if ts == 1:
        return 0
    if ts == T:
        return D
    return None


def rmsnorm_adaln(x, scale, shift, eps=1e-6):
    """rms_norm(x) * (1 + scale) + shift, fused. Falls back to eager when unsupported."""
    if not (_HAS_TRITON and x.is_cuda and x.dim() == 3):
        return _eager_rmsnorm_adaln(x, scale, shift, eps)
    B, T, D = x.shape
    if B != 1:  # AdaLN row indexing below assumes a single batch (always true here)
        return _eager_rmsnorm_adaln(x, scale, shift, eps)
    ss = _row_stride_for(scale, B, T, D)
    hs = _row_stride_for(shift, B, T, D)
    if ss is None or hs is None or ss != hs:
        return _eager_rmsnorm_adaln(x, scale, shift, eps)
    xc = x.contiguous()
    sc = scale.contiguous()
    hc = shift.contiguous()
    y = torch.empty_like(xc)
    BLOCK = 1024
    _rmsnorm_adaln_kernel[(B * T,)](
        xc.view(-1, D),
        sc.view(-1, D),
        hc.view(-1, D),
        y.view(-1, D),
        D,
        ss,
        D,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return y.view(B, T, D)


def gate_residual(x, y, gate):
    """x + y * gate, fused. Falls back to eager when unsupported."""
    if not (_HAS_TRITON and x.is_cuda and x.dim() == 3):
        return x + y * gate
    B, T, D = x.shape
    if B != 1 or y.shape != x.shape:
        return x + y * gate
    gs = _row_stride_for(gate, B, T, D)
    if gs is None:
        return x + y * gate
    xc = x.contiguous()
    yc = y.contiguous()
    gc = gate.contiguous()
    o = torch.empty_like(xc)
    _gate_residual_kernel[(B * T,)](
        xc.view(-1, D),
        yc.view(-1, D),
        gc.view(-1, D),
        o.view(-1, D),
        D,
        gs,
        D,
        BLOCK=1024,
        num_warps=8,
    )
    return o.view(B, T, D)


def _eager_rmsnorm_adaln(x, scale, shift, eps):
    return (
        torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)
        * (1 + scale)
        + shift
    )


# ------------------------------------------------------- SPLIT RoPE (what LTX-2.3 uses)
if _HAS_TRITON:

    @triton.jit
    def _rope_split_kernel_row(X, C, S, O, Tt, H, R, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        k = tl.arange(0, BLOCK)
        m = k < (H * R)
        h = k // R
        j = k % R
        xb = t * (H * 2 * R) + h * (2 * R) + j
        cb = h * (Tt * R) + t * R + j
        a = tl.load(X + xb, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X + xb + R, mask=m, other=0.0).to(tl.float32)
        c = tl.load(C + cb, mask=m, other=0.0).to(tl.float32)
        s = tl.load(S + cb, mask=m, other=0.0).to(tl.float32)
        tl.store(O + xb, a * c - b * s, mask=m)
        tl.store(O + xb + R, b * c + a * s, mask=m)

    @triton.jit
    def _rope_split_kernel(X, C, S, O, Tt, H, R, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        t = pid // H
        h = pid % H
        xb = t * (H * 2 * R) + h * (2 * R)
        cb = h * (Tt * R) + t * R
        j = tl.arange(0, BLOCK)
        m = j < R
        a = tl.load(X + xb + j, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X + xb + R + j, mask=m, other=0.0).to(tl.float32)
        c = tl.load(C + cb + j, mask=m, other=0.0).to(tl.float32)
        s = tl.load(S + cb + j, mask=m, other=0.0).to(tl.float32)
        tl.store(O + xb + j, a * c - b * s, mask=m)
        tl.store(O + xb + R + j, b * c + a * s, mask=m)


def rope_split(x, cos, sin):
    """Fused SPLIT RoPE for x=(1,T,H*2R), cos/sin=(1,H,T,R).

    eager path = rearrange + mul + 2x addcmul_ + rearrange + swapaxes().reshape()
                 (the final reshape after swapaxes forces a full copy)
    fused      = 1 read + 1 write, output already in (1,T,H*2R) layout.
    Exact same math:  out[:R] = a*cos - b*sin ;  out[R:] = b*cos + a*sin
    """
    if not (_HAS_TRITON and x.is_cuda):
        return None
    if x.dim() != 3 or cos.dim() != 4 or sin.shape != cos.shape:
        return None
    b_, t_, hhd = x.shape
    b2, h_, t2, r_ = cos.shape
    if b_ != 1 or b2 != 1 or t_ != t2 or hhd != h_ * 2 * r_:
        return None
    xc = x.contiguous()
    cc = cos.contiguous()
    sc = sin.contiguous()
    o = torch.empty_like(xc)
    need = h_ * r_
    BLOCK = 1
    while BLOCK < need:
        BLOCK *= 2
    if BLOCK <= 4096:  # one program per token row: far fewer, much fatter programs
        _rope_split_kernel_row[(t_,)](
            xc, cc, sc, o, t_, h_, r_, BLOCK=BLOCK, num_warps=8
        )
    else:
        BLOCK = 1
        while BLOCK < r_:
            BLOCK *= 2
        _rope_split_kernel[(t_ * h_,)](
            xc, cc, sc, o, t_, h_, r_, BLOCK=BLOCK, num_warps=2
        )
    return o


# --------------------------------------------- plain RMSNorm with weight (q_norm/k_norm)
if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_w_kernel(X, W, Y, stride_x, D, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        xp = X + row * stride_x
        yp = Y + row * stride_x
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, D, BLOCK):
            c = off + tl.arange(0, BLOCK)
            m = c < D
            a = tl.load(xp + c, mask=m, other=0.0).to(tl.float32)
            acc += a * a
        inv = 1.0 / tl.sqrt(tl.sum(acc) / D + eps)
        for off in range(0, D, BLOCK):
            c = off + tl.arange(0, BLOCK)
            m = c < D
            a = tl.load(xp + c, mask=m, other=0.0).to(tl.float32)
            w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
            tl.store(yp + c, a * inv * w, mask=m)


def rmsnorm_w(x, mod):
    """Fused torch.nn.RMSNorm(weight) - replaces pow+mean+rsqrt+mul (4 kernels, 3 full
    passes over x) with 2 reads + 1 write. Used for q_norm/k_norm (192 calls/step).
    Returns None to fall back."""
    if not (_HAS_TRITON and x.is_cuda):
        return None
    w = getattr(mod, "weight", None)
    if w is None or w.dim() != 1 or w.shape[0] != x.shape[-1]:
        return None
    D = x.shape[-1]
    n = x.numel() // D
    eps = getattr(mod, "eps", None)
    if eps is None:
        return None
    xc = x.contiguous()
    wc = w.contiguous()
    y = torch.empty_like(xc)
    _rmsnorm_w_kernel[(n,)](
        xc.view(-1, D), wc, y.view(-1, D), D, D, float(eps), BLOCK=1024, num_warps=8
    )
    return y.view_as(x)
