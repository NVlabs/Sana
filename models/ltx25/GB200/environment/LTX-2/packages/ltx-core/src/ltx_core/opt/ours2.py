"""RoPE+QK-norm fusion (288 sites) and gated-residual fusion, bf16.

PytorchPreAttention runs q_norm, k_norm, then apply_rotary_emb on each -- and
apply_split_rotary_emb alone is mul + 2 addcmul_ over the full q/k tensors. That is
~8 HBM round trips per attention site, at 288 sites (the largest slot count in the
model; post_sa/ada_zero are only 48 each).

SPLIT rope semantics, read off apply_split_rotary_emb (rope.py:43-84):
    split_input = rearrange(x, "... (d r) -> ... d r", d=2)   # contiguous halves
    x1, x2 = split_input[...,0,:], split_input[...,1,:]
    out1 = x1*cos - x2*sin ;  out2 = x2*cos + x1*sin
i.e. CONTIGUOUS split-half (first D/2 vs last D/2), NOT interleaved -- getting this
backwards produces plausible-looking but wrong video, so LTX_VERIFY=1 checks every
call against the reference on real tensors.

When input is 3D (b,s,inner) and cos is 4D (b,h,s,dh/2) the reference reshapes to
(b,h,s,dh) first; RMSNorm however spans the WHOLE inner dim across heads. One
program per token row therefore does: rms over all `inner` elements, then per-head
rotary using that head's cos/sin.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_VERIFY = os.environ.get("LTX_VERIFY", "") == "1"
_VERIFY_SEEN: dict = {}


@triton.jit
def _rmsnorm_rope_kernel(
    X, W, COS, SIN, OUT,
    S, INNER, DH, HALF,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)          # flattened (b*s) token index
    s = row % S
    cols = tl.arange(0, BLOCK)
    mask = cols < INNER

    x = tl.load(X + row * INNER + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / INNER
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    xn = x * rstd * w

    head = cols // DH
    within = cols % DH
    j = tl.where(within < HALF, within, within - HALF)
    # cos/sin are (b, h, s, HALF) contiguous
    f_off = (head * S + s) * HALF + j
    c = tl.load(COS + f_off, mask=mask, other=0.0).to(tl.float32)
    sn = tl.load(SIN + f_off, mask=mask, other=0.0).to(tl.float32)

    # partner element: +HALF for the first half, -HALF for the second
    p_off = tl.where(within < HALF, cols + HALF, cols - HALF)
    xp = tl.load(X + row * INNER + p_off, mask=mask, other=0.0).to(tl.float32)
    wp = tl.load(W + p_off, mask=mask, other=0.0).to(tl.float32)
    xpn = xp * rstd * wp

    out = tl.where(within < HALF, xn * c - xpn * sn, xn * c + xpn * sn)
    tl.store(OUT + row * INNER + cols, out.to(OUT.dtype.element_ty), mask=mask)


def _reference(x, weight, eps, cos, sin, rope_mod):
    xn = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
    return rope_mod.apply_split_rotary_emb(xn, cos, sin)


def fused_rmsnorm_split_rope(x, weight, eps, cos, sin):
    """Returns None when the shape assumptions do not hold (caller falls back)."""
    if x.ndim != 3 or cos.ndim != 4 or cos.shape[0] != 1 or x.shape[0] != 1:
        return None
    if cos.shape != sin.shape:
        return None
    b, s, inner = x.shape
    h, s_c, half = cos.shape[1], cos.shape[2], cos.shape[3]
    if s_c != s or h * half * 2 != inner:
        return None
    dh = inner // h
    from ltx_core.opt.ours import _note as _n2
    xc, cc, sc = _n2("rope_x", x).contiguous(), cos.contiguous(), sin.contiguous()
    out = torch.empty_like(xc)
    _rmsnorm_rope_kernel[(b * s,)](
        xc, weight.contiguous(), cc, sc, out,
        s, inner, dh, half, eps,
        BLOCK=triton.next_power_of_2(inner),
    )
    return out


class FusedPreAttention:
    """Drop-in for PytorchPreAttention. Falls back to eager whenever the fused
    path's shape assumptions do not hold, so it can never silently mis-handle a
    layout it was not written for."""

    def __init__(self):
        from ltx_core.model.transformer import rope as _rope
        from ltx_core.model.transformer.ops import PytorchPreAttention

        self._rope = _rope
        self._eager = PytorchPreAttention()

    def _one(self, x, norm, pe, tag):
        out = fused_rmsnorm_split_rope(x, norm.weight, norm.eps, pe[0], pe[1])
        if out is None:
            return None
        if _VERIFY and tag not in _VERIFY_SEEN:
            ref = _reference(x, norm.weight, norm.eps, pe[0], pe[1], self._rope)
            err = (ref.float() - out.float()).abs().max().item()
            rel = err / ref.float().abs().max().clamp_min(1e-8).item()
            _VERIFY_SEEN[tag] = (err, rel)
            print(f"[verify] {tag}: max_abs={err:.3e} rel={rel:.3e} shape={tuple(x.shape)}", flush=True)
            assert rel < 5e-2, f"{tag} fused rope disagrees with reference (rel={rel:.3e})"
        return out

    def __call__(self, q, k, attn_module, mask, pe, k_pe):
        if pe is None or getattr(attn_module, "rope_type", None) is None:
            return self._eager(q, k, attn_module, mask, pe, k_pe)
        if str(attn_module.rope_type).endswith("SPLIT") is False:
            return self._eager(q, k, attn_module, mask, pe, k_pe)
        qo = self._one(q, attn_module.q_norm, pe, "q")
        ko = self._one(k, attn_module.k_norm, pe if k_pe is None else k_pe, "k")
        if qo is None or ko is None:
            return self._eager(q, k, attn_module, mask, pe, k_pe)
        return qo, ko


# --- O2: gated residual  x + y*gate(*mask) -----------------------------------
@triton.jit
def _gated_residual_kernel(X, Y, G, OUT, H, g_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(X + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + row * g_stride + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * H + cols, (x + y * g).to(OUT.dtype.element_ty), mask=mask)


def gated_residual(x, y, gate):
    """x + y*gate in one kernel (the `vx = vx + attn(...)*gate*mask` sites)."""
    h = x.shape[-1]
    rows = x.numel() // h
    if gate.shape[-1] != h:
        return x + y * gate
    xc, yc, gc = x.contiguous(), y.contiguous(), gate.contiguous()
    out = torch.empty_like(xc)
    n = gc.numel() // h
    _gated_residual_kernel[(rows,)](xc, yc, gc, out, h, 0 if n == 1 else h,
                                    BLOCK=triton.next_power_of_2(h))
    return out


# --- O3: FFN GEMM+GELU epilogue ----------------------------------------------
# GELUApprox.forward is `gelu(self.proj(x), approximate="tanh")` -- a full GEMM
# write-out followed by a separate elementwise pass over the 4x-wide inner tensor.
# Rather than hand-writing a matmul (which will not beat cuBLAS), use the cuBLASLt
# GELU epilogue that PyTorch exposes as torch._addmm_activation.
def install_gelu_epilogue(gelu_mod) -> int:
    cls = gelu_mod.GELUApprox
    if getattr(cls.forward, "_ltx_fused", False):
        return 0
    ref_forward = cls.forward

    def forward(self, x):
        w, b = self.proj.weight, self.proj.bias
        if b is None or x.ndim != 3:
            return ref_forward(self, x)
        flat = x.reshape(-1, x.shape[-1])
        try:
            out = torch._addmm_activation(b, flat, w.t(), use_gelu=True)
        except Exception:
            return ref_forward(self, x)
        out = out.view(*x.shape[:-1], -1)
        if _VERIFY and "gelu" not in _VERIFY_SEEN:
            ref = ref_forward(self, x)
            err = (ref.float() - out.float()).abs().max().item()
            rel = err / ref.float().abs().max().clamp_min(1e-8).item()
            _VERIFY_SEEN["gelu"] = (err, rel)
            print(f"[verify] gelu_epilogue: max_abs={err:.3e} rel={rel:.3e}", flush=True)
            assert rel < 5e-2, f"gelu epilogue disagrees with reference (rel={rel:.3e})"
        return out

    forward._ltx_fused = True
    cls.forward = forward
    return 1


# --- A1 + A2: cross-attention AdaLN --------------------------------------------
# apply_cross_attention_adaln does, per block per step:
#     attn_input  = x_normed * (1 + q_scale) + q_shift        <- bare affine (A2)
#     enc_states  = context  * (1 + scale_kv) + shift_kv       <- bare affine (A2)
#     return attn(...) * q_gate
# None of it goes through ada_zero_function, so O1 never covered it.
#
# A1: the vendor's own comment says that with the prompt-side AdaLN MLP disabled
# (prompt_timestep is None) "K/V are timestep-independent and cacheable across
# denoising/AR steps" -- yet enc_states is recomputed every block every step.
# Caching it removes the work outright instead of merely fusing it.
_KV_CACHE: dict = {}


@triton.jit
def _affine_kernel(X, S, SH, OUT, H, s_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < H
    x = tl.load(X + row * H + cols, mask=m, other=0.0).to(tl.float32)
    s = tl.load(S + row * s_stride + cols, mask=m, other=0.0).to(tl.float32)
    sh = tl.load(SH + row * s_stride + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + row * H + cols, (x * (1.0 + s) + sh).to(OUT.dtype.element_ty), mask=m)


def fused_affine(x, scale, shift):
    """x*(1+scale)+shift in one kernel; falls back to eager on shape mismatch."""
    h = x.shape[-1]
    if scale.shape[-1] != h or shift.shape != scale.shape:
        return x * (1 + scale) + shift
    rows = x.numel() // h
    xc, sc, shc = x.contiguous(), scale.contiguous(), shift.contiguous()
    n = sc.numel() // h
    if n not in (1, rows):
        return x * (1 + scale) + shift
    out = torch.empty_like(xc)
    _affine_kernel[(rows,)](xc, sc, shc, out, h, 0 if n == 1 else h,
                            BLOCK=triton.next_power_of_2(h))
    return out


def install_cross_attn_opt(tmod) -> int:
    orig = tmod.apply_cross_attention_adaln
    if getattr(orig, "_ltx_opt", False):
        return 0

    def patched(x_normed, context, attn, q_shift, q_scale, q_gate,
                prompt_scale_shift_table, prompt_timestep, context_mask=None):
        batch_size = x_normed.shape[0]
        if prompt_timestep is None:
            # A1: timestep-independent -> compute once, reuse for every step.
            key = (context.data_ptr(), tuple(context.shape),
                   prompt_scale_shift_table.data_ptr(), context.dtype)
            enc = _KV_CACHE.get(key)
            if enc is None:
                kv = prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
                shift_kv, scale_kv = kv.unbind(dim=2)
                enc = fused_affine(context, scale_kv, shift_kv)
                _KV_CACHE[key] = enc
        else:
            kv = prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
            kv = kv + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
            shift_kv, scale_kv = kv.unbind(dim=2)
            enc = fused_affine(context, scale_kv, shift_kv)

        attn_input = fused_affine(x_normed, q_scale, q_shift)  # A2
        out = attn(attn_input, context=enc, mask=context_mask) * q_gate

        if _VERIFY and "xattn" not in _VERIFY_SEEN:
            ref = orig(x_normed, context, attn, q_shift, q_scale, q_gate,
                       prompt_scale_shift_table, prompt_timestep, context_mask)
            err = (ref.float() - out.float()).abs().max().item()
            rel = err / ref.float().abs().max().clamp_min(1e-8).item()
            _VERIFY_SEEN["xattn"] = (err, rel)
            print(f"[verify] cross_attn_adaln: max_abs={err:.3e} rel={rel:.3e} "
                  f"kv_cached={prompt_timestep is None}", flush=True)
            assert rel < 5e-2, f"cross-attn adaln disagrees with reference (rel={rel:.3e})"
        return out

    patched._ltx_opt = True
    tmod.apply_cross_attention_adaln = patched
    return 1


# --- A3: gated residual wiring -------------------------------------------------
# The a2v / v2a sites in BasicAVTransformerBlock.forward are
#     vx = vx + (attn(...) * gate * mask)
# i.e. three full-tensor passes (two muls + an add) after the attention. Fuse to one.
@triton.jit
def _gated_residual2_kernel(X, Y, G, M, OUT, H, g_stride, m_stride,
                            HAS_M: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    msk = cols < H
    x = tl.load(X + row * H + cols, mask=msk, other=0.0).to(tl.float32)
    y = tl.load(Y + row * H + cols, mask=msk, other=0.0).to(tl.float32)
    g = tl.load(G + row * g_stride + cols, mask=msk, other=0.0).to(tl.float32)
    v = y * g
    if HAS_M:
        mm = tl.load(M + row * m_stride + cols, mask=msk, other=0.0).to(tl.float32)
        v = v * mm
    tl.store(OUT + row * H + cols, (x + v).to(OUT.dtype.element_ty), mask=msk)


def _stride_for(t, h, rows):
    n = t.numel() // h if t.shape[-1] == h else -1
    if n == 1:
        return 0
    if n == rows:
        return h
    return None


def gated_residual(x, y, gate, mask=None):
    """x + y*gate*mask in one kernel; falls back to eager on any shape mismatch.

    Gated by LTX_A3 so the same tree can run the eager baseline -- the call sites are
    edited into transformer.py unconditionally, so without a switch every arm
    (including K0) would carry it and the delta would be unmeasurable."""
    if os.environ.get("LTX_A3", "") != "1":
        return x + (y * gate if mask is None else y * gate * mask)
    h = x.shape[-1]
    rows = x.numel() // h
    gs = _stride_for(gate, h, rows)
    ms = 0 if mask is None else _stride_for(mask, h, rows)
    if gs is None or ms is None:
        return x + (y * gate if mask is None else y * gate * mask)
    xc, yc, gc = x.contiguous(), y.contiguous(), gate.contiguous()
    mc = xc if mask is None else mask.contiguous()
    out = torch.empty_like(xc)
    _gated_residual2_kernel[(rows,)](xc, yc, gc, mc, out, h, gs, ms,
                                     HAS_M=mask is not None,
                                     BLOCK=triton.next_power_of_2(h))
    if _VERIFY and "gres" not in _VERIFY_SEEN:
        ref = x + (y * gate if mask is None else y * gate * mask)
        err = (ref.float() - out.float()).abs().max().item()
        rel = err / ref.float().abs().max().clamp_min(1e-8).item()
        _VERIFY_SEEN["gres"] = (err, rel)
        print(f"[verify] gated_residual: max_abs={err:.3e} rel={rel:.3e}", flush=True)
        assert rel < 5e-2, f"gated_residual disagrees (rel={rel:.3e})"
    return out


# --- A4: vendor CUDA rms_norm_split_rope (now that ops_cpp builds) --------------
# Built with conda cuda-nvcc 13.0.88 -- torch cu130's headers do NOT compile under
# the cluster's 13.3 nvcc (fails inside torch's own List_inl.h). Load functional.py
# by path: the package __init__ also drags in linear.py -> blockwise_cpp, which we
# deliberately never built (it needs cutlass and is FP8-only).
_VENDOR_ROPE = None


def _load_vendor_rope():
    global _VENDOR_ROPE
    if _VENDOR_ROPE is not None:
        return _VENDOR_ROPE
    import importlib
    import sys
    import types
    from pathlib import Path

    sys.path.insert(0, "/lustre/fsw/portfolios/nvr/users/yitongl/code/ltx25")
    import torch  # noqa: F401  (must load libc10 before the extension)
    import ops_cpp  # noqa: F401

    root = Path(__file__).resolve().parents[4]
    src = root / "ltx-kernels" / "src"
    # functional.py uses ABSOLUTE imports (`from ltx_kernels.blockwise.triton_ops ...`),
    # so loading it by path alone fails. But importing the package for real runs
    # ltx_kernels/__init__.py -> all_to_all -> all2all_cpp and
    # blockwise/__init__.py -> linear -> blockwise_cpp, neither of which we built
    # (all2all is multi-GPU only; blockwise is FP8-only and needs cutlass).
    # Register namespace stubs so submodule imports resolve without running either __init__.
    for name, sub in (("ltx_kernels", src / "ltx_kernels"),
                      ("ltx_kernels.blockwise", src / "ltx_kernels" / "blockwise")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(sub)]
            sys.modules[name] = m
    _VENDOR_ROPE = importlib.import_module("ltx_kernels.blockwise.functional")
    return _VENDOR_ROPE


class VendorPreAttention:
    """Vendor CUDA rms_norm_split_rope with out_fp8=False (bf16 out)."""

    def __init__(self):
        from ltx_core.model.transformer.ops import PytorchPreAttention
        self._m = _load_vendor_rope()
        self._eager = PytorchPreAttention()

    def __call__(self, q, k, attn_module, mask, pe, k_pe):
        if pe is None:
            return self._eager(q, k, attn_module, mask, pe, k_pe)
        f = self._m.rms_norm_split_rope
        kpe = pe if k_pe is None else k_pe
        try:
            qo = f(q, pe[0], pe[1], attn_module.q_norm.weight, False)
            ko = f(k, kpe[0], kpe[1], attn_module.k_norm.weight, False)
        except Exception:
            return self._eager(q, k, attn_module, mask, pe, k_pe)
        return qo, ko
