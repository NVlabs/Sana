# Fast H3 VAE decoder (non-causal, 24 layers), from the StreamRender-H3 fast H3 VAE work (NVIDIA); see fast_vae/__init__.py.
# Copied from the delivery code; changed: imports, and quack GEMMs run with tuned=False (quack autotunes every new
# GEMM shape in-process otherwise, which takes minutes at these sizes).
"""NonCausalFast v3: v2 plus three changes that trade fp32 exactness (unneeded at a 30 dB target) for speed.

* input: latent de-normalisation, post_quant_conv (1x1x1 conv) and x_embedder are one 24 -> D affine map, one GEMM;
* head: final LayerNorm writes bf16 once and the head is a single bf16 GEMM (v2 splits hi/lo for fp32 accuracy);
* everything else (window / global FA4 varlen attention, fused qkv post kernel, quack SwiGLU) is unchanged.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .noncausal_fast import BF16, NonCausalFast, _resid_norm_kernel, qgemm, qgemm_act, _flash_attn_fwd, _nc_qkv_post_kernel


@triton.jit
def _resid_ln_bf16_kernel(x, y, scale, w, b, out, D: tl.constexpr, EPS: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    xv = tl.load(x + r.to(tl.int64) * D + c)
    yv = tl.load(y + r.to(tl.int64) * D + c).to(tl.float32)
    xv = xv + yv * tl.load(scale + c)
    mean = tl.sum(xv, 0) / D
    xc = xv - mean
    rs = tl.rsqrt(tl.sum(xc * xc, 0) / D + EPS)
    tl.store(out + r.to(tl.int64) * D + c, (xc * rs * tl.load(w + c) + tl.load(b + c)).to(tl.bfloat16))


class NonCausalFast3(NonCausalFast):
    def __init__(self, ref, T, H, W, *, cuda_graph: bool = True):
        r = ref
        dev = r.register_tokens.device
        # z_norm -> raw = z*std + mean -> pq = Wc raw + bc -> e = We pq + be   =>   e = A z + c
        Wc = r.post_quant_conv.weight.detach().float().reshape(24, 24)
        bc = r.post_quant_conv.bias.detach().float()
        We = r.x_embedder.weight.detach().float()                 # D, 24
        be = r.x_embedder.bias.detach().float()
        std, mean = r.latents_std.float().to(dev), r.latents_mean.float().to(dev)
        A = We @ Wc * std[None, :]                                # D, 24
        c = We @ (Wc @ mean + bc) + be
        self._A, self._c = A.contiguous(), c.contiguous()
        self._whead_bf = r.proj_out.weight.detach().to(BF16).contiguous()
        self._hln = None
        super().__init__(ref, T, H, W, cuda_graph=cuda_graph)

    @torch.no_grad()
    def _step(self):
        r, T, H, W, N, M = self.r, self.T, self.H, self.W, self.N, self.M
        if self._hln is None:
            self._hln = torch.zeros(N, self.D, dtype=BF16, device=self.dev)
            self._zt = torch.zeros(N, 24, device=self.dev)
        self._zt.copy_(self.z_in[0].permute(0, 2, 3, 1).reshape(N, 24))
        torch.addmm(self._c, self._zt, self._A.t(), out=self.x[:N])
        self.x[:N].copy_(self.x[:N].to(BF16))                     # same bf16 rounding of the embedding as v2
        self.x[N:].copy_(self.g0)
        lw0 = self.lw[0]
        _resid_norm_kernel[(M,)](self.x, self.ybuf, self.ident, lw0["s1"], lw0["n1"], self.h, D=self.D, EPS=lw0["eps1"], ADD=False, num_warps=8)
        for li in range(self.L):
            lw, g = self.lw[li], self.geo[self.shift[li]]
            torch.addmm(lw["bqkv"], self.h, lw["wqkv"].t(), out=self.qkv)
            _nc_qkv_post_kernel[(M,)](self.qkv, self.qo, self.kp, self.vp, self.kgp, self.vgp, g["cos"], g["sin"], g["qpos"],
                                      g["kpos"], self.kgpos, g["gwin"], N, self.S, g["maxw"],
                                      HEADS=self.heads, DH=self.dh, RD=g["rd"], EPS=lw["epsq"], num_warps=4)
            _flash_attn_fwd(self.qo[:N], self.kp[:g["total_k"]], self.vp[:g["total_k"]], cu_seqlens_q=g["cu_q"], cu_seqlens_k=g["cu_k"],
                            max_seqlen_q=g["max_q"], max_seqlen_k=g["max_k"], out=self.obuf[:N].view(N, self.heads, self.dh))
            if li < self.L - 1:
                _flash_attn_fwd(self.qo[N:], self.kgp, self.vgp, cu_seqlens_q=self.g_cuq, cu_seqlens_k=self.g_cuk,
                                max_seqlen_q=self.S, max_seqlen_k=self.g_maxk, out=self.obuf[N:].view(M - N, self.heads, self.dh))
            qgemm(self.obuf, lw["wo"].t(), bias=lw["bo"], out=self.ybuf, tuned=False)
            _resid_norm_kernel[(M,)](self.x, self.ybuf, g["src"], lw["s1"], lw["n2"], self.h, D=self.D, EPS=lw["eps2"], ADD=True, num_warps=8)
            qgemm_act(self.h, lw["w1"].t(), bias=lw["b1"], activation="swiglu", store_preact=False, postact_out=self.abuf, tuned=False)
            torch.addmm(lw["b2"], self.abuf, lw["w2"].t(), out=self.ybuf)
            if li < self.L - 1:
                nxt = self.lw[li + 1]
                _resid_norm_kernel[(M,)](self.x, self.ybuf, self.ident, lw["s2"], nxt["n1"], self.h, D=self.D, EPS=nxt["eps1"], ADD=True, num_warps=8)
            else:
                _resid_ln_bf16_kernel[(N,)](self.x, self.ybuf, lw["s2"], self.ln_w, self.ln_b, self._hln, D=self.D, EPS=self.ln_eps, num_warps=8)
        torch.mm(self._hln, self._whead_bf.t(), out_dtype=torch.float32, out=self.G)
