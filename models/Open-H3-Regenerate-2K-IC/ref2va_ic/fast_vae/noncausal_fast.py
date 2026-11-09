# Fast H3 VAE decoder (non-causal, 24 layers), from the StreamRender-H3 fast H3 VAE work (NVIDIA); see fast_vae/__init__.py.
# Copied from the delivery code; changed: imports, and quack GEMMs run with tuned=False (quack autotunes every new
# GEMM shape in-process otherwise, which takes minutes at these sizes).
"""Deployment engine for NonCausalViTDecoder: same math (bf16 GEMMs, fp32 residual), batch 1, fixed (T, H, W).

* every (temporal window, spatial window) is one FlashAttention-4 varlen sequence: queries = its valid tokens,
  keys = its valid tokens + the global tokens of the native blocks it overlaps (no padding, no mask);
* global-token attention is one more varlen call: per native block, queries = its 5 global tokens, keys = all
  tokens of the block + its global tokens;
* the 5 global rows of every block are appended to the token rows, so each linear layer is a single GEMM;
* residual add + RMSNorm, SwiGLU in the up-projection GEMM (quack), fp32 head as one bf16 GEMM on a hi/lo split
  (fused_kernels.py);
* the whole decode is one CUDA graph.
"""
from __future__ import annotations

import site
import sys

import torch
from torch import nn

if "cutlass" not in sys.modules:
    for p in list(sys.path):
        if p.endswith("site-packages"):
            site.addsitedir(p)
from flash_attn.cute.interface import _flash_attn_fwd
from quack.gemm_interface import gemm as qgemm, gemm_act as qgemm_act

from .fused_kernels import _interleave, _resid_ln_split_kernel, _resid_norm_kernel
from .decoder import OUT_CH, PATCH, PATCH_T, NonCausalViTDecoder

BF16 = torch.bfloat16
import triton
import triton.language as tl


@triton.jit
def _nc_qkv_post_kernel(qkv, qo, kp, vp, kgp, vgp, cos, sin, qpos, kpos, kgpos, gwin, N, S, MAXW,
                        HEADS: tl.constexpr, DH: tl.constexpr, RD: tl.constexpr, EPS: tl.constexpr):
    """One row of qkv: qk-RMSNorm, rotary (token rows only), then scatter q to its window-ordered slot, k/v to its
    slot in the window key pool and in the global-attention key pool. A global row is also copied into every
    window that overlaps its block (gwin[b, :] = slot bases, -1 = unused)."""
    r = tl.program_id(0)
    D3: tl.constexpr = 3 * DH
    ROW: tl.constexpr = HEADS * DH
    HALF: tl.constexpr = RD // 2
    hh = tl.arange(0, HEADS)[:, None]
    d = tl.arange(0, DH)[None, :]
    src = qkv + r.to(tl.int64) * (HEADS * D3)
    dp = tl.where(d < HALF, d + HALF, tl.where(d < RD, d - HALF, d))
    sg = tl.where(d < HALF, -1.0, 1.0)
    oq = hh * D3 + d
    op = hh * D3 + dp
    q = tl.load(src + oq).to(tl.float32)
    qp = tl.load(src + op).to(tl.float32)
    k = tl.load(src + DH + oq).to(tl.float32)
    kq = tl.load(src + DH + op).to(tl.float32)
    v = tl.load(src + 2 * DH + oq)
    rq = tl.rsqrt(tl.sum(q * q, axis=1) / DH + EPS)[:, None]
    rk = tl.rsqrt(tl.sum(k * k, axis=1) / DH + EPS)[:, None]
    qn = (q * rq).to(tl.bfloat16).to(tl.float32)
    kn = (k * rk).to(tl.bfloat16).to(tl.float32)
    oo = hh * DH + d
    if r < N:
        qr = (qp * rq).to(tl.bfloat16).to(tl.float32) * sg
        kr = (kq * rk).to(tl.bfloat16).to(tl.float32) * sg
        rm = d < RD
        dc = tl.where(rm, d, 0)
        c = tl.load(cos + r.to(tl.int64) * RD + dc)
        s_ = tl.load(sin + r.to(tl.int64) * RD + dc)
        qv = tl.where(rm, (qn * c).to(tl.bfloat16).to(tl.float32) + (qr * s_).to(tl.bfloat16).to(tl.float32), qn)
        kv = tl.where(rm, (kn * c).to(tl.bfloat16).to(tl.float32) + (kr * s_).to(tl.bfloat16).to(tl.float32), kn)
        qv = qv.to(tl.bfloat16)
        kv = kv.to(tl.bfloat16)
        qd = tl.load(qpos + r).to(tl.int64)
        tl.store(qo + qd * ROW + oo, qv)
        kd = tl.load(kpos + r).to(tl.int64)
        tl.store(kp + kd * ROW + oo, kv)
        tl.store(vp + kd * ROW + oo, v)
    else:
        qv = qn.to(tl.bfloat16)
        kv = kn.to(tl.bfloat16)
        tl.store(qo + r.to(tl.int64) * ROW + oo, qv)
        i = r - N
        b = i // S
        slot = i % S
        for w in range(MAXW):
            base = tl.load(gwin + b * MAXW + w)
            if base >= 0:
                pos = (base + slot).to(tl.int64)
                tl.store(kp + pos * ROW + oo, kv)
                tl.store(vp + pos * ROW + oo, v)
    kg = tl.load(kgpos + r).to(tl.int64)
    tl.store(kgp + kg * ROW + oo, kv)
    tl.store(vgp + kg * ROW + oo, v)


@triton.jit
def _nc_unpatch_kernel(G, bias, out, H, W, NF, TB, NTRIM, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    py = pid % 16
    tmp = pid // 16
    pt = tmp % 4
    tmp = tmp // 4
    oc = tmp % 3
    tmp = tmp // 3
    y = tmp % H
    t = tmp // H
    drop = ((t % TB) == 0) & (pt < NTRIM)
    if not drop:
        fo = t * 4 + pt - NTRIM * (t // TB + 1)
        e = tl.arange(0, BLOCK)
        em = e < W * 16
        x = e // 16
        px = e % 16
        ch = oc * 1024 + pt * 256 + py * 16 + px
        gi = ((t * H + y) * W + x).to(tl.int64) * 3072 + ch
        v = tl.load(G + gi, mask=em, other=0.0) + tl.load(bias + ch, mask=em, other=0.0)
        orow = ((oc * NF + fo) * (16 * H) + y * 16 + py).to(tl.int64) * (16 * W)
        tl.store(out + orow + e, v, mask=em)


class NonCausalFast(nn.Module):
    def __init__(self, ref: NonCausalViTDecoder, T: int, H: int, W: int, *, cuda_graph: bool = True):
        super().__init__()
        r = ref
        self.r, self.T, self.H, self.W = r, T, H, W
        self.heads, self.dh, self.D = r.heads, r.dh, r.heads * r.dh
        dev = r.register_tokens.device
        self.dev = dev
        tb = r.block
        self.nb = nb = (T + tb - 1) // tb
        self.S = S = r.register_tokens.shape[1] + 1
        self.N = N = T * H * W
        self.M = M = N + nb * S
        self.L = L = len(r.blocks)
        self.lw = []
        for blk in r.blocks:
            assert isinstance(blk.norm1, nn.RMSNorm) and blk.attn.norm_q is not None
            self.lw.append(dict(
                wqkv=blk.attn.to_qkv.weight.detach().to(BF16).contiguous(), bqkv=blk.attn.to_qkv.bias.detach().to(BF16),
                wo=blk.attn.to_out.weight.detach().to(BF16).contiguous(), bo=blk.attn.to_out.bias.detach().to(BF16),
                w1=_interleave(blk.ff.w1.weight.detach().to(BF16)), b1=_interleave(blk.ff.w1.bias.detach().to(BF16)),
                w2=blk.ff.w2.weight.detach().to(BF16).contiguous(), b2=blk.ff.w2.bias.detach().to(BF16),
                n1=blk.norm1.weight.detach().float().contiguous(), n2=blk.norm2.weight.detach().float().contiguous(),
                s1=blk.scale1.detach().float().contiguous(), s2=blk.scale2.detach().float().contiguous(),
                eps1=blk.norm1.eps if blk.norm1.eps is not None else torch.finfo(torch.float32).eps,
                eps2=blk.norm2.eps if blk.norm2.eps is not None else torch.finfo(torch.float32).eps,
                epsq=blk.attn.norm_q.eps if blk.attn.norm_q.eps is not None else torch.finfo(torch.float32).eps))
        self.inter = r.blocks[0].ff.w2.weight.shape[1]
        wh = r.proj_out.weight.detach().float(); whi = wh.to(BF16); wlo = (wh - whi.float()).to(BF16)
        self.whead = torch.cat([whi, whi, wlo], 1).contiguous()
        self.bhead = r.proj_out.bias.detach().float().contiguous()
        self.ln_w, self.ln_b, self.ln_eps = r.norm_out.weight.detach().float().contiguous(), r.norm_out.bias.detach().float().contiguous(), r.norm_out.eps
        self.g0 = torch.cat([r.register_tokens[0].detach(), r.register_tokens.new_zeros(1, self.D)], 0).to(BF16).float().repeat(nb, 1).contiguous()
        i32 = dict(dtype=torch.int32, device=dev)
        # global-token attention: per block, keys = the block's tokens (x order) + its S global rows ---------
        HW = H * W
        kgpos = torch.empty(M, dtype=torch.int64); cu_q, cu_k = [0], [0]
        for b in range(nb):
            t0, t1 = b * tb * HW, min((b + 1) * tb, T) * HW
            kgpos[t0:t1] = torch.arange(t1 - t0) + cu_k[-1]
            kgpos[N + b * S: N + (b + 1) * S] = torch.arange(S) + cu_k[-1] + (t1 - t0)
            cu_q.append(cu_q[-1] + S); cu_k.append(cu_k[-1] + (t1 - t0) + S)
        self.kgpos = kgpos.to(**i32)
        self.g_cuq, self.g_cuk = torch.tensor(cu_q, **i32), torch.tensor(cu_k, **i32)
        self.g_maxk = max(b - a for a, b in zip(cu_k, cu_k[1:]))
        self.kgp = torch.zeros(cu_k[-1], self.heads, self.dh, dtype=BF16, device=dev)
        self.vgp = torch.zeros_like(self.kgp)
        # window attention geometry per shift -------------------------------------------------------------
        self.shift = [r._shifts(li) for li in range(L)]
        self.geo = {sh: self._geometry(*sh, dev) for sh in sorted(set(self.shift))}
        tk = max(g["total_k"] for g in self.geo.values())
        self.kp = torch.zeros(tk, self.heads, self.dh, dtype=BF16, device=dev)
        self.vp = torch.zeros_like(self.kp)
        # buffers ------------------------------------------------------------------------------------------
        self.z_in = torch.zeros(1, T, r.latents_mean.numel(), H, W, device=dev)
        self.x = torch.zeros(M, self.D, device=dev)
        self.h = torch.zeros(M, self.D, dtype=BF16, device=dev)
        self.qkv = torch.zeros(M, 3 * self.D, dtype=BF16, device=dev)
        self.qo = torch.zeros(M, self.heads, self.dh, dtype=BF16, device=dev)
        self.obuf = torch.zeros(M, self.D, dtype=BF16, device=dev)
        self.ybuf = torch.zeros(M, self.D, dtype=BF16, device=dev)
        self.abuf = torch.zeros(M, self.inter, dtype=BF16, device=dev)
        self.hsplit = torch.zeros(N, 3 * self.D, dtype=BF16, device=dev)
        self.G = torch.zeros(N, self.whead.shape[0], device=dev)
        self.ident = torch.arange(M, **i32)
        self.NF = T * PATCH_T - r.frames_to_trim * nb
        self.cuda_graph, self.graph = cuda_graph, None
        with torch.no_grad():
            self._step()
        torch.cuda.synchronize()
        if cuda_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()
            torch.cuda.synchronize()

    def _geometry(self, s, st, dev):
        r, T, H, W, tb, ws, S, N = self.r, self.T, self.H, self.W, self.r.block, self.r.window, self.S, self.N
        HW = H * W
        cos, sin = r._rope(T, H, W, s, st, dev)                      # (1, N, 1, rd)
        rd = cos.shape[-1]
        qpos = torch.empty(N, dtype=torch.int64); kpos = torch.empty(N, dtype=torch.int64)
        gw = [[] for _ in range(self.nb)]
        cu_q, cu_k = [0], [0]
        Tp = ((T + st + tb - 1) // tb) * tb; Hp = ((H + s + ws - 1) // ws) * ws; Wp = ((W + s + ws - 1) // ws) * ws
        for j in range(Tp // tb):
            t_lo, t_hi = max(j * tb - st, 0), min((j + 1) * tb - st, T)
            if t_hi <= t_lo: continue
            blocks = sorted({t_lo // tb, (t_hi - 1) // tb})
            for i in range(Hp // ws):
                y_lo, y_hi = max(i * ws - s, 0), min((i + 1) * ws - s, H)
                if y_hi <= y_lo: continue
                for k in range(Wp // ws):
                    x_lo, x_hi = max(k * ws - s, 0), min((k + 1) * ws - s, W)
                    if x_hi <= x_lo: continue
                    tt, yy, xx = torch.meshgrid(torch.arange(t_lo, t_hi), torch.arange(y_lo, y_hi), torch.arange(x_lo, x_hi), indexing="ij")
                    toks = (tt * HW + yy * W + xx).flatten(); n = len(toks)
                    qpos[toks] = torch.arange(n) + cu_q[-1]
                    kpos[toks] = torch.arange(n) + cu_k[-1]
                    for jj, b in enumerate(blocks):
                        gw[b].append(cu_k[-1] + n + jj * S)
                    cu_q.append(cu_q[-1] + n); cu_k.append(cu_k[-1] + n + len(blocks) * S)
        assert cu_q[-1] == N and len(torch.unique(qpos)) == N
        maxw = max(len(x) for x in gw)
        gwin = torch.full((self.nb, maxw), -1, dtype=torch.int64)
        for b, x in enumerate(gw):
            gwin[b, :len(x)] = torch.tensor(x)
        i32 = dict(dtype=torch.int32, device=dev)
        src = torch.cat([qpos, torch.arange(N, self.M)])
        lens_q = [b - a for a, b in zip(cu_q, cu_q[1:])]; lens_k = [b - a for a, b in zip(cu_k, cu_k[1:])]
        return {"cos": cos[0, :, 0].to(BF16).float().contiguous(), "sin": sin[0, :, 0].to(BF16).float().contiguous(), "rd": rd,
                "qpos": qpos.to(**i32), "kpos": kpos.to(**i32), "src": src.to(**i32), "gwin": gwin.to(**i32), "maxw": maxw,
                "cu_q": torch.tensor(cu_q, **i32), "cu_k": torch.tensor(cu_k, **i32), "total_k": cu_k[-1],
                "max_q": max(lens_q), "max_k": max(lens_k)}

    @torch.no_grad()
    def _step(self):
        r, T, H, W, N, M = self.r, self.T, self.H, self.W, self.N, self.M
        with torch.autocast("cuda", dtype=BF16):
            raw = self.z_in * r.latents_std.view(1, 1, -1, 1, 1) + r.latents_mean.view(1, 1, -1, 1, 1)
            raw = r.post_quant_conv(raw.transpose(1, 2))
        e = r.x_embedder(raw.permute(0, 2, 3, 4, 1).float()).reshape(N, -1)
        self.x[:N].copy_(e.to(BF16))
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
            if li < self.L - 1:                     # the last layer's global tokens are never read
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
                _resid_ln_split_kernel[(N,)](self.x, self.ybuf, lw["s2"], self.ln_w, self.ln_b, self.hsplit, D=self.D, EPS=self.ln_eps, num_warps=8)
        torch.mm(self.hsplit, self.whead.t(), out_dtype=torch.float32, out=self.G)

    @torch.no_grad()
    def decode(self, z):
        """z: normalized latents (1, T, 24, H, W), t = 0 a block start -> decode_base pixels (1, 3, F, 16H, 16W)."""
        assert tuple(z.shape) == tuple(self.z_in.shape)
        self.z_in.copy_(z)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step()
        H, W = self.H, self.W
        out = torch.empty(1, OUT_CH, self.NF, H * PATCH, W * PATCH, device=self.dev)
        _nc_unpatch_kernel[(self.T * H * OUT_CH * PATCH_T * PATCH,)](self.G, self.bhead, out, H, W, self.NF, self.r.block,
                                                                    self.r.frames_to_trim, BLOCK=triton.next_power_of_2(W * PATCH), num_warps=4)
        return out
