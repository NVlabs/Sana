"""Deployment decoder for CausalViTDecoder: same weights, same math as decode_chunk (batch 1, fixed H x W), fast kernels.

* window attention: FlashAttention-4 (CuTe DSL, Blackwell) varlen. Every window is one short sequence; padding
  positions are simply not packed, so no mask is needed. Queries: the chunk's valid tokens of the window.
* K/V cache: one packed pool per layer, per window [suffix keys | ring slot 0 .. slot nslot-1] (valid tokens only).
  RoPE is applied once when a latent enters the ring; each chunk writes its own slots in place. At stream start the
  written slots are a prefix of the segment, so `seqused_k` cuts off the empty ones.
* suffix (register) queries attend every valid cached/current token once plus the suffix keys (window 0 copy):
  a small matmul + softmax over the pool with a bias for empty slots and the duplicate suffix copies.
* block linear weights are held in bf16 (no per-call autocast casts); the residual stream is fp32 as in the
  reference (bf16 branch output times fp32 residual scale promotes to fp32).
* the whole chunk step is captured in one CUDA graph; every chunk has the same shapes.
"""
from __future__ import annotations

import copy
import site
import sys

import torch
from torch import nn

if "cutlass" not in sys.modules:                      # the cutlass DSL is put on sys.path by a .pth file
    for p in list(sys.path):
        if p.endswith("site-packages"):
            site.addsitedir(p)
from flash_attn.cute.interface import flash_attn_varlen_func

from causal_vit_decoder import OUT_CH, PATCH, PATCH_T, CausalViTDecoder
from dev.yanzuolu.projects.minimax_h3.modeling.video_vae.vit_utils import _apply_rotary_pos_emb_impl

torch._dynamo.config.cache_size_limit = 128
BF16 = torch.bfloat16


def _attn_pre(blk, x, suf, cos, sin, scos, ssin, qperm, heads: int, dh: int):
    """Norm, qkv, qk-norm, RoPE; tokens reordered window by window. x: (N, D) fp32, suf: (S, D) fp32."""
    N = x.shape[0]
    q, k, v = blk.attn.to_qkv(blk.norm1(x).to(BF16)).view(N, heads, 3 * dh).chunk(3, -1)
    qs, ks, vs = blk.attn.to_qkv(blk.norm1(suf).to(BF16)).view(-1, heads, 3 * dh).chunk(3, -1)
    if blk.attn.norm_q is not None:
        q, qs = blk.attn.norm_q(q.float()).to(BF16), blk.attn.norm_q(qs.float()).to(BF16)
        k, ks = blk.attn.norm_k(k.float()).to(BF16), blk.attn.norm_k(ks.float()).to(BF16)
    q = _apply_rotary_pos_emb_impl(q.unsqueeze(0), (cos, sin))[0]
    k = _apply_rotary_pos_emb_impl(k.unsqueeze(0), (cos, sin))[0]
    qs = _apply_rotary_pos_emb_impl(qs.unsqueeze(0), (scos, ssin))[0]      # suffix: time of the chunk's first latent
    ks = _apply_rotary_pos_emb_impl(ks.unsqueeze(0), (scos, ssin))[0]
    return q[qperm].contiguous(), k[qperm].contiguous(), v[qperm].contiguous(), qs.contiguous(), ks.contiguous(), vs.contiguous()


def _attn_post(blk, x, suf, O, kpool, vpool, qs, kbias, inv, heads: int, dh: int):
    """Merge window outputs, suffix attention over the pool, output projection, feed-forward."""
    o = O.reshape(-1, heads * dh)[inv]                                                    # N, D in token order
    sc = torch.matmul(qs.transpose(0, 1), kpool.permute(1, 2, 0)).float() * dh ** -0.5   # heads, S, total_k
    p = (sc + kbias).softmax(-1).to(BF16)
    os_ = torch.matmul(p, vpool.transpose(0, 1)).transpose(0, 1).reshape(-1, heads * dh)  # S, D
    x = x + blk.attn.to_out(o).float() * blk.scale1
    suf = suf + blk.attn.to_out(os_).float() * blk.scale1
    x = x + blk.ff(blk.norm2(x).to(BF16)).float() * blk.scale2
    suf = suf + blk.ff(blk.norm2(suf).to(BF16)).float() * blk.scale2
    return x, suf


class StreamingCausalViT(nn.Module):
    def __init__(self, ref: CausalViTDecoder, H: int, W: int, *, cuda_graph: bool = True):
        super().__init__()
        r = ref
        self.r, self.H, self.W, self.C = r, H, W, r.chunk
        self.heads, self.dh = r.heads, r.dh
        assert r.cache_latents % r.chunk == 0
        self.nslot = r.cache_latents + r.chunk
        self.S = r.register_tokens.shape[1] + 1
        dev = r.register_tokens.device
        self.blocks = copy.deepcopy(r.blocks)
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                m.to(BF16)
        self.blocks.requires_grad_(False)
        self.shifts = [(r.window // 2) if (r.shift and li % 2 == 1) else 0 for li in range(len(self.blocks))]
        self.geo = {s: self._geometry(s, dev) for s in sorted(set(self.shifts))}
        self.kpool = [torch.zeros(self.geo[s]["total_k"], self.heads, self.dh, dtype=BF16, device=dev) for s in self.shifts]
        self.vpool = [torch.zeros_like(k) for k in self.kpool]
        self.slot_bias = torch.zeros(self.nslot + 1, device=dev)
        self.z_in = torch.zeros(1, self.C, r.latents_mean.numel(), H, W, device=dev)
        self.cos, self.sin, self.scos, self.ssin = {}, {}, None, None
        self.out = None
        self._pre = torch.compile(_attn_pre, dynamic=False)
        self._post = torch.compile(_attn_post, dynamic=False)
        self.cuda_graph, self.graph = cuda_graph, None
        self.init_state()

    def _geometry(self, s, dev):
        H, W, C, ws, S, ns = self.H, self.W, self.C, self.r.window, self.S, self.nslot
        Hp, Wp = ((H + s + ws - 1) // ws) * ws, ((W + s + ws - 1) // ws) * ws
        nh, nw = Hp // ws, Wp // ws
        yy = torch.arange(Hp).view(nh, 1, ws, 1) - s
        xx = torch.arange(Wp).view(1, nw, 1, ws) - s
        yy, xx = torch.broadcast_tensors(yy, xx)
        valid = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).reshape(nh * nw, ws * ws)
        flat = (yy * W + xx).reshape(nh * nw, ws * ws)
        keepw = valid.any(1)                                    # drop windows that are entirely padding
        valid, flat = valid[keepw], flat[keepw]
        n = valid.sum(1)                                        # valid tokens per window
        nW = len(n)
        qperm, widx = [], {a: [] for a in range(0, ns, C)}
        seg = S + ns * n
        start = torch.cumsum(seg, 0) - seg
        for w in range(nW):
            toks = flat[w][valid[w]]
            for t in range(C):
                qperm.append(t * H * W + toks)
                for a in widx:
                    widx[a].append(start[w] + S + (a + t) * n[w] + torch.arange(n[w]))
        qperm = torch.cat(qperm)
        inv = torch.empty_like(qperm); inv[qperm] = torch.arange(len(qperm))
        total_k = int(seg.sum())
        row_slot = torch.full((total_k,), ns, dtype=torch.long)           # suffix rows -> slot_bias[ns] = 0
        dup = torch.zeros(total_k)
        for w in range(nW):
            row_slot[start[w] + S: start[w] + seg[w]] = torch.arange(ns).repeat_interleave(int(n[w]))
            if w > 0:
                dup[start[w]: start[w] + S] = float("-inf")            # suffix keys counted once (window 0)
        sidx = (start[:, None] + torch.arange(S)[None]).reshape(-1)
        i32 = dict(dtype=torch.int32, device=dev)
        return {"nW": nW, "n": n.to(dev), "qperm": qperm.to(dev), "inv": inv.to(dev),
                "widx": {a: v_.to(dev) for a, v_ in ((a, torch.cat(l)) for a, l in widx.items())},
                "sidx": sidx.to(dev), "total_k": total_k, "row_slot": row_slot.to(dev), "dup": dup.to(dev),
                "cu_q": torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(C * n, 0)]).to(**i32),
                "cu_k": torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(seg, 0)]).to(**i32),
                "max_q": int(C * n.max()), "max_k": int(seg.max()),
                "seqused": torch.zeros(nW, **i32), "widx_cur": torch.zeros(C * H * W, dtype=torch.long, device=dev),
                "kbias": torch.zeros(total_k, device=dev)}

    def init_state(self):
        for k, v in zip(self.kpool, self.vpool):
            k.zero_(); v.zero_()
        self.t = 0
        return self

    @torch.no_grad()
    def _step(self):
        r, C, H, W = self.r, self.C, self.H, self.W
        with torch.autocast("cuda", dtype=BF16):
            raw = self.z_in * r.latents_std.view(1, 1, -1, 1, 1) + r.latents_mean.view(1, 1, -1, 1, 1)
            raw = r.post_quant_conv(raw.transpose(1, 2))
        x = r.x_embedder(raw.permute(0, 2, 3, 4, 1).float()).to(BF16).float().reshape(C * H * W, -1)
        suf = torch.cat([r.register_tokens[0], r.register_tokens.new_zeros(1, x.shape[-1])], 0).to(BF16).float()
        for g in self.geo.values():
            g["kbias"].copy_(g["dup"] + self.slot_bias[g["row_slot"]])
        for li, blk in enumerate(self.blocks):
            s = self.shifts[li]; g = self.geo[s]
            Q, K, V, qs, ks, vs = self._pre(blk, x, suf, self.cos[s], self.sin[s], self.scos, self.ssin, g["qperm"], self.heads, self.dh)
            kp, vp = self.kpool[li], self.vpool[li]
            kp.index_copy_(0, g["widx_cur"], K); vp.index_copy_(0, g["widx_cur"], V)
            kp.index_copy_(0, g["sidx"], ks.unsqueeze(0).expand(g["nW"], -1, -1, -1).reshape(-1, self.heads, self.dh))
            vp.index_copy_(0, g["sidx"], vs.unsqueeze(0).expand(g["nW"], -1, -1, -1).reshape(-1, self.heads, self.dh))
            O = flash_attn_varlen_func(Q, kp, vp, cu_seqlens_q=g["cu_q"], cu_seqlens_k=g["cu_k"], max_seqlen_q=g["max_q"],
                                       max_seqlen_k=g["max_k"], seqused_k=g["seqused"])
            O = O[0] if isinstance(O, tuple) else O
            x, suf = self._post(blk, x, suf, O, kp, vp, qs, g["kbias"], g["inv"], self.heads, self.dh)
        out = r.proj_out(r.norm_out(x))
        out = out.view(C, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(3, 0, 4, 1, 5, 2, 6)
        return out.reshape(1, OUT_CH, C * PATCH_T, H * PATCH, W * PATCH)

    def _set_inputs(self, z):
        r, C = self.r, self.C
        self.z_in.copy_(z)
        a = self.t % self.nslot
        filled = min(self.t + C, self.nslot)                   # written slots form the prefix 0..filled-1 of the ring
        sb = torch.zeros(self.nslot + 1)
        sb[filled:self.nslot] = float("-inf")
        self.slot_bias.copy_(sb)
        for g in self.geo.values():
            g["widx_cur"].copy_(g["widx"][a])
            g["seqused"].copy_((self.S + filled * g["n"]).to(torch.int32))
        scos, ssin = r._suffix_ids(self.t, self.S, torch.float32, self.z_in.device)
        if self.scos is None:
            self.scos, self.ssin = scos.clone(), ssin.clone()
        else:
            self.scos.copy_(scos); self.ssin.copy_(ssin)
        for s in self.geo:
            cos, sin = r._ids(self.t, C, self.H, self.W, s, torch.float32, self.z_in.device)
            if s not in self.cos:
                self.cos[s], self.sin[s] = cos.clone(), sin.clone()
            else:
                self.cos[s].copy_(cos); self.sin[s].copy_(sin)

    @torch.no_grad()
    def decode_chunk(self, z):
        """z: normalized latents (1, C, 24, H, W). Returns decode_base-space pixels (1, 3, frames, 16H, 16W)."""
        assert z.shape[0] == 1 and z.shape[1] == self.C and tuple(z.shape[-2:]) == (self.H, self.W)
        self._set_inputs(z)
        if not self.cuda_graph or self.t < self.nslot:          # eager warm-up compiles every kernel first
            out = self._step()
        else:
            if self.graph is None:
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):              # capture records only; buffers untouched
                    self.out = self._step()
            self.graph.replay()
            out = self.out
        keep = [i for i in range(self.C * PATCH_T)
                if not ((self.t + i // PATCH_T) % self.r.block_latents == 0 and i % PATCH_T < self.r.frames_to_trim)]
        self.t += self.C
        return out[:, :, keep] if len(keep) < self.C * PATCH_T else out.clone()
