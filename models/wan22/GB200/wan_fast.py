"""Owned forward for diffusers' WanTransformer3DModel (Wan2.2 TI2V-5B and T2V-A14B), single GB200.

Every denoising step still runs the full block stack for both CFG branches; nothing is skipped.

    lossless-to-rounding (both precisions)
      * CFG batching: the cond call computes [cond, uncond] as one batch-2 forward; the uncond call
        that follows returns the stashed half.
      * per-request caches: the text projection and every block's cross-attention K/V depend only on
        the prompt embeddings, so they are computed once per request instead of once per step.
      * TI2V-5B's per-token timesteps are all equal for text-to-video; the per-token (S, 6, D)
        modulation is collapsed to one row per batch item (identical values).
      * fused Triton kernels (wan_kern): residual-gate + LayerNorm + modulate, q/k RMSNorm + RoPE,
        GELU-tanh, with the next block's input norm absorbing the previous block's FFN residual.
    precision
      * precision="fp8": every block linear as MXFP8 (E4M3, E8M0 per 32); the norm / GELU kernels
        emit the quantized activation directly.
    attention
      * attention="sol": Sol-Attn (tau 1.0, diag) over a model-level Morton3D token order, dense for
        the first `dense_steps` denoising steps and the layers in `dense_layers`.
"""
import math
import os

import torch
import torch.nn.functional as F

import wan_kern


class SolPolicy:
    def __init__(self, enabled, tau=1.0, thresh_type="diag", dense_steps=10, dense_layers=(0,)):
        self.enabled = enabled
        self.tau, self.thresh_type = tau, thresh_type
        self.dense_steps, self.dense_layers = dense_steps, set(dense_layers)
        self.step = -1
        self.sparse_calls = self.dense_calls = 0
        self._kernel = None

    def kernel(self):
        if self._kernel is None:
            from sol_attn import sol_attn
            self._kernel = sol_attn
        return self._kernel


def _dense_attn(q, k, v):
    """(B, S, H, D) in and out; cuDNN SDPA on Blackwell."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


class _CudnnFP8Attn:
    """cuDNN FP8 SDPA (cudnn-frontend sdpa_fp8): Q, K, V as E4M3 with per-tensor dynamic scales, softmax
    probabilities scaled by 448, BF16 output. One graph per shape."""

    def __init__(self):
        self.graphs = {}

    def _graph(self, B, S, H, D):
        key = (B, S, H, D)
        if key not in self.graphs:
            import cudnn
            dims, strides = [B, H, S, D], [S * H * D, D, H * D, 1]
            g = cudnn.pygraph(io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT,
                              compute_data_type=cudnn.data_type.FLOAT)
            t = {n: g.tensor(name=n, dim=dims, stride=strides, data_type=cudnn.data_type.FP8_E4M3) for n in "qkv"}
            sc = {n: g.tensor(name=n, dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
                  for n in ("dq", "dk", "dv", "ds", "ss", "so")}
            o, _, amax_s, amax_o = g.sdpa_fp8(q=t["q"], k=t["k"], v=t["v"], descale_q=sc["dq"], descale_k=sc["dk"],
                                              descale_v=sc["dv"], descale_s=sc["ds"], scale_s=sc["ss"],
                                              scale_o=sc["so"], is_inference=True, attn_scale=1.0 / math.sqrt(D),
                                              use_causal_mask=False)
            o.set_output(True).set_dim(dims).set_stride(strides).set_data_type(cudnn.data_type.BFLOAT16)
            for a in (amax_s, amax_o):
                a.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            g.check_support()
            g.build_plans()
            dev = "cuda"
            consts = {"ds": torch.full((1, 1, 1, 1), 1.0 / 448.0, device=dev),
                      "ss": torch.full((1, 1, 1, 1), 448.0, device=dev),
                      "so": torch.ones((1, 1, 1, 1), device=dev),
                      "amax": torch.zeros((2, 1, 1, 1, 1), device=dev)}
            ws = torch.empty(g.get_workspace_size(), device=dev, dtype=torch.uint8)
            self.graphs[key] = (g, t, sc, o, amax_s, amax_o, consts, ws)
        return self.graphs[key]

    def __call__(self, q, k, v):
        B, S, H, D = q.shape
        g, t, sc, o, amax_s, amax_o, c, ws = self._graph(B, S, H, D)
        pack = {sc["ds"]: c["ds"], sc["ss"]: c["ss"], sc["so"]: c["so"], amax_s: c["amax"][0], amax_o: c["amax"][1]}
        for name, x in (("q", q), ("k", k), ("v", v)):
            pack[t[name]], pack[sc["d" + name]] = wan_kern.fp8_quant_pertensor(x.contiguous())
        out = torch.empty_like(q)
        pack[o] = out
        g.execute(pack, ws)
        return out


# Multi-GPU layout: CFG parallel 2 (cond on the first half of the ranks, uncond on the second) x Ulysses
# over each half.  Filled by init_parallel(); single-GPU defaults otherwise.
_PG = dict(P=1, rank=0, group=None, cfg=1, cfg_rank=0, cfg_group=None)


def init_parallel(cfg_parallel=True):
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return dict(_PG)
    W, r = dist.get_world_size(), dist.get_rank()
    cfg = 2 if cfg_parallel and W % 2 == 0 else 1
    U = W // cfg
    for c in range(cfg):                       # every rank must create every group, in the same order
        g = dist.new_group(list(range(c * U, (c + 1) * U)))
        if r // U == c:
            _PG.update(P=U, rank=r % U, group=g)
    if cfg == 2:
        for i in range(U):
            g = dist.new_group([i, i + U])
            if r % U == i:
                _PG.update(cfg_group=g)
    _PG.update(cfg=cfg, cfg_rank=r // U)
    return {k: v for k, v in _PG.items() if "group" not in k}


_SYMM = {}


def _symm():
    """Symmetric-memory Ulysses for this rank's group (WAN_SYMM=1, the default), else None."""
    if os.environ.get("WAN_SYMM", "1") != "1" or _PG["P"] < 2:
        return None
    if "x" not in _SYMM:
        from wan_symm import SymmUlysses
        _SYMM["x"] = SymmUlysses(_PG["group"], _PG["P"], _PG["rank"])
    return _SYMM["x"]


def _sp():
    """(size, rank, group) of this rank's Ulysses group."""
    return _PG["P"], _PG["rank"], _PG["group"]


def _halo(x, hdim, ph):
    """Extend this rank's row shard by `ph` neighbour rows on each side (zeros at the image edge, which is
    what the conv's own zero padding would have supplied)."""
    import torch.distributed as dist
    W, r = dist.get_world_size(), dist.get_rank()
    top = x.narrow(hdim, 0, ph).contiguous()
    bot = x.narrow(hdim, x.size(hdim) - ph, ph).contiguous()
    from_up, from_down = torch.zeros_like(top), torch.zeros_like(bot)
    ops = []
    if r > 0:
        ops += [dist.P2POp(dist.isend, top, r - 1), dist.P2POp(dist.irecv, from_up, r - 1)]
    if r < W - 1:
        ops += [dist.P2POp(dist.isend, bot, r + 1), dist.P2POp(dist.irecv, from_down, r + 1)]
    for q in dist.batch_isend_irecv(ops):
        q.wait()
    return torch.cat([from_up, x, from_down], hdim)


def _gather_rows(x, hdim, heights):
    """All-gather row shards of uneven height `heights` (per rank) along `hdim`."""
    import torch.distributed as dist
    hmax = max(heights)
    if x.size(hdim) < hmax:
        pad = [0, 0] * (x.dim() - 1 - hdim) + [0, hmax - x.size(hdim)]
        x = F.pad(x, pad)
    buf = torch.empty((len(heights),) + tuple(x.shape), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(buf, x.contiguous())
    return torch.cat([buf[i].narrow(hdim, 0, h) for i, h in enumerate(heights)], hdim)


def install_parallel_vae(vae):
    """Exact row-sharded VAE decode (the SGLang scheme): each rank decodes a contiguous band of latent rows;
    3x3 convs exchange one halo row with their neighbours, the mid-block frame attention runs on gathered
    rows, and the bands are gathered at the end.  Every other decoder op is per pixel or per row, so the
    result equals the single-GPU untiled decode up to conv algorithm choice."""
    import types
    import torch.distributed as dist
    import torch.nn as nn
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock, WanCausalConv3d

    W, r = dist.get_world_size(), dist.get_rank()
    st = {"lat": None}                                   # latent row counts per rank, set per decode

    def heights_at(local_h):                             # per-rank heights at the current resolution
        f = local_h // st["lat"][r]
        return [h * f for h in st["lat"]]

    def conv3d_fwd(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = _halo(x, 3, padding[2])
        padding[2] = padding[3] = 0
        return nn.Conv3d.forward(self, F.pad(x, padding))

    def conv2d_fwd(self, x):
        ph, pw = self.padding
        x = F.pad(_halo(x, 2, ph), (pw, pw, 0, 0))
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

    def attn_fwd(self, x, _orig=None):
        full = _orig(_gather_rows(x, 3, heights_at(x.size(3))))
        off = sum(heights_at(x.size(3))[:r])
        return full.narrow(3, off, x.size(3))

    n = {"conv3d": 0, "conv2d": 0, "attn": 0}
    for m in vae.decoder.modules():
        if isinstance(m, WanCausalConv3d) and m._padding[2] > 0:
            m.forward = types.MethodType(conv3d_fwd, m); n["conv3d"] += 1
        elif type(m) is nn.Conv2d and m.padding[0] > 0:
            assert m.stride == (1, 1)
            m.forward = types.MethodType(conv2d_fwd, m); n["conv2d"] += 1
        elif isinstance(m, WanAttentionBlock):
            orig = m.forward
            m.forward = lambda x, _o=orig, _m=m: attn_fwd(_m, x, _orig=_o); n["attn"] += 1

    orig_decode = vae._decode
    vae.disable_tiling()

    def _decode(z, return_dict=True):
        H = z.shape[3]
        st["lat"] = [H // W + (1 if i < H % W else 0) for i in range(W)]
        off = sum(st["lat"][:r])
        out = orig_decode(z[:, :, :, off:off + st["lat"][r]].contiguous(), return_dict=False)[0]
        out = _gather_rows(out, 3, heights_at(out.size(3)))
        from diffusers.models.autoencoders.vae import DecoderOutput
        return DecoderOutput(sample=out) if return_dict else (out,)

    vae._decode = _decode
    return {"vae_parallel": W, "vae_scheme": "row-shard+halo", **{f"vae_{k}": v for k, v in n.items()}}


def _a2a_to_heads(x, P, group):
    """(B, S/P, H, D) token shard -> (B, S, H/P, D) head shard (Ulysses)."""
    import torch.distributed as dist
    B, Sl, H, Dh = x.shape
    x = x.view(B, Sl, P, H // P, Dh).permute(2, 0, 1, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out.permute(1, 0, 2, 3, 4).reshape(B, P * Sl, H // P, Dh)


_A2A_EV = []          # (start, end) CUDA events per all-to-all when WAN_A2A_TIMING=1


def _timed(fn):
    if os.environ.get("WAN_A2A_TIMING", "0") != "1":
        return fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    _A2A_EV.append((s, e))
    return out


def a2a_ms_and_reset():
    torch.cuda.synchronize()
    ms = sum(s.elapsed_time(e) for s, e in _A2A_EV)
    n = len(_A2A_EV)
    _A2A_EV.clear()
    return ms, n


def _a2a_qkv_to_heads(q, k, v, P, group):
    """q, k, v (B, S/P, H, D) token shards -> three (B, S, H/P, D) head shards with ONE all-to-all."""
    import torch.distributed as dist
    B, Sl, H, Dh = q.shape
    x = torch.stack((q, k, v)).view(3, B, Sl, P, H // P, Dh).permute(3, 0, 1, 2, 4, 5).contiguous()
    out = torch.empty_like(x)
    _timed(lambda: dist.all_to_all_single(out, x, group=group))
    out = out.permute(1, 2, 0, 3, 4, 5).reshape(3, B, P * Sl, H // P, Dh)
    return out[0], out[1], out[2]


def _a2a_to_tokens(y, P, group):
    """(B, S, H/P, D) head shard -> (B, S/P, H, D) token shard."""
    import torch.distributed as dist
    B, S, Hp, Dh = y.shape
    Sl = S // P
    y = y.reshape(B, P, Sl, Hp, Dh).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(y)
    _timed(lambda: dist.all_to_all_single(out, y, group=group))
    return out.permute(1, 2, 0, 3, 4).reshape(B, Sl, P * Hp, Dh)


def _linear(mod, x):
    return mod(x)


def _to_fp8(blocks):
    import mxfp8
    for blk in blocks:
        a1, a2 = blk.attn1, blk.attn2
        a1.to_q, a1.to_k, a1.to_v = (mxfp8.MXFP8Linear(m) for m in (a1.to_q, a1.to_k, a1.to_v))
        a1.to_out[0] = mxfp8.MXFP8Linear(a1.to_out[0])
        a2.to_q = mxfp8.MXFP8Linear(a2.to_q)
        a2.to_out[0] = mxfp8.MXFP8Linear(a2.to_out[0])
        blk.ffn.net[0].proj = mxfp8.MXFP8Linear(blk.ffn.net[0].proj)
        blk.ffn.net[2] = mxfp8.MXFP8Linear(blk.ffn.net[2])
    torch.cuda.empty_cache()


class WanFast:
    """Holds the per-transformer state; `forward(x, t, enc)` with x (B,C,F,H,W), t (B,), enc (B,L,4096)."""

    def __init__(self, tr, precision="bf16", sol: SolPolicy | None = None):
        self.tr = tr
        self.fp8 = precision == "fp8"
        self.sol = sol or SolPolicy(False)
        if self.fp8:
            _to_fp8(tr.blocks)
        self._rope_key = None
        self._text_key = None
        self.heads = tr.blocks[0].attn1.heads
        self.dim = tr.blocks[0].attn1.inner_dim
        self.head_dim = self.dim // self.heads
        self.eps = tr.blocks[0].norm1.eps
        # fp32 copies of the modulation tables, made once
        self.tables = [blk.scale_shift_table.detach().float() for blk in tr.blocks]
        import os
        self.gelu_epilogue = os.environ.get("WAN_GELU_EPILOGUE", "1") == "1"
        self.fp8_attn = _CudnnFP8Attn() if os.environ.get("WAN_FP8_ATTN", "0") == "1" else None
        self.out_table = tr.scale_shift_table.detach().float()

    # -- per-shape / per-request caches --------------------------------------------------------
    def _rope(self, x):
        key = tuple(x.shape[2:])
        if key != self._rope_key:
            fc, fs = self.tr.rope(x)                         # (1, S, 1, head_dim) float64
            cos = fc[0, :, 0, 0::2].float().contiguous()     # pair i uses cos[2i] == cos[2i+1]
            sin = fs[0, :, 0, 1::2].float().contiguous()
            self.perm = self.inv = None
            if self.sol.enabled:
                from sol_attn_backend import _morton3d_perm
                p_t, p_h, p_w = self.tr.config.patch_size
                grid = (x.shape[2] // p_t, x.shape[3] // p_h, x.shape[4] // p_w)
                self.perm, self.inv = _morton3d_perm(grid, x.device)
                cos, sin = cos.index_select(0, self.perm).contiguous(), sin.index_select(0, self.perm).contiguous()
            self.cos, self.sin, self._rope_key = cos, sin, key
        return self.cos, self.sin

    def _text(self, enc):
        key = (enc.data_ptr(), tuple(enc.shape))
        if key != self._text_key:
            tr = self.tr
            ctx = tr.condition_embedder.text_embedder(enc)                  # (B, L, D)
            B, L, _ = ctx.shape
            flat = ctx.reshape(B * L, -1)
            self.cross = []
            for blk in tr.blocks:
                a2 = blk.attn2
                k = a2.to_k(flat)
                v = a2.to_v(flat)
                k, _ = wan_kern.qk_rms_rope(k.contiguous(), None, a2.norm_k.weight, None, None, None, L,
                                            a2.norm_k.eps, rope=False)
                self.cross.append((k.view(B, L, self.heads, self.head_dim),
                                   v.view(B, L, self.heads, self.head_dim)))
            self._text_key = key
        return self.cross

    # -- attention cores -----------------------------------------------------------------------
    def _self_attn(self, q, k, v, layer):
        sol = self.sol
        dump = os.environ.get("WAN_DUMP_QKV")       # "step,layer,path": save real q/k/v once for analysis
        if dump:
            st, ly, path = dump.split(",")
            if sol.step == int(st) and layer == int(ly) and not os.path.exists(path):
                torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu(), "step": sol.step, "layer": layer,
                            "perm": None if self.perm is None else self.perm.cpu()}, path)
        if sol.enabled and sol.step >= sol.dense_steps and layer not in sol.dense_layers:
            sol.sparse_calls += 1
            if os.environ.get("WAN_SPARSE_IMPL", "sol") == "fa4sol":   # Sol routing on FA4 block-sparse tiles
                if getattr(self, "_fa4sol", None) is None:
                    from sol_fa4 import FA4Sol
                    self._fa4sol = FA4Sol()
                return self._fa4sol(q, k, v)
            return sol.kernel()(q, k, v, tau=sol.tau, thresh_type=sol.thresh_type, kv_splits=1)
        sol.dense_calls += 1
        if self.fp8_attn is not None:
            return self.fp8_attn(q, k, v)
        return _dense_attn(q, k, v)

    # -- forward -------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x, t, enc):
        tr = self.tr
        B = x.shape[0]
        p_t, p_h, p_w = tr.config.patch_size
        Fp, Hp, Wp = x.shape[2] // p_t, x.shape[3] // p_h, x.shape[4] // p_w
        S = Fp * Hp * Wp
        cos, sin = self._rope(x)
        cross = self._text(enc)

        h = tr.patch_embedding(x).flatten(2).transpose(1, 2)               # (B, S, D)
        if self.perm is not None:
            h = h.index_select(1, self.perm)
        P, rank, group = _sp()
        S_full = S
        if P > 1:                                                          # Ulysses: keep my token chunk
            if S % P:
                raise RuntimeError(f"sequence {S} not divisible by {P} ranks")
            S = S // P
            h = h[:, rank * S:(rank + 1) * S]
            cos, sin = cos[rank * S:(rank + 1) * S], sin[rank * S:(rank + 1) * S]
        h = h.contiguous().view(B * S, -1)

        ce = tr.condition_embedder
        tp = ce.timesteps_proj(t)
        tdt = next(iter(ce.time_embedder.parameters())).dtype
        temb = ce.time_embedder(tp.to(tdt) if tp.dtype != tdt else tp).type_as(enc)    # (B, D)
        tproj = ce.time_proj(ce.act_fn(temb)).unflatten(1, (6, -1)).float()             # (B, 6, D)

        H, Dh, eps = self.heads, self.head_dim, self.eps
        fp8 = self.fp8
        pending = None
        for i, blk in enumerate(tr.blocks):
            mod = (self.tables[i] + tproj).contiguous()                         # (B, 6, D) fp32
            shift, scale, gate, c_shift, c_scale, c_gate = (mod[:, j] for j in range(6))
            if pending is None:
                _, n1 = wan_kern.ln_mod(h, S, eps, scale=scale, shift=shift, out_fp8=fp8)
            else:
                h, n1 = wan_kern.ln_mod(h, S, eps, branch=pending[0], gate=pending[1], scale=scale, shift=shift,
                                        out_fp8=fp8)
            a1 = blk.attn1
            q, k, v = a1.to_q(n1), a1.to_k(n1), a1.to_v(n1)
            q, k = wan_kern.qk_rms_rope_v2(q, k, a1.norm_q.weight, a1.norm_k.weight, cos, sin, S, a1.norm_q.eps,
                                        head_dim=Dh)
            if P > 1 and B == 1 and _symm() is not None:
                o = _symm()(q, k, v, lambda qa, ka, va, i=i: self._self_attn(qa, ka, va, i))
            elif P > 1:
                if os.environ.get("WAN_A2A_PACKED", "0") == "1":
                    qa, ka, va = _a2a_qkv_to_heads(q.view(B, S, H, Dh), k.view(B, S, H, Dh), v.view(B, S, H, Dh),
                                                   P, group)
                else:
                    qa, ka, va = (_timed(lambda t=t: _a2a_to_heads(t.view(B, S, H, Dh), P, group))
                                  for t in (q, k, v))
                o = _a2a_to_tokens(self._self_attn(qa, ka, va, i), P, group)
            else:
                o = self._self_attn(q.view(B, S, H, Dh), k.view(B, S, H, Dh), v.view(B, S, H, Dh), i)
            o = o.reshape(B * S, H * Dh)
            o = a1.to_out[0](wan_kern.mx_quant(o) if fp8 else o)
            n2w = blk.norm2
            h, n2 = wan_kern.ln_mod(h, S, n2w.eps, branch=o, gate=gate, weight=n2w.weight, bias=n2w.bias,
                                    out_fp8=fp8)
            a2 = blk.attn2
            q2 = a2.to_q(n2)
            q2, _ = wan_kern.qk_rms_rope_v2(q2, None, a2.norm_q.weight, None, None, None, S, a2.norm_q.eps, rope=False)
            k2, v2 = cross[i]
            o2 = _dense_attn(q2.view(B, S, H, Dh), k2, v2).reshape(B * S, H * Dh)
            o2 = a2.to_out[0](wan_kern.mx_quant(o2) if fp8 else o2)
            h, n3 = wan_kern.ln_mod(h, S, eps, branch=o2, scale=c_scale, shift=c_shift, out_fp8=fp8)
            pj = blk.ffn.net[0].proj
            if not fp8 and self.gelu_epilogue:
                # GELU-tanh in the GEMM epilogue (cuBLASLt/CUTLASS): the pre-activation never reaches HBM
                act = torch._addmm_activation(pj.bias, n3, pj.weight.t(), use_gelu=True)
            else:
                act = wan_kern.gelu_tanh(pj(n3), out_fp8=fp8)
            ff = blk.ffn.net[2](act)
            pending = (ff, c_gate)

        # last block's FFN residual, then the output norm (no affine) with its own modulation
        omod = (self.out_table + temb.float().unsqueeze(1)).contiguous()        # (B, 2, D): shift, scale
        h, hn = wan_kern.ln_mod(h, S, tr.norm_out.eps, branch=pending[0], gate=pending[1],
                                scale=omod[:, 1], shift=omod[:, 0])
        out = tr.proj_out(hn).view(B, S, -1)
        if P > 1:
            import torch.distributed as dist
            full = torch.empty((P,) + tuple(out.shape), dtype=out.dtype, device=out.device)
            dist.all_gather_into_tensor(full, out.contiguous(), group=group)
            out = full.permute(1, 0, 2, 3).reshape(B, S_full, -1)
            S = S_full
        if self.inv is not None:
            out = out.index_select(1, self.inv)
        out = out.reshape(B, Fp, Hp, Wp, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return out.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def install_pipeline(pipe, precision="bf16", attention="dense", dense_steps=10, dense_layers=(0,), tau=1.0):
    """Replace every transformer's forward in a WanPipeline with the batched-CFG fast forward."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    sol = SolPolicy(attention == "sol", tau=tau, dense_steps=dense_steps, dense_layers=dense_layers)
    state = {"pos": None, "neg": None, "stash": None, "enc2": None}
    models = [m for m in (pipe.transformer, getattr(pipe, "transformer_2", None)) if m is not None]
    fasts = {id(m): WanFast(m, precision, sol) for m in models}

    original_encode = pipe.encode_prompt

    def encode_prompt(*a, **kw):
        pe, ne = original_encode(*a, **kw)
        state.update(pos=pe, neg=ne, stash=None, enc2=None, ts_checked=False)
        sol.step = -1
        for f in fasts.values():
            f._text_key = None          # a new request may reuse the old embeddings' address
        return pe, ne

    pipe.encode_prompt = encode_prompt

    def scalar_t(timestep, B):
        if timestep.ndim == 2:                       # TI2V per-token timesteps (all equal for T2V)
            t0 = timestep[:, :1]
            if not state.get("ts_checked"):         # one host sync per request, not per step
                if not bool((timestep == t0).all()):
                    raise RuntimeError("per-token timesteps differ; image-conditioned TI2V is not served")
                state["ts_checked"] = True
            timestep = t0[:, 0]
        return timestep.expand(B) if timestep.numel() == 1 else timestep

    def make(m):
        fast = fasts[id(m)]

        def forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=None,
                    return_dict=True, attention_kwargs=None):
            if encoder_hidden_states_image is not None:
                raise RuntimeError("image embeddings are not served")
            wrap = (lambda o: Transformer2DModelOutput(sample=o)) if return_dict else (lambda o: (o,))
            st = state["stash"]
            if st is not None and st[0] is hidden_states and encoder_hidden_states is state["neg"]:
                state["stash"] = None
                return wrap(st[1])
            if encoder_hidden_states is state["pos"] and state["neg"] is not None \
                    and state["neg"].shape == state["pos"].shape:
                sol.step += 1
                if _PG["cfg"] == 2:                  # CFG parallel: my half computes one branch
                    import torch.distributed as dist
                    enc = state["pos"] if _PG["cfg_rank"] == 0 else state["neg"]
                    o = fast.forward(hidden_states, scalar_t(timestep, 1), enc).contiguous()
                    out = torch.empty((2,) + tuple(o.shape[1:]), dtype=o.dtype, device=o.device)
                    dist.all_gather_into_tensor(out, o, group=_PG["cfg_group"])
                    state["stash"] = (hidden_states, out[1:2])
                    return wrap(out[0:1])
                if state["enc2"] is None:
                    state["enc2"] = torch.cat([state["pos"], state["neg"]])
                x2 = torch.cat([hidden_states, hidden_states])
                out = fast.forward(x2, scalar_t(timestep, 1).expand(2), state["enc2"])
                state["stash"] = (hidden_states, out[1:2])
                return wrap(out[0:1])
            return wrap(fast.forward(hidden_states, scalar_t(timestep, hidden_states.shape[0]),
                                     encoder_hidden_states))
        return forward

    for m in models:
        m.forward = make(m)
    return {"precision": precision, "attention": attention, "sol": sol, "models": len(models)}
