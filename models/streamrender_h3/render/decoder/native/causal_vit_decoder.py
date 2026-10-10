"""Causal streaming decoder built from (a subset of) the MiniMax H3 VAE ViT decoder.

Takes the released H3 VAE, keeps chosen transformer layers with their original
weights, and decodes a latent stream chunk by chunk:

* time: strictly chunk-causal. A chunk of `chunk` latents attends to itself and to
  the K/V of the previous `cache_latents` latents (per layer cache). Nothing from a
  later chunk is ever visible.
* space: local attention in `window` x `window` token windows (16 tokens = 256 px,
  the tile size the released VAE decodes with). Odd layers shift the window grid by
  half a window so neighbouring windows exchange information without overlapping tiles.
* suffix (4 register tokens + zero cls token): one global set per chunk. It attends to
  every token of the current chunk and the cached latents, and every window attends to
  it. It is rebuilt for each chunk, never cached, so it cannot leak future content.
* rotary positions: spatial ids are window-local and length-normalised by the window
  size, as in the released tile decode; the time id uses the absolute latent index with
  a fixed normaliser so cached keys keep their position. The suffix tokens carry the time
  id of the chunk's first latent (spatial ids 0), so every attention score depends only on
  time differences and a long stream never leaves the trained range. (With zero suffix ids,
  window-query x suffix-key scores depended on absolute time and block-start frames went
  dark from latent 10 on.)
* output: each token is projected to 4 frames x 16 x 16 pixels (released head); the
  first latent of every native 5-latent block drops its 3 alignment frames.

The output is in the same pixel space as `MiniMaxH3VideoVAE.decode_base`; apply
`vae.processor.revert_tensor` to get RGB.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from dev.yanzuolu.projects.minimax_h3.modeling.video_vae.attention import _apply_qk_norm
from dev.yanzuolu.projects.minimax_h3.modeling.video_vae.vit_utils import _apply_rotary_pos_emb_impl

PATCH, PATCH_T, OUT_CH = 16, 4, 3


class CausalViTDecoder(nn.Module):
    def __init__(self, vae, keep_layers, *, window: int = 16, chunk: int = 2, cache_latents: int = 4,
                 shift: bool = True, time_norm: float = 7.0, frames_to_trim: int = 3, block_latents: int = 5):
        super().__init__()
        dec = vae.decoder
        self.keep_layers = list(keep_layers)
        self.window, self.chunk, self.cache_latents, self.shift = window, chunk, cache_latents, shift
        self.time_norm, self.frames_to_trim, self.block_latents = time_norm, frames_to_trim, block_latents
        self.post_quant_conv = copy.deepcopy(vae.post_quant_conv)
        self.x_embedder = copy.deepcopy(dec.x_embedder)
        self.register_tokens = nn.Parameter(dec.register_tokens.detach().clone())
        self.blocks = nn.ModuleList([copy.deepcopy(dec.transformer_blocks[i]) for i in self.keep_layers])
        self.norm_out = copy.deepcopy(dec.norm_out)
        self.proj_out = copy.deepcopy(dec.proj_out)
        self.pos_embed = copy.deepcopy(dec.pos_embed)
        self.heads = self.blocks[0].attn.heads
        self.dh = self.blocks[0].attn.dim_head
        self.register_buffer("latents_mean", vae.latents_mean.detach().clone().float(), persistent=False)
        self.register_buffer("latents_std", vae.latents_std.detach().clone().float(), persistent=False)
        for p in self.pos_embed.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- helpers
    def init_state(self):
        return {"t": 0, "k": [None] * len(self.blocks), "v": [None] * len(self.blocks)}

    def _ids(self, t0, T, H, W, s, dtype, device):
        """Rotary ids (T*H*W, 3): absolute time (fixed normaliser), window-local h/w for shift s."""
        t = (torch.arange(t0, t0 + T, device=device, dtype=torch.float32) + 0.5) / self.time_norm * 2 - 1
        ly = ((torch.arange(H, device=device) + s) % self.window).float()
        lx = ((torch.arange(W, device=device) + s) % self.window).float()
        hy = (ly + 0.5) / self.window * 2 - 1
        wx = (lx + 0.5) / self.window * 2 - 1
        grid = torch.stack(torch.meshgrid(t, hy, wx, indexing="ij"), -1).reshape(1, -1, 3)
        cos, sin = self.pos_embed(grid.to(dtype))
        return cos, sin

    def _suffix_ids(self, t0, n, dtype, device):
        """Rotary ids for the n suffix tokens: time of the chunk's first latent, spatial 0."""
        grid = torch.zeros(1, n, 3, device=device, dtype=torch.float32)
        grid[..., 0] = (t0 + 0.5) / self.time_norm * 2 - 1
        return self.pos_embed(grid.to(dtype))

    def _partition(self, x, s):
        """x: (B, T, H, W, ...) -> windows (B*nW, T*ws*ws, ...), valid mask (nW, T*ws*ws), layout info."""
        B, T, H, W = x.shape[:4]
        ws = self.window
        ph0 = pw0 = s
        Hp = ((H + s + ws - 1) // ws) * ws
        Wp = ((W + s + ws - 1) // ws) * ws
        rest = x.shape[4:]
        xp = x.new_zeros(B, T, Hp, Wp, *rest)
        xp[:, :, ph0:ph0 + H, pw0:pw0 + W] = x
        valid = torch.zeros(Hp, Wp, dtype=torch.bool, device=x.device)
        valid[ph0:ph0 + H, pw0:pw0 + W] = True
        nh, nw = Hp // ws, Wp // ws
        xw = xp.view(B, T, nh, ws, nw, ws, *rest).permute(0, 2, 4, 1, 3, 5, *range(6, 6 + len(rest)))
        xw = xw.reshape(B * nh * nw, T * ws * ws, *rest)
        vw = valid.view(nh, ws, nw, ws).permute(0, 2, 1, 3).reshape(nh * nw, 1, ws * ws).expand(-1, T, -1).reshape(nh * nw, T * ws * ws)
        return xw, vw, (B, T, H, W, Hp, Wp, nh, nw)

    def _unpartition(self, xw, info, s):
        B, T, H, W, Hp, Wp, nh, nw = info
        ws = self.window
        rest = xw.shape[2:]
        x = xw.view(B, nh, nw, T, ws, ws, *rest).permute(0, 3, 1, 4, 2, 5, *range(6, 6 + len(rest)))
        x = x.reshape(B, T, Hp, Wp, *rest)
        return x[:, :, s:s + H, s:s + W]

    # ---------------------------------------------------------------- core
    def decode_chunk(self, z, state, *, noncausal_reference: bool = False):
        """z: normalized latents (B, C, 24, H, W) for the next C latents of the stream.
        Returns decode_base-space pixels (B, 3, frames, 16H, 16W) for these latents."""
        B, C, _, H, W = z.shape
        dev = z.device
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else z.dtype
        raw = z.float() * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2))                    # B,24,C,H,W
        with torch.autocast("cuda", enabled=False):
            x = self.x_embedder(raw.permute(0, 2, 3, 4, 1).float())           # B,C,H,W,D (fp32 like the released embedder)
        x = x.to(compute_dtype)
        D = x.shape[-1]
        n_suffix = self.register_tokens.shape[1] + 1
        suf = torch.cat([self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, D)], 1).to(compute_dtype)
        t0 = state["t"]
        for li, blk in enumerate(self.blocks):
            s = (self.window // 2) if (self.shift and li % 2 == 1 and not noncausal_reference) else 0
            # attention ------------------------------------------------------------
            h = blk.norm1(x.float()).to(x.dtype)
            hs = blk.norm1(suf.float()).to(suf.dtype)
            qkv = blk.attn.to_qkv(h).view(B, C, H, W, self.heads, 3 * self.dh)
            q, k, v = qkv.chunk(3, dim=-1)
            qkv_s = blk.attn.to_qkv(hs).view(B, n_suffix, self.heads, 3 * self.dh)
            qs, ks, vs = qkv_s.chunk(3, dim=-1)
            if blk.attn.norm_q is not None:
                q, qs = _apply_qk_norm(blk.attn.norm_q, q), _apply_qk_norm(blk.attn.norm_q, qs)
                k, ks = _apply_qk_norm(blk.attn.norm_k, k), _apply_qk_norm(blk.attn.norm_k, ks)
            cos, sin = self._ids(t0, C, H, W, s, torch.float32, dev)
            q = _apply_rotary_pos_emb_impl(q.reshape(B, C * H * W, self.heads, self.dh), (cos, sin)).view(B, C, H, W, self.heads, self.dh)
            k = _apply_rotary_pos_emb_impl(k.reshape(B, C * H * W, self.heads, self.dh), (cos, sin)).view(B, C, H, W, self.heads, self.dh)
            if not noncausal_reference:                                   # released decoder: suffix ids are zero
                scos, ssin = self._suffix_ids(t0, n_suffix, torch.float32, dev)
                qs = _apply_rotary_pos_emb_impl(qs, (scos, ssin))
                ks = _apply_rotary_pos_emb_impl(ks, (scos, ssin))
            k_all, v_all = k, v
            if state["k"][li] is not None and not noncausal_reference:
                k_all = torch.cat([state["k"][li], k], 1)
                v_all = torch.cat([state["v"][li], v], 1)
            Tk = k_all.shape[1]
            # windows: queries of the chunk, keys of cached + current latents of the same window
            qw, _, info = self._partition(q, s)
            kw, vmask, _ = self._partition(k_all, s)
            vw, _, _ = self._partition(v_all, s)
            nW = qw.shape[0] // B
            ksw = ks.unsqueeze(1).expand(B, nW, n_suffix, self.heads, self.dh).reshape(B * nW, n_suffix, self.heads, self.dh)
            vsw = vs.unsqueeze(1).expand(B, nW, n_suffix, self.heads, self.dh).reshape(B * nW, n_suffix, self.heads, self.dh)
            K = torch.cat([kw, ksw], 1).transpose(1, 2)
            V = torch.cat([vw, vsw], 1).transpose(1, 2)
            keymask = torch.cat([vmask, torch.ones(nW, n_suffix, dtype=torch.bool, device=dev)], 1)
            keymask = keymask.repeat(B, 1)[:, None, None, :]
            ow = F.scaled_dot_product_attention(qw.transpose(1, 2), K, V, attn_mask=keymask).transpose(1, 2)
            o = self._unpartition(ow, info, s).reshape(B, C, H, W, self.heads * self.dh)
            # suffix queries attend all cached + current tokens and the suffix
            Kall = torch.cat([k_all.reshape(B, Tk * H * W, self.heads, self.dh), ks], 1).transpose(1, 2)
            Vall = torch.cat([v_all.reshape(B, Tk * H * W, self.heads, self.dh), vs], 1).transpose(1, 2)
            os_ = F.scaled_dot_product_attention(qs.transpose(1, 2), Kall, Vall).transpose(1, 2).reshape(B, n_suffix, -1)
            x = x + blk.attn.to_out(o) * blk.scale1
            suf = suf + blk.attn.to_out(os_) * blk.scale1
            # feed forward ----------------------------------------------------------
            x = x + blk.ff(blk.norm2(x.float()).to(x.dtype)) * blk.scale2
            suf = suf + blk.ff(blk.norm2(suf.float()).to(suf.dtype)) * blk.scale2
            # cache update (keep the last `cache_latents` latents of K/V)
            if not noncausal_reference and self.cache_latents > 0:
                state["k"][li] = k_all[:, -self.cache_latents:]
                state["v"][li] = v_all[:, -self.cache_latents:]
        x = self.norm_out(x)
        with torch.autocast("cuda", enabled=False):
            out = self.proj_out(x.float())                                    # B,C,H,W,3*4*16*16
        out = out.view(B, C, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(0, 4, 1, 5, 2, 6, 3, 7)
        out = out.reshape(B, OUT_CH, C * PATCH_T, H * PATCH, W * PATCH)
        keep = torch.ones(C * PATCH_T, dtype=torch.bool, device=dev)
        for i in range(C):
            if (t0 + i) % self.block_latents == 0:
                keep[i * PATCH_T: i * PATCH_T + self.frames_to_trim] = False
        state["t"] = t0 + C
        return out[:, :, keep]

    def forward(self, z, state=None):
        """Decode a whole latent sequence z (B, T, 24, H, W) chunk by chunk (training / offline eval).
        Identical to calling decode_chunk on consecutive chunks in deployment."""
        state = self.init_state() if state is None else state
        outs = [self.decode_chunk(z[:, i:i + self.chunk], state) for i in range(0, z.shape[1], self.chunk)]
        return torch.cat(outs, 2)
