# Fast H3 VAE decoder (non-causal, 24 layers), from the StreamRender-H3 fast H3 VAE work (NVIDIA); see fast_vae/__init__.py.
# Copied from the delivery code with only the imports changed.
"""General (non-causal) fast H3 VAE decoder: the released H3 VAE ViT decoder pruned to 24 layers, decoding whole
native 5-latent blocks with window attention and no tiling / temporal-overlap recomputation.

* time: native 5-latent H3 blocks (17 video frames). Attention is bidirectional inside a temporal window of 5
  latent frames. Odd layers shift the temporal windows by 2 latent frames (Video-Swin style), so neighbouring
  blocks exchange information inside the network; nothing is recomputed and nothing is cross-faded.
* space: 16x16-token windows (256 px, the released tile size); odd layers shift them by half a window.
* global tokens: every native 5-latent block owns 5 global tokens (4 register + 1 class, from the released decoder).
  They attend to every token of their block; every window attends to the global tokens of the blocks it overlaps.
* rotary positions: spatial ids are window-local (as in the released tile decode); time ids are window-local too,
  so nothing depends on the absolute position in the video and any length decodes the same way.
* output: per token a 3 x 4 x 16 x 16 pixel patch; the first latent of every block keeps 1 of its 4 frames,
  exactly as the released decoder, so 5 latent frames give 17 video frames.

Input z: normalized latents (B, T, 24, H, W), t = 0 is the start of a native block. Output: decode_base-space
pixels (B, 3, F, 16H, 16W) in the ImageNet-normalized space of the released VAE.

Built from the config (no released-VAE object needed); the delivered weights overwrite every parameter.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


# ---- the released H3 VAE ViT decoder modules the fast decoder is made of (same parameter names, so the delivered
# weights load strictly); dim 2048 = 32 heads x 64, SwiGLU FFN x4, RMSNorm eps 1e-5, 3-axis RoPE on 48 of 64 channels.
def _apply_qk_norm(norm, x):
    """qk RMSNorm without affine, computed in fp32 and cast back (the released recipe)."""
    return norm(x.float()).to(x.dtype)


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary_pos_emb_impl(t, rotary_pos_emb):
    """Rotate the first rot_dim channels of t (B, N, heads, dh) with cos/sin (B, N, 1, rot_dim)."""
    cos, sin = (r.to(t.dtype) for r in rotary_pos_emb[:2])
    rot = cos.shape[-1]
    t_rot, t_pass = t[..., :rot], t[..., rot:]
    t_rot = t_rot * cos + _rotate_half(t_rot) * sin
    return torch.cat((t_rot, t_pass), dim=-1) if t_pass.shape[-1] else t_rot


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.register_buffer("inv_freq", 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32), persistent=False)

    def forward(self, ids):
        with torch.autocast("cuda", enabled=False):
            angles = 2.0 * math.pi * ids[:, :, :, None] * self.inv_freq.to(ids.device)[None, None, None, :]
            angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
            return torch.cos(angles).to(ids.dtype), torch.sin(angles).to(ids.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head)   # per head [q | k | v]
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * dim * mult)    # [gate | up]
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, eps):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, dim_head, eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim)
        self.scale2 = nn.Parameter(torch.zeros(dim))

PATCH, PATCH_T, OUT_CH = 16, 4, 3
KEEP_24 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 31, 32, 35]
# greedy single-layer removal on the released 36-layer decoder (vae_prune/layer_scan.py)
KEEP_16 = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 14, 15, 20, 31, 32, 35]
KEEP_12 = [0, 1, 2, 3, 9, 10, 14, 15, 20, 31, 32, 35]
KEEP_8 = [0, 1, 2, 15, 20, 31, 32, 35]
KEEP_6 = [0, 1, 2, 15, 20, 35]          # KEEP_8 minus the two layers with the smallest single-removal PSNR drop
KEEP = {24: KEEP_24, 16: KEEP_16, 12: KEEP_12, 8: KEEP_8, 6: KEEP_6}


class NonCausalViTDecoder(nn.Module):
    def __init__(self, latents_mean, latents_std, keep_layers=KEEP_24, *, window: int = 16, block: int = 5, t_shift: int = 2,
                 frames_to_trim: int = 3, time_norm: float | None = None, heads: int = 32, dim_head: int = 64,
                 rope_theta: float = 100.0, rope_dim_ratio: float = 0.75, eps: float = 1e-5, latent_channels: int = 24,
                 num_register_tokens: int = 4):
        super().__init__()
        dim = heads * dim_head
        self.keep_layers = list(keep_layers)
        self.window, self.block, self.t_shift, self.frames_to_trim = window, block, t_shift, frames_to_trim
        # time ids: (window-local index + 0.5) / time_norm * 2 - 1 (the released decoder uses 7 for its 5 + 2 chunks)
        self.time_norm = float(time_norm if time_norm is not None else block)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.x_embedder = nn.Linear(latent_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dim_head, eps) for _ in self.keep_layers])
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = nn.Linear(dim, OUT_CH * PATCH_T * PATCH * PATCH)
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta)
        self.heads, self.dh = heads, dim_head
        self.register_buffer("latents_mean", torch.as_tensor(latents_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.as_tensor(latents_std, dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------ geometry
    def _shifts(self, li):
        odd = li % 2 == 1
        return (self.window // 2 if odd else 0), (self.t_shift if odd else 0)

    def _rope(self, T, H, W, s, st, device):
        """cos/sin for tokens in (T, H, W) order with window-local time and space ids for shifts (st, s)."""
        lt = ((torch.arange(T, device=device) + st) % self.block).float()
        ly = ((torch.arange(H, device=device) + s) % self.window).float()
        lx = ((torch.arange(W, device=device) + s) % self.window).float()
        t = (lt + 0.5) / self.time_norm * 2 - 1
        hy = (ly + 0.5) / self.window * 2 - 1
        wx = (lx + 0.5) / self.window * 2 - 1
        grid = torch.stack(torch.meshgrid(t, hy, wx, indexing="ij"), -1).reshape(1, -1, 3)
        return self.pos_embed(grid.float())

    def _partition(self, x, s, st):
        """x: (B, T, H, W, ...) -> windows (B*nW, Tw*ws*ws, ...), valid (nW, Tw*ws*ws), info.
        Window order is (time window, row, col)."""
        B, T, H, W = x.shape[:4]
        ws, tb = self.window, self.block
        rest = x.shape[4:]
        Tp = ((T + st + tb - 1) // tb) * tb
        Hp = ((H + s + ws - 1) // ws) * ws
        Wp = ((W + s + ws - 1) // ws) * ws
        xp = x.new_zeros(B, Tp, Hp, Wp, *rest)
        xp[:, st:st + T, s:s + H, s:s + W] = x
        valid = torch.zeros(Tp, Hp, Wp, dtype=torch.bool, device=x.device)
        valid[st:st + T, s:s + H, s:s + W] = True
        nt, nh, nw = Tp // tb, Hp // ws, Wp // ws
        r = len(rest)
        xw = xp.view(B, nt, tb, nh, ws, nw, ws, *rest).permute(0, 1, 3, 5, 2, 4, 6, *range(7, 7 + r))
        xw = xw.reshape(B * nt * nh * nw, tb * ws * ws, *rest)
        vw = valid.view(nt, tb, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, tb * ws * ws)
        return xw, vw, (B, T, H, W, Tp, Hp, Wp, nt, nh, nw)

    def _unpartition(self, xw, info, s, st):
        B, T, H, W, Tp, Hp, Wp, nt, nh, nw = info
        ws, tb = self.window, self.block
        rest = xw.shape[2:]
        r = len(rest)
        x = xw.view(B, nt, nh, nw, tb, ws, ws, *rest).permute(0, 1, 4, 2, 5, 3, 6, *range(7, 7 + r))
        x = x.reshape(B, Tp, Hp, Wp, *rest)
        return x[:, st:st + T, s:s + H, s:s + W]

    def _layer(self, li, blk, x, g):
        """One transformer layer on the tokens x (B,T,H,W,D) and the per-block global tokens g (B,nb,S,D)."""
        B, T, H, W, nb, S = self._geom
        dev = x.device
        tb = self.block
        s, st = self._shifts(li)
        h = blk.norm1(x.float()).to(x.dtype)
        hg = blk.norm1(g.float()).to(g.dtype)
        q, k, v = blk.attn.to_qkv(h).view(B, T, H, W, self.heads, 3 * self.dh).chunk(3, -1)
        qg, kg, vg = blk.attn.to_qkv(hg).view(B, nb, S, self.heads, 3 * self.dh).chunk(3, -1)
        if blk.attn.norm_q is not None:
            q, qg = _apply_qk_norm(blk.attn.norm_q, q), _apply_qk_norm(blk.attn.norm_q, qg)
            k, kg = _apply_qk_norm(blk.attn.norm_k, k), _apply_qk_norm(blk.attn.norm_k, kg)
        cos, sin = self._rope(T, H, W, s, st, dev)
        q = _apply_rotary_pos_emb_impl(q.reshape(B, T * H * W, self.heads, self.dh), (cos, sin)).view(B, T, H, W, self.heads, self.dh)
        k = _apply_rotary_pos_emb_impl(k.reshape(B, T * H * W, self.heads, self.dh), (cos, sin)).view(B, T, H, W, self.heads, self.dh)
        # ---- window attention: tokens of a window + global tokens of the native blocks the window overlaps
        qw, vmask, info = self._partition(q, s, st)
        kw, _, _ = self._partition(k, s, st)
        vw, _, _ = self._partition(v, s, st)
        nt, nh, nw = info[7:]
        nW = nt * nh * nw
        # native blocks overlapped by temporal window j: latent range [j*tb - st, (j+1)*tb - st)
        tw_lo = (torch.arange(nt, device=dev) * tb - st).clamp(min=0)
        tw_hi = ((torch.arange(nt, device=dev) + 1) * tb - st - 1).clamp(max=T - 1)
        b0, b1 = tw_lo // tb, tw_hi // tb                         # at most 2 blocks per window
        gk_idx = torch.stack([b0, b1], 1)                          # nt, 2
        gk_valid = torch.stack([torch.ones_like(b0, dtype=torch.bool), b1 != b0], 1)   # nt, 2
        kgw = kg[:, gk_idx].reshape(B, nt, 2 * S, self.heads, self.dh)
        vgw = vg[:, gk_idx].reshape(B, nt, 2 * S, self.heads, self.dh)
        kgw = kgw[:, :, None, None].expand(B, nt, nh, nw, 2 * S, self.heads, self.dh).reshape(B * nW, 2 * S, self.heads, self.dh)
        vgw = vgw[:, :, None, None].expand(B, nt, nh, nw, 2 * S, self.heads, self.dh).reshape(B * nW, 2 * S, self.heads, self.dh)
        gmask = gk_valid[:, :, None].expand(nt, 2, S).reshape(nt, 2 * S)
        gmask = gmask[:, None, None].expand(nt, nh, nw, 2 * S).reshape(nW, 2 * S)
        Kc = torch.cat([kw, kgw], 1).transpose(1, 2)
        Vc = torch.cat([vw, vgw], 1).transpose(1, 2)
        kmask = torch.cat([vmask, gmask], 1).repeat(B, 1)[:, None, None, :]
        ow = F.scaled_dot_product_attention(qw.transpose(1, 2), Kc, Vc, attn_mask=kmask).transpose(1, 2)
        o = self._unpartition(ow, info, s, st).reshape(B, T, H, W, self.heads * self.dh)
        # ---- global tokens: attend to every token of their own native block + the block's global tokens
        Tp = nb * tb
        kb = k.new_zeros(B, Tp, H, W, self.heads, self.dh); kb[:, :T] = k
        vb = v.new_zeros(B, Tp, H, W, self.heads, self.dh); vb[:, :T] = v
        kb = kb.view(B, nb, tb * H * W, self.heads, self.dh)
        vb = vb.view(B, nb, tb * H * W, self.heads, self.dh)
        tvalid = torch.arange(Tp, device=dev) < T
        bmask = tvalid.view(nb, tb, 1).expand(nb, tb, H * W).reshape(nb, tb * H * W)
        Kg = torch.cat([kb, kg], 2).reshape(B * nb, tb * H * W + S, self.heads, self.dh).transpose(1, 2)
        Vg = torch.cat([vb, vg], 2).reshape(B * nb, tb * H * W + S, self.heads, self.dh).transpose(1, 2)
        gm = torch.cat([bmask, torch.ones(nb, S, dtype=torch.bool, device=dev)], 1).repeat(B, 1)[:, None, None, :]
        og = F.scaled_dot_product_attention(qg.reshape(B * nb, S, self.heads, self.dh).transpose(1, 2), Kg, Vg,
                                            attn_mask=gm).transpose(1, 2).reshape(B, nb, S, -1)
        x = x + blk.attn.to_out(o) * blk.scale1
        g = g + blk.attn.to_out(og) * blk.scale1
        x = x + blk.ff(blk.norm2(x.float()).to(x.dtype)) * blk.scale2
        g = g + blk.ff(blk.norm2(g.float()).to(g.dtype)) * blk.scale2
        return x, g

    # ------------------------------------------------------------------ forward
    def forward(self, z):
        B, T, _, H, W = z.shape
        dev = z.device
        tb, ws = self.block, self.window
        nb = (T + tb - 1) // tb                                     # native blocks
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else z.dtype
        raw = z.float() * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2))
        with torch.autocast("cuda", enabled=False):
            x = self.x_embedder(raw.permute(0, 2, 3, 4, 1).float())   # B,T,H,W,D
        x = x.to(compute_dtype)
        D = x.shape[-1]
        S = self.register_tokens.shape[1] + 1
        g = torch.cat([self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, D)], 1).to(compute_dtype)
        g = g.unsqueeze(1).expand(B, nb, S, D).contiguous()          # B, nb, S, D  (one set per native block)
        self._geom = (B, T, H, W, nb, S)
        for li, blk in enumerate(self.blocks):
            x, g = self._layer(li, blk, x, g)
        x = self.norm_out(x)
        with torch.autocast("cuda", enabled=False):
            out = self.proj_out(x.float())                                  # B,T,H,W,3*4*16*16
        out = out.view(B, T, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(0, 4, 1, 5, 2, 6, 3, 7)
        out = out.reshape(B, OUT_CH, T * PATCH_T, H * PATCH, W * PATCH)
        keep = torch.ones(T * PATCH_T, dtype=torch.bool, device=dev)
        for t in range(0, T, tb):
            keep[t * PATCH_T: t * PATCH_T + self.frames_to_trim] = False
        return out[:, :, keep]
