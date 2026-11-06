"""The globally sharded reference encode is bitwise the stock `_encode` (fake VAE with the diffusers tiling / blending)."""
import math
import types

import pytest
import torch

from sol_hyperflow_flash import vae_par as VP


class FakeVAE(torch.nn.Module):
    """Stock AutoencoderKLMiniMaxH3 tiling, blending and temporal chunking (diffusers code, small geometry)."""

    def __init__(self):
        super().__init__()
        self.spatial_compression_ratio = 4
        self.config = types.SimpleNamespace(clip_length=5, token_drop=1)
        self.use_tiling = True
        self.tile_sample_min_height = self.tile_sample_min_width = 16
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = 4
        self.encoder = torch.nn.Sequential(torch.nn.AvgPool3d((1, 4, 4)), torch.nn.Conv3d(3, 6, 1))
        self.quant_conv = torch.nn.Conv3d(6, 6, 1)

    def _split_tiles(self, length, tile_size, min_overlap):
        if tile_size >= length:
            return [0], [length], []
        n = math.ceil(length / tile_size)
        while tile_size * n - min_overlap * (n - 1) - length < 0:
            n += 1
        ov = [min_overlap] * (n - 1)
        for i in range((tile_size * n - sum(ov) - length) // self.spatial_compression_ratio):
            ov[i % (n - 1)] += self.spatial_compression_ratio
        st = [0]
        for i in range(n - 1):
            st.append(st[-1] + tile_size - ov[i])
        return st, [tile_size] * n, ov

    def _blend(self, a, b, e, dim):
        e = min(a.shape[dim], b.shape[dim], e)
        pos = torch.arange(e, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = e
        sa, sb = [slice(None)] * a.ndim, [slice(None)] * b.ndim
        sa[dim], sb[dim] = slice(-e, None), slice(0, e)
        bl = a[tuple(sa)] * (1 - pos / e).view(shape) + b[tuple(sb)] * (pos / e).view(shape)
        if e == b.shape[dim]:
            return bl
        sr = [slice(None)] * b.ndim
        sr[dim] = slice(e, None)
        return torch.cat([bl, b[tuple(sr)]], dim=dim)

    def _stitch_tiles(self, tiles, hov, wov):
        rows = []
        for i, row in enumerate(tiles):
            out = []
            for j, t in enumerate(row):
                if i > 0:
                    t = self._blend(tiles[i - 1][j], t, hov[i - 1], dim=-2)
                if j > 0:
                    t = self._blend(row[j - 1], t, wov[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    t = t[..., :-hov[i], :]
                if j < len(row) - 1:
                    t = t[..., :, :-wov[j]]
                out.append(t)
            rows.append(torch.cat(out, dim=-1))
        return torch.cat(rows, dim=-2)

    def _encode_clip(self, x):
        yi, yl, yo = self._split_tiles(x.shape[-2], self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        xi, xl, xo = self._split_tiles(x.shape[-1], self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        rows = [[self.quant_conv(self.encoder(x[..., a:a + al, b:b + bl])) for b, bl in zip(xi, xl)] for a, al in zip(yi, yl)]
        r = self.spatial_compression_ratio
        return self._stitch_tiles(rows, [o // r for o in yo], [o // r for o in xo])

    def _encode(self, x):
        L, T = self.config.clip_length, x.shape[2]
        if T % L:
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, (-T) % L, 1, 1)], dim=2)
        m = torch.cat([self._encode_clip(x[:, :, i * L:(i + 1) * L]) for i in range(x.shape[2] // L)], dim=2)
        return m[:, :, :-self.config.token_drop] if self.config.token_drop else m


@pytest.mark.parametrize("world", [1, 2, 3, 5, 8, 64])
def test_global_encode_matches_stock(world):
    fv = FakeVAE()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        x = torch.randn(1, 3, 12, 40, 56, generator=gen)
        want = fv._encode(x)
        locs = [VP.global_encode_local(torch, fv, x, r, world) for r in range(world)]
        assert len({tuple(lc.shape) for lc, _ in locs}) == 1          # equal count per rank: fixed-size all-gather
        flat = torch.cat([lc for lc, _ in locs])[:locs[0][1]["n"]]
        assert torch.equal(VP.global_encode_assemble(torch, fv, flat, locs[0][1]), want)


@pytest.mark.parametrize("world", [1, 2, 3, 7])
def test_single_clip_tile_split_matches_stock(world):
    fv = FakeVAE()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        img = torch.randn(1, 3, 1, 40, 56, generator=gen)
        want = fv._encode_clip(img)
        locs = [VP.clip_encode_local(torch, fv, img, r, world) for r in range(world)]
        flat = torch.cat([lc for lc, _ in locs])[:locs[0][1]["n"]]
        assert torch.equal(VP.global_encode_assemble(torch, fv, flat, locs[0][1]), want)
