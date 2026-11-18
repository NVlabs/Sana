"""Faster tiled decode for AutoencoderKLMiniMaxH3: same tiles, same stitching, tiles batched and optionally
spread over ranks. install(vae, batch=..., distributed=...) patches vae._decode_clip in place."""
import types

import torch
import torch.distributed as dist


def install(vae, batch=26, distributed=False):
    original = vae._decode_clip

    def _decode_clip(self, z):
        if not self.use_tiling:
            return original(z)
        ratio = self.spatial_compression_ratio
        ys, ylen, yover = self._split_tiles(z.shape[-2] * ratio, self.tile_sample_min_height,
                                            self.tile_sample_min_overlap_height)
        xs, xlen, xover = self._split_tiles(z.shape[-1] * ratio, self.tile_sample_min_width,
                                            self.tile_sample_min_overlap_width)
        coords = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
        crops = [z[..., ys[i] // ratio: ys[i] // ratio + ylen[i] // ratio,
                   xs[j] // ratio: xs[j] // ratio + xlen[j] // ratio] for i, j in coords]
        if len({tuple(c.shape) for c in crops}) != 1:
            return original(z)
        world, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
        per = -(-len(crops) // world)
        mine = crops[rank * per:(rank + 1) * per]
        outs = []
        for k in range(0, len(mine), batch):
            stacked = torch.cat(mine[k:k + batch], dim=0)
            outs.append(self.decoder(self.post_quant_conv(stacked)))
        local = torch.cat(outs, dim=0) if outs else None
        if world > 1:
            if local.shape[0] < per:
                local = torch.cat([local, local.new_zeros((per - local.shape[0],) + local.shape[1:])])
            gathered = torch.empty((world * per,) + tuple(local.shape[1:]), dtype=local.dtype, device=local.device)
            dist.all_gather_into_tensor(gathered, local.contiguous())
            local = gathered
        tiles = [local[k:k + 1] for k in range(len(crops))]
        rows = [tiles[i * len(xs):(i + 1) * len(xs)] for i in range(len(ys))]
        return self._stitch_tiles(rows, yover, xover)

    vae._decode_clip = types.MethodType(_decode_clip, vae)
    return vae
