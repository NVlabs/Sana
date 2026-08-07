"""Batch the video VAE's tile decodes instead of launching them one at a time.

Once the cache and Sol-Attn are on, the denoising loop stops being the whole request: the
fixed remainder is 31.8 s of which the video decode is 29.4 s — 92.6%, and a sixth of the
whole request. Inside it, `_decode_clip` walks a grid of tiles and calls the decoder once per
tile. At 832x480 that is a 3x4 grid, twelve sequential launches of about 2.45 s each.

Every tile is independent and `_split_tiles` gives them all the same size — the slack goes
into the overlaps, not into shorter edge tiles — so they can go through the decoder as one
batch. This is the same change Sol-Engine's `vae_shard.py` makes, where it measures 1.93x and
reports the result bit-identical.

Bit-identical is a claim about arithmetic, not about kernels: batching can move cuDNN and
cuBLAS onto different algorithms, and their reductions need not associate the same way.
`--check` verifies it here rather than inheriting the claim.
"""

from __future__ import annotations

import paths

paths.setup()

import torch


def _batched_decode_clip(self, z: torch.Tensor) -> torch.Tensor:
    """`_decode_clip`, with the tile loop replaced by batched decoder calls."""
    if not self.use_tiling:
        return self.decoder(self.post_quant_conv(z))

    height = z.shape[-2] * self.spatial_compression_ratio
    width = z.shape[-1] * self.spatial_compression_ratio
    y_indices, y_lengths, y_overlaps = self._split_tiles(
        height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
    )
    x_indices, x_lengths, x_overlaps = self._split_tiles(
        width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
    )

    ratio = self.spatial_compression_ratio
    tiles = [
        z[..., i_pos // ratio : i_pos // ratio + i_len // ratio,
             j_pos // ratio : j_pos // ratio + j_len // ratio]
        for i_pos, i_len in zip(y_indices, y_lengths)
        for j_pos, j_len in zip(x_indices, x_lengths)
    ]

    # A cap rather than one batch of everything: the ViT decoder's activations scale with the
    # batch, and this runs on a part whose memory is shared with the host.
    limit = getattr(self, "_h3_tile_batch", 0) or len(tiles)
    decoded = []
    for start in range(0, len(tiles), limit):
        group = tiles[start : start + limit]
        batch = torch.cat(group, dim=0)
        out = self.decoder(self.post_quant_conv(batch))
        decoded.extend(out.split(group[0].shape[0], dim=0))

    columns = len(x_indices)
    rows = [decoded[i * columns : (i + 1) * columns] for i in range(len(y_indices))]
    return self._stitch_tiles(rows, y_overlaps, x_overlaps)


def patch_batched_tiles(vae, batch: int = 0) -> None:
    """Route this VAE's tiled decode through the batched implementation.

    `batch` caps how many tiles go through the decoder at once; 0 means all of them.
    """
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    vae._h3_tile_batch = batch
    if getattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles", False):
        return
    AutoencoderKLMiniMaxH3._h3_decode_clip_original = AutoencoderKLMiniMaxH3._decode_clip
    AutoencoderKLMiniMaxH3._decode_clip = _batched_decode_clip
    AutoencoderKLMiniMaxH3._h3_batched_tiles = True


def unpatch_batched_tiles() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    if getattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles", False):
        AutoencoderKLMiniMaxH3._decode_clip = AutoencoderKLMiniMaxH3._h3_decode_clip_original
        AutoencoderKLMiniMaxH3._h3_batched_tiles = False
