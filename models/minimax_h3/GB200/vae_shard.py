"""Shard the MiniMax-H3 video VAE decode across the context-parallel ranks.

Context parallelism shards the block stack but not the decode, so every rank decodes the whole
video and all but one rank's share of that work is thrown away. Unsharded the decode is 7.55 s at
1344x768/124f; sharded over eight ranks it is 1.16 s.

The decode is already built out of independent pieces, so nothing has to be restructured:

    _decode        splits the latents into temporal chunks and cross-fades the decoded frames.
                   Each `_decode_clip` call reads only its own latent slice and carries no state
                   forward, so the chunks are independent — but they are left alone here, because
                   the spatial axis divides more evenly.
    _decode_clip   lays tiles over the frame and calls the decoder once per tile, then blends the
                   overlaps in `_stitch_tiles`. Every tile is independent, and `_split_tiles`
                   returns `[tile_size] * num_tiles`, so every tile is exactly the same size.

At 1344x768 with 256-pixel tiles that is 4 x 7 = 28 tiles per clip, and 37 latent frames make 7
clips, so 196 tile decodes in total.

**Batching (`batched=True`, the default).** Splitting a clip's tiles across eight ranks leaves each
rank four tiles of `(1, 24, 7, 16, 16)` — small enough that four separate decoder launches spend
most of their time not computing. Feeding all four through as one batch is the same arithmetic on
the same kernels and measures 1.93x faster (1.1597 s -> 0.6021 s), bit-identical to the unsharded
decode. It is legal because the tiles are all the same size and because the decoder is
batch-independent: it is a ViT over `(B, S, C)` tokens whose norms reduce over the last dimension
only, whose register and cls tokens are per-batch replicas, and whose RoPE is derived from the tile
geometry rather than the batch. The one batch-mixing module in the file,
`MiniMaxH3VideoGroupNorm`, is used by the encoder's ResNet blocks and never by the decoder.

**Compile (`compile_mode`, off by default).** All 196 tile decodes share one static shape and the
collective sits outside the tile loop, so there is no graph/collective conflict. Compiling on top
of batching gives 3.23x over production (0.3593 s) and drops peak memory from 18.8 GB to 15.6 GB.
This is *not* bit-identical — inductor reassociates, and the measured deviation against the eager
decode is `max_abs` 0.021 — so it is opt-in rather than default. `max-autotune-no-cudagraphs` was
measured at 0.3507 s, 2.4% better than `default` for a much longer compile; plain `max-autotune` is
never used, because cudagraphs would hand back a reused static buffer.

Sharding itself is lossless in the strict sense, unlike most of this tree: the same tiles are
decoded by the same code and blended by the same `_stitch_tiles`, only on different devices.
Nothing is skipped, no precision changes, and no reduction order changes — a tile's arithmetic does
not depend on which GPU runs it.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _undo_only(undo):
    """Uninstall for the paths that never patched `_decode_clip` (single rank, or no distributed)."""
    def uninstall():
        for u in reversed(undo):
            u()
    return uninstall


def install(vae, group=None, batched: bool | None = None, compile_mode: str | None = None):
    """Patch `_decode_clip` to compute this rank's tiles and gather the rest. Returns uninstall.

    `batched` defaults to on (`H3_VAE_BATCHED=0` disables it, reproducing the one-launch-per-tile
    loop). `compile_mode` defaults to off (`H3_VAE_COMPILE=default` or
    `max-autotune-no-cudagraphs` enables it).
    """
    if batched is None or os.environ.get("H3_VAE_BATCHED") == "0":   # serve.py passes True
        batched = os.environ.get("H3_VAE_BATCHED", "1") == "1"
    if compile_mode is None:
        compile_mode = os.environ.get("H3_VAE_COMPILE") or None

    undo = []
    if compile_mode:
        eager_decoder = vae.decoder
        vae.decoder = torch.compile(eager_decoder, mode=compile_mode, dynamic=False)

        def _restore_decoder():
            vae.decoder = eager_decoder
            torch._dynamo.reset()
        undo.append(_restore_decoder)

    if dist.is_available() and dist.is_initialized():
        group = group or dist.group.WORLD
        world, rank = dist.get_world_size(group), dist.get_rank(group)
    else:
        world, rank = 1, 0
    # One rank still batches: unpatched, `_decode_clip` decodes its 28 tiles one launch at a time
    # (1792 tokens each, far too few rows to fill the GPU's GEMMs).
    if world == 1 and not batched:
        return _undo_only(undo)

    original = vae._decode_clip
    original_decode = vae._decode

    def tile_geometry(z: torch.Tensor):
        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        coords = [
            (y_indices[i], y_lengths[i], x_indices[j], x_lengths[j])
            for i in range(len(y_indices))
            for j in range(len(x_indices))
        ]
        return coords, (len(y_indices), len(x_indices)), y_overlaps, x_overlaps

    def decode_tiles(clips: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode every tile of every clip (all clips the same shape), sharded over the ranks.

        Returns one stitched clip per input clip. Tiles are numbered clip-major, so rank r's
        contiguous block may span a clip boundary; the gather puts them back in order.
        """
        coords, (n_y, n_x), y_overlaps, x_overlaps = tile_geometry(clips[0])
        ratio = vae.spatial_compression_ratio
        tiles = [(c, t) for c in range(len(clips)) for t in coords]
        # Contiguous blocks, not round-robin: `all_gather` concatenates rank 0's block, then rank
        # 1's, and so on, so contiguous assignment is what puts the tiles back in tile order for
        # free. The list is padded up to an equal count per rank so the gather stays a plain
        # fixed-size one; the padding entries are recomputed duplicates of the last tile, at most
        # world-1 of them, and are dropped after the gather.
        per_rank = (len(tiles) + world - 1) // world
        padded = tiles + [tiles[-1]] * (per_rank * world - len(tiles))
        mine = padded[rank * per_rank: (rank + 1) * per_rank]

        slices = [
            clips[c][..., y // ratio: y // ratio + y_len // ratio,
                     x // ratio: x // ratio + x_len // ratio]
            for c, (y, y_len, x, x_len) in mine
        ]
        if batched:
            # At one rank all 196 tiles would go through at once; cap the launch (equal chunks keep
            # one static shape for the compiled decoder). The decoder is batch-independent.
            n_chunks = (len(slices) + max_tiles - 1) // max_tiles
            size = (len(slices) + n_chunks - 1) // n_chunks
            local_stack = torch.cat([
                vae.decoder(vae.post_quant_conv(torch.cat(slices[i: i + size], dim=0)))
                for i in range(0, len(slices), size)], dim=0)
        else:
            local_stack = torch.stack(
                [vae.decoder(vae.post_quant_conv(s)) for s in slices], dim=0).squeeze(1)

        if world == 1:
            flat = local_stack
        else:
            gathered = [torch.empty_like(local_stack) for _ in range(world)]
            dist.all_gather(gathered, local_stack, group=group)
            flat = torch.cat(gathered, dim=0)[: len(tiles)]

        # `_stitch_tiles` only touches dims -2/-1, so a missing batch axis would not raise — it
        # would return `(3, T, H, W)` and `_decode` would then slice the height axis where it means
        # to slice frames. Restore the axis explicitly.
        out, index = [], 0
        for _ in clips:
            rows = []
            for _ in range(n_y):
                row = []
                for _ in range(n_x):
                    row.append(flat[index].unsqueeze(0))
                    index += 1
                rows.append(row)
            out.append(vae._stitch_tiles(rows, y_overlaps, x_overlaps))
        return out

    def sharded_decode_clip(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return original(z)
        return decode_tiles([z])[0]

    # Batch across the temporal clips too (`H3_VAE_CLIP_BATCH=0` disables): `_decode` calls
    # `_decode_clip` once per clip on independent latent slices, so all clips' tiles go through one
    # decoder launch (and one gather) and `_decode`'s cross-fade then consumes the precomputed clips
    # in order. At 4 ranks that is 49 tiles per launch instead of 7, 7x fewer gathers.
    clip_batch = os.environ.get("H3_VAE_CLIP_BATCH", "1") == "1"
    max_tiles = max(1, int(os.environ.get("H3_VAE_MAX_TILES", "98")))

    def clip_slices(z: torch.Tensor) -> list[torch.Tensor]:
        """The latent slices `_decode` hands to `_decode_clip`, in call order (same padding)."""
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        return [z[:, :, i * tokens_chunk_size: i * tokens_chunk_size + tokens_chunk_size + vae.token_overlap]
                for i in range(num_chunks)]

    def batched_decode(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return original_decode(z)
        clips = clip_slices(z)
        if len({c.shape for c in clips}) != 1:
            return original_decode(z)
        decoded = iter(decode_tiles(clips))
        shapes = iter([c.shape for c in clips])

        def precomputed_clip(zc: torch.Tensor) -> torch.Tensor:
            assert zc.shape == next(shapes), "clip order/shape differs from clip_slices"
            return next(decoded)

        vae._decode_clip = precomputed_clip
        try:
            out = original_decode(z)
        finally:
            vae._decode_clip = sharded_decode_clip
        assert next(decoded, None) is None, "unused precomputed clips"
        return out

    vae._decode_clip = sharded_decode_clip
    if clip_batch:
        vae._decode = batched_decode

    def uninstall():
        vae._decode_clip = original
        vae._decode = original_decode
        for u in reversed(undo):
            u()

    return uninstall
