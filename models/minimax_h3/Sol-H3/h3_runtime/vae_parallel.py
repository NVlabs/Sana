"""Shard the MiniMax-H3 video VAE decode across the context-parallel ranks.

Context parallelism shards the block stack but not the decode, so every rank decodes the whole
video and all but one rank's share of that work is thrown away. Unsharded the decode is 7.55 s at
1344x768/124f; sharded over eight ranks it is 1.16 s.

The decode is already built out of independent pieces, so nothing has to be restructured:

    _decode        splits the latents into temporal chunks and cross-fades the decoded frames.
                   Each `_decode_clip` call reads only its own latent slice and carries no state
                   forward, so all temporal/spatial tiles can share one balanced global batch.
    _decode_clip   lays tiles over the frame and calls the decoder once per tile, then blends the
                   overlaps in `_stitch_tiles`. Every tile is independent, and `_split_tiles`
                   returns `[tile_size] * num_tiles`, so every tile is exactly the same size.

At 1344x768 with 256-pixel tiles that is 4 x 7 = 28 tiles per clip, and 37 latent frames make 7
clips, so 196 tile decodes in total.

**Batching (`batched=True`, the default).** Splitting a clip's tiles across eight ranks leaves each
rank four tiles of `(1, 24, 7, 16, 16)` — small enough that four separate decoder launches spend
most of their time not computing. Feeding all four through as one batch is the same arithmetic on
the same kernels and measures 1.93x faster (1.1597 s -> 0.6021 s), bit-identical to the unsharded
decode. With global batching enabled, the 196 real tiles are balanced as 25 per rank with only four
padded duplicates in total, instead of four per rank for each of seven clips (28 duplicates), and
the decoder plus gather run once rather than seven times. The larger compiled batch selects a
different GEMM implementation, so this final step is not bit-identical: its measured deviation from
the per-clip path is max-abs 0.013672, relative L2 0.000284, and 75.75 dB PSNR over the decoder's
[-1, 1] range. It is legal because the tiles are all the same size and because the decoder is
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

Per-clip sharding is lossless in the strict sense: the same tiles are decoded by the same code and
blended by the same `_stitch_tiles`, only on different devices. Global batching remains optional via
`H3_VAE_GLOBAL_BATCH=0` because its different compiled batch shape introduces the small, measured
rounding drift above. Nothing is skipped and the stitch/blend order remains unchanged.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist


def _undo_only(undo):
    """Uninstall for the paths that never patched `_decode_clip` (single rank, or no distributed)."""
    def uninstall():
        for u in reversed(undo):
            u()
    return uninstall


def install(vae, group=None, batched: bool | None = None, compile_mode: str | None = None,
            global_batch: bool | None = None, encode_parallel: bool = False):
    """Shard equal VAE tiles across ranks and gather them in canonical order. Returns uninstall.

    `batched` defaults to on (`H3_VAE_BATCHED=0` disables it, reproducing the one-launch-per-tile
    loop). `compile_mode` defaults to off (`H3_VAE_COMPILE=default` or
    `max-autotune-no-cudagraphs` enables it). `global_batch` defaults to on;
    `H3_VAE_GLOBAL_BATCH=0` restores the bit-exact per-clip gather path. `encode_parallel`
    distributes Ref2VA reference tiles without changing their encode or stitch order.
    """
    if batched is None:
        batched = os.environ.get("H3_VAE_BATCHED", "1") == "1"
    if compile_mode is None:
        compile_mode = os.environ.get("H3_VAE_COMPILE") or None
    if global_batch is None:
        global_batch = os.environ.get("H3_VAE_GLOBAL_BATCH", "1") == "1"
    global_gate_pending = os.environ.get("H3_VAE_GLOBAL_GATE", "0") == "1"

    undo = []
    if compile_mode:
        eager_decoder = vae.decoder
        vae.decoder = torch.compile(eager_decoder, mode=compile_mode, dynamic=False)

        def _restore_decoder():
            vae.decoder = eager_decoder
            torch._dynamo.reset()
        undo.append(_restore_decoder)

    if not (dist.is_available() and dist.is_initialized()):
        return _undo_only(undo)

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world == 1:
        return _undo_only(undo)

    original_encode_clip = vae._encode_clip
    original_decode_clip = vae._decode_clip
    original_decode = vae._decode

    def geometry(z: torch.Tensor):
        height = z.shape[-2] * vae.spatial_compression_ratio
        width = z.shape[-1] * vae.spatial_compression_ratio
        y_parts = vae._split_tiles(
            height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        x_parts = vae._split_tiles(
            width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        y_indices, y_lengths, y_overlaps = y_parts
        x_indices, x_lengths, x_overlaps = x_parts
        coords = [
            (y_indices[i], y_lengths[i], x_indices[j], x_lengths[j])
            for i in range(len(y_indices))
            for j in range(len(x_indices))
        ]
        return coords, y_indices, y_overlaps, x_indices, x_overlaps

    def encode_geometry(x: torch.Tensor):
        y_parts = vae._split_tiles(
            x.shape[-2], vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
        )
        x_parts = vae._split_tiles(
            x.shape[-1], vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
        )
        y_indices, y_lengths, y_overlaps = y_parts
        x_indices, x_lengths, x_overlaps = x_parts
        coords = [
            (y_indices[i], y_lengths[i], x_indices[j], x_lengths[j])
            for i in range(len(y_indices))
            for j in range(len(x_indices))
        ]
        ratio = vae.spatial_compression_ratio
        return (
            coords,
            y_indices,
            [overlap // ratio for overlap in y_overlaps],
            x_indices,
            [overlap // ratio for overlap in x_overlaps],
        )

    def gather_tiles(local_stack: torch.Tensor, logical_tiles: int) -> torch.Tensor:
        """Fixed-size NCCL gather directly into its final rank-major flat layout."""
        local_stack = local_stack.contiguous()
        flat = torch.empty(
            (world * local_stack.shape[0], *local_stack.shape[1:]),
            dtype=local_stack.dtype,
            device=local_stack.device,
        )
        dist.all_gather_into_tensor(flat, local_stack, group=group)
        return flat[:logical_tiles]

    def stitch(flat: torch.Tensor, start: int, y_indices, y_overlaps,
               x_indices, x_overlaps) -> torch.Tensor:
        rows = []
        index = start
        for _ in y_indices:
            row = []
            for _ in x_indices:
                row.append(flat[index].unsqueeze(0))
                index += 1
            rows.append(row)
        return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    def sharded_encode_clip(x: torch.Tensor) -> torch.Tensor:
        if not encode_parallel or not vae.use_tiling or x.shape[0] != 1:
            return original_encode_clip(x)

        coords, y_indices, y_overlaps, x_indices, x_overlaps = encode_geometry(x)
        per_rank = (len(coords) + world - 1) // world
        padded = coords + [coords[-1]] * (per_rank * world - len(coords))
        mine = padded[rank * per_rank : (rank + 1) * per_rank]
        local = []
        for y, y_len, x_pos, x_len in mine:
            tile = x[..., y : y + y_len, x_pos : x_pos + x_len]
            local.append(vae.quant_conv(vae.encoder(tile)))
        flat = gather_tiles(torch.cat(local, dim=0), len(coords))
        return stitch(flat, 0, y_indices, y_overlaps, x_indices, x_overlaps)

    vae._encode_clip = sharded_encode_clip

    def sharded_decode_clip(z: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling or z.shape[0] != 1:
            return original_decode_clip(z)

        coords, y_indices, y_overlaps, x_indices, x_overlaps = geometry(z)
        ratio = vae.spatial_compression_ratio
        # Contiguous blocks, not round-robin: `all_gather` concatenates rank 0's block, then rank
        # 1's, and so on, so contiguous assignment is what puts the tiles back in tile order for
        # free. The list is padded up to an equal count per rank so the gather stays a plain
        # fixed-size one; the padding entries are recomputed duplicates of the last tile, at most
        # world-1 of them, and are dropped after the gather.
        per_rank = (len(coords) + world - 1) // world
        padded = coords + [coords[-1]] * (per_rank * world - len(coords))
        mine = padded[rank * per_rank: (rank + 1) * per_rank]

        slices = [
            z[..., y // ratio: y // ratio + y_len // ratio,
              x // ratio: x // ratio + x_len // ratio]
            for y, y_len, x, x_len in mine
        ]
        if batched:
            local_stack = vae.decoder(vae.post_quant_conv(torch.cat(slices, dim=0)))
        else:
            local_stack = torch.stack(
                [vae.decoder(vae.post_quant_conv(s)) for s in slices], dim=0).squeeze(1)

        flat = gather_tiles(local_stack, len(coords))

        # `_stitch_tiles` only touches dims -2/-1, so a missing batch axis would not raise — it
        # would return `(3, T, H, W)` and `_decode` would then slice the height axis where it means
        # to slice frames. Restore the axis explicitly.
        return stitch(flat, 0, y_indices, y_overlaps, x_indices, x_overlaps)

    vae._decode_clip = sharded_decode_clip

    def globally_sharded_decode(z: torch.Tensor) -> torch.Tensor:
        nonlocal global_gate_pending
        if not global_batch or not batched or not vae.use_tiling or z.shape[0] != 1:
            return original_decode(z)

        original_z = z
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        temporal_ratio = vae.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        coords, y_indices, y_overlaps, x_indices, x_overlaps = geometry(z)
        ratio = vae.spatial_compression_ratio
        tile_inputs = []
        for chunk_index in range(num_chunks):
            start = chunk_index * tokens_chunk_size
            clip = z[:, :, start : start + tokens_chunk_size + vae.token_overlap]
            tile_inputs.extend(
                clip[..., y // ratio : y // ratio + y_len // ratio,
                     x // ratio : x // ratio + x_len // ratio]
                for y, y_len, x, x_len in coords
            )

        logical_tiles = len(tile_inputs)
        per_rank = (logical_tiles + world - 1) // world
        padded = tile_inputs + [tile_inputs[-1]] * (per_rank * world - logical_tiles)
        mine = padded[rank * per_rank : (rank + 1) * per_rank]
        local_stack = vae.decoder(vae.post_quant_conv(torch.cat(mine, dim=0)))
        flat = gather_tiles(local_stack, logical_tiles)

        decoded_chunks = []
        overlap = None
        tiles_per_clip = len(coords)
        for chunk_index in range(num_chunks):
            clip = stitch(
                flat,
                chunk_index * tiles_per_clip,
                y_indices,
                y_overlaps,
                x_indices,
                x_overlaps,
            )
            for part in range(int(token_drop > 0) + 1):
                frame_start = part * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, vae.frame_pre_padding :]
                if part == 0:
                    if overlap is not None:
                        chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        dec = torch.cat(decoded_chunks, dim=2)
        if pad_tokens > 0:
            intra_tail = vae.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0
                else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        if global_gate_pending:
            reference = original_decode(original_z)
            difference = dec.float() - reference.float()
            maximum = float(difference.abs().max().item())
            mse = float(difference.square().mean().item())
            rel_l2 = float(
                (torch.linalg.vector_norm(difference)
                 / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)).item()
            )
            psnr_range2 = math.inf if mse == 0.0 else 10.0 * math.log10(4.0 / mse)
            limit = float(os.environ.get("H3_VAE_GLOBAL_GATE_MAX_ABS", "0.02"))
            passed = maximum <= limit
            verdict = torch.tensor(int(passed), device=dec.device, dtype=torch.int32)
            dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group)
            passed = bool(verdict.item())
            if rank == 0:
                print(
                    "[h3opt.vae] global batch correctness gate "
                    f"{'PASS' if passed else 'FAIL'} max_abs={maximum} "
                    f"mse={mse} rel_l2={rel_l2} psnr_range2_db={psnr_range2} "
                    f"max_abs_limit={limit}",
                    flush=True,
                )
            if not passed:
                raise RuntimeError(
                    f"global VAE batching exceeded max_abs limit: {maximum} > {limit}"
                )
            global_gate_pending = False
        return dec

    vae._decode = globally_sharded_decode

    def uninstall():
        vae._decode = original_decode
        vae._decode_clip = original_decode_clip
        vae._encode_clip = original_encode_clip
        for u in reversed(undo):
            u()

    return uninstall
