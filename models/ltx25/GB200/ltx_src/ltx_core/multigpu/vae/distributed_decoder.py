"""Distributed video decoder that partitions the latent across ranks.
Tiles are assigned to ranks via round-robin, so the number of tiles
may exceed the number of GPUs (e.g. 16 tiles on 4 GPUs = 4 tiles per
rank).  Each rank decodes its assigned tiles sequentially.  Workers
put their list of decoded tiles into a ``mp.Queue`` (CUDA IPC —
zero-copy handle sharing).  The driver collects all tiles, blends
overlap zones, and returns temporal batches distributed across devices.
The tiling configuration comes from ``MGPUConfig.vae_tiling`` (set at
construction time), NOT from the pipeline's SGPU tiling kwarg.  MGPU
tiling controls parallelism; SGPU tiling controls single-GPU VRAM
management — they are independent concerns.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from einops import rearrange
from torch.multiprocessing import Queue

from ltx_core.model.disposable import Disposable
from ltx_core.model.video_vae.video_vae import (
    VideoDecoder,
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
)
from ltx_core.tiling import (
    Tile,
    TileCountConfig,
    TilingConfig,
    compute_summed_weights,
    create_tiles,
    masks_are_complementary,
    scale_by_masks_1d,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

logger = logging.getLogger(__name__)


def _sgpu_has_temporal_tiling(tiling_config: TilingConfig | None, num_frames: int) -> bool:
    """True when SGPU tiling would produce more than one temporal chunk for ``num_frames``."""
    return tiling_config is not None and tiling_config.video_chunks_number(num_frames) > 1


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedTile:
    """A VAE-decoded tile with pixel-space placement.
    Attributes:
        pixels: ``[F_tile, H_tile, W_tile, C]`` in the decoder's native dtype.
        pixel_tile: Carries ``out_coords`` (f, h, w slices) and separable ``masks_1d``.
    """

    pixels: torch.Tensor
    pixel_tile: Tile


# ------------------------------------------------------------------
# Tile construction helpers
# ------------------------------------------------------------------


def _to_decoded_tile(
    pixels: torch.Tensor,
    tile: Tile,
) -> DecodedTile:
    """Wrap ``decode_video`` output ``[F, H, W, C]`` in ``[0, 1]`` as a :class:`DecodedTile`."""
    return DecodedTile(pixels=pixels, pixel_tile=tile)


def gather_frames(
    tiles: list[DecodedTile],
    total_frames: int,
    output_height: int,
    output_width: int,
    num_temporal_batches: int,
    world_size: int,
    weights: torch.Tensor | None = None,
    device_fn: Callable[[int], str | torch.device] | None = None,
) -> Iterator[torch.Tensor]:
    """Assemble decoded tiles into temporal batches distributed across GPUs.
    Each temporal batch is allocated on the device returned by *device_fn(batch_index)*.
    By default batches are placed round-robin on ``cuda:0`` … ``cuda:<world_size-1>``.
    Blending multiplies by separable ``mf*mh*mw``. When *weights* is ``None``, masks
    are assumed complementary and the denominator pass is skipped; otherwise
    ``output`` is divided by *weights*.
    """
    if device_fn is None:
        device_fn = lambda b: f"cuda:{b % world_size}"  # noqa: E731

    batch_size = (total_frames + num_temporal_batches - 1) // num_temporal_batches

    for b in range(num_temporal_batches):
        batch_range = slice(b * batch_size, min((b + 1) * batch_size, total_frames))
        batch_len = batch_range.stop - batch_range.start
        if batch_len <= 0:
            break

        device = device_fn(b)
        dtype = tiles[0].pixels.dtype
        output = torch.zeros(batch_len, output_height, output_width, 3, device=device, dtype=dtype)

        for tile in tiles:
            f_slice, h_slice, w_slice = tile.pixel_tile.out_coords

            overlap = slice(max(batch_range.start, f_slice.start), min(batch_range.stop, f_slice.stop))
            if overlap.start >= overlap.stop:
                continue

            tile_frames = slice(overlap.start - f_slice.start, overlap.stop - f_slice.start)
            out_frames = slice(overlap.start - batch_range.start, overlap.stop - batch_range.start)

            # Blend weights stay float32 so the multiply promotes bf16/fp16 pixels.
            mf, mh, mw = (m.to(device=device, dtype=torch.float32) for m in tile.pixel_tile.masks_1d)
            mf = mf[tile_frames]
            pix = tile.pixels[tile_frames].to(device=device, non_blocking=True)
            # Channel axis is not on the pixel Tile; length-1 ones broadcasts over C.
            output[out_frames, h_slice, w_slice, :] += scale_by_masks_1d(
                pix, (mf, mh, mw, torch.ones(1, device=device, dtype=torch.float32))
            )

        if weights is not None:
            batch_weights = weights[batch_range.start : batch_range.stop].to(device=device)
            output.div_(batch_weights[:, :, :, None])
        yield output


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class DistributedVideoDecoder(torch.nn.Module, Disposable):
    """Distributed VAE decoder with queue-based tile collection.
    Wraps any :class:`~ltx_core.model.video_vae.video_vae.VideoDecoder`
    (conv or diffusion) and decodes each MGPU tile via public
    ``decode_video`` (already ``[F, H, W, C]`` in ``[0, 1]``). Full-video
    assembly is :func:`gather_frames` on the driver.
    All ranks decode their latent tile in parallel.  Workers send
    their :class:`DecodedTile` to the driver rank via the shared
    ``mp.Queue`` (CUDA IPC — zero-copy).  The driver collects all
    tiles, blends overlapping regions, and returns temporal batches
    as an iterator.
    Parameters
    ----------
    decoder:
        The real (local) ``VideoDecoder`` instance.
    queue:
        ``mp.Queue`` shared across all ranks for CUDA IPC tile transfer.
    vae_group:
        NCCL process group for the VAE ranks. Used to derive
        ``rank`` and ``world_size`` within the group.
    vae_tiling:
        MGPU tiling config that determines how the latent is split.
    driver_rank:
        Group-local rank of the driver process (the rank that collects
        and assembles tiles).
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int = 0,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.queue = queue
        self.vae_group = vae_group
        self.rank = dist.get_rank(vae_group)
        self.world_size = dist.get_world_size(vae_group)
        self.vae_tiling = vae_tiling
        self.driver_rank = driver_rank

    @property
    def video_downscale_factors(self) -> SpatioTemporalScaleFactors:
        return self.decoder.video_downscale_factors

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Non-tiled path: full local ``decode_video``, returned as ``[B, C, F, H, W]``.
        Uses the public decoder API (Conv: single-pass; Diff: full Euler loop)
        rather than ``decoder.forward``, which is one diffusion step on Diff.
        """
        chunks = list(self.decoder.decode_video(sample, tiling_config=None, generator=generator))
        pixels = torch.cat(chunks, dim=0)  # [F, H, W, C] in [0, 1]
        return rearrange(pixels, "f h w c -> 1 c f h w")

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        device_fn: Callable[[int], str | torch.device] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Distributed decode — all ranks decode, driver assembles.
        Not a generator so that worker side-effects (decode + queue.put)
        execute eagerly regardless of whether the caller iterates.
        1. Each rank decodes its latent tile (with optional intra-GPU tiling).
        2. Workers send their :class:`DecodedTile` to the driver via the queue.
        3. The driver collects all tiles, blends overlaps, and returns
           temporal batches distributed across GPUs.
        """
        latent_shape = VideoLatentShape.from_torch_shape(latent.shape)
        scale = self.decoder.video_downscale_factors
        full_shape = latent_shape.upscale(scale)

        # Phase 1: each rank decodes its assigned tiles.
        my_tiles = self._decode_tiles(latent, latent_shape, scale, generator, tiling_config)

        # Phase 2: workers send tiles to driver.
        if self.rank != self.driver_rank:
            self.queue.put((self.rank, my_tiles))
            return iter([])

        # Phase 3: driver collects and assembles.
        all_tiles = self._collect_tiles(my_tiles)
        if masks_are_complementary(
            [t.pixel_tile for t in all_tiles],
            (full_shape.frames, full_shape.height, full_shape.width),
        ):
            weights = None
        else:
            logger.warning(
                "VAE blend masks are not complementary; falling back to dense [F,H,W] "
                "weight normalization (expensive for large videos)."
            )
            weights = compute_summed_weights(
                [t.pixel_tile for t in all_tiles],
                (full_shape.frames, full_shape.height, full_shape.width),
            )
        batches = gather_frames(
            all_tiles,
            full_shape.frames,
            full_shape.height,
            full_shape.width,
            self.world_size,
            self.world_size,
            weights,
            device_fn=device_fn,
        )
        return batches

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_tiles(
        self,
        latent: torch.Tensor,
        latent_shape: VideoLatentShape,
        scale: SpatioTemporalScaleFactors,
        generator: torch.Generator | None,
        tiling_config: TilingConfig | None = None,
    ) -> list[DecodedTile]:
        """Decode this rank's assigned latent tiles and convert to :class:`DecodedTile` list."""
        t_split, h_split, w_split = self.vae_tiling.to_splitters(scale, causal_temporal=True)
        all_tiles = create_tiles(
            torch.Size([latent_shape.frames, latent_shape.height, latent_shape.width]),
            splitters=[t_split, h_split, w_split],
            mappers=[
                to_mapping_operation(map_temporal_slice, scale.time),
                to_mapping_operation(map_spatial_slice, scale.height),
                to_mapping_operation(map_spatial_slice, scale.width),
            ],
        )
        my_tiles = [t for i, t in enumerate(all_tiles) if i % self.world_size == self.rank]
        if self.vae_tiling.frames.num_tiles > 1 and tiling_config is not None:
            for tile in my_tiles:
                latent_f = tile.in_coords[0].stop - tile.in_coords[0].start
                pixel_f = (latent_f - 1) * scale.time + 1
                if _sgpu_has_temporal_tiling(tiling_config, pixel_f):
                    raise ValueError(
                        "Cannot combine multi-GPU temporal tiling (vae_tiling.frames.num_tiles > 1) "
                        "with single-GPU temporal tiling that would split this rank's tile "
                        f"({pixel_f} frames → {tiling_config.video_chunks_number(pixel_f)} chunks). "
                        "Use only one to avoid causal decoding conflicts."
                    )
        decoded = []
        for tile in my_tiles:
            # One MGPU tile only — cat stitches SGPU temporal yields of this tile,
            # not the full video (full assembly is gather_frames on the driver).
            latent_slice = latent[:, :, tile.in_coords[0], tile.in_coords[1], tile.in_coords[2]]
            chunks = list(self.decoder.decode_video(latent_slice, tiling_config, generator=generator))
            pixels = torch.cat(chunks, dim=0)  # [F, H, W, C] in [0, 1]
            decoded.append(_to_decoded_tile(pixels, tile))
        return decoded

    def _collect_tiles(self, driver_tiles: list[DecodedTile]) -> list[DecodedTile]:
        """Collect tiles from all workers via the queue. Returns flat list of all tiles.
        Sorted by rank so the downstream reduction in ``gather_frames``
        (in-place ``+=`` over overlapping pixel regions) processes tiles in a
        fixed order. Queue-arrival order would otherwise vary run-to-run and
        yield 1-ulp bf16 drift from non-associative floating-point summation.
        """
        per_rank: dict[int, list[DecodedTile]] = {self.driver_rank: driver_tiles}
        for _ in range(self.world_size - 1):
            worker_rank, worker_tiles = self.queue.get()
            per_rank[worker_rank] = worker_tiles
        result: list[DecodedTile] = []
        for rank in sorted(per_rank):
            result.extend(per_rank[rank])
        return result
