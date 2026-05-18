"""Video modality tiling helpers.
Provides :class:`VideoModalityTilingHelper` — a stateless helper that
tiles and blends video :class:`Modality` token sequences by
spatial/temporal region.  Tile geometry is represented by the existing
:class:`Tile` NamedTuple from :mod:`ltx_core.tiling`; no distributed
primitives are required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from diffusion.refiner.vendor.ltx_core.model.transformer.modality import Modality
from diffusion.refiner.vendor.ltx_core.tiling import (
    Tile,
    TileCountConfig,
    create_tiles,
    identity_mapping_operation,
    split_by_count,
)
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import VideoLatentShape


@dataclass(frozen=True)
class TilingContext:
    """Opaque context produced by :meth:`VideoModalityTilingHelper.tile_modality`.
    Carries the token-level keep mask and per-conditioning-token blend
    weights needed by :meth:`~VideoModalityTilingHelper.blend`.
    """

    keep_mask: torch.Tensor
    cond_blend_weights: torch.Tensor | None
    """``(num_kept_cond,)`` — weight for each kept conditioning token,
    equal to ``1 / num_tiles_that_keep_this_token``.  ``None`` when
    there are no conditioning tokens."""


class VideoModalityTilingHelper:
    """Stateless helper that tiles and blends video :class:`Modality` sequences.
    Constructed once with a :class:`TileCountConfig` and
    :class:`VideoLatentTools`.  Tiles are computed at construction and
    available via the :attr:`tiles` property.  Use :meth:`tile_modality`
    and :meth:`blend` with any tile from that list.
    Usage::
        helper = VideoModalityTilingHelper(tiling, video_tools)
        for tile in helper.tiles:
            tiled_mod, ctx = helper.tile_modality(modality, tile)
            result = run_model(tiled_mod)
            helper.blend(result, tile, ctx, output=output)
    """

    def __init__(self, tiling: TileCountConfig, video_tools: VideoLatentTools) -> None:
        self._patchifier = video_tools.patchifier
        self._latent_shape = video_tools.target_shape
        self._num_generated_tokens = self._patchifier.get_token_count(self._latent_shape)
        self._tiles = create_tiles(
            torch.Size([self._latent_shape.frames, self._latent_shape.height, self._latent_shape.width]),
            splitters=[
                split_by_count(tiling.frames.num_tiles, tiling.frames.overlap),
                split_by_count(tiling.height.num_tiles, tiling.height.overlap),
                split_by_count(tiling.width.num_tiles, tiling.width.overlap),
            ],
            mappers=[identity_mapping_operation] * 3,
        )

    @property
    def tiles(self) -> list[Tile]:
        """All tiles for the configured tiling layout."""
        return self._tiles

    # -- tile modality -----------------------------------------------------

    def tile_modality(self, modality: Modality, tile: Tile) -> tuple[Modality, TilingContext]:
        """Slice *modality* to the tokens covered by *tile*.
        Selects generated tokens belonging to the tile's spatial region
        and conditioning tokens that overlap with the tile (or have
        negative time coordinates).
        Returns:
            A ``(tiled_modality, context)`` tuple.  Pass *context* to
            :meth:`blend` together with the model output.
        """
        keep_mask = self._keep_mask(modality, tile)

        tile_attention_mask = None
        if modality.attention_mask is not None:
            keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(1)
            tile_attention_mask = modality.attention_mask[:, keep_indices, :][:, :, keep_indices]

        tiled = replace(
            modality,
            latent=modality.latent[:, keep_mask, :],
            timesteps=modality.timesteps[:, keep_mask],
            positions=modality.positions[:, :, keep_mask, :],
            attention_mask=tile_attention_mask,
        )

        cond_blend_weights = None
        num_total = modality.latent.shape[1]
        if num_total > self._num_generated_tokens:
            cond_keep = keep_mask[self._num_generated_tokens :]
            # Count how many tiles keep each conditioning token.
            cond_counts = torch.zeros(cond_keep.sum(), dtype=torch.float32)
            for t in self._tiles:
                other_mask = self._keep_mask(modality, t)
                other_cond = other_mask[self._num_generated_tokens :]
                # Map other tile's kept cond tokens into this tile's kept subset.
                cond_counts += other_cond[cond_keep].float()
            cond_blend_weights = 1.0 / cond_counts

        return tiled, TilingContext(keep_mask=keep_mask, cond_blend_weights=cond_blend_weights)

    # -- blend -------------------------------------------------------------

    def blend(
        self,
        tile_to_blend: torch.Tensor,
        tile: Tile,
        context: TilingContext,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend-weight tile results and accumulate into the full token space.
        Premultiplied (blend-weighted) data is **added** to *output*,
        allowing multiple tiles to be accumulated into the same buffer.
        Args:
            tile_to_blend: Denoised tile tensor ``(B, num_tile_tokens, D)``,
                where the first ``_tile_generated_token_count(tile)``
                entries are generated tokens and the remainder are
                conditioning tokens.
            tile: The :class:`Tile` that was used in :meth:`tile_modality`.
            context: The :class:`TilingContext` returned by :meth:`tile_modality`.
            output: Optional pre-allocated output tensor.  When provided
                its shape must be ``(B, num_total_tokens, D)`` and the
                blended tile is **added** into it.  When ``None`` a new
                zero-filled tensor is created.
        Returns:
            The output tensor with the blended tile added at the correct
            positions.
        """
        batch, _, dim = tile_to_blend.shape
        num_tile_gen = self._tile_generated_token_count(tile)
        gen_indices = self._generated_token_indices(tile)

        num_total_tokens = context.keep_mask.shape[0]
        expected_shape = (batch, num_total_tokens, dim)

        if output is not None:
            if output.shape != expected_shape:
                raise ValueError(f"Expected output shape {expected_shape}, got {output.shape}")
            result = output
        else:
            result = torch.zeros(*expected_shape, device=tile_to_blend.device, dtype=tile_to_blend.dtype)

        # Blend mask is (tile_F, tile_H, tile_W) — one weight per token in row-major order.
        blend_weights = tile.blend_mask.reshape(-1).to(device=tile_to_blend.device, dtype=tile_to_blend.dtype)
        tile_gen = tile_to_blend[:, :num_tile_gen, :] * blend_weights[None, :, None]

        result[:, gen_indices, :] += tile_gen

        # Scatter kept conditioning tokens, weighted by 1/N where N is
        # the number of tiles that keep each token (so they sum to 1).
        if num_total_tokens > self._num_generated_tokens and context.cond_blend_weights is not None:
            cond_keep = context.keep_mask[self._num_generated_tokens :]
            cond_indices = self._num_generated_tokens + cond_keep.nonzero(as_tuple=False).squeeze(1)
            weights = context.cond_blend_weights.to(device=tile_to_blend.device, dtype=tile_to_blend.dtype)
            result[:, cond_indices, :] += tile_to_blend[:, num_tile_gen:, :] * weights[None, :, None]

        return result

    # -- private -----------------------------------------------------------

    def _tile_generated_token_count(self, tile: Tile) -> int:
        """Number of generated tokens in *tile*."""
        frame_slice, height_slice, width_slice = tile.in_coords
        tile_shape = VideoLatentShape(
            batch=self._latent_shape.batch,
            channels=self._latent_shape.channels,
            frames=frame_slice.stop - frame_slice.start,
            height=height_slice.stop - height_slice.start,
            width=width_slice.stop - width_slice.start,
        )
        return self._patchifier.get_token_count(tile_shape)

    def _generated_token_indices(self, tile: Tile) -> torch.Tensor:
        """Flat token indices of *tile*'s generated tokens in the full sequence."""
        frame_slice, height_slice, width_slice = tile.in_coords
        f = torch.arange(frame_slice.start, frame_slice.stop)
        h = torch.arange(height_slice.start, height_slice.stop)
        w = torch.arange(width_slice.start, width_slice.stop)
        return (
            f[:, None, None] * self._latent_shape.height * self._latent_shape.width
            + h[None, :, None] * self._latent_shape.width
            + w[None, None, :]
        ).reshape(-1)

    def _keep_mask(self, modality: Modality, tile: Tile) -> torch.Tensor:
        """Boolean mask ``(num_total_tokens,)`` — True for tokens the tile processes.
        Generated tokens are selected by grid position.  Conditioning
        tokens are kept when their ``[start, end)`` intervals overlap
        the tile in all three dimensions, or when they have a negative
        time coordinate (reference tokens).
        """
        num_total = modality.latent.shape[1]
        mask = torch.zeros(num_total, dtype=torch.bool)

        gen_indices = self._generated_token_indices(tile)
        mask[gen_indices] = True

        if num_total > self._num_generated_tokens:
            gen_positions = modality.positions[:, :, gen_indices, :]  # (B, 3, num_tile_gen, 2)
            tile_start = gen_positions[..., 0].amin(dim=2)  # (B, 3)
            tile_end = gen_positions[..., 1].amax(dim=2)  # (B, 3)

            cond_positions = modality.positions[:, :, self._num_generated_tokens :, :]  # (B, 3, num_cond, 2)

            overlaps = (cond_positions[..., 0] < tile_end.unsqueeze(2)) & (
                cond_positions[..., 1] > tile_start.unsqueeze(2)
            )  # (B, 3, num_cond)
            overlaps_all_dims = overlaps.all(dim=1)  # (B, num_cond)

            has_negative_time = cond_positions[:, 0, :, 0] < 0  # (B, num_cond)

            keep_cond = (overlaps_all_dims | has_negative_time).any(dim=0)  # (num_cond,)
            mask[self._num_generated_tokens :] = keep_cond

        return mask
