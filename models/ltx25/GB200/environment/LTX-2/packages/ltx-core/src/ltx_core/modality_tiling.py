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

from ltx_core.model.transformer.modality import Modality
from ltx_core.tiling import Tile, TileCountConfig, create_tiles, identity_mapping_operation, split_by_count
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape


@dataclass(frozen=True)
class TilingContext:
    """Opaque context produced by :meth:`VideoModalityTilingHelper.tile_modality`.
    Carries the token-level keep indices and per-conditioning-token blend
    weights needed by :meth:`~VideoModalityTilingHelper.blend`.
    """

    keep_indices: torch.Tensor
    """``(num_kept,)`` int64 — sorted indices of tokens the tile processes."""
    num_total_tokens: int
    """Total number of tokens in the full (untiled) sequence."""
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

    def tile_modality(
        self, modality: Modality, tile: Tile, *, normalize_positions: bool = True
    ) -> tuple[Modality, TilingContext]:
        """Slice *modality* to the tokens covered by *tile*.
        Selects generated tokens belonging to the tile's spatial region
        and conditioning tokens that overlap with the tile (or have
        negative time coordinates).
        Args:
            normalize_positions: When True, shift all positions so the
                tile's generated tokens start at zero in every dimension.
        Returns:
            A ``(tiled_modality, context)`` tuple.  Pass *context* to
            :meth:`blend` together with the model output.
        """
        device = modality.positions.device
        gen_indices = self._generated_token_indices(tile, device=device)
        num_total = modality.latent.shape[1]

        cond_blend_weights: torch.Tensor | None = None
        if num_total > self._num_generated_tokens:
            keep_per_tile_cond = self._all_tiles_cond_keep(modality)  # (num_tiles, num_cond) bool
            tile_idx = next((i for i, t in enumerate(self._tiles) if t.in_coords == tile.in_coords), None)
            if tile_idx is None:
                raise RuntimeError(
                    f"Tile with in_coords={tile.in_coords} is not in this helper's tile set; "
                    f"pass a tile obtained from `helper.tiles`."
                )
            my_cond_keep = keep_per_tile_cond[tile_idx]
            cond_indices = self._num_generated_tokens + my_cond_keep.nonzero(as_tuple=False).squeeze(1)
            keep_indices = torch.cat([gen_indices, cond_indices])
            total_keepers = keep_per_tile_cond.sum(dim=0).float()  # (num_cond,)
            cond_blend_weights = 1.0 / total_keepers[my_cond_keep]
        else:
            keep_indices = gen_indices

        tile_attention_mask = None
        if modality.attention_mask is not None:
            tile_attention_mask = modality.attention_mask[:, keep_indices, :][:, :, keep_indices]

        positions = modality.positions[:, :, keep_indices, :]
        if normalize_positions:
            num_tile_gen = self._tile_generated_token_count(tile)
            gen_pos = positions[:, :, :num_tile_gen, :]  # (B, 3, num_tile_gen, 2)
            offset = gen_pos[..., 0].amin(dim=2, keepdim=True).unsqueeze(-1)  # (B, 3, 1, 1)
            positions = positions - offset

        tiled = replace(
            modality,
            latent=modality.latent[:, keep_indices, :],
            timesteps=modality.timesteps[:, keep_indices],
            positions=positions,
            attention_mask=tile_attention_mask,
        )

        return tiled, TilingContext(
            keep_indices=keep_indices, num_total_tokens=num_total, cond_blend_weights=cond_blend_weights
        )

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
        gen_indices = self._generated_token_indices(tile, device=tile_to_blend.device)

        num_total_tokens = context.num_total_tokens
        expected_shape = (batch, num_total_tokens, dim)

        if output is not None:
            if output.shape != expected_shape:
                raise RuntimeError(f"Expected output shape {expected_shape}, got {output.shape}")
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
            cond_indices = context.keep_indices[context.keep_indices >= self._num_generated_tokens]
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

    def _generated_token_indices(self, tile: Tile, device: torch.device | None = None) -> torch.Tensor:
        """Flat token indices of *tile*'s generated tokens in the full sequence."""
        frame_slice, height_slice, width_slice = tile.in_coords
        f = torch.arange(frame_slice.start, frame_slice.stop, device=device)
        h = torch.arange(height_slice.start, height_slice.stop, device=device)
        w = torch.arange(width_slice.start, width_slice.stop, device=device)
        return (
            f[:, None, None] * self._latent_shape.height * self._latent_shape.width
            + h[None, :, None] * self._latent_shape.width
            + w[None, None, :]
        ).reshape(-1)

    def _all_tiles_cond_keep(self, modality: Modality) -> torch.Tensor:
        """Vectorized (num_tiles, num_cond) bool: which tiles keep each conditioning token.
        A conditioning token is kept by a tile when its ``[start, end)`` interval
        overlaps the tile in all three dimensions, or when it has a negative time
        coordinate (reference token).
        """
        cond_positions = modality.positions[:, :, self._num_generated_tokens :, :]  # (B, 3, num_cond, 2)
        device = cond_positions.device

        # Per-tile (start, end) bounds along each axis; small Python loop (num_tiles <= ~16).
        starts_list: list[torch.Tensor] = []
        ends_list: list[torch.Tensor] = []
        for t in self._tiles:
            gen_idx = self._generated_token_indices(t, device=device)
            gen_positions = modality.positions[:, :, gen_idx, :]  # (B, 3, num_tile_gen, 2)
            starts_list.append(gen_positions[..., 0].amin(dim=2))  # (B, 3)
            ends_list.append(gen_positions[..., 1].amax(dim=2))  # (B, 3)
        tile_starts = torch.stack(starts_list, dim=0)  # (num_tiles, B, 3)
        tile_ends = torch.stack(ends_list, dim=0)  # (num_tiles, B, 3)

        cond_starts = cond_positions[..., 0]  # (B, 3, num_cond)
        cond_ends = cond_positions[..., 1]  # (B, 3, num_cond)
        # Broadcast: (1, B, 3, num_cond) vs (num_tiles, B, 3, 1) -> (num_tiles, B, 3, num_cond).
        overlaps = (cond_starts[None] < tile_ends[..., None]) & (cond_ends[None] > tile_starts[..., None])
        overlaps_all_dims = overlaps.all(dim=2)  # (num_tiles, B, num_cond)
        has_negative_time = (cond_positions[:, 0, :, 0] < 0)[None]  # (1, B, num_cond)
        return (overlaps_all_dims | has_negative_time).any(dim=1)  # (num_tiles, num_cond)
