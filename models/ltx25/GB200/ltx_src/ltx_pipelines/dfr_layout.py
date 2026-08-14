"""DFR canvas layout: keyframe segment grid, temporal tile ranges, and the latent stitch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch

from ltx_core.types import VIDEO_SCALE_FACTORS

SEGMENT_CANDIDATES = (24, 32)

# Lead-in for non-first tiles, in canvas segments. A tile's local latent 0 is an *image* latent (one
# pixel frame) and its local latent 1 was denoised against that image; neither may be spliced into the
# mid-canvas stream. One segment of lead-in puts both inside the discarded prefix while keeping the
# window start on a keyframe anchor, which the KeyFrame@0 lock requires.
TILE_LEAD_SEGMENTS = 1


class TileRange(NamedTuple):
    """One temporal denoise tile in global pixel / latent coordinates.
    ``pixel_start`` / ``pixel_end`` are inclusive, ``latent_end_exclusive`` is half-open. Non-first
    tiles start ``TILE_LEAD_SEGMENTS`` segments before the region they own, so the seam shared with the
    previous tile falls inside the window. ``anchor_kf_global`` are the seam keyframes in the window,
    which includes ``pixel_start`` on every tile but the first -- frame 0 is not a keyframe, so the
    first tile's window start contributes no anchor. ``slot_kf_global`` are the mid-segment positions
    this window invents.
    """

    pixel_start: int
    pixel_end: int
    latent_start: int
    latent_end_exclusive: int
    anchor_kf_global: tuple[int, ...]
    slot_kf_global: tuple[int, ...]
    drop_latent_prefix: int


def choose_segment_length(content_frames: int) -> int:
    """Pick the keyframe segment length from ``SEGMENT_CANDIDATES``, preferring whichever pads least.
    ``content_frames`` is ``num_frames - 1``. Ties keep the larger segment.
    """
    if content_frames < 1:
        raise ValueError(f"content_frames must be >= 1, got {content_frames}")

    def _pad(segment: int) -> int:
        return (segment - content_frames % segment) % segment

    best = int(SEGMENT_CANDIDATES[0])
    best_pad = _pad(best)
    for candidate in SEGMENT_CANDIDATES[1:]:
        segment = int(candidate)
        pad = _pad(segment)
        if pad < best_pad or (pad == best_pad and segment > best):
            best, best_pad = segment, pad
    return best


def resolve_canvas(
    num_frames: int,
    *,
    temporal_scale: int = VIDEO_SCALE_FACTORS.time,
) -> tuple[int, int, list[int]]:
    """Pad ``(num_frames - 1)`` up to a multiple of the segment length, returning ``(N', S, positions)``.
    Positions are ``[S, 2S, ..., N' - 1]``: frame 0 is excluded (already a single-pixel-frame token
    under causal encoding) and the terminal frame is included.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(f"num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0 (got {num_frames})")

    content = num_frames - 1
    if content == 0:
        raise ValueError("The canvas needs at least 2 pixel frames")

    segment = choose_segment_length(content)
    content_padded = content + (segment - content % segment) % segment
    positions = [segment * index for index in range(1, content_padded // segment + 1)]
    return content_padded + 1, segment, positions


def pixel_to_latent_index(pixel_frame: int, temporal_scale: int = VIDEO_SCALE_FACTORS.time) -> int:
    """Map an x8-border pixel frame to its latent index."""
    if pixel_frame < 0:
        raise ValueError(f"pixel_frame must be >= 0, got {pixel_frame}")
    if pixel_frame != 0 and pixel_frame % temporal_scale != 0:
        raise ValueError(f"pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border")
    return pixel_frame // temporal_scale


def _owned_segment_counts(n_segments: int, num_tiles: int) -> list[int]:
    """Split ``n_segments`` into ``num_tiles`` contiguous owned runs, largest first."""
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    base, remainder = divmod(n_segments, num_tiles)
    return [base + (1 if index < remainder else 0) for index in range(num_tiles)]


def _build_tile(
    boundaries: Sequence[int],
    own_lo: int,
    own_hi: int,
    *,
    lead_segments: int,
    temporal_scale: int,
) -> TileRange:
    """One window owning segments ``[own_lo, own_hi)``, preceded by ``lead_segments`` of lead-in."""
    window_lo = max(0, own_lo - lead_segments)
    pixel_start = int(boundaries[window_lo])
    pixel_end = int(boundaries[own_hi])
    latent_start = pixel_to_latent_index(pixel_start, temporal_scale)

    # Handover is exactly at the shared keyframe: the previous tile keeps the seam latent (the 8-frame
    # token ending on the KF mark) and this tile resumes strictly after it, so the prefix drops the
    # lead-in plus that seam latent.
    drop_latent_prefix = pixel_to_latent_index(int(boundaries[own_lo]), temporal_scale) - latent_start
    if own_lo > 0:
        drop_latent_prefix += 1

    return TileRange(
        pixel_start=pixel_start,
        pixel_end=pixel_end,
        latent_start=latent_start,
        latent_end_exclusive=pixel_to_latent_index(pixel_end, temporal_scale) + 1,
        anchor_kf_global=tuple(int(boundaries[index]) for index in range(window_lo, own_hi + 1) if boundaries[index]),
        slot_kf_global=tuple(
            (int(boundaries[index]) + int(boundaries[index + 1])) // 2 for index in range(window_lo, own_hi)
        ),
        drop_latent_prefix=drop_latent_prefix,
    )


def tile_ranges(
    seam_positions: Sequence[int],
    num_frames: int,
    num_tiles: int,
    *,
    temporal_scale: int = VIDEO_SCALE_FACTORS.time,
    lead_segments: int = TILE_LEAD_SEGMENTS,
) -> list[TileRange]:
    """Partition the canvas into ``num_tiles`` keyframe-seam tiles, gapless, with a lead-in.
    Owned segment runs are contiguous; each non-first window additionally reaches ``lead_segments``
    back so it denoises through the shared seam keyframe. ``num_tiles`` is clamped to the segment count.
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    if not seam_positions:
        raise ValueError("seam_positions must be non-empty")
    if int(seam_positions[-1]) != num_frames - 1:
        raise ValueError(f"Last seam must be the terminal frame {num_frames - 1}, got {seam_positions[-1]}")
    if lead_segments < 1:
        raise ValueError(f"lead_segments must be >= 1, got {lead_segments}")

    boundaries = [0, *[int(position) for position in seam_positions]]
    for index in range(1, len(boundaries)):
        span = boundaries[index] - boundaries[index - 1]
        if span <= 0:
            raise ValueError(f"seam_positions must be strictly increasing, got {list(seam_positions)}")
        if span % temporal_scale != 0:
            raise ValueError(f"Segment span {span} is not a multiple of temporal scale {temporal_scale}")
        if span // temporal_scale < 2:
            raise ValueError(f"Segment span {span} is under 2 latent frames, too short to carry a tile lead-in")

    n_segments = len(boundaries) - 1
    tiles: list[TileRange] = []
    own_lo = 0
    for index, count in enumerate(_owned_segment_counts(n_segments, min(num_tiles, n_segments))):
        tiles.append(
            _build_tile(
                boundaries,
                own_lo,
                own_lo + count,
                lead_segments=lead_segments if index > 0 else 0,
                temporal_scale=temporal_scale,
            )
        )
        own_lo += count
    return tiles


def stitch_tile_latents(
    tile_latents: Sequence[torch.Tensor],
    ranges: Sequence[TileRange],
) -> torch.Tensor:
    """Concatenate tile video latents along T, each tile contributing ``latent[drop_latent_prefix:]``."""
    if len(tile_latents) != len(ranges):
        raise ValueError(f"Expected {len(ranges)} tile latents, got {len(tile_latents)}")
    if not tile_latents:
        raise ValueError("tile_latents must be non-empty")

    pieces: list[torch.Tensor] = []
    for latent, tile in zip(tile_latents, ranges, strict=True):
        if latent.ndim != 5:
            raise ValueError(f"Expected tile latent (B, C, T, H, W), got {tuple(latent.shape)}")
        expected_t = tile.latent_end_exclusive - tile.latent_start
        if latent.shape[2] != expected_t:
            raise ValueError(
                f"Tile latent T={latent.shape[2]} != expected {expected_t} "
                f"for range [{tile.latent_start}, {tile.latent_end_exclusive})"
            )
        if not 0 <= tile.drop_latent_prefix < latent.shape[2]:
            raise ValueError(f"drop_latent_prefix={tile.drop_latent_prefix} invalid for tile T={latent.shape[2]}")
        pieces.append(latent[:, :, tile.drop_latent_prefix :])
    return torch.cat(pieces, dim=2)


def remap_positions_to_local(positions: Sequence[int], pixel_start: int) -> list[int]:
    """Shift global pixel indices into a tile-local frame (``local = global - pixel_start``)."""
    return [int(position) - pixel_start for position in positions]
