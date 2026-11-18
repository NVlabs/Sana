from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple, Sequence

import torch


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    """
    Generate a 1D trapezoidal blending mask with linear ramps.
    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.
    Returns:
        A 1D tensor of shape `(length,)` with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


def compute_rectangular_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
) -> torch.Tensor:
    """
    Generate a 1D rectangular (pulse) mask.
    Args:
        length: Output length of the mask.
        left_ramp: Number of elements at the start of the mask to set to 0.
        right_ramp: Number of elements at the end of the mask to set to 0.
    Returns:
        A 1D tensor of shape `(length,)` with values 0 or 1.
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    mask = torch.ones(length)
    if left_ramp > 0:
        mask[:left_ramp] = 0
    if right_ramp > 0:
        mask[-right_ramp:] = 0
    return mask


@dataclass(frozen=True)
class DimensionInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


@dataclass(frozen=True)
class DimensionIntervals:
    """Intervals which a single dimension of the latent space is split into.
    Each interval is defined by its start, end, left ramp, and right ramp.
    The start and end are the indices of the first and last element (exclusive) in the interval.
    Ramps are regions of the interval where the value of the mask tensor is
    interpolated between 0 and 1 for blending with neighboring intervals.
    The left ramp and right ramp values are the lengths of the left and right ramps.
    """

    intervals: list[DimensionInterval]


@dataclass(frozen=True)
class LatentIntervals:
    """Intervals which the latent tensor of given shape is split into.
    Each dimension of the latent space is split into intervals based on the length along said dimension.
    """

    original_shape: torch.Size
    dimension_intervals: tuple[DimensionIntervals, ...]


# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(intervals=[DimensionInterval(start=0, end=length, left_ramp=0, right_ramp=0)])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def untiled_mask_1d() -> torch.Tensor:
    """Length-1 ones that broadcast over an untiled axis (historical ``None`` mask)."""
    return torch.ones(1)


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor]]:
    return [slice(0, None)], [untiled_mask_1d()]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------


def _grow_last_tile_to_min(intervals: list[DimensionInterval], min_tile_size: int) -> list[DimensionInterval]:
    """Grow a short last tile left to ``min_tile_size``; widen penultimate ``right_ramp``."""
    if len(intervals) <= 1:
        return list(intervals)
    last = intervals[-1]
    if last.end - last.start >= min_tile_size:
        return list(intervals)
    new_start = last.end - min_tile_size
    prev = intervals[-2]
    new_overlap = prev.end - new_start
    return [
        *intervals[:-2],
        replace(prev, right_ramp=new_overlap),
        replace(last, start=new_start, left_ramp=new_overlap),
    ]


def _validate_tile_intervals(intervals: list[DimensionInterval], *, dim_size: int, min_tile_size: int) -> None:
    """Validate coverage, ramp/overlap consistency, and ``min_tile_size``."""
    if not intervals or intervals[0].start != 0 or intervals[-1].end != dim_size:
        raise ValueError(f"tiles must cover [0, {dim_size})")
    for i, iv in enumerate(intervals):
        length = iv.end - iv.start
        if length < min_tile_size:
            raise ValueError(f"tile {i} length {length} is below min_tile_size={min_tile_size}")
        if iv.left_ramp < 0 or iv.right_ramp < 0 or iv.left_ramp > length or iv.right_ramp > length:
            raise ValueError(f"tile {i} has invalid ramps: left={iv.left_ramp}, right={iv.right_ramp}, length={length}")
        if i == 0:
            continue
        overlap = intervals[i - 1].end - iv.start
        if overlap < 0 or intervals[i - 1].right_ramp != overlap or iv.left_ramp != overlap:
            raise ValueError(f"tiles {i - 1}/{i}: ramp/overlap mismatch (overlap={overlap})")


def split_by_size(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    """Split a dimension into overlapping tiles of a given size.
    Tiles are sized ``size`` with ``overlap`` shared elements between
    consecutive tiles.  The last tile may be shorter if the dimension
    doesn't divide evenly.  If ``min_tile_size`` is set and the last tile is
    shorter, it is grown leftward (penultimate ``right_ramp`` widens); the
    result is validated and invalid layouts raise ``ValueError``.
    Args:
        size: Target tile size (in axis units).
        overlap: Overlap between consecutive tiles.
        min_tile_size: Optional minimum tile length. ``None`` keeps legacy
            short-last-tile behavior.
    Returns:
        A split operation that divides a dimension into tiles.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")

    def split(dimension_size: int) -> DimensionIntervals:
        if min_tile_size is not None and dimension_size < min_tile_size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        intervals = [
            DimensionInterval(start=0, end=size, left_ramp=0, right_ramp=overlap),
            *(
                DimensionInterval(
                    start=i * (size - overlap),
                    end=i * (size - overlap) + size,
                    left_ramp=overlap,
                    right_ramp=overlap,
                )
                for i in range(1, amount - 1)
            ),
            DimensionInterval(
                start=(amount - 1) * (size - overlap), end=dimension_size, left_ramp=overlap, right_ramp=0
            ),
        ]
        if min_tile_size is not None:
            intervals = _grow_last_tile_to_min(intervals, min_tile_size)
            _validate_tile_intervals(intervals, dim_size=dimension_size, min_tile_size=min_tile_size)
        return DimensionIntervals(intervals=intervals)

    return split


def split_temporal_causal(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    """Split a temporal axis into overlapping tiles with causal handling.
    Each tile after the first is shifted back by 1 and its left ramp is
    increased by 1, ensuring causal continuity through the blend ramps.
    Args:
        size: Tile size in axis units.
        overlap: Overlap between tiles in the same units.
        min_tile_size: Optional floor forwarded to :func:`split_by_size`.
    Returns:
        Split operation that divides temporal dimension with causal handling.
    """
    non_causal_split = split_by_size(size, overlap, min_tile_size=min_tile_size)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_temporal(tile_size_frames: int, overlap_frames: int) -> SplitOperation:
    """Split a temporal axis in video frame space into overlapping tiles.
    Args:
        tile_size_frames: Tile length in frames.
        overlap_frames: Overlap between consecutive tiles in frames.
    Returns:
        Split operation that takes frame count and returns DimensionIntervals in frame indices.
    """
    non_causal_split = split_by_size(tile_size_frames, overlap_frames)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= tile_size_frames:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        modified_intervals = [
            replace(interval, end=interval.end + 1, right_ramp=0) for interval in dim_intervals.intervals[:-1]
        ] + [replace(dim_intervals.intervals[-1], right_ramp=0)]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_by_count_temporal_causal(num_tiles: int, overlap: int = 0) -> SplitOperation:
    """Split a temporal dimension by count with causal handling.
    Wraps :func:`split_by_count` with the same causal adjustment as
    :func:`split_temporal_causal`: each tile after the first is shifted
    back by 1 and its left ramp is increased by 1.
    Args:
        num_tiles: Number of tiles. Must be >= 1.
        overlap: Overlap between adjacent tiles (default 0).
    Returns:
        A split operation that divides a temporal dimension into tiles.
    """
    non_causal_split = split_by_count(num_tiles, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_by_count(num_tiles: int, overlap: int = 0) -> SplitOperation:
    """Split a dimension into a given number of tiles with overlap.
    Computes the tile size as
    ``(dim_size + overlap * (num_tiles - 1)) // num_tiles`` so that
    ``num_tiles`` tiles of that size with ``overlap`` shared elements
    cover the dimension evenly.  Delegates to :func:`split_by_size` for
    the actual interval construction.
    When the total ``dim_size + overlap * (num_tiles - 1)`` is not evenly
    divisible by ``num_tiles``, the first ``remainder`` tiles each absorb
    one extra unit.
    Args:
        num_tiles: Number of tiles. Must be >= 1.
        overlap: Overlap between adjacent tiles (default 0). Must be >= 0
            and less than the computed tile size.
    Returns:
        A split operation that divides a dimension into tiles.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    def split(dim_size: int) -> DimensionIntervals:
        if num_tiles > dim_size:
            raise ValueError(
                f"num_tiles ({num_tiles}) exceeds dim_size ({dim_size}). Cannot assign at least 1 unit per tile."
            )
        if num_tiles == 1:
            return DEFAULT_SPLIT_OPERATION(dim_size)

        total = dim_size + overlap * (num_tiles - 1)
        tile_size = total // num_tiles
        remainder = total % num_tiles

        base_intervals = split_by_size(tile_size, overlap)(dim_size - remainder).intervals

        # First `remainder` tiles each absorb 1 extra unit; shift subsequent boundaries.
        intervals: list[DimensionInterval] = []
        for i, iv in enumerate(base_intervals):
            shift = min(i, remainder)
            grow = 1 if i < remainder else 0
            intervals.append(replace(iv, start=iv.start + shift, end=iv.end + shift + grow))

        return DimensionIntervals(intervals=intervals)

    return split


# ---------------------------------------------------------------------------
# Mapping operations
# ---------------------------------------------------------------------------


def identity_mapping_operation(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor]]:
    """Map each DimensionInterval to an output region at the same position, with trapezoidal blend masks.
    For every interval the output start/end matches the input start/end and a
    1-D blending mask is built from the interval's left_ramp and right_ramp.
    """
    out_slices: list[slice] = []
    masks: list[torch.Tensor] = []
    for iv in intervals.intervals:
        out_slices.append(slice(iv.start, iv.end))
        masks.append(compute_trapezoidal_mask_1d(iv.end - iv.start, iv.left_ramp, iv.right_ramp))
    return out_slices, masks


class Tile(NamedTuple):
    """
    Represents a single tile.
    Attributes:
        in_coords:
            Tuple of slices specifying where to cut the tile from the INPUT tensor.
        out_coords:
            Tuple of slices specifying where this tile's OUTPUT should be placed in the reconstructed OUTPUT tensor.
        masks_1d:
            Per-dimension masks in OUTPUT units.
            Untiled axes use a length-1 ones tensor (broadcasts). These are used
            for separable blending (and for the dense ``blend_mask`` property).
    Methods:
        blend_mask:
            Create a single N-D mask from the per-dimension masks.
    """

    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[torch.Tensor, ...]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def scale_by_masks_1d(x: torch.Tensor, masks_1d: Sequence[torch.Tensor]) -> torch.Tensor:
    """Multiply ``x`` by separable 1d masks with broadcasting.
    ``len(masks_1d)`` must equal ``x.ndim``. Prefer float32 masks so bf16/fp16 ``x`` promotes.
    Length-1 masks (untiled axes) broadcast over that dimension.
    """
    if len(masks_1d) != x.ndim:
        raise ValueError(f"masks_1d length {len(masks_1d)} != x.ndim {x.ndim}")
    out = x
    for axis, mask in enumerate(masks_1d):
        view_shape = [1] * x.ndim
        view_shape[axis] = -1
        out = out * mask.reshape(*view_shape)
    return out


def masks_are_complementary(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
    *,
    atol: float = 1e-5,
) -> bool:
    """Return whether per-axis 1d blend masks partition unity (sum to 1).
    Checks each axis independently over the unique out-slices on that axis
    (cartesian tile products would otherwise multi-count the same 1d interval).
    When True, weighted accumulation needs no denominator.
    """
    if not tiles:
        return True
    ndim = len(full_shape)
    for tile in tiles:
        if len(tile.out_coords) != ndim or len(tile.masks_1d) != ndim:
            raise ValueError(
                f"Tile out_coords/masks_1d rank {len(tile.out_coords)}/{len(tile.masks_1d)} != full_shape rank {ndim}"
            )
    for axis, length in enumerate(full_shape):
        # Explicit CPU float32: masks may live on CUDA; a non-CPU default device
        # must not place ``acc`` on GPU (device-mismatch on ``acc[sl] +=``).
        acc = torch.zeros(length, dtype=torch.float32, device="cpu")
        seen: set[tuple[int | None, int | None]] = set()
        for tile in tiles:
            sl = tile.out_coords[axis]
            key = (sl.start, sl.stop)
            if key in seen:
                continue
            seen.add(key)
            # Length-1 untiled masks broadcast over ``acc[sl]``.
            acc[sl] += tile.masks_1d[axis].detach().float().cpu()
        if not torch.allclose(acc, torch.ones(length, dtype=torch.float32), atol=atol, rtol=0.0):
            return False
    return True


def compute_summed_weights(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
) -> torch.Tensor:
    """Build the dense denominator for weighted blending over ``full_shape``.
    Uses separable per-axis mask broadcasts — never ``Tile.blend_mask``.
    Requires concrete ``out_coords`` (``stop`` not ``None``) on every axis.
    Always builds on CPU float32 so CUDA masks / a non-CPU default device cannot
    place a multi-GB ``[F,H,W]`` tensor on GPU.
    """
    weights = torch.zeros(*full_shape, dtype=torch.float32, device="cpu")
    for tile in tiles:
        masks = tuple(m.detach().float().cpu() for m in tile.masks_1d)
        region_shape = tuple(s.stop - s.start for s in tile.out_coords)
        region = torch.ones(region_shape, dtype=torch.float32, device="cpu")
        weights[tile.out_coords] += scale_by_masks_1d(region, masks)
    return weights.clamp(min=1e-8)


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: list[MappingOperation],
) -> list[Tile]:
    full_dim_input_slices: list[list[slice]] = []
    full_dim_output_slices: list[list[slice]] = []
    full_dim_masks_1d: list[list[torch.Tensor]] = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        input_slices = [slice(interval.start, interval.end) for interval in dimension_intervals.intervals]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        n_intervals = len(input_slices)
        if len(output_slices) != n_intervals or len(masks_1d) != n_intervals:
            raise ValueError(
                f"Axis {axis_index}: mapper produced {len(output_slices)} output slices and "
                f"{len(masks_1d)} masks for {n_intervals} input intervals"
            )
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    return [
        Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d)
        for in_coord, out_coord, mask_1d in zip(
            itertools.product(*full_dim_input_slices),
            itertools.product(*full_dim_output_slices),
            itertools.product(*full_dim_masks_1d),
            strict=True,
        )
    ]


def create_tiles(
    latent_shape: torch.Size,
    splitters: list[SplitOperation],
    mappers: list[MappingOperation],
) -> list[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in latent shape, "
            f"got {len(splitters)} and {len(latent_shape)}"
        )
    if len(mappers) != len(latent_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in latent shape, "
            f"got {len(mappers)} and {len(latent_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)


def group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    """Group consecutive tiles that share the same temporal ``out_coords`` slice.
    Assumes ``tiles`` is ordered with the temporal axis varying slowest (true
    for every tile list this codebase builds via ``itertools.product`` with
    the temporal axis first), so equal temporal slices are always contiguous.
    """
    if not tiles:
        return []

    groups = []
    current_slice = tiles[0].out_coords[2]
    current_group = []

    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]

    if current_group:
        groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Video-grid tiling configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionTilingConfig:
    """Tiling parameters for a single dimension of the patchified grid.
    Attributes:
        num_tiles: Number of tiles along this dimension.
        overlap: Overlap between adjacent tiles, in latent grid units.
            Adjacent tiles share ``overlap`` grid cells at their
            boundary, producing an overlap zone blended with
            trapezoidal masks.
    """

    num_tiles: int
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")

    @classmethod
    def from_tile_size(cls, dim_size: int, tile_size: int, overlap: int = 0) -> DimensionTilingConfig:
        """Create config by computing ``num_tiles`` from dimension size and tile size.
        Args:
            dim_size: Total length of the dimension.
            tile_size: Desired tile size.
            overlap: Overlap between consecutive tiles.
        Returns:
            A ``DimensionTilingConfig`` with the computed ``num_tiles``.
        """
        split_op = split_by_size(tile_size, overlap)
        intervals = split_op(dim_size)
        return cls(num_tiles=len(intervals.intervals), overlap=overlap)


@dataclass(frozen=True)
class TileCountConfig:
    """Tiling layout for a ``(F, H, W)`` grid.
    Specifies tile *counts* per dimension (as opposed to tile *sizes*
    which are used by the single-GPU VAE ``TilingConfig``).
    Attributes:
        frames: Tiling along the temporal (frames) dimension.
        height: Tiling along the latent height dimension.
        width: Tiling along the latent width dimension.
    """

    frames: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0)
    height: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0)
    width: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0)


def balanced_tile_split(num_tiles: int) -> tuple[int, int]:
    """Factor ``num_tiles`` into ``(small, large)`` as square as possible.
    ``small`` is the largest divisor not exceeding the square root, so
    ``small * large == num_tiles`` and ``small <= large``. E.g. 2 -> (1, 2),
    4 -> (2, 2), 8 -> (2, 4), 16 -> (4, 4). The caller decides which tiled
    dimension gets which factor.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    small = next(d for d in range(math.isqrt(num_tiles), 0, -1) if num_tiles % d == 0)
    return small, num_tiles // small
