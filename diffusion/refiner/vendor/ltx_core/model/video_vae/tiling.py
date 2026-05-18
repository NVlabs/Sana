from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.
    Args:
        tile_size_in_pixels (int): Size of each tile in pixels. Must be at least 64 and divisible by 32.
        tile_overlap_in_pixels (int, optional): Overlap between tiles in pixels. Must be divisible by 32. Defaults to 0.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles (chunks of frames) with optional overlap.
    Args:
        tile_size_in_frames (int): Number of frames in each tile. Must be at least 16 and divisible by 8.
        tile_overlap_in_frames (int, optional): Number of overlapping frames between consecutive tiles.
            Must be divisible by 8. Defaults to 0.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap.
    Attributes:
        spatial_config: Configuration for splitting spatial dimensions into tiles.
        temporal_config: Configuration for splitting temporal dimension into tiles.
    """

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )
