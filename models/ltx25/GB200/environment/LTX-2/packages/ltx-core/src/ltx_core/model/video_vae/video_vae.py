import logging
from typing import Any, Callable, Iterator, List, Protocol, Tuple

import torch
from torch import nn

from ltx_core.model.common.normalization import PixelNorm
from ltx_core.model.disposable import Disposable
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.video_vae.attention import AttnBlock3D
from ltx_core.model.video_vae.convolution import make_conv_nd
from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from ltx_core.model.video_vae.ops import PerChannelStatistics, patchify
from ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from ltx_core.model.video_vae.sampling import SpaceToDepthDownsample
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    compute_rectangular_mask_1d,
    compute_trapezoidal_mask_1d,
    create_tiles,
    masks_are_complementary,
    scale_by_masks_1d,
    split_temporal,
)
from ltx_core.tiling import (
    split_by_size as split_in_spatial,
)
from ltx_core.tiling import (
    split_temporal_causal as split_in_temporal,
)
from ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoLatentShape

logger: logging.Logger = logging.getLogger(__name__)


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
    attention: AttentionFunction | AttentionCallable,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn":
        block = AttnBlock3D(in_channels=in_channels, attention=attention)
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module, Disposable):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Encoder. Encodes video frames into a latent representation.
    The encoder compresses the input video through a series of downsampling operations controlled by
    patch_size and encoder_blocks. The output is a normalized latent tensor with shape (B, 128, F', H', W').
    Compression Behavior:
        The total compression is determined by:
        1. Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        2. Sequential compression through encoder_blocks based on their stride patterns
        Compression blocks apply 2x compression in specified dimensions:
            - "compress_time" / "compress_time_res": temporal only
            - "compress_space" / "compress_space_res": spatial only (H and W)
            - "compress_all" / "compress_all_res": all dimensions (F, H, W)
            - "res_x" / "res_x_y": no compression
        Standard LTX Video configuration:
            - patch_size=4
            - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
            - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
            - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
            - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels. For RGB images, this is 3.
        out_channels: The number of output channels (latent channels). For latent channels, this is 128.
        encoder_blocks: The list of blocks to construct the encoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: The patch size for initial spatial compression. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var: The log variance mode. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int]] | List[Tuple[str, dict[str, Any]]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.PYTORCH,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        # Spatiotemporal downscaling derived from the block list (see SpatioTemporalScaleFactors.from_blocks).
        self.video_scale_factors = SpatioTemporalScaleFactors.from_blocks(encoder_blocks, patch_size)

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
                attention=attention,
            )

            self.down_blocks.append(block)

        # out
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""
        Encode video frames into normalized latent representation.
        Args:
            sample: Input video (B, C, F, H, W). F should be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).
                If not, the encoder crops the last frames to the nearest valid length.
                Should be normalized to [-1, 1] range before encoding.
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        # Validate frame count (crop to nearest valid length if needed)
        temporal_factor = self.video_scale_factors.time
        frames_count = sample.shape[2]
        if ((frames_count - 1) % temporal_factor) != 0:
            frames_to_crop = (frames_count - 1) % temporal_factor
            logger.warning(
                "Invalid number of frames %s for encode; cropping last %s frames to satisfy 1 + %s*k.",
                frames_count,
                frames_to_crop,
                temporal_factor,
            )
            sample = sample[:, :, :-frames_to_crop, ...]

        # Initial spatial compression: trade spatial resolution for channel depth
        # This reduces H,W by patch_size and increases channels, making convolutions more efficient
        # Example: (B, 3, F, 512, 512) -> (B, 48, F, 128, 128) with patch_size=4
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform Variance: model outputs N means and 1 shared log-variance channel.
            # We need to expand the single logvar to match the number of means channels
            # to create a format compatible with PER_CHANNEL (means + logvar, each with N channels).
            # Sample shape: (B, N+1, ...) where N = latent_channels (e.g., 128 means + 1 logvar = 129)
            # Target shape: (B, 2*N, ...) where first N are means, last N are logvar

            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2 channels "
                    f"(N means + 1 logvar), got {sample.shape[1]}"
                )

            # Extract means (first N channels) and logvar (last 1 channel)
            means = sample[:, :-1, ...]  # (B, N, ...)
            logvar = sample[:, -1:, ...]  # (B, 1, ...)

            # Repeat logvar N times to match means channels
            # Use expand/repeat pattern that works for both 4D and 5D tensors
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)  # (B, N, ...)

            # Concatenate to create (B, 2*N, ...) format: [means, repeated_logvar]
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def tiled_encode(
        self,
        video: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        """Encode video to latent using tiled processing of the given video tensor.
        Device Handling:
            - Input video can be on CPU or GPU
            - Accumulation buffers are created on model's device
            - Each tile is automatically moved to model's device before encoding
            - Output latent is returned on model's device
        Args:
            video: Input video tensor (B, 3, F, H, W) in range [-1, 1]
            tiling_config: Tiling configuration for the video tensor
        Returns:
            Latent tensor (B, 128, F', H', W') on model's device
            where F' = 1 + (F-1)/8, H' = H/32, W' = W/32
        """
        # Detect model device and dtype
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # Extract shape components
        batch, _, frames, height, width = video.shape

        # Check frame count and crop if needed
        if (frames - 1) % self.video_scale_factors.time != 0:
            frames_to_crop = (frames - 1) % self.video_scale_factors.time
            logger.warning(
                f"Number of frames {frames} of input video is not ({self.video_scale_factors.time} * k + 1), "
                f"last {frames_to_crop} frames will be cropped"
            )
            video = video[:, :, :-frames_to_crop, ...]
            # Update frames after cropping
            frames = video.shape[2]

        # Calculate output latent shape (inverse of upscale)
        latent_shape = VideoLatentShape(
            batch=batch,
            channels=self.latent_channels,  # 128 for standard VAE
            frames=(frames - 1) // self.video_scale_factors.time + 1,
            height=height // self.video_scale_factors.height,
            width=width // self.video_scale_factors.width,
        )

        # Prepare tiles (operates on VIDEO dimensions)
        tiles = prepare_tiles_for_encoding(video, tiling_config, scale_factors=self.video_scale_factors)
        complementary = masks_are_complementary(tiles, latent_shape.to_torch_shape())

        # Initialize accumulation buffers on model device
        latent_buffer = torch.zeros(
            latent_shape.to_torch_shape(),
            device=model_device,
            dtype=model_dtype,
        )
        weights_buffer: torch.Tensor | None = None if complementary else torch.zeros_like(latent_buffer)

        # Process each tile
        for tile in tiles:
            # Extract video tile from input (may be on CPU)
            video_tile = video[tile.in_coords]

            # Move tile to model device if needed
            if video_tile.device != model_device or video_tile.dtype != model_dtype:
                video_tile = video_tile.to(device=model_device, dtype=model_dtype)

            # Encode tile to latent (output on model device)
            latent_tile = self.forward(video_tile)

            masks = tuple(m.to(device=model_device, dtype=torch.float32) for m in tile.masks_1d)
            latent_buffer[tile.out_coords] += scale_by_masks_1d(latent_tile, masks)
            if weights_buffer is not None:
                strength = torch.ones(latent_tile.shape, device=model_device, dtype=torch.float32)
                weights_buffer[tile.out_coords] += scale_by_masks_1d(strength, masks)

            del latent_tile, video_tile

        if weights_buffer is None:
            return latent_buffer
        weights_buffer = weights_buffer.clamp(min=1e-8)
        return latent_buffer / weights_buffer


def prepare_tiles_for_encoding(
    video: torch.Tensor,
    tiling_config: TilingConfig | None = None,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> List[Tile]:
    """Prepare tiles for VAE encoding.
    Args:
        video: Input video tensor (B, 3, F, H, W) in range [-1, 1]
        tiling_config: Tiling configuration for the video tensor
    Returns:
        List of tiles for the video tensor
    """

    splitters = [DEFAULT_SPLIT_OPERATION] * len(video.shape)
    mappers = [DEFAULT_MAPPING_OPERATION] * len(video.shape)
    minimum_spatial_overlap_px = 64
    minimum_temporal_overlap_frames = 16

    if tiling_config is not None and tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config

        tile_size_px = cfg.tile_size_in_pixels
        overlap_px = cfg.tile_overlap_in_pixels

        # Set minimum spatial overlap to 64 pixels in order to allow cutting padding from
        # the front and back of the tiles and concatenate tiles without artifacts.
        # The encoder uses symmetric padding (pad=1) in H and W at each conv layer. At tile
        # boundaries, convs see padding (zeros/reflect) instead of real neighbor pixels, causing
        # incorrect context near edges.
        # For each overlap we discard 1 latent per edge (32px at scale 32) and concatenate tiles at a
        # shared region with the next tile.
        if overlap_px < minimum_spatial_overlap_px:
            logger.warning(
                f"Overlap pixels {overlap_px} in spatial tiling is less than \
            {minimum_spatial_overlap_px}, setting to minimum required {minimum_spatial_overlap_px}"
            )
            overlap_px = minimum_spatial_overlap_px

        # Define split and map operations for the spatial dimensions

        # Height axis (H)
        splitters[3] = split_in_spatial(tile_size_px, overlap_px)
        mappers[3] = to_mapping_operation(map_spatial_interval_to_latent, scale=scale_factors.height)

        # Width axis (W)
        splitters[4] = split_in_spatial(tile_size_px, overlap_px)
        mappers[4] = to_mapping_operation(map_spatial_interval_to_latent, scale=scale_factors.width)

    if tiling_config is not None and tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        tile_size_frames = cfg.tile_size_in_frames
        overlap_frames = cfg.tile_overlap_in_frames

        if overlap_frames < minimum_temporal_overlap_frames:
            logger.warning(f"Overlap frames {overlap_frames} is less than 16, setting to minimum required 16")
            overlap_frames = minimum_temporal_overlap_frames

        splitters[2] = split_temporal(tile_size_frames, overlap_frames)
        mappers[2] = to_mapping_operation(map_temporal_interval_to_latent, scale=scale_factors.time)

    return create_tiles(video.shape, splitters, mappers)


def latent_tile_splitters(
    latent_shape: torch.Size,
    tiling_config: TilingConfig,
    scale_factors: SpatioTemporalScaleFactors,
    min_tile_size: Tuple[int, int, int] | None = None,
) -> List[SplitOperation]:
    """Convert a pixel/frame-space ``TilingConfig`` into per-axis latent-grid split
    operations for a ``(B, C, F, H, W)`` latent tensor.
    Shared by ``ConvVideoDecoder`` (which maps a latent tile straight to
    pixel-space output) and ``DiffusionVideoDecoder`` (which forward-propagates
    a latent tile through its NA-upsample stages before reaching pixel space) --
    both need the exact same pixel/frame-to-latent-grid conversion so a given
    ``TilingConfig`` describes the same latent tile grid regardless of decoder.
    Axes without a requested tiling config get ``DEFAULT_SPLIT_OPERATION``
    (single tile covering the whole axis).
    ``min_tile_size``, when set, is a ``(T, H, W)`` floor in *latent-grid* units
    forwarded to the per-axis splitters (DiffVAE passes each stage's NA
    ``kernel_size`` requirement here so remnant tiles cannot undershoot natten).
    """
    splitters: List[SplitOperation] = [DEFAULT_SPLIT_OPERATION] * len(latent_shape)
    min_t = min_h = min_w = None
    if min_tile_size is not None:
        min_t, min_h, min_w = min_tile_size

    if tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config
        long_side = max(latent_shape[3], latent_shape[4])

        def enable_on_axis(axis_idx: int, factor: int, axis_min: int | None) -> None:
            size = cfg.tile_size_in_pixels // factor
            overlap = cfg.tile_overlap_in_pixels // factor
            axis_length = latent_shape[axis_idx]
            lower_threshold = max(2, overlap + 1)
            tile_size = max(lower_threshold, round(size * axis_length / long_side))
            splitters[axis_idx] = split_in_spatial(tile_size, overlap, min_tile_size=axis_min)

        enable_on_axis(3, scale_factors.height, min_h)
        enable_on_axis(4, scale_factors.width, min_w)

    if tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        tile_size = cfg.tile_size_in_frames // scale_factors.time
        overlap = cfg.tile_overlap_in_frames // scale_factors.time
        splitters[2] = split_in_temporal(tile_size, overlap, min_tile_size=min_t)

    return splitters


class VideoDecoder(Protocol):
    """Structural interface for video VAE decoders.
    Implementations decode a latent tensor into pixel-space video chunks
    (e.g. ``ConvVideoDecoder``, ``DiffusionVideoDecoder``,
    ``DistributedVideoDecoder``).
    """

    video_downscale_factors: SpatioTemporalScaleFactors

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode a video latent tensor, yielding float chunks ``[f, h, w, c]`` in ``[0, 1]``."""
        ...


def get_video_chunks_number(num_frames: int, tiling_config: TilingConfig | None = None) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.
    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.
    Returns:
        Number of video chunks.
    """
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    frame_stride = cfg.tile_size_in_frames - cfg.tile_overlap_in_frames
    return (num_frames - 1 + frame_stride - 1) // frame_stride


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor]],
    scale: int,
) -> MappingOperation:
    """Create a mapping operation over a set of tiling intervals.
    The given mapping function is applied to each interval in the input dimension. The result function is used for
    creating tiles in the output dimension.
    Args:
        map_func: Mapping function to create the mapping operation from
        scale: Scale factor for the transformation, used as an argument for the mapping function
    Returns:
        Mapping operation that takes a set of tiling intervals and returns a set of slices and masks in the output
        dimension.
    """

    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor] = []
        for interval in intervals.intervals:
            output_slice, mask_1d = map_func(
                interval.start, interval.end, interval.left_ramp, interval.right_ramp, scale
            )
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp = 0 if left_ramp == 0 else 1 + (left_ramp - 1) * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, True)


def map_temporal_interval_to_latent(
    begin: int, end: int, left_ramp: int, right_ramp: int | None = None, scale: int = 1
) -> Tuple[slice, torch.Tensor]:
    """
    Map temporal interval in video frame space to latent space.
    Args:
        begin: Start position in video frame space
        end: End position in video frame space
        left_ramp: Left ramp size in video frame space
        right_ramp: Right ramp size in video frame space
        scale: Scale factor for transformation
    Returns:
        Tuple of (output_slice, blend_mask)
    """
    start = begin // scale
    stop = (end - 1) // scale + 1

    left_ramp_latents = 0 if left_ramp == 0 else 1 + (left_ramp - 1) // scale
    right_ramp_latents = right_ramp // scale

    if right_ramp_latents != 0:
        raise ValueError("For tiled encoding, temporal tiles are expected to have a right ramp equal to 0")

    mask_1d = compute_rectangular_mask_1d(stop - start, left_ramp_latents, right_ramp_latents)

    return slice(start, stop), mask_1d


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp = left_ramp * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, False)


def map_spatial_interval_to_latent(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> Tuple[slice, torch.Tensor]:
    """Map spatial interval in pixel space to latent space.
       Args:
        begin: Start position in pixel space
        end: End position in pixel space
        left_ramp: Left ramp size in pixel space
        right_ramp: Right ramp size in pixel space
        scale: Scale factor for transformation
    Returns:
        Tuple of (output_slice, blend_mask)
    """
    start = begin // scale
    stop = end // scale
    left_ramp = max(0, left_ramp // scale - 1)

    right_ramp = 0 if right_ramp == 0 else 1

    mask_1d = compute_rectangular_mask_1d(stop - start, left_ramp, right_ramp)
    return slice(start, stop), mask_1d


from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder  # noqa: E402
from ltx_core.model.video_vae.diffusion_video_decoder import (  # noqa: E402
    DiffusionVideoDecoder,
    _propagate_interval_through_upsample_hops,
)

__all__ = [
    "ConvVideoDecoder",
    "DiffusionVideoDecoder",
    "VideoDecoder",
    "VideoEncoder",
    "_propagate_interval_through_upsample_hops",
    "get_video_chunks_number",
    "latent_tile_splitters",
    "map_spatial_interval_to_latent",
    "map_spatial_slice",
    "map_temporal_interval_to_latent",
    "map_temporal_slice",
    "prepare_tiles_for_encoding",
    "to_mapping_operation",
]
