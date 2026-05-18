import logging
from typing import Any, Callable, Iterator, List, Tuple

import torch
from einops import rearrange
from torch import nn

from diffusion.refiner.vendor.ltx_core.model.common.normalization import PixelNorm
from diffusion.refiner.vendor.ltx_core.model.transformer.timestep_embedding import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)
from diffusion.refiner.vendor.ltx_core.model.video_vae.convolution import make_conv_nd
from diffusion.refiner.vendor.ltx_core.model.video_vae.enums import (
    LogVarianceType,
    NormLayerType,
    PaddingModeType,
)
from diffusion.refiner.vendor.ltx_core.model.video_vae.ops import PerChannelStatistics, patchify, unpatchify
from diffusion.refiner.vendor.ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from diffusion.refiner.vendor.ltx_core.model.video_vae.sampling import (
    DepthToSpaceUpsample,
    SpaceToDepthDownsample,
)
from diffusion.refiner.vendor.ltx_core.model.video_vae.tiling import TilingConfig
from diffusion.refiner.vendor.ltx_core.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    Tile,
    compute_rectangular_mask_1d,
    compute_trapezoidal_mask_1d,
    create_tiles,
)
from diffusion.refiner.vendor.ltx_core.tiling import split_by_size as split_in_spatial
from diffusion.refiner.vendor.ltx_core.tiling import (
    split_temporal,
)
from diffusion.refiner.vendor.ltx_core.tiling import split_temporal_causal as split_in_temporal
from diffusion.refiner.vendor.ltx_core.types import (
    VIDEO_SCALE_FACTORS,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)

logger: logging.Logger = logging.getLogger(__name__)


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
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
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):
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
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

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
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        # Validate frame count (crop to nearest valid length if needed)
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            frames_to_crop = (frames_count - 1) % 8
            logger.warning(
                "Invalid number of frames %s for encode; cropping last %s frames to satisfy 1 + 8*k.",
                frames_count,
                frames_to_crop,
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
        if (frames - 1) % VIDEO_SCALE_FACTORS.time != 0:
            frames_to_crop = (frames - 1) % VIDEO_SCALE_FACTORS.time
            logger.warning(
                f"Number of frames {frames} of input video is not ({VIDEO_SCALE_FACTORS.time} * k + 1), "
                f"last {frames_to_crop} frames will be cropped"
            )
            video = video[:, :, :-frames_to_crop, ...]
            # Update frames after cropping
            frames = video.shape[2]

        # Calculate output latent shape (inverse of upscale)
        latent_shape = VideoLatentShape(
            batch=batch,
            channels=self.latent_channels,  # 128 for standard VAE
            frames=(frames - 1) // VIDEO_SCALE_FACTORS.time + 1,
            height=height // VIDEO_SCALE_FACTORS.height,
            width=width // VIDEO_SCALE_FACTORS.width,
        )

        # Prepare tiles (operates on VIDEO dimensions)
        tiles = prepare_tiles_for_encoding(video, tiling_config)

        # Initialize accumulation buffers on model device
        latent_buffer = torch.zeros(
            latent_shape.to_torch_shape(),
            device=model_device,
            dtype=model_dtype,
        )
        weights_buffer = torch.zeros_like(latent_buffer)

        # Process each tile
        for tile in tiles:
            # Extract video tile from input (may be on CPU)
            video_tile = video[tile.in_coords]

            # Move tile to model device if needed
            if video_tile.device != model_device or video_tile.dtype != model_dtype:
                video_tile = video_tile.to(device=model_device, dtype=model_dtype)

            # Encode tile to latent (output on model device)
            latent_tile = self.forward(video_tile)

            # Move blend mask to model device
            mask = tile.blend_mask.to(
                device=model_device,
                dtype=model_dtype,
            )

            # Weighted accumulation in latent space
            latent_buffer[tile.out_coords] += latent_tile * mask
            weights_buffer[tile.out_coords] += mask

            del latent_tile, mask, video_tile

        # Normalize by accumulated weights
        weights_buffer = weights_buffer.clamp(min=1e-8)
        return latent_buffer / weights_buffer


def prepare_tiles_for_encoding(
    video: torch.Tensor,
    tiling_config: TilingConfig | None = None,
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
        mappers[3] = to_mapping_operation(map_spatial_interval_to_latent, scale=VIDEO_SCALE_FACTORS.height)

        # Width axis (W)
        splitters[4] = split_in_spatial(tile_size_px, overlap_px)
        mappers[4] = to_mapping_operation(map_spatial_interval_to_latent, scale=VIDEO_SCALE_FACTORS.width)

    if tiling_config is not None and tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        tile_size_frames = cfg.tile_size_in_frames
        overlap_frames = cfg.tile_overlap_in_frames

        if overlap_frames < minimum_temporal_overlap_frames:
            logger.warning(f"Overlap frames {overlap_frames} is less than 16, setting to minimum required 16")
            overlap_frames = minimum_temporal_overlap_frames

        splitters[2] = split_temporal(tile_size_frames, overlap_frames)
        mappers[2] = to_mapping_operation(map_temporal_interval_to_latent, scale=VIDEO_SCALE_FACTORS.time)

    return create_tiles(video.shape, splitters, mappers)


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
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
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


class VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Decoder. Decodes latent representation into video frames.
    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.
    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling
    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
        First frame removed after temporal upsampling in both modes. Output shape unchanged.
        Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512) for both modes.
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels (latent channels). Default is 128.
        out_channels: The number of output channels. For RGB images, this is 3.
        decoder_blocks: The list of blocks to construct the decoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: Final spatial expansion factor. For standard LTX Video, use 4 for 4x spatial expansion:
                    H -> Hx4, W -> Wx4. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal: Whether to use causal convolutions. For standard LTX Video, use False for symmetric padding.
                When True, uses causal padding (past/current frames only).
        timestep_conditioning: Whether to condition the decoder on timestep for denoising.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        base_channels: int = 128,
    ):
        super().__init__()

        # Spatiotemporal downscaling between decoded video space and VAE latents.
        # According to the LTXV paper, the standard configuration downsamples
        # video inputs by a factor of 8 in the temporal dimension and 32 in
        # each spatial dimension (height and width). This parameter determines how
        # many video frames and pixels correspond to a single latent cell.
        self.video_downscale_factors = SpatioTemporalScaleFactors(
            time=8,
            width=32,
            height=32,
        )

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # LTX VAE decoder architecture uses 3 upsampler blocks with multiplier equals to 2.
        # Hence the total feature_channels is multiplied by 8 (2^3).
        feature_channels = base_channels * 8

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Decode latent representation into video frames.
        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True). Uses default 0.05 if None.
            generator: Random generator for deterministic noise injection (if inject_noise=True in blocks).
        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512).
            Note: First frame is removed after temporal upsampling regardless of causal mode.
            When causal=False, allows future frame dependencies in convolutions but maintains same output shape.
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )

            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion: reverse the initial patchify from encoder
        # Moves pixels from channels back to spatial dimensions
        # Example: (B, 48, F, 128, 128) -> (B, 3, F, 512, 512) with patch_size=4
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample

    def _prepare_tiles(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> List[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION] * len(latent.shape)
        mappers = [DEFAULT_MAPPING_OPERATION] * len(latent.shape)
        if tiling_config is not None and tiling_config.spatial_config is not None:
            cfg = tiling_config.spatial_config
            long_side = max(latent.shape[3], latent.shape[4])

            def enable_on_axis(axis_idx: int, factor: int) -> None:
                size = cfg.tile_size_in_pixels // factor
                overlap = cfg.tile_overlap_in_pixels // factor
                axis_length = latent.shape[axis_idx]
                lower_threshold = max(2, overlap + 1)
                tile_size = max(lower_threshold, round(size * axis_length / long_side))
                splitters[axis_idx] = split_in_spatial(tile_size, overlap)
                mappers[axis_idx] = to_mapping_operation(map_spatial_slice, scale=factor)

            enable_on_axis(3, self.video_downscale_factors.height)
            enable_on_axis(4, self.video_downscale_factors.width)

        if tiling_config is not None and tiling_config.temporal_config is not None:
            cfg = tiling_config.temporal_config
            tile_size = cfg.tile_size_in_frames // self.video_downscale_factors.time
            overlap = cfg.tile_overlap_in_frames // self.video_downscale_factors.time
            splitters[2] = split_in_temporal(tile_size, overlap)
            mappers[2] = to_mapping_operation(map_temporal_slice, scale=self.video_downscale_factors.time)

        return create_tiles(latent.shape, splitters, mappers)

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode a latent tensor into video frames using tiled processing.
        Splits the latent tensor into tiles, decodes each tile individually,
        and yields video chunks as they become available.
        Args:
            latent: Input latent tensor (B, C, F', H', W').
            tiling_config: Tiling configuration for the latent tensor.
            timestep: Optional timestep for decoder conditioning.
            generator: Optional random generator for deterministic decoding.
        Yields:
            Video chunks (B, C, T, H, W) by temporal slices;
        """

        # Calculate full video shape from latent shape to get spatial dimensions
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
        tiles = self._prepare_tiles(latent, tiling_config)

        temporal_groups = self._group_tiles_by_temporal_slice(tiles)

        # State for temporal overlap handling
        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]

            # Calculate the shape of the temporal buffer for this group of tiles.
            # The temporal length depends on whether this is the first tile (starts at 0) or not.
            # - First tile: (frames - 1) * scale + 1
            # - Subsequent tiles: frames * scale
            # This logic is handled by TemporalAxisMapping and reflected in out_coords.
            temporal_tile_buffer_shape = full_video_shape._replace(
                frames=curr_temporal_slice.stop - curr_temporal_slice.start,
            )

            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=latent.device,
                dtype=latent.dtype,
            )

            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
            )

            # Blend with previous temporal chunk if it exists
            if previous_chunk is not None:
                # Check if current temporal slice overlaps with previous temporal slice
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)

                    # The overlap is already masked before it reaches this step. Each tile is accumulated into buffer
                    # with its trapezoidal mask, and curr_weights accumulates the same mask. In the overlap blend we add
                    # the masked values (buffer[...]) and the corresponding weights (curr_weights[...]) into the
                    # previous buffers, then later normalize by weights.
                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[:, :, slice(0, overlap_len), :, :]
                    previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[
                        :, :, slice(0, overlap_len), :, :
                    ]

                    buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]
                    curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[
                        :, :, temporal_overlap_slice, :, :
                    ]

                # Yield the non-overlapping part of the previous chunk
                previous_weights = previous_weights.clamp(min=1e-8)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                yield (previous_chunk / previous_weights)[:, :, :yield_len, :, :]

            # Update state for next iteration
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice

        # Yield any remaining chunk
        if previous_chunk is not None:
            previous_weights = previous_weights.clamp(min=1e-8)
            yield previous_chunk / previous_weights

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode a video latent tensor, yielding uint8 chunks ``[f, h, w, c]``.
        Subclasses (e.g. ``DistributedVideoDecoder``) may override this to
        control eagerness or distribution across ranks.
        """

        def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
            frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            frames = rearrange(frames[0], "c f h w -> f h w c")
            return frames

        if tiling_config is not None:
            for frames in self.tiled_decode(latent, tiling_config, generator=generator):
                yield convert_to_uint8(frames)
        else:
            decoded = self(latent, generator=generator)
            yield convert_to_uint8(decoded)

    def _group_tiles_by_temporal_slice(self, tiles: List[Tile]) -> List[List[Tile]]:
        """Group tiles by their temporal output slice."""
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

        # Add the final group
        if current_group:
            groups.append(current_group)

        return groups

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """
        Decode and accumulate all tiles of a temporal group into a local buffer.
        The buffer is local to the group and always starts at time 0; temporal coordinates
        are rebased by subtracting temporal_slice.start.
        """
        temporal_slice = group_tiles[0].out_coords[2]

        weights = torch.zeros_like(buffer)

        for tile in group_tiles:
            decoded_tile = self.forward(latent[tile.in_coords], timestep, generator)
            mask = tile.blend_mask.to(device=buffer.device, dtype=buffer.dtype)
            temporal_offset = tile.out_coords[2].start - temporal_slice.start
            # Use the tile's output coordinate length, not the decoded tile's length,
            # as the decoder may produce a different number of frames than expected
            expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
            decoded_temporal_len = decoded_tile.shape[2]

            # Ensure we don't exceed the buffer or decoded tile bounds
            actual_temporal_len = min(expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset)

            chunk_coords = (
                slice(None),  # batch
                slice(None),  # channels
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],  # height
                tile.out_coords[4],  # width
            )

            # Slice decoded_tile and mask to match the actual length we're writing
            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask

            buffer[chunk_coords] += decoded_slice * mask_slice
            weights[chunk_coords] += mask_slice

        return weights


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
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor | None]],
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

    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor | None] = []
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
