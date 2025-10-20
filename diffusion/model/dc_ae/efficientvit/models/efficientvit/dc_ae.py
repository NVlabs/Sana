# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf
from torch import Tensor

from ...models.nn.act import build_act
from ...models.nn.norm import build_norm
from ...models.nn.ops import (
    ChannelDuplicatingPixelUnshuffleUpSampleLayer,
    ConvLayer,
    ConvPixelShuffleUpSampleLayer,
    ConvPixelUnshuffleDownSampleLayer,
    EfficientViTBlock,
    IdentityLayer,
    InterpolateConvUpSampleLayer,
    OpSequential,
    PixelUnshuffleChannelAveragingDownSampleLayer,
    ResBlock,
    ResidualBlock,
)

__all__ = ["DCAE", "dc_vae_f32", "dc_ae_f32c32", "dc_ae_f64c128", "dc_ae_f128c512"]


@dataclass
class EncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: str = "trms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_shortcut: Optional[str] = "averaging"
    double_latent: bool = False
    temporal_downsample: tuple[bool, ...] = ()


@dataclass
class DecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: str = "duplicating"
    out_norm: str = "trms2d"
    out_act: str = "relu"
    temporal_upsample: tuple[bool, ...] = ()


@dataclass
class DCVAEConfig:
    is_training: bool = False  # NOTE: set to True in vae train config
    use_spatial_tiling: bool = False
    use_temporal_tiling: bool = False
    spatial_tile_size: int = 256
    temporal_tile_size: int = 32
    tile_overlap_factor: float = 0.25
    time_compression_ratio: int = 1
    spatial_compression_ratio: Optional[int] = None


@dataclass
class DCAEConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    decoder: DecoderConfig = field(
        default_factory=lambda: DecoderConfig(in_channels="${..in_channels}", latent_channels="${..latent_channels}")
    )
    use_quant_conv: bool = False

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"

    scaling_factor: Optional[float] = None
    # cache_dir
    cache_dir: Optional[str] = None
    # video
    video: Optional[DCVAEConfig] = None


def build_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str], is_video: bool
) -> nn.Module:
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            is_video=is_video,
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(), is_video=is_video
        )
    elif block_type == "EViTS5_GLU":
        assert in_channels == out_channels
        block = EfficientViTBlock(
            in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,), is_video=is_video
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, input_width: int, is_video: bool
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
            is_video=is_video,
        )
        stage.append(block)
    return stage


def build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    is_video: bool,
    temporal_downsample: bool = False,
) -> nn.Module:
    """
    Spatial downsample is always performed. Temporal downsample is optional.
    """

    if block_type == "Conv":
        if is_video:
            if temporal_downsample:
                stride = (2, 2, 2)
            else:
                stride = (1, 2, 2)
        else:
            stride = 2
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            use_bias=True,
            norm=None,
            act_func=None,
            is_video=is_video,
        )
    elif block_type == "ConvPixelUnshuffle":
        if is_video:
            raise NotImplementedError("ConvPixelUnshuffle downsample is not supported for video")
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2, temporal_downsample=temporal_downsample
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    is_video: bool,
    temporal_upsample: bool = False,
) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        if is_video:
            raise NotImplementedError("ConvPixelShuffle upsample is not supported for video")
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            is_video=is_video,
            temporal_upsample=temporal_upsample,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2, temporal_upsample=temporal_upsample
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_encoder_project_in_block(
    in_channels: int, out_channels: int, factor: int, downsample_block_type: str, is_video: bool
):
    if factor == 1:
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            is_video=is_video,
        )
    elif factor == 2:
        if is_video:
            raise NotImplementedError("Downsample during project_in is not supported for video")
        block = build_downsample_block(
            block_type=downsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
        )
    else:
        raise ValueError(f"downsample factor {factor} is not supported for encoder project in")
    return block


def build_encoder_project_out_block(
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    shortcut: Optional[str],
    is_video: bool,
):
    block = OpSequential(
        [
            build_norm(norm),
            build_act(act),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                is_video=is_video,
            ),
        ]
    )
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for encoder project out")
    return block


def build_decoder_project_in_block(in_channels: int, out_channels: int, shortcut: Optional[str], is_video: bool):
    block = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        use_bias=True,
        norm=None,
        act_func=None,
        is_video=is_video,
    )
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=1
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for decoder project in")
    return block


def build_decoder_project_out_block(
    in_channels: int,
    out_channels: int,
    factor: int,
    upsample_block_type: str,
    norm: Optional[str],
    act: Optional[str],
    is_video: bool,
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
    ]
    if factor == 1:
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                is_video=is_video,
            )
        )
    elif factor == 2:
        if is_video:
            raise NotImplementedError("Upsample during project_out is not supported for video")
        layers.append(
            build_upsample_block(
                block_type=upsample_block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
            )
        )
    else:
        raise ValueError(f"upsample factor {factor} is not supported for decoder project out")
    return OpSequential(layers)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, is_video: bool):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )

        self.project_in = build_encoder_project_in_block(
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            factor=1 if cfg.depth_list[0] > 0 else 2,
            downsample_block_type=cfg.downsample_block_type,
            is_video=is_video,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            stage = build_stage_main(
                width=width,
                depth=depth,
                block_type=block_type,
                norm=cfg.norm,
                act=cfg.act,
                input_width=width,
                is_video=is_video,
            )

            if stage_id < num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=cfg.downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width,
                    shortcut=cfg.downsample_shortcut,
                    is_video=is_video,
                    temporal_downsample=cfg.temporal_downsample[stage_id] if cfg.temporal_downsample != [] else False,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            in_channels=cfg.width_list[-1],
            out_channels=2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
            shortcut=cfg.out_shortcut,
            is_video=is_video,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, is_video: bool):
        super().__init__()
        self.cfg = cfg
        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)

        self.project_in = build_decoder_project_in_block(
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
            shortcut=cfg.in_shortcut,
            is_video=is_video,
        )

        self.stages: list[OpSequential] = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type=cfg.upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1],
                    shortcut=cfg.upsample_shortcut,
                    is_video=is_video,
                    temporal_upsample=cfg.temporal_upsample[stage_id] if cfg.temporal_upsample != [] else False,
                )
                stage.append(upsample_block)

            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=block_type,
                    norm=norm,
                    act=act,
                    input_width=(
                        width if cfg.upsample_match_channel else cfg.width_list[min(stage_id + 1, num_stages - 1)]
                    ),
                    is_video=is_video,
                )
            )
            self.stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            factor=1 if cfg.depth_list[0] > 0 else 2,
            upsample_block_type=cfg.upsample_block_type,
            norm=cfg.out_norm,
            act=cfg.out_act,
            is_video=is_video,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class DCAE(nn.Module):
    def __init__(self, cfg: DCAEConfig):
        super().__init__()
        self.cfg = cfg
        is_video = cfg.video is not None
        self.encoder = Encoder(cfg.encoder, is_video)
        self.decoder = Decoder(cfg.decoder, is_video)

        if is_video:
            # video
            self.scaling_factor = cfg.scaling_factor
            self.time_compression_ratio = cfg.video.time_compression_ratio
            self.video_spatial_compression_ratio = cfg.video.spatial_compression_ratio
            self.use_spatial_tiling = cfg.video.use_spatial_tiling
            self.use_temporal_tiling = cfg.video.use_temporal_tiling
            self.spatial_tile_size = cfg.video.spatial_tile_size
            self.temporal_tile_size = cfg.video.temporal_tile_size
            assert (
                cfg.video.spatial_tile_size // cfg.video.spatial_compression_ratio
            ), f"spatial tile size {cfg.video.spatial_tile_size} must be divisible by spatial compression of {cfg.video.spatial_compression_ratio}"
            self.spatial_tile_latent_size = cfg.video.spatial_tile_size // cfg.video.spatial_compression_ratio
            assert (
                cfg.video.temporal_tile_size // cfg.video.time_compression_ratio
            ), f"temporal tile size {cfg.video.temporal_tile_size} must be divisible by temporal compression of {cfg.video.time_compression_ratio}"
            self.temporal_tile_latent_size = cfg.video.temporal_tile_size // cfg.video.time_compression_ratio
            self.tile_overlap_factor = cfg.video.tile_overlap_factor

        if self.cfg.pretrained_path is not None:
            self.load_model()

    def load_model(self):
        if self.cfg.pretrained_source == "dc-ae":
            state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)["state_dict"]
            self.load_state_dict(state_dict)
        else:
            raise NotImplementedError

    @property
    def spatial_compression_ratio(self) -> int:
        if self.video_spatial_compression_ratio:
            return self.video_spatial_compression_ratio
        return 2 ** (self.decoder.num_stages - 1)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        net_size = int(self.spatial_tile_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.spatial_tile_latent_size * self.tile_overlap_factor)
        row_limit = self.spatial_tile_latent_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], net_size):
            row = []
            for j in range(0, x.shape[-1], net_size):
                tile = x[:, :, :, i : i + self.spatial_tile_size, j : j + self.spatial_tile_size]
                tile = self.video_encode(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=-2)

    def temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        overlap_size = int(self.temporal_tile_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.temporal_tile_latent_size * self.tile_overlap_factor)
        t_limit = self.temporal_tile_latent_size - blend_extent

        # Split the video into tiles and encode them separately.
        row = []
        for i in range(0, x.shape[2], overlap_size):
            tile = x[:, :, i : i + self.temporal_tile_size, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.spatial_tile_size or tile.shape[-2] > self.spatial_tile_size
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.video_encode(tile)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def video_encode(self, x: torch.Tensor) -> torch.Tensor:
        x_ret = []
        for i in range(x.shape[0]):
            x_ret.append(self.encoder(x[i : i + 1]))
        return torch.cat(x_ret, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE scaling factor is used outside
        if self.cfg.video:
            if self.use_temporal_tiling and x.shape[2] > self.temporal_tile_size:
                return self.temporal_tiled_encode(x)
            elif self.use_spatial_tiling and (
                x.shape[-1] > self.spatial_tile_size or x.shape[-2] > self.spatial_tile_size
            ):
                return self.spatial_tiled_encode(x)
            else:
                return self.video_encode(x)
        x = self.encoder(x)
        return x

    def video_decode(self, z: torch.Tensor) -> torch.Tensor:
        x_ret = []
        for i in range(z.shape[0]):
            x_ret.append(self.decoder(z[i : i + 1]))
        return torch.cat(x_ret, dim=0)

    def spatial_tiled_decode(self, z: torch.FloatTensor) -> torch.Tensor:
        net_size = int(self.spatial_tile_latent_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.spatial_tile_size * self.tile_overlap_factor)
        row_limit = self.spatial_tile_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], net_size):
            row = []
            for j in range(0, z.shape[-1], net_size):
                tile = z[:, :, :, i : i + self.spatial_tile_latent_size, j : j + self.spatial_tile_latent_size]
                decoded = self.video_decode(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=-2)

    def temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        overlap_size = int(self.temporal_tile_latent_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.temporal_tile_size * self.tile_overlap_factor)
        t_limit = self.temporal_tile_size - blend_extent

        row = []
        for i in range(0, z.shape[2], overlap_size):
            tile = z[:, :, i : i + self.temporal_tile_latent_size, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.spatial_tile_latent_size or tile.shape[-2] > self.spatial_tile_latent_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.video_decode(tile)
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE scaling factor is used outside
        if self.cfg.video:
            if self.use_temporal_tiling and x.shape[2] > self.temporal_tile_latent_size:
                return self.temporal_tiled_decode(x)
            elif self.use_spatial_tiling and (
                x.shape[-1] > self.spatial_tile_latent_size or x.shape[-2] > self.spatial_tile_latent_size
            ):
                return self.spatial_tiled_decode(x)
            else:
                return self.video_decode(x)

        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor, global_step: int) -> tuple[Any, Tensor, dict[Any, Any]]:
        x = self.encoder(x)
        x = self.decoder(x)
        return x, torch.tensor(0), {}


def dc_vae_f32(name: str, pretrained_path: str) -> DCAEConfig:
    if name in ["dc-vae-f32t4c128"]:
        cfg_str = (
            "video.time_compression_ratio=4 "
            "video.use_temporal_tiling=True "
            "video.use_spatial_tiling=True "
            "video.spatial_compression_ratio=32 "
            "video.spatial_tile_size=512 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[2,2,2,3,3,3] "
            "encoder.downsample_block_type=Conv "
            "encoder.norm=rms3d "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[3,3,3,3,3,3] "
            "decoder.upsample_block_type=InterpolateConv "
            "decoder.norm=rms3d decoder.act=silu decoder.out_norm=rms3d "
            "encoder.temporal_downsample=[False,False,False,True,True,False] "
            "decoder.temporal_upsample=[False,False,False,True,True,False] "
            "latent_channels=128 "
            "scaling_factor=0.493"
        )
    elif name in ["dc-vae-f32t1c128"]:
        cfg_str = (
            "video.time_compression_ratio=1 "
            "video.use_temporal_tiling=True "
            "video.use_spatial_tiling=True "
            "video.spatial_compression_ratio=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[2,2,2,3,3,3] "
            "encoder.downsample_block_type=Conv "
            "encoder.norm=rms3d "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[3,3,3,3,3,3] "
            "decoder.upsample_block_type=InterpolateConv "
            "decoder.norm=rms3d decoder.act=silu decoder.out_norm=rms3d "
            "encoder.temporal_downsample=[False,False,False,True,True,False] "
            "decoder.temporal_upsample=[False,False,False,True,True,False] "
            "latent_channels=128 "
            "scaling_factor=0.493"
        )
    elif name in ["dc-vae-f32t4c128-nospatialtiling"]:
        cfg_str = (
            "video.time_compression_ratio=4 "
            "video.use_temporal_tiling=True "
            "video.use_spatial_tiling=False "
            "video.spatial_compression_ratio=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[2,2,2,3,3,3] "
            "encoder.downsample_block_type=Conv "
            "encoder.norm=rms3d "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[3,3,3,3,3,3] "
            "decoder.upsample_block_type=InterpolateConv "
            "decoder.norm=rms3d decoder.act=silu decoder.out_norm=rms3d "
            "encoder.temporal_downsample=[False,False,False,True,True,False] "
            "decoder.temporal_upsample=[False,False,False,True,True,False] "
            "latent_channels=128 "
            "scaling_factor=0.493"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg


def dc_ae_f32c32(name: str, pretrained_path: str) -> DCAEConfig:
    if name in ["dc-ae-f32c32-in-1.0", "dc-ae-f32c32-mix-1.0"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[0,4,8,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[0,5,10,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu]"
        )
    elif name in ["dc-ae-f32c32-sana-1.0", "dc-ae-f32c32-sana-1.1"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[2,2,2,3,3,3] "
            "encoder.downsample_block_type=Conv "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024] decoder.depth_list=[3,3,3,3,3,3] "
            "decoder.upsample_block_type=InterpolateConv "
            "decoder.norm=trms2d decoder.act=silu "
            "scaling_factor=0.41407"
        )
    elif name in ["dc-ae-lite-f32c32-sana-1.1"]:
        cfg_str = (
            "latent_channels=32 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViTS5_GLU,EViTS5_GLU,EViTS5_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024] encoder.depth_list=[2,2,2,3,3,3] "
            "encoder.downsample_block_type=Conv "
            "decoder.block_type=ResBlock "
            "decoder.width_list=[128,256,512,512,1024,1024] "
            "decoder.depth_list=[0,3,5,4,4,4] "
            "decoder.out_act=silu "
            "pretrained_source=dc-ae "
            "scaling_factor=0.41407"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg


def dc_ae_f64c128(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    if name in ["dc-ae-f64c128-in-1.0", "dc-ae-f64c128-mix-1.0"]:
        cfg_str = (
            "latent_channels=128 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024,2048] encoder.depth_list=[0,4,8,2,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024,2048] decoder.depth_list=[0,5,10,2,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu,silu]"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg


def dc_ae_f128c512(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    if name in ["dc-ae-f128c512-in-1.0", "dc-ae-f128c512-mix-1.0"]:
        cfg_str = (
            "latent_channels=512 "
            "encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "encoder.width_list=[128,256,512,512,1024,1024,2048,2048] encoder.depth_list=[0,4,8,2,2,2,2,2] "
            "decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
            "decoder.width_list=[128,256,512,512,1024,1024,2048,2048] decoder.depth_list=[0,5,10,2,2,2,2,2] "
            "decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu,silu,silu]"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    return cfg
