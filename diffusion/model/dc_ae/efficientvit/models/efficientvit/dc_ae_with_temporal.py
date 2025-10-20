# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

from ..nn.act import build_act
from ..nn.norm import build_norm
from ..nn.ops import IdentityLayer
from ..nn.ops_3d import (
    ChannelDuplicatingPixelShuffleUpSampleLayer3d,
    ConvLayer3d,
    ConvPixelShuffleUpSampleLayer3d,
    ConvPixelUnshuffleDownSampleLayer3d,
    OpSequential3d,
    PixelUnshuffleChannelAveragingDownSampleLayer3d,
    ResBlock3d,
    ResidualBlock3d,
)


@dataclass
class DCAEWithTemporalEncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING

    project_in_block_type: str = "${.downsample_block_type}"
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock3d@3@1"  # spatial kernel size 3, temporal kernel size 1
    norm: Any = "trms2d"
    act: str = "silu"
    downsample_block_type: Any = "ConvPixelUnshuffle@2@1@3@1"  # spatial factor 2, temporal factor 1, spatial kernel size 3, temporal kernel size 1
    downsample_shortcut: Optional[str] = "averaging"
    project_out_block_type: str = "ConvLayer3d@3@1"  # spatial kernel size 3, temporal kernel size 1

    zero_out: bool = False


@dataclass
class DCAEWithTemporalDecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING

    project_in_block_type: str = "ConvLayer3d@3@1"  # spatial kernel size 3, temporal kernel size 1

    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock3d@3@1"  # spatial kernel size 3, temporal kernel size 1
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: Any = (
        "ConvPixelShuffle@2@1@3@1"  # spatial factor 2, temporal factor 1, spatial kernel size 3, temporal kernel size 1
    )
    upsample_shortcut: str = "duplicating"
    project_out_block_type: str = "${.upsample_block_type}"
    out_norm: str = "trms2d"
    out_act: str = "silu"

    zero_out: bool = False


@dataclass
class DCAEWithTemporalConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: DCAEWithTemporalEncoderConfig = field(
        default_factory=lambda: DCAEWithTemporalEncoderConfig(
            in_channels="${..in_channels}",
            latent_channels="${..latent_channels}",
        )
    )
    decoder: DCAEWithTemporalDecoderConfig = field(
        default_factory=lambda: DCAEWithTemporalDecoderConfig(
            in_channels="${..in_channels}",
            latent_channels="${..latent_channels}",
        )
    )

    num_pad_frames: int = 0

    pretrained_path: Optional[str] = None

    zero_out: bool = False
    use_feature_cache: bool = False

    encode_temporal_tile_size: Optional[int] = None
    encode_temporal_tile_latent_size: Optional[int] = None
    decode_temporal_tile_size: Optional[int] = None
    decode_temporal_tile_latent_size: Optional[int] = None

    # spatial tiling
    encode_temporal_tile_overlap_factor: float = 0.0
    decode_temporal_tile_overlap_factor: float = 0.0

    spatial_tile_size: Optional[int] = None
    spatial_tile_overlap_factor: float = 0.0

    # scaling factor
    scaling_factor: Optional[float] = None
    # cache_dir
    cache_dir: Optional[str] = None

    verbose: bool = False


def build_downsample_block(
    block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str], zero_out: bool = False
) -> nn.Module:
    cfg = block_type.split("@")
    block_name, spatial_factor, temporal_factor = cfg[0], int(cfg[1]), int(cfg[2])
    if block_name in ["ConvPixelUnshuffle", "CausalConvPixelUnshuffle"]:
        kwargs = {}
        if len(cfg) >= 7:
            kwargs["spatial_padding_mode"] = cfg[5]
            kwargs["temporal_padding_mode"] = cfg[6]
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelUnshuffle",
            **kwargs,
        )
    elif block_name == "ChunkedCausalConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            causal_chunk_length=int(cfg[5]),
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(
    block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str], zero_out: bool = False
) -> nn.Module:
    cfg = block_type.split("@")
    block_name, spatial_factor, temporal_factor = cfg[0], int(cfg[1]), int(cfg[2])
    if block_name in ["ConvPixelShuffle", "CausalConvPixelShuffle"]:
        kwargs = {}
        if len(cfg) >= 7:
            kwargs["spatial_padding_mode"] = cfg[5]
            kwargs["temporal_padding_mode"] = cfg[6]
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelShuffle",
            **kwargs,
        )
    elif block_name in ["ChunkedCausalConvPixelShuffle"]:
        kwargs = {}
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[4]), int(cfg[3]), int(cfg[3])),
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
            zero_out=zero_out,
            causal_chunk_length=int(cfg[5]),
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=spatial_factor,
            temporal_factor=temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_block(block_type: str, channels: int, norm: Optional[str], act: Optional[str], zero_out: bool) -> nn.Module:
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name in ["ResBlock3d", "CausalResBlock3d"]:
        kwargs = {}
        if len(cfg) >= 5:
            kwargs["spatial_padding_mode"] = cfg[3]
            kwargs["temporal_padding_mode"] = cfg[4]
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal=block_name == "CausalResBlock3d",
            **kwargs,
        )
        block = ResidualBlock3d(main_block, IdentityLayer())
    elif block_name in ["ChunkedCausalResBlock3d"]:
        kwargs = {}
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal_chunk_length=int(cfg[3]),
        )
        block = ResidualBlock3d(main_block, IdentityLayer())
    else:
        raise ValueError(f"block_name {block_name} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: str | list[str], norm: str, act: str, zero_out: bool = False
) -> list[nn.Module]:
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            channels=width,
            norm=norm,
            act=act,
            zero_out=zero_out,
        )
        stage.append(block)
    return stage


def build_encoder_project_in_block(block_type: str, in_channels: int, out_channels: int):
    block = build_downsample_block(
        block_type=block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
    )
    return block


def build_encoder_project_out_block(block_type: str, in_channels: int, out_channels: int):
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name in ["ConvLayer3d", "CausalConvLayer3d"]:
        kwargs = {}
        if len(cfg) >= 5:
            kwargs["spatial_padding_mode"] = cfg[3]
            kwargs["temporal_padding_mode"] = cfg[4]
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name in ["ChunkedCausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=int(cfg[3]),
        )
    else:
        raise ValueError(f"encoder project out block name {block_name} is not supported")
    return block


def build_decoder_project_in_block(block_type: str, in_channels: int, out_channels: int):
    cfg = block_type.split("@")
    block_name = cfg[0]
    if block_name in ["ConvLayer3d", "CausalConvLayer3d"]:
        kwargs = {}
        if len(cfg) >= 5:
            kwargs["spatial_padding_mode"] = cfg[3]
            kwargs["temporal_padding_mode"] = cfg[4]
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name in ["ChunkedCausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(int(cfg[2]), int(cfg[1]), int(cfg[1])),
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=int(cfg[3]),
        )
    else:
        raise ValueError(f"decoder project in block name {block_name} is not supported")
    return block


def build_decoder_project_out_block(
    block_type: str, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
        build_upsample_block(block_type=block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None),
    ]
    return OpSequential3d(layers)


class DCAEWithTemporalEncoder(nn.Module):
    def __init__(self, cfg: DCAEWithTemporalEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.project_in = build_encoder_project_in_block(
            block_type=cfg.project_in_block_type,
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
        )

        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.downsample_block_type, str) or (
            isinstance(cfg.downsample_block_type, list) and len(cfg.downsample_block_type) == num_stages - 1
        )

        self.stages: list[OpSequential3d] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            stage = build_stage_main(
                width=width,
                depth=depth,
                block_type=block_type,
                norm=norm,
                act=cfg.act,
                zero_out=cfg.zero_out,
            )
            if stage_id < num_stages - 1 and depth > 0:
                downsample_block_type = (
                    cfg.downsample_block_type[stage_id]
                    if isinstance(cfg.downsample_block_type, list)
                    else cfg.downsample_block_type
                )
                downsample_block = build_downsample_block(
                    block_type=downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1],
                    shortcut=cfg.downsample_shortcut,
                    zero_out=cfg.zero_out,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential3d(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            block_type=cfg.project_out_block_type,
            in_channels=cfg.width_list[-1],
            out_channels=cfg.latent_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.project_in(x, feature_cache, feature_key + "project_in." if feature_key is not None else None)
        for stage_id, stage in enumerate(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x, feature_cache, feature_key + f"stages.{stage_id}." if feature_key is not None else None)
        x = self.project_out(x, feature_cache, feature_key + "project_out." if feature_key is not None else None)
        return x


class DCAEWithTemporalDecoder(nn.Module):
    def __init__(self, cfg: DCAEWithTemporalDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.project_in = build_decoder_project_in_block(
            block_type=cfg.project_in_block_type,
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
        )

        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, str) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)
        assert isinstance(cfg.upsample_block_type, str) or (
            isinstance(cfg.upsample_block_type, list) and len(cfg.upsample_block_type) == num_stages - 1
        )
        self.stages: list[OpSequential3d] = []
        self.spatial_compression_ratio = 1
        self.temporal_compression_ratio = 1
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block_type = (
                    cfg.upsample_block_type[stage_id]
                    if isinstance(cfg.upsample_block_type, list)
                    else cfg.upsample_block_type
                )
                upsample_block = build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width,
                    shortcut=cfg.upsample_shortcut,
                    zero_out=cfg.zero_out,
                )
                stage.append(upsample_block)
                self.spatial_compression_ratio *= int(upsample_block_type.split("@")[1])
                self.temporal_compression_ratio *= int(upsample_block_type.split("@")[2])

            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width, depth=depth, block_type=block_type, norm=norm, act=act, zero_out=cfg.zero_out
                )
            )
            self.stages.insert(0, OpSequential3d(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_decoder_project_out_block(
            block_type=cfg.project_out_block_type,
            in_channels=cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            out_channels=cfg.in_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )
        self.spatial_compression_ratio *= int(cfg.project_out_block_type.split("@")[1])
        self.temporal_compression_ratio *= int(cfg.project_out_block_type.split("@")[2])

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.project_in(x, feature_cache, feature_key + "project_in." if feature_key is not None else None)
        for stage_id, stage in reversed(list(enumerate(self.stages))):
            if len(stage.op_list) == 0:
                continue
            x = stage(x, feature_cache, feature_key + f"stages.{stage_id}." if feature_key is not None else None)
        x = self.project_out(x, feature_cache, feature_key + "project_out." if feature_key is not None else None)
        return x


class DCAEWithTemporal(nn.Module):
    def __init__(self, cfg: DCAEWithTemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = DCAEWithTemporalEncoder(cfg.encoder)
        self.decoder = DCAEWithTemporalDecoder(cfg.decoder)

    @property
    def spatial_compression_ratio(self) -> int:
        return self.decoder.spatial_compression_ratio

    @property
    def temporal_compression_ratio(self) -> int:
        return self.decoder.temporal_compression_ratio

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            blend_ratio = x / blend_extent
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - blend_ratio) + b[:, :, x, :, :] * blend_ratio
        return b

    def blend_w(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - y / blend_extent) + b[..., y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[..., x] = a[..., -blend_extent + x] * (1 - x / blend_extent) + b[..., x] * (x / blend_extent)
        return b

    def temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        overlap_size = int(self.cfg.encode_temporal_tile_size * (1 - self.cfg.encode_temporal_tile_overlap_factor))
        blend_extent = int(self.cfg.encode_temporal_tile_latent_size * self.cfg.encode_temporal_tile_overlap_factor)
        t_limit = self.cfg.encode_temporal_tile_latent_size - blend_extent

        feature_cache = {} if self.cfg.use_feature_cache else None

        # Split the video into tiles and encode them separately.
        row = []
        for i in tqdm(range(0, x.shape[2], overlap_size), desc="Tiled Encode", disable=not self.cfg.verbose):
            tile = x[:, :, i : i + self.cfg.encode_temporal_tile_size, :, :]
            tile = self.encoder(tile, feature_cache, f"encoder." if self.cfg.use_feature_cache else None)
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        spatial_tile_size = self.cfg.spatial_tile_size
        spatial_tile_stride = round((1 - self.cfg.spatial_tile_overlap_factor) * spatial_tile_size)
        spatial_tile_latent_size = spatial_tile_size // self.spatial_compression_ratio
        spatial_tile_latent_stride = spatial_tile_stride // self.spatial_compression_ratio
        blend_size = spatial_tile_latent_size - spatial_tile_latent_stride

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, spatial_tile_stride):
            row = []
            for j in range(0, width, spatial_tile_stride):
                tile = x[..., i : i + spatial_tile_size, j : j + spatial_tile_size]
                if (
                    tile.shape[-2] % self.spatial_compression_ratio != 0
                    or tile.shape[-1] % self.spatial_compression_ratio != 0
                ):
                    pad_h = (self.spatial_compression_ratio - tile.shape[-2]) % self.spatial_compression_ratio
                    pad_w = (self.spatial_compression_ratio - tile.shape[-1]) % self.spatial_compression_ratio
                    tile = F.pad(tile, (0, pad_w, 0, pad_h))
                if self.cfg.encode_temporal_tile_size is not None:
                    tile = self.temporal_tiled_encode(tile)
                else:
                    tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_w(rows[i - 1][j], tile, blend_size)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_size)
                result_row.append(tile[..., :spatial_tile_latent_stride, :spatial_tile_latent_stride])
            result_rows.append(torch.cat(result_row, dim=-1))

        encoded = torch.cat(result_rows, dim=-2)[..., :latent_height, :latent_width]
        return encoded

    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        # Pad image data
        if x.ndim == 4:
            x = x[:, :, None]
        if x.shape[2] == 1:
            x = x.repeat(1, 1, self.temporal_compression_ratio, 1, 1)

        if self.cfg.num_pad_frames > 0:
            x = F.pad(x, (0, 0, 0, 0, self.cfg.num_pad_frames, 0), mode="replicate")

        if self.cfg.spatial_tile_size is not None:
            x = self.spatial_tiled_encode(x)
        elif self.cfg.encode_temporal_tile_size is not None:
            x = self.temporal_tiled_encode(x)
        else:
            x = self.encoder(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encode_single(x[i : i + 1]) for i in range(x.shape[0])], dim=0)

    def temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        overlap_size = int(
            self.cfg.decode_temporal_tile_latent_size * (1 - self.cfg.decode_temporal_tile_overlap_factor)
        )
        blend_extent = int(self.cfg.decode_temporal_tile_size * self.cfg.decode_temporal_tile_overlap_factor)
        t_limit = self.cfg.decode_temporal_tile_size - blend_extent

        feature_cache = {} if self.cfg.use_feature_cache else None

        row = []
        for i in tqdm(range(0, z.shape[2], overlap_size), desc="Tiled Decode", disable=not self.cfg.verbose):
            tile = z[:, :, i : i + self.cfg.decode_temporal_tile_latent_size, :, :]
            decoded = self.decoder(tile, feature_cache, f"decoder." if self.cfg.use_feature_cache else None)
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        height, width = z.shape[-2:]

        spatial_tile_size = self.cfg.spatial_tile_size
        spatial_tile_stride = round((1 - self.cfg.spatial_tile_overlap_factor) * spatial_tile_size)
        spatial_tile_latent_size = spatial_tile_size // self.spatial_compression_ratio
        spatial_tile_latent_stride = spatial_tile_stride // self.spatial_compression_ratio
        blend_size = spatial_tile_size - spatial_tile_stride

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, spatial_tile_latent_stride):
            row = []
            for j in range(0, width, spatial_tile_latent_stride):
                tile = z[..., i : i + spatial_tile_latent_size, j : j + spatial_tile_latent_size]
                if self.cfg.decode_temporal_tile_size is not None:
                    tile = self.temporal_tiled_decode(tile)
                else:
                    tile = self.decoder(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_w(rows[i - 1][j], tile, blend_size)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_size)
                result_row.append(tile[..., :spatial_tile_stride, :spatial_tile_stride])
            result_rows.append(torch.cat(result_row, dim=-1))

        decoded = torch.cat(result_rows, dim=-2)

        return decoded

    def decode_single(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.spatial_tile_size is not None:
            z = self.spatial_tiled_decode(z)
        elif self.cfg.decode_temporal_tile_size is not None:
            z = self.temporal_tiled_decode(z)
        else:
            z = self.decoder(z)
        if self.cfg.num_pad_frames > 0:
            z = z[:, :, self.cfg.num_pad_frames :, :, :]
        return z

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.decode_single(x[i : i + 1]) for i in range(x.shape[0])], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        z = self.encode(x)
        out = self.decode(z)
        return out


def st_dc_ae_f32t4c32_chunked_causal(name: str, pretrained_path: str) -> DCAEWithTemporalConfig:
    spatial_tile_overlap_factor = 0.0
    depth_list = "[0,5,10,4,4,4,4]"
    if name == "st-dc-ae-f32t4c32":
        chunk_size = 32
        scaling_factor = 0.7389
        spatial_tile_size = "null"
    elif name == "st-dc-ae-f32t4c32-chunk40":
        chunk_size = 40
        scaling_factor = 0.8018
        spatial_tile_size = "null"
    elif name == "st-dc-ae-f32t4c32-chunk40-ivj":
        chunk_size = 40
        scaling_factor = 0.7241
        spatial_tile_size = "null"
    elif name == "st-dc-ae-f32t4c32-chunk40-spatial-tile-512":
        chunk_size = 40
        scaling_factor = 0.8018
        spatial_tile_size = 512
        spatial_tile_overlap_factor = 0.25
    elif name in [
        "st-dc-ae-f32t4c32-chunked-causal-40-0.1",
        "st-dc-ae-f32t4c32-chunked-causal-40-0.2",
        "st-dc-ae-f32t4c32-chunked-causal-40-0.3",
    ]:
        chunk_size, spatial_tile_size = 40, "null"
        scaling_factor = 0.8018
    elif name == "st-dc-ae-f32t4c32-chunked-causal-40-0.4":
        chunk_size = 40
        depth_list = "[0,4,4,4,4,4,4]"
        scaling_factor = 1.2041
        spatial_tile_size = "null"
    else:
        raise ValueError(f"model {name} is not supported")

    cfg_str = (
        f"latent_channels=32 use_feature_cache=True encode_temporal_tile_size={chunk_size} encode_temporal_tile_latent_size={chunk_size//4} decode_temporal_tile_size={chunk_size} decode_temporal_tile_latent_size={chunk_size//4} "
        f"spatial_tile_size={spatial_tile_size} spatial_tile_overlap_factor={spatial_tile_overlap_factor} "
        f"encoder.project_in_block_type=ChunkedCausalConvPixelUnshuffle@2@1@3@3@{chunk_size} "
        f"encoder.depth_list={depth_list} encoder.width_list=[128,256,512,512,1024,1024,1024] "
        f"encoder.block_type=[ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size//4}] "
        f"encoder.downsample_block_type=[ChunkedCausalConvPixelUnshuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelUnshuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelUnshuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelUnshuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelUnshuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelUnshuffle@1@4@3@3@{chunk_size}] "
        f"encoder.project_out_block_type=ChunkedCausalConvLayer3d@3@3@{chunk_size//4} "
        f"decoder.depth_list={depth_list} decoder.width_list=[128,256,512,512,1024,1024,1024] "
        f"decoder.project_in_block_type=ChunkedCausalConvLayer3d@3@3@{chunk_size//4} "
        f"decoder.block_type=[ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size},ChunkedCausalResBlock3d@3@3@{chunk_size//4}] "
        f"decoder.upsample_block_type=[ChunkedCausalConvPixelShuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelShuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelShuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelShuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelShuffle@2@1@3@3@{chunk_size},ChunkedCausalConvPixelShuffle@1@4@3@3@{chunk_size//4}] "
        f"decoder.project_out_block_type=ChunkedCausalConvPixelShuffle@2@1@3@3@{chunk_size} "
        f"scaling_factor={scaling_factor}"
    )

    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: DCAEWithTemporalConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DCAEWithTemporalConfig), cfg)
    )
    cfg.pretrained_path = pretrained_path
    return cfg
