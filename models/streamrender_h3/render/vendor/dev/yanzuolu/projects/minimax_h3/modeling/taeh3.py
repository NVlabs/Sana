# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 Ollin Boer Bohan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MiniMax H3 tiny autoencoder with the project's video-VAE interface.

The network and H3 temporal mapping are adapted from ``madebyollin/taehv`` at
revision ``62f7591f59dfbb4c3c02b7a621d180a9eeaba26c``. The adapter deliberately
keeps checkpoint loading outside the constructor so the normal model weight
stage can strictly load either the released weights or a fine-tuned derivative.

TAEH3 consumes RGB in ``[0, 1]`` and directly emits normalized diffusion
latents. The identity pixel processor and identity latent statistics below make
that contract compatible with the full H3 VAE call sites without special cases
in a meta model.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .time_request import H3_VIDEO_TEMPORAL_MAPPING
from .video_vae.processor import VAEProcessor

_WorkItem = namedtuple("_WorkItem", ("tensor", "block_index"))
_H3_CLIP_LENGTH = 17
_H3_PREFIX_FRAMES = 3
_H3_TOKEN_DROP = 3
_H3_TOKENS_PER_CLIP = 5
_H3_FRAMES_PER_TOKEN = 4
_H3_SPATIAL_RATIO = 16


def _conv(in_channels: int, out_channels: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tanh(tensor / 3) * 3


class _MemoryBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(in_channels * 2, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
            nn.ReLU(inplace=True),
            _conv(out_channels, out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, tensor: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([tensor, past], dim=1)) + self.skip(tensor))


class _TemporalPool(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = tensor.shape
        return self.conv(tensor.reshape(-1, self.stride * channels, height, width))


class _TemporalGrow(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = tensor.shape
        tensor = self.conv(tensor)
        return tensor.reshape(-1, channels, height, width)


def _apply_parallel(model: nn.Sequential, tensor: torch.Tensor) -> torch.Tensor:
    """Apply temporal memory blocks in parallel over each sequence."""
    if tensor.ndim != 5:
        raise ValueError(f"TAEH3 expects NTCHW input, got shape {tuple(tensor.shape)}")
    batch, frames, channels, height, width = tensor.shape
    tensor = tensor.reshape(batch * frames, channels, height, width)

    for block in model:
        if isinstance(block, _MemoryBlock):
            flat_frames, channels, height, width = tensor.shape
            frames = flat_frames // batch
            sequence = tensor.reshape(batch, frames, channels, height, width)
            past = F.pad(sequence, (0, 0, 0, 0, 0, 0, 1, 0))[:, :frames]
            tensor = block(tensor, past.reshape(tensor.shape))
        else:
            tensor = block(tensor)

    flat_frames, channels, height, width = tensor.shape
    frames = flat_frames // batch
    return tensor.view(batch, frames, channels, height, width)


def _step_sequential(
    model: nn.Sequential,
    memory: list[Any],
    work_queue: list[_WorkItem],
) -> torch.Tensor | None:
    """Advance a causal block graph until it emits one output frame."""
    while work_queue:
        tensor, block_index = work_queue.pop(0)
        if block_index == len(model):
            return tensor.unsqueeze(1)
        block = model[block_index]
        if isinstance(block, _MemoryBlock):
            past = tensor * 0 if memory[block_index] is None else memory[block_index]
            output = block(tensor, past)
            memory[block_index] = tensor
            work_queue.insert(0, _WorkItem(output, block_index + 1))
        elif isinstance(block, _TemporalPool):
            if memory[block_index] is None:
                memory[block_index] = []
            memory[block_index].append(tensor)
            if len(memory[block_index]) > block.stride:
                raise ValueError(
                    f"temporal pool received more than {block.stride} pending frames"
                )
            if len(memory[block_index]) == block.stride:
                batch, channels, height, width = tensor.shape
                pooled = block(
                    torch.cat(memory[block_index], dim=1).view(
                        batch * block.stride, channels, height, width
                    )
                )
                memory[block_index] = []
                work_queue.insert(0, _WorkItem(pooled, block_index + 1))
        elif isinstance(block, _TemporalGrow):
            output = block(tensor)
            flat_frames, channels, height, width = output.shape
            chunks = output.view(
                flat_frames // block.stride,
                block.stride * channels,
                height,
                width,
            ).chunk(block.stride, dim=1)
            for next_tensor in reversed(chunks):
                work_queue.insert(0, _WorkItem(next_tensor, block_index + 1))
        else:
            work_queue.insert(0, _WorkItem(block(tensor), block_index + 1))
    return None


def _apply_sequential(model: nn.Sequential, tensor: torch.Tensor) -> torch.Tensor:
    """Apply temporal memory blocks with constant activation memory in time."""
    if tensor.ndim != 5:
        raise ValueError(f"TAEH3 expects NTCHW input, got shape {tuple(tensor.shape)}")
    work_queue = [_WorkItem(frame, 0) for frame in tensor.unbind(1)]
    memory: list[Any] = [None] * len(model)
    outputs = []
    while work_queue:
        output = _step_sequential(model, memory, work_queue)
        if output is not None:
            outputs.append(output)
    if not outputs:
        raise ValueError("TAEH3 input did not contain enough frames to produce output")
    return torch.cat(outputs, dim=1)


def _apply(
    model: nn.Sequential,
    tensor: torch.Tensor,
    *,
    parallel: bool,
) -> torch.Tensor:
    return _apply_parallel(model, tensor) if parallel else _apply_sequential(model, tensor)


class MiniMaxH3TAE(nn.Module):
    """TAEH3 exposed as a drop-in ``video_vae`` model node.

    ``encode_videos`` and ``decode_base`` use the same NCTHW-facing contract as
    ``MiniMaxH3VideoVAE``. ``encode_video`` and ``decode_video`` expose the
    upstream NTCHW contract for parity tests and direct use.
    """

    video_temporal_mapping = H3_VIDEO_TEMPORAL_MAPPING

    def __init__(
        self,
        *,
        parallel_encode: bool = True,
        parallel_decode: bool = True,
    ) -> None:
        super().__init__()
        self.parallel_encode = bool(parallel_encode)
        self.parallel_decode = bool(parallel_decode)
        self.patch_size = 2
        self.latent_channels = 24
        self.image_channels = 3
        self.vae_ratio = _H3_SPATIAL_RATIO
        self.vae_ratio_t = _H3_FRAMES_PER_TOKEN
        self.t_downscale = _H3_FRAMES_PER_TOKEN
        self.t_upscale = _H3_FRAMES_PER_TOKEN
        self.frames_to_trim = self.t_upscale - 1

        self.encoder = nn.Sequential(
            _conv(12, 64),
            nn.ReLU(inplace=True),
            _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _TemporalPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _TemporalPool(64, 1),
            _conv(64, 64, stride=2, bias=False),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _conv(64, self.latent_channels),
        )
        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(self.latent_channels, 256),
            nn.ReLU(inplace=True),
            _MemoryBlock(256, 256),
            _MemoryBlock(256, 256),
            _MemoryBlock(256, 256),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(256, 1),
            _conv(256, 128, bias=False),
            _MemoryBlock(128, 128),
            _MemoryBlock(128, 128),
            _MemoryBlock(128, 128),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(128, 2),
            _conv(128, 64, bias=False),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            _MemoryBlock(64, 64),
            nn.Upsample(scale_factor=2),
            _TemporalGrow(64, 2),
            _conv(64, 64, bias=False),
            nn.ReLU(inplace=True),
            _conv(64, 12),
        )

        self.processor = VAEProcessor(
            vae_ratio=self.vae_ratio,
            vae_ratio_t=self.vae_ratio_t,
            clip_length=_H3_CLIP_LENGTH,
            frame_overlap=5,
            token_overlap=2,
            tokens_chunk_size=_H3_TOKENS_PER_CLIP,
            isolated_last_frame=False,
            latent_patch_size=1,
            crop_mode="top_left",
            pixel_norm_type="raw",
            use_3d_conv=True,
        )
        self.register_buffer(
            "latents_mean", torch.zeros(self.latent_channels), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.ones(self.latent_channels), persistent=False
        )

    @staticmethod
    def _validate_spatial_shape(tensor: torch.Tensor, name: str) -> None:
        height, width = tensor.shape[-2:]
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} spatial dimensions must be positive")
        if height % _H3_SPATIAL_RATIO or width % _H3_SPATIAL_RATIO:
            raise ValueError(
                f"{name} height and width must be divisible by {_H3_SPATIAL_RATIO}, "
                f"got {(height, width)}"
            )

    @staticmethod
    def _validate_video_frame_count(frame_count: int) -> None:
        quotient, remainder = divmod(frame_count, _H3_CLIP_LENGTH)
        if not (
            (quotient >= 1 and remainder == 0)
            or (frame_count >= 5 and remainder == 5)
        ):
            raise ValueError(
                f"MiniMax H3 video frame count must match 17n or 17n+5, "
                f"got {frame_count}"
            )

    @staticmethod
    def _validate_latent_frame_count(frame_count: int) -> None:
        quotient, remainder = divmod(frame_count, _H3_TOKENS_PER_CLIP)
        if not (
            (quotient >= 1 and remainder == 0)
            or (frame_count >= 2 and remainder == 2)
        ):
            raise ValueError(
                f"MiniMax H3 latent frame count must match 5n or 5n+2, "
                f"got {frame_count}"
            )

    @staticmethod
    def _pixel_unshuffle(tensor: torch.Tensor) -> torch.Tensor:
        return F.pixel_unshuffle(tensor, 2)

    @staticmethod
    def _pixel_shuffle(tensor: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(tensor, 2).clamp_(0, 1)

    def _encode_image_tensor(
        self,
        tensor: torch.Tensor,
        *,
        parallel: bool,
    ) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.image_channels:
            raise ValueError(
                f"TAEH3 expects NCHW RGB images, got shape {tuple(tensor.shape)}"
            )
        self._validate_spatial_shape(tensor, "TAEH3 image")
        sequence = F.pad(tensor.unsqueeze(1), (0, 0, 0, 0, 0, 0, 3, 0))
        sequence = self._pixel_unshuffle(sequence)
        latent = _apply(self.encoder, sequence, parallel=parallel)
        if latent.shape[1] != 1:
            raise RuntimeError(
                f"TAEH3 image encoding produced {latent.shape[1]} latent frames"
            )
        return latent

    def encode_video(
        self,
        tensor: torch.Tensor,
        *,
        parallel: bool | None = None,
    ) -> torch.Tensor:
        """Encode ``NTCHW`` RGB in ``[0, 1]`` to normalized H3 latents."""
        if tensor.ndim != 5 or tensor.shape[2] != self.image_channels:
            raise ValueError(
                f"TAEH3 expects NTCHW RGB video, got shape {tuple(tensor.shape)}"
            )
        self._validate_spatial_shape(tensor, "TAEH3 video")
        self._validate_video_frame_count(int(tensor.shape[1]))
        parallel = self.parallel_encode if parallel is None else bool(parallel)

        batch = tensor.shape[0]
        has_tail = tensor.shape[1] % _H3_CLIP_LENGTH == 5
        trailing = (-tensor.shape[1]) % _H3_CLIP_LENGTH
        if trailing:
            tensor = torch.cat(
                [tensor, tensor[:, -1:].expand(-1, trailing, -1, -1, -1)], dim=1
            )
        chunks = tensor.reshape(batch, -1, _H3_CLIP_LENGTH, *tensor.shape[2:])
        chunks = F.pad(chunks, (0, 0, 0, 0, 0, 0, _H3_PREFIX_FRAMES, 0))
        chunks = self._pixel_unshuffle(chunks)

        if parallel:
            encoded = _apply_parallel(self.encoder, chunks.flatten(0, 1))
            encoded = encoded.reshape(batch, -1, *encoded.shape[2:])
        else:
            encoded = torch.cat(
                [_apply_sequential(self.encoder, chunk) for chunk in chunks.unbind(1)],
                dim=1,
            )
        return encoded[:, :-_H3_TOKEN_DROP] if has_tail else encoded

    def _decode_image_latent(
        self,
        tensor: torch.Tensor,
        *,
        parallel: bool,
    ) -> torch.Tensor:
        decoded = _apply(self.decoder, tensor, parallel=parallel)
        return self._pixel_shuffle(decoded[:, self.frames_to_trim :])

    def decode_video(
        self,
        tensor: torch.Tensor,
        *,
        parallel: bool | None = None,
    ) -> torch.Tensor:
        """Decode normalized H3 ``NTCHW`` latents to RGB in ``[0, 1]``."""
        if tensor.ndim != 5 or tensor.shape[2] != self.latent_channels:
            raise ValueError(
                f"TAEH3 expects NTCHW latents with 24 channels, got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
            raise ValueError("TAEH3 latent spatial dimensions must be positive")
        self._validate_latent_frame_count(int(tensor.shape[1]))
        parallel = self.parallel_decode if parallel is None else bool(parallel)

        decoded = _apply(self.decoder, tensor, parallel=parallel)
        chunk_frames = _H3_TOKENS_PER_CLIP * self.t_upscale
        trailing = (-decoded.shape[1]) % chunk_frames
        if trailing:
            decoded = F.pad(decoded, (0, 0, 0, 0, 0, 0, 0, trailing))
        decoded = decoded.unflatten(1, (-1, chunk_frames))
        decoded = decoded[:, :, self.frames_to_trim :].flatten(1, 2)
        if trailing:
            decoded = decoded[:, :-trailing]
        return self._pixel_shuffle(decoded)

    @staticmethod
    def _ensure_list(values: Any) -> list[Any]:
        return values if isinstance(values, list) else [values]

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_images(
        self,
        images: Any,
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
    ) -> list[torch.Tensor]:
        """Encode PIL, numpy, or tensor images with full-VAE-compatible output."""
        del verbose
        images = self._ensure_list(images)
        if not images:
            raise ValueError("encode_images requires at least one image")

        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            images = [
                np.asarray(image) if isinstance(image, Image.Image) else image
                for image in images
            ]

        if isinstance(images[0], np.ndarray):
            images = torch.split(
                self.processor.convert_numpy_to_tensor(images, self._model_device()),
                1,
                dim=0,
            )
            transform_input = True

        prepared = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"unsupported image type {type(image).__name__}")
            if transform_input:
                image = self.processor.transform_tensor(image)
            elif image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[1] != self.image_channels:
                raise ValueError(f"expected NCHW image, got shape {tuple(image.shape)}")
            self._validate_spatial_shape(image, "TAEH3 image")
            prepared.append(image)

        if len({tuple(image.shape) for image in prepared}) == 1:
            encoded = self._encode_image_tensor(
                torch.cat(prepared, dim=0), parallel=self.parallel_encode
            )
            outputs = [
                latent.permute(1, 0, 2, 3) for latent in encoded.unbind(0)
            ]
        else:
            outputs = [
                self._encode_image_tensor(image, parallel=self.parallel_encode)[0].permute(
                    1, 0, 2, 3
                )
                for image in prepared
            ]
        if use_fp16_latent:
            outputs = [latent.to(torch.float16) for latent in outputs]
        return [latent.contiguous() for latent in outputs]

    @torch.no_grad()
    def encode_videos(
        self,
        videos: Any,
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
        encode_prefix: bool = False,
    ) -> list[torch.Tensor]:
        """Encode CTHW videos and return one CTHW latent tensor per video."""
        del verbose
        if encode_prefix:
            raise ValueError("TAEH3 does not support continuation-prefix encoding")
        videos = self._ensure_list(videos)
        if not videos:
            raise ValueError("encode_videos requires at least one video")

        if isinstance(videos[0], np.ndarray):
            videos = [
                self.processor.convert_numpy_to_tensor(video, self._model_device())
                for video in videos
            ]
            transform_input = True

        prepared = []
        for video in videos:
            if not isinstance(video, torch.Tensor):
                raise TypeError(f"unsupported video type {type(video).__name__}")
            if transform_input:
                video = self.processor.transform_tensor(video).transpose(0, 1)
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if video.ndim != 5 or video.shape[1] != self.image_channels:
                raise ValueError(f"expected NCTHW video, got shape {tuple(video.shape)}")
            if video.shape[0] != 1:
                raise ValueError("each encode_videos list item must contain one video")
            self._validate_spatial_shape(video, "TAEH3 video")
            self._validate_video_frame_count(int(video.shape[2]))
            prepared.append(video)

        outputs = []
        same_shape = len({tuple(video.shape) for video in prepared}) == 1
        if same_shape:
            batch = torch.cat(prepared, dim=0).permute(0, 2, 1, 3, 4)
            encoded = self.encode_video(batch)
            outputs = [latent.permute(1, 0, 2, 3) for latent in encoded.unbind(0)]
        else:
            for video in prepared:
                encoded = self.encode_video(video.permute(0, 2, 1, 3, 4))[0]
                outputs.append(encoded.permute(1, 0, 2, 3))
        if use_fp16_latent:
            outputs = [latent.to(torch.float16) for latent in outputs]
        return [latent.contiguous() for latent in outputs]

    @torch.no_grad()
    def decode_base(
        self,
        latent: torch.Tensor,
        frame_num: int | None = None,
        process_image: bool = False,
    ) -> torch.Tensor:
        """Decode NCTHW normalized latents with the full H3 VAE return layout."""
        if latent.ndim == 4:
            if not process_image:
                raise ValueError("rank-4 latent input requires process_image=True")
            latent = latent.unsqueeze(2)
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(f"expected NCTHW latent, got shape {tuple(latent.shape)}")
        ntchw = latent.permute(0, 2, 1, 3, 4)
        if process_image:
            decoded = self._decode_image_latent(
                ntchw, parallel=self.parallel_decode
            ).permute(0, 2, 1, 3, 4)
            return decoded.squeeze(2)

        decoded = self.decode_video(ntchw).permute(0, 2, 1, 3, 4)
        if frame_num is not None and frame_num < decoded.shape[2]:
            decoded = decoded[:, :, -frame_num:]
        return decoded.contiguous()


EntryClass = MiniMaxH3TAE

__all__ = ["MiniMaxH3TAE", "EntryClass"]
