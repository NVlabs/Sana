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
"""TAE-H3 model node with continuous temporal encoding and decoding.

The temporal rules follow ``StreamingTAEHV`` from ``madebyollin/taehv`` at
revision ``011dfc2112197741c540e0bdd5b7b67bcc930771``. Encoding pads the final
RGB frame to complete each four-frame group. Decoding trims the first three
output frames once across the complete latent sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .taeh3 import MiniMaxH3TAE, _apply, _step_sequential, _WorkItem
from .time_request import CONTINUOUS_VIDEO_TEMPORAL_MAPPING
from .video_vae.processor import VAEProcessor


class _StreamingTAEProcessor(VAEProcessor):
    def get_suitable_video_length(self, video_length: int, verbose: bool = False) -> int:
        """Retain every frame of a nonempty input sequence."""
        del verbose
        video_length = int(video_length)
        if video_length <= 0:
            raise ValueError("continuous TAE video frame count must be positive")
        return video_length

    def get_latent_length(self, video_length: int) -> int:
        """Count latent frames after completing the final four-frame group."""
        video_length = self.get_suitable_video_length(video_length)
        return (video_length + self.vae_ratio_t - 1) // self.vae_ratio_t


class _TAEStream:
    """Independent NTCHW encoder and decoder state over one codec's weights.

    RGB uses [0, 1], and latents use the codec's normalized coordinates. Four
    RGB frames produce one latent. Decoder memory persists across calls and
    removes only the stream's first three RGB frames. Flush completes pending
    encoder frames without resetting state. Set the codec's device and dtype
    before creating a stream, and reset after changing either.
    """

    def __init__(self, codec: MiniMaxH3StreamingTAE) -> None:
        self.codec = codec
        self.reset()

    def reset(self) -> None:
        """Begin independent encoder and decoder sequences."""
        self._encoder_memory: list[Any] = [None] * len(self.codec.encoder)
        self._decoder_memory: list[Any] = [None] * len(self.codec.decoder)
        self._encoder_frames = 0
        self._encoder_last_frame: torch.Tensor | None = None
        self._decoder_trim = self.codec.frames_to_trim

    @staticmethod
    def _run(
        model: torch.nn.Sequential, memory: list[Any], tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        queue = [_WorkItem(frame, 0) for frame in tensor.unbind(1)]
        outputs = []
        while queue:
            output = _step_sequential(model, memory, queue)
            if output is not None:
                outputs.append(output)
        return torch.cat(outputs, dim=1) if outputs else None

    def encode(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Consume NTCHW RGB and return every completed latent frame."""
        if tensor.ndim != 5 or tensor.shape[2] != self.codec.image_channels:
            raise ValueError(f"continuous TAE expects NTCHW RGB video, got {tuple(tensor.shape)}")
        self.codec._validate_spatial_shape(tensor, "continuous TAE video")
        self.codec._validate_video_frame_count(int(tensor.shape[1]))
        tensor = self.codec._pixel_unshuffle(tensor)
        self._encoder_frames += int(tensor.shape[1])
        self._encoder_last_frame = tensor[:, -1:]
        return self._run(self.codec.encoder, self._encoder_memory, tensor)

    def flush_encoder(self) -> torch.Tensor | None:
        """Repeat the last RGB frame to finish a pending four-frame group."""
        trailing = (-self._encoder_frames) % self.codec.t_downscale
        if not trailing:
            return None
        tensor = self._encoder_last_frame.expand(-1, trailing, -1, -1, -1)
        self._encoder_frames += trailing
        return self._run(self.codec.encoder, self._encoder_memory, tensor)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Consume NTCHW latents and return all newly available NTCHW RGB."""
        if tensor.ndim != 5 or tensor.shape[2] != self.codec.latent_channels:
            raise ValueError(f"continuous TAE expects NTCHW latents with 24 channels, got {tuple(tensor.shape)}")
        if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
            raise ValueError("continuous TAE latent spatial dimensions must be positive")
        self.codec._validate_latent_frame_count(int(tensor.shape[1]))
        decoded = self._run(self.codec.decoder, self._decoder_memory, tensor)
        trim = min(self._decoder_trim, int(decoded.shape[1]))
        self._decoder_trim -= trim
        return self.codec._pixel_shuffle(decoded[:, trim:])

    def flush_decoder(self) -> None:
        """Decoding emits every available RGB frame during each decode call."""
        return None


class MiniMaxH3StreamingTAE(MiniMaxH3TAE):
    """Continuous TAE-H3 with the standard ``video_vae`` model interface.

    RGB uses [0, 1], and 24-channel latents are already normalized. T positive
    RGB frames encode to ceil(T / 4) latents by repeating the final RGB frame
    when necessary. L positive latents decode to 4L - 3 RGB frames. An image
    follows the same rules as a one-frame video.

    Whole-video calls process independent sequences. Parallel flags select
    temporal batching or sequential execution with local memory. create_stream
    retains memory between input fragments with the same temporal mapping.
    The inherited network and checkpoint parameter names are preserved.
    """

    video_temporal_mapping = CONTINUOUS_VIDEO_TEMPORAL_MAPPING

    def __init__(
        self,
        *,
        parallel_encode: bool = True,
        parallel_decode: bool = True,
    ) -> None:
        super().__init__(
            parallel_encode=parallel_encode, parallel_decode=parallel_decode
        )
        self.processor = _StreamingTAEProcessor(
            vae_ratio=self.vae_ratio,
            vae_ratio_t=self.vae_ratio_t,
            clip_length=self.t_downscale,
            frame_overlap=0,
            token_overlap=0,
            tokens_chunk_size=1,
            isolated_last_frame=False,
            latent_patch_size=1,
            crop_mode="top_left",
            pixel_norm_type="raw",
            use_3d_conv=True,
        )

    def create_stream(self) -> _TAEStream:
        """Create independent NTCHW state sharing this codec's current weights."""
        return _TAEStream(self)

    @staticmethod
    def _validate_video_frame_count(frame_count: int) -> None:
        if frame_count <= 0:
            raise ValueError("continuous TAE video frame count must be positive")

    @staticmethod
    def _validate_latent_frame_count(frame_count: int) -> None:
        if frame_count <= 0:
            raise ValueError("continuous TAE latent frame count must be positive")

    def _encode_image_tensor(
        self, tensor: torch.Tensor, *, parallel: bool,
    ) -> torch.Tensor:
        return self.encode_video(tensor.unsqueeze(1), parallel=parallel)

    def encode_video(
        self, tensor: torch.Tensor, *, parallel: bool | None = None,
    ) -> torch.Tensor:
        """Encode NTCHW RGB into ceil(T / 4) normalized latent frames."""
        if tensor.ndim != 5 or tensor.shape[2] != self.image_channels:
            raise ValueError(
                f"continuous TAE expects NTCHW RGB video, got {tuple(tensor.shape)}"
            )
        self._validate_spatial_shape(tensor, "continuous TAE video")
        self._validate_video_frame_count(int(tensor.shape[1]))
        parallel = self.parallel_encode if parallel is None else bool(parallel)
        trailing = (-tensor.shape[1]) % self.t_downscale
        if trailing:
            tensor = torch.cat(
                [tensor, tensor[:, -1:].expand(-1, trailing, -1, -1, -1)], dim=1
            )
        return _apply(self.encoder, self._pixel_unshuffle(tensor), parallel=parallel)

    def decode_video(
        self, tensor: torch.Tensor, *, parallel: bool | None = None,
    ) -> torch.Tensor:
        """Decode NTCHW latents into 4L - 3 RGB frames with one startup trim."""
        if tensor.ndim != 5 or tensor.shape[2] != self.latent_channels:
            raise ValueError(
                f"continuous TAE expects NTCHW latents with 24 channels, got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
            raise ValueError("continuous TAE latent spatial dimensions must be positive")
        self._validate_latent_frame_count(int(tensor.shape[1]))
        parallel = self.parallel_decode if parallel is None else bool(parallel)
        decoded = _apply(self.decoder, tensor, parallel=parallel)
        return self._pixel_shuffle(decoded[:, self.frames_to_trim :])

    @torch.no_grad()
    def encode_images(
        self,
        images: Any,
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
    ) -> list[torch.Tensor]:
        """Encode PIL/NumPy RGB images or CHW/NCHW tensors.

        Tensor inputs use [0, 1]. transform_input applies the pixel transform
        along the explicit channel axis without changing the input layout.
        """
        images = self._ensure_list(images)
        if transform_input:
            images = [
                self.processor.transform(image) if isinstance(image, torch.Tensor) else image
                for image in images
            ]
        return super().encode_images(
            images, transform_input=False,
            use_fp16_latent=use_fp16_latent, verbose=verbose,
        )

    @torch.no_grad()
    def encode_videos(
        self,
        videos: Any,
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
        encode_prefix: bool = False,
    ) -> list[torch.Tensor]:
        """Encode NumPy THWC RGB videos or CTHW/NCTHW tensors.

        NumPy pixels are converted from [0, 255] to [0, 1]. Tensor inputs use
        [0, 1]. transform_input applies the pixel transform at the explicit
        channel axis. Return one CTHW latent tensor per video.
        """
        prepared = []
        for video in self._ensure_list(videos):
            if isinstance(video, np.ndarray):
                video = self.processor.convert_numpy_to_tensor(
                    video, self._model_device()
                )
                video = self.processor.transform(video).transpose(0, 1)
            elif transform_input and isinstance(video, torch.Tensor):
                if video.ndim == 4:
                    video = self.processor.transform(video.transpose(0, 1)).transpose(0, 1)
                elif video.ndim == 5:
                    batch, channels, frames, height, width = video.shape
                    pixels = video.transpose(1, 2).reshape(
                        batch * frames, channels, height, width
                    )
                    video = self.processor.transform(pixels).reshape(
                        batch, frames, channels, height, width
                    ).transpose(1, 2)
            prepared.append(video)
        return super().encode_videos(
            prepared, transform_input=False, use_fp16_latent=use_fp16_latent,
            verbose=verbose, encode_prefix=encode_prefix,
        )


EntryClass = MiniMaxH3StreamingTAE

__all__ = ["MiniMaxH3StreamingTAE", "EntryClass"]
