# SPDX-License-Identifier: MIT
"""Incremental TAE-H3 codec with the native H3 17-frame/5-latent clock.

Unlike ``taeh3_streaming.MiniMaxH3StreamingTAE``, this class keeps the native
H3 temporal mapping used to train the 1001a checkpoint.  Encoder state resets
at every 17-RGB-frame native block and prepends the three alignment slots used
by the ordinary TAE-H3 encoder.  Decoder memory remains continuous, while the
three invalid leading RGB frames are removed at every five-latent boundary.

The stream accepts arbitrary positive fragments.  A caller can therefore push
the 1001a steady sequence of 3/2 native latents and receive 9/8 RGB frames
without buffering until five latents are available.
"""

from __future__ import annotations

from typing import Any

import torch

from .taeh3 import MiniMaxH3TAE, _WorkItem, _step_sequential


class _NativeH3TAEStream:
    """One independent native-H3 encoder/decoder stream sharing codec weights."""

    pixel_frames_per_block = 17
    latent_frames_per_block = 5
    prefix_pixel_frames = 3
    # Force output adapters to dispatch a multi-latent DiT chunk one latent at
    # a time. The decoder never sees a future latent while producing RGB for
    # the current latent.
    requires_single_latent_dispatch = True
    requires_single_frame_encode_dispatch = True

    def __init__(self, codec: "MiniMaxH3NativeStreamingTAE") -> None:
        self.codec = codec
        self.reset()

    def reset(self) -> None:
        self._encoder_memory: list[Any] = [None] * len(self.codec.encoder)
        self._decoder_memory: list[Any] = [None] * len(self.codec.decoder)
        self._encoder_block_position = 0
        self._decoder_block_position = 0
        self.last_encode_latent_counts: list[int] = []
        self.last_decode_frame_counts: list[int] = []

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

    def _start_encoder_block(self, pixels: torch.Tensor) -> None:
        self._encoder_memory = [None] * len(self.codec.encoder)
        prefix = pixels[:, :1].new_zeros(
            pixels.shape[0], self.prefix_pixel_frames, pixels.shape[2],
            pixels.shape[3], pixels.shape[4],
        )
        prefix = self.codec._pixel_unshuffle(prefix)
        emitted = self._run(self.codec.encoder, self._encoder_memory, prefix)
        if emitted is not None:
            raise RuntimeError("native H3 encoder prefix unexpectedly emitted a latent")

    @torch.no_grad()
    def encode_one(self, pixel: torch.Tensor) -> torch.Tensor | None:
        """Encode exactly one present RGB frame without seeing future frames."""
        if pixel.ndim != 5 or pixel.shape[2] != self.codec.image_channels:
            raise ValueError(
                f"native streaming TAE expects NTCHW RGB, got {tuple(pixel.shape)}"
            )
        if pixel.shape[1] != 1:
            raise ValueError(
                "strict native streaming encode_one requires exactly one RGB frame, "
                f"got T={pixel.shape[1]}"
            )
        self.codec._validate_spatial_shape(pixel, "native streaming TAE video")
        if self._encoder_block_position == 0:
            self._start_encoder_block(pixel)

        current = self.codec._pixel_unshuffle(pixel)
        encoded = self._run(self.codec.encoder, self._encoder_memory, current)
        self._encoder_block_position = (
            self._encoder_block_position + 1
        ) % self.pixel_frames_per_block
        if encoded is not None and encoded.shape[1] != 1:
            raise RuntimeError(
                "strict native H3 encoder may emit at most one latent per RGB frame, "
                f"got {encoded.shape[1]}"
            )
        return encoded

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor | None:
        """Compatibility wrapper that executes strictly one RGB frame at a time."""
        if pixels.ndim != 5 or pixels.shape[2] != self.codec.image_channels:
            raise ValueError(f"native streaming TAE expects NTCHW RGB, got {tuple(pixels.shape)}")
        self.codec._validate_spatial_shape(pixels, "native streaming TAE video")
        if pixels.shape[1] <= 0:
            return None

        outputs = []
        self.last_encode_latent_counts = []
        for frame_index in range(int(pixels.shape[1])):
            encoded = self.encode_one(pixels[:, frame_index:frame_index + 1])
            count = 0 if encoded is None else int(encoded.shape[1])
            self.last_encode_latent_counts.append(count)
            if count:
                outputs.append(encoded)
        return torch.cat(outputs, dim=1) if outputs else None

    def flush_encoder(self) -> None:
        """The online benchmark intentionally ignores an incomplete tail."""
        return None

    @torch.no_grad()
    def decode_one(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode exactly one causal latent without access to a future latent.

        One latent creates four raw decoder frames. At the start of every
        native five-latent H3 block, the first three alignment frames are
        removed, yielding the native 1/4/4/4/4 publication cadence.
        """
        if latent.ndim != 5 or latent.shape[2] != self.codec.latent_channels:
            raise ValueError(
                "native streaming TAE expects NTCHW latents with 24 channels, "
                f"got {tuple(latent.shape)}"
            )
        if latent.shape[1] != 1:
            raise ValueError(
                "strict native streaming decode_one requires exactly one latent, "
                f"got T={latent.shape[1]}"
            )

        decoded = self._run(self.codec.decoder, self._decoder_memory, latent)
        if decoded is None or decoded.shape[1] != self.codec.t_upscale:
            count = None if decoded is None else int(decoded.shape[1])
            raise RuntimeError(
                "native H3 decoder must emit four raw frames per latent, "
                f"got {count}"
            )
        if self._decoder_block_position == 0:
            decoded = decoded[:, self.codec.frames_to_trim:]
        self._decoder_block_position = (
            self._decoder_block_position + 1
        ) % self.latent_frames_per_block
        return self.codec._pixel_shuffle(decoded)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper that executes strictly one latent at a time."""
        if latents.ndim != 5 or latents.shape[2] != self.codec.latent_channels:
            raise ValueError(
                "native streaming TAE expects NTCHW latents with 24 channels, "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[1] <= 0:
            return latents.new_empty(
                latents.shape[0], 0, self.codec.image_channels,
                latents.shape[-2] * self.codec.vae_ratio,
                latents.shape[-1] * self.codec.vae_ratio,
            )

        outputs = []
        self.last_decode_frame_counts = []
        for latent_index in range(int(latents.shape[1])):
            decoded = self.decode_one(latents[:, latent_index:latent_index + 1])
            self.last_decode_frame_counts.append(int(decoded.shape[1]))
            outputs.append(decoded)
        return torch.cat(outputs, dim=1)

    def flush_decoder(self) -> None:
        return None


class _FiveLatentH3TAEStream(_NativeH3TAEStream):
    """Buffer across DiT chunks and launch decode only for five latents."""

    requires_single_latent_dispatch = False
    decode_block_latents = 5

    def reset(self) -> None:
        super().reset()
        self._pending_decode_latents: list[torch.Tensor] = []
        self.last_decode_block_sizes: list[int] = []

    def _decode_block(self, block: torch.Tensor, *, final_tail: bool) -> torch.Tensor:
        decoded = self._run(self.codec.decoder, self._decoder_memory, block)
        if decoded is None:
            raise RuntimeError("five-latent TAE decoder did not emit RGB")
        expected = int(block.shape[1]) * self.codec.t_upscale
        if decoded.shape[1] != expected:
            raise RuntimeError(
                f"TAE decode emitted {decoded.shape[1]} raw frames, expected {expected}"
            )
        decoded = decoded[:, self.codec.frames_to_trim:]
        if not final_tail and block.shape[1] != self.decode_block_latents:
            raise RuntimeError("steady TAE decode requires exactly five latents")
        return self.codec._pixel_shuffle(decoded)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        if latents.ndim != 5 or latents.shape[2] != self.codec.latent_channels:
            raise ValueError(
                "native streaming TAE expects NTCHW latents with 24 channels, "
                f"got {tuple(latents.shape)}"
            )
        self._pending_decode_latents.extend(
            latents[:, index:index + 1]
            for index in range(int(latents.shape[1]))
        )
        outputs = []
        self.last_decode_block_sizes = []
        while len(self._pending_decode_latents) >= self.decode_block_latents:
            block = torch.cat(
                self._pending_decode_latents[:self.decode_block_latents], dim=1
            )
            del self._pending_decode_latents[:self.decode_block_latents]
            outputs.append(self._decode_block(block, final_tail=False))
            self.last_decode_block_sizes.append(self.decode_block_latents)
        return torch.cat(outputs, dim=1) if outputs else None

    @torch.no_grad()
    def flush_decoder(self) -> torch.Tensor | None:
        if not self._pending_decode_latents:
            return None
        block = torch.cat(self._pending_decode_latents, dim=1)
        self._pending_decode_latents.clear()
        self.last_decode_block_sizes = [int(block.shape[1])]
        return self._decode_block(block, final_tail=True)


class _OfficialWholeVideoH3TAEStream(_NativeH3TAEStream):
    """Buffer every generated latent and use the official whole-video decode."""

    requires_single_latent_dispatch = False

    def reset(self) -> None:
        super().reset()
        self._whole_video_latents: list[torch.Tensor] = []

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5 or latents.shape[2] != self.codec.latent_channels:
            raise ValueError(
                "official whole-video TAE expects NTCHW latents with 24 channels, "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[1]:
            self._whole_video_latents.append(latents.detach().clone())
        return None

    @torch.no_grad()
    def flush_decoder(self) -> torch.Tensor | None:
        if not self._whole_video_latents:
            return None
        latents = torch.cat(self._whole_video_latents, dim=1)
        self._whole_video_latents.clear()
        # This is the upstream/basic TAEH3 path: one call over the complete
        # latent sequence, with MemBlocks evaluated in parallel.
        return self.codec.decode_video(latents, parallel=True)


class MiniMaxH3NativeStreamingTAE(MiniMaxH3TAE):
    """TAE-H3 with variable-fragment streams and the original H3 clock."""

    def create_stream(self) -> _NativeH3TAEStream:
        return _NativeH3TAEStream(self)


class MiniMaxH3FiveLatentDecodeTAE(MiniMaxH3NativeStreamingTAE):
    """Strict RGB streaming encode with five-latent block decode."""

    def create_stream(self) -> _FiveLatentH3TAEStream:
        return _FiveLatentH3TAEStream(self)


class MiniMaxH3OfficialWholeVideoDecodeTAE(MiniMaxH3NativeStreamingTAE):
    """Strict streaming encode plus official whole-sequence parallel decode."""

    def create_stream(self) -> _OfficialWholeVideoH3TAEStream:
        return _OfficialWholeVideoH3TAEStream(self)


EntryClass = MiniMaxH3NativeStreamingTAE

__all__ = [
    "MiniMaxH3NativeStreamingTAE",
    "MiniMaxH3FiveLatentDecodeTAE",
    "MiniMaxH3OfficialWholeVideoDecodeTAE",
    "EntryClass",
]
