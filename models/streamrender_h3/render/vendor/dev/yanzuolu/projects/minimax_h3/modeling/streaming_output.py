# SPDX-License-Identifier: Apache-2.0
"""Incremental RGB and predictive-context audio decoding on a shared media clock."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from dev.yanzuolu.projects.minimax_h3.modeling.audio_vae.alias_free import Activation1d
from dev.yanzuolu.projects.minimax_h3.modeling.audio_vae.bigvgan import AMPBlock1, BigVGAN
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS


def _conv_input_span(
    module: nn.Conv1d | nn.ConvTranspose1d, span: tuple[int, int],
) -> tuple[int, int]:
    """Back-project an output interval through a temporal convolution."""
    start, stop = span
    stride, padding, dilation, kernel = (
        module.stride[0], module.padding[0], module.dilation[0], module.kernel_size[0]
    )
    support = (kernel - 1) * dilation
    if isinstance(module, nn.ConvTranspose1d):
        lower = start + padding - support
        return -((-lower) // stride), (stop - 1 + padding) // stride + 1
    return start * stride - padding, (stop - 1) * stride - padding + support + 1


def _activation_input_span(
    module: Activation1d, span: tuple[int, int],
) -> tuple[int, int]:
    """Include both finite anti-alias filters around a pointwise activation."""
    if module.up_ratio != module.down_ratio:
        raise ValueError("audio activation must preserve temporal resolution")
    down = module.downsample.lowpass
    left_pad = down.pad_left if down.padding else 0
    start = span[0] * down.stride - left_pad
    stop = (span[1] - 1) * down.stride - left_pad + down.kernel_size
    up = module.upsample
    lower = start + up.pad_left - up.kernel_size + 1
    return (
        -((-lower) // up.stride) - up.pad,
        (stop - 1 + up.pad_left) // up.stride + 1 - up.pad,
    )


def _residual_input_span(module: AMPBlock1, span: tuple[int, int]) -> tuple[int, int]:
    for index in reversed(range(len(module.convs1))):
        branch = _conv_input_span(module.convs2[index], span)
        branch = _activation_input_span(module.activations[2 * index + 1], branch)
        branch = _conv_input_span(module.convs1[index], branch)
        branch = _activation_input_span(module.activations[2 * index], branch)
        span = min(span[0], branch[0]), max(span[1], branch[1])
    return span


@dataclass(frozen=True)
class AudioDecoderContext:
    samples_per_latent: int
    left_latents: int
    right_latents: int


def bigvgan_decoder_context(audio_vae: Any) -> AudioDecoderContext:
    """Derive sufficient latent context from the actual BigVGAN module graph.

    Convolution bounds include the up/down filters inside every alias-free
    activation. Context is rounded outward to whole latent frames so cropped
    decoder calls retain the full network's upsampling phase.
    """
    decoder = audio_vae.decoder
    if not isinstance(decoder, BigVGAN):
        raise TypeError("incremental audio decoding requires the MiniMax H3 BigVGAN decoder")
    projection = audio_vae.dec_in_proj
    if not isinstance(projection, nn.Conv1d) or projection.stride != (1,):
        raise ValueError("audio decoder input projection must have unit temporal stride")
    hop = math.prod(layer.stride[0] for group in decoder.ups for layer in group)
    span = _conv_input_span(decoder.conv_post, (0, hop))
    span = _activation_input_span(decoder.activation_post, span)
    for index in reversed(range(decoder.num_upsamples)):
        branches = [
            _residual_input_span(decoder.resblocks[index * decoder.num_kernels + branch], span)
            for branch in range(decoder.num_kernels)
        ]
        span = min(value[0] for value in branches), max(value[1] for value in branches)
        for layer in reversed(decoder.ups[index]):
            span = _conv_input_span(layer, span)
    span = _conv_input_span(decoder.conv_pre, span)
    span = _conv_input_span(projection, span)
    return AudioDecoderContext(hop, max(0, -span[0]), max(0, span[1] - 1))


@dataclass(frozen=True)
class StreamingAVChunk:
    """New RGB [1,3,F,H,W] and stereo waveform [2,S], with global offsets."""

    video: torch.Tensor | None
    audio: torch.Tensor | None
    video_start: int
    audio_start: int
    fps: int = MINIMAX_H3_SUPPORTED_FPS
    sample_rate: int = 32000
    is_final: bool = False
    audio_generated_stop: int = 0
    audio_committed_stop: int = 0

    @property
    def video_time(self) -> float:
        return self.video_start / self.fps

    @property
    def audio_time(self) -> float:
        return self.audio_start / self.sample_rate


class StreamingAVDecoder:
    """Own independent codec state for one sample of a ragged streaming batch.

    Codecs exposing create_stream supply incremental NTCHW RGB decoding. Other
    video codecs decode the complete latent sequence at flush, preserving their
    temporal attention, padding and overlap rules. Such codecs have final-only
    video output rather than an assumed per-latent frame cadence.

    Audio decodes continuous committed history, the current chunk and retained
    generated right context together, then keeps only the current PCM. The
    right context is preserved across calls and only new tail predictions
    extend it. ``audio_lookahead_latents`` controls the retained left context.
    ``audio_right_lookahead_latents`` controls the right context and defaults
    to the left length when omitted. The default 17-latent context is an
    empirical policy, while audio_context reports the complete graph's
    conservative context bound independently.

    Publication follows the RGB clock. Fractional audio-latent overhang is held
    until the next video boundary, and flush preserves the actual final audio
    length. A final-only video codec also publishes its audio at flush. Latents
    passed to push are normalized. Each instance belongs to one sample.
    """

    def __init__(
        self, video_vae: Any, audio_vae: Any, *, audio_lookahead_latents: int = 17,
        audio_right_lookahead_latents: int | None = None,
    ) -> None:
        if not isinstance(audio_lookahead_latents, int) or isinstance(audio_lookahead_latents, bool) or audio_lookahead_latents < 0:
            raise ValueError("audio_lookahead_latents must be a nonnegative integer")
        if audio_right_lookahead_latents is None:
            audio_right_lookahead_latents = audio_lookahead_latents
        if not isinstance(audio_right_lookahead_latents, int) or isinstance(audio_right_lookahead_latents, bool) or audio_right_lookahead_latents < 0:
            raise ValueError("audio_right_lookahead_latents must be a nonnegative integer")
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        factory = getattr(video_vae, "create_stream", None)
        self.video_stream = factory() if callable(factory) else None
        self._video_stream_device: torch.device | None = None
        self.audio_context = bigvgan_decoder_context(audio_vae)
        self.sample_rate = int(audio_vae.sample_rate)
        if self.sample_rate != 32000 or self.audio_context.samples_per_latent != 800:
            raise ValueError("MiniMax H3 audio requires 32 kHz output and 40 Hz latents")
        self.audio_lookahead_latents = audio_lookahead_latents
        self.audio_right_lookahead_latents = audio_right_lookahead_latents
        self._video_chunks: list[torch.Tensor] = []
        self._audio_history: torch.Tensor | None = None
        self._audio_future: torch.Tensor | None = None
        self._pending_pcm: torch.Tensor | None = None
        self._audio_committed = 0
        self._audio_generated_stop = 0
        self._video_emitted = 0
        self._audio_emitted = 0
        self._closed = False
        self.last_video_decode_frame_counts: list[int] = []

    @property
    def video_is_incremental(self) -> bool:
        return self.video_stream is not None

    @property
    def audio_lookahead_seconds(self) -> float:
        return self.audio_right_lookahead_latents * self.audio_context.samples_per_latent / self.sample_rate

    def _video_denormalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(dtype=self.video_vae.param_dtype).unsqueeze(0)
        return value * self.video_vae.latents_std.to(value).view(1, -1, 1, 1, 1) + self.video_vae.latents_mean.to(value).view(1, -1, 1, 1, 1)

    def _decode_current_audio(self, audio: torch.Tensor, future: torch.Tensor | None) -> None:
        """Decode current PCM from continuous generated latents without revising context."""
        if audio.ndim != 3 or audio.shape[0] != 2:
            raise ValueError("audio input must be normalized stereo [2,C,T] latents")
        if future is None:
            future = audio[:, :, :0]
        if future.ndim != 3 or future.shape[:2] != audio.shape[:2]:
            raise ValueError("audio lookahead must have the committed audio's stereo/channel shape")
        if future.shape[2] > self.audio_right_lookahead_latents:
            raise ValueError("audio lookahead exceeds the configured latent context")
        current_count = int(audio.shape[2])
        self._audio_generated_stop = self._audio_committed + current_count + int(future.shape[2])
        supplied = torch.cat((audio, future), dim=2)
        if self._audio_future is not None:
            overlap = min(self._audio_future.shape[2], supplied.shape[2])
            assert torch.equal(self._audio_future[:, :, :overlap], supplied[:, :, :overlap]), (
                "previously generated audio context must be preserved across steps"
            )
            supplied = torch.cat((self._audio_future[:, :, :overlap], supplied[:, :, overlap:]), dim=2)
        audio, future = supplied[:, :, :current_count], supplied[:, :, current_count:]
        self._audio_future = future.detach().clone()
        if current_count == 0:
            return
        past = audio[:, :, :0] if self._audio_history is None else self._audio_history
        left_count = int(past.shape[2])
        committed = torch.cat((past, audio), dim=2)
        value = torch.cat((committed, future), dim=2).to(dtype=self.audio_vae.param_dtype)
        value = value * self.audio_vae.latents_std.to(value).view(1, -1, 1) + self.audio_vae.latents_mean.to(value).view(1, -1, 1)
        decoded = self.audio_vae.decode(value).squeeze(1)
        hop = self.audio_context.samples_per_latent
        assert decoded.shape == (2, value.shape[-1] * hop)
        core = decoded[:, left_count * hop:(left_count + current_count) * hop].contiguous()
        self._pending_pcm = core if self._pending_pcm is None else torch.cat((self._pending_pcm, core), dim=1)
        self._audio_committed += current_count
        self._audio_history = (
            committed[:, :, -self.audio_lookahead_latents:].detach().clone()
            if self.audio_lookahead_latents else committed.new_empty((*committed.shape[:2], 0))
        )

    def _publish_audio(self, *, final: bool) -> torch.Tensor | None:
        if self._pending_pcm is None:
            return None
        available = self._audio_emitted + int(self._pending_pcm.shape[-1])
        stop = available if final else min(
            available, self._video_emitted * self.sample_rate // MINIMAX_H3_SUPPORTED_FPS
        )
        count = stop - self._audio_emitted
        if count <= 0:
            return None
        output = self._pending_pcm[:, :count].contiguous()
        self._pending_pcm = (
            self._pending_pcm[:, count:].contiguous()
            if count < self._pending_pcm.shape[1] else None
        )
        self._audio_emitted = stop
        return output

    def _event(
        self, video: torch.Tensor | None, audio: torch.Tensor | None,
        video_start: int, audio_start: int, *, is_final: bool = False,
    ) -> StreamingAVChunk:
        return StreamingAVChunk(
            video, audio, video_start, audio_start, sample_rate=self.sample_rate, is_final=is_final,
            audio_generated_stop=self._audio_generated_stop, audio_committed_stop=self._audio_committed,
        )

    @torch.no_grad()
    def push(
        self, video: torch.Tensor | None = None, audio: torch.Tensor | None = None,
        *, audio_lookahead: torch.Tensor | None = None,
    ) -> StreamingAVChunk:
        if self._closed:
            raise RuntimeError("cannot append latents after the decoder was flushed")
        video_start, audio_start = self._video_emitted, self._audio_emitted
        frames = None
        if video is not None:
            if video.ndim != 4:
                raise ValueError("video input must be normalized [C,T,H,W] latents")
            if video.shape[1]:
                if self.video_stream is None:
                    self._video_chunks.append(video.detach().clone())
                else:
                    value = self._video_denormalize(video).transpose(1, 2)
                    self._video_stream_device = value.device
                    dtype = self.video_vae.param_dtype
                    with torch.autocast(value.device.type, dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
                        if getattr(self.video_stream, "requires_single_latent_dispatch", False):
                            decoded_pieces = []
                            self.last_video_decode_frame_counts = []
                            for latent_index in range(int(value.shape[1])):
                                # Exactly one present latent is visible to this
                                # decoder call; no future latent from the same
                                # DiT chunk can participate in the computation.
                                decoded_piece = self.video_stream.decode(
                                    value[:, latent_index:latent_index + 1]
                                )
                                count = 0 if decoded_piece is None else int(decoded_piece.shape[1])
                                self.last_video_decode_frame_counts.append(count)
                                if count:
                                    decoded_pieces.append(decoded_piece)
                            decoded = (
                                torch.cat(decoded_pieces, dim=1)
                                if decoded_pieces else None
                            )
                        else:
                            self.last_video_decode_frame_counts = []
                            decoded = self.video_stream.decode(value)
                    if decoded is not None and decoded.shape[1]:
                        frames = decoded.transpose(1, 2).contiguous()
        if audio is not None:
            self._decode_current_audio(audio, audio_lookahead)
        elif audio_lookahead is not None:
            raise ValueError("audio lookahead requires a committed audio tensor")
        if frames is not None:
            self._video_emitted += int(frames.shape[2])
        waveform = self._publish_audio(final=False)
        return self._event(frames, waveform, video_start, audio_start)

    @torch.no_grad()
    def flush(self) -> StreamingAVChunk:
        video_start, audio_start = self._video_emitted, self._audio_emitted
        if self._closed:
            return self._event(None, None, video_start, audio_start, is_final=True)
        frames = None
        if self.video_stream is not None:
            if self._video_stream_device is None:
                decoded = None
            else:
                dtype = self.video_vae.param_dtype
                with torch.autocast(self._video_stream_device.type, dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
                    decoded = self.video_stream.flush_decoder()
            if decoded is not None and decoded.shape[1]:
                frames = decoded.transpose(1, 2).contiguous()
        elif self._video_chunks:
            value = self._video_denormalize(torch.cat(self._video_chunks, dim=1))
            frames = self.video_vae.processor.revert_tensor(self.video_vae.decode_base(value))
            self._video_chunks.clear()
        if frames is not None:
            self._video_emitted += int(frames.shape[2])
        waveform = self._publish_audio(final=True)
        self._audio_history = None
        self._audio_future = None
        self._closed = True
        return self._event(frames, waveform, video_start, audio_start, is_final=True)


__all__ = [
    "AudioDecoderContext",
    "StreamingAVChunk",
    "StreamingAVDecoder",
    "bigvgan_decoder_context",
]
