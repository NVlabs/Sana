"""Shared AAC mux helpers for SDR and HLG PyAV encoders."""

from __future__ import annotations

from fractions import Fraction

import av
import torch

from ltx_core.types import Audio


def validate_audio_waveform(audio: Audio) -> None:
    """Raise ValueError if the waveform is empty or not stereo ``(2, N)`` / ``(N, 2)``."""
    samples = audio.waveform
    if samples.numel() == 0:
        raise ValueError("audio.waveform is empty; pass audio=None for no audio.")
    if samples.ndim != 2 or 2 not in samples.shape:
        raise ValueError(f"audio.waveform must be stereo (2, N) or (N, 2); got shape {tuple(samples.shape)}.")


def normalize_audio_waveform(samples: torch.Tensor) -> torch.Tensor:
    """Transpose a validated stereo waveform to channel-last ``(N, 2)``."""
    return samples.T if samples.shape[1] != 2 else samples


def prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """Add a stereo AAC stream to ``container``."""
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    """Resample ``frame_in`` to the encoder format and mux packets."""
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def write_audio(container: av.container.Container, audio_stream: av.audio.AudioStream, audio: Audio) -> None:
    """Mux ``audio`` onto an already-open AAC stream."""
    samples = normalize_audio_waveform(audio.waveform)

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio.sampling_rate

    resample_audio(container, audio_stream, frame_in)
