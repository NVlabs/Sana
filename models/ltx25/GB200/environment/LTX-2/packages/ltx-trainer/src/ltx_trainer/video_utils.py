"""Video I/O utilities using PyAV.
This module provides functions for reading and writing video files using PyAV,
with optional audio support.
"""

from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
from torch import Tensor

VideoFormat = Literal["CFHW", "FCHW"]


def get_video_frame_count(video_path: str | Path) -> int:
    """Get the number of frames in a video file.
    Tries three approaches in order: stream metadata, duration*fps estimate,
    full decode. The estimate may be off by a few frames for VFR videos or
    containers with edit lists — exact for the min_frames filtering use case.
    Args:
        video_path: Path to the video file
    Returns:
        Number of frames in the video
    """
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]

        if video_stream.frames > 0:
            return video_stream.frames

        # Fast estimate from container metadata (avoids full decode).
        # Uses Fraction arithmetic to prevent float precision loss.
        rate = video_stream.average_rate or video_stream.base_rate
        if video_stream.duration and video_stream.time_base and rate:
            duration = Fraction(video_stream.duration) * Fraction(video_stream.time_base)
            return round(duration * Fraction(rate))

        # Last resort: full decode (very slow for 4K)
        return sum(1 for _ in container.decode(video=0))


def read_video(video_path: str | Path, max_frames: int | None = None) -> tuple[Tensor, float]:
    """Load frames from a video file using PyAV.
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to read. If None, reads all frames.
    Returns:
        Video tensor with shape [F, C, H, W] in range [0, 1] and frames per second (fps).
    """
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)

        frames = []
        for frame in container.decode(video=0):
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    frames_np = np.stack(frames, axis=0)  # [F, H, W, C]
    video = torch.from_numpy(frames_np).float().div(255.0)  # [F, H, W, C] in [0, 1]
    return video.permute(0, 3, 1, 2), fps  # [F, C, H, W]


def save_video(
    video_tensor: torch.Tensor,
    output_path: Path | str,
    fps: float = 24.0,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    video_format: VideoFormat | None = None,
) -> None:
    """Save a video tensor to a file using PyAV, optionally with audio.
    Args:
        video_tensor: Video tensor of shape [C, F, H, W] or [F, C, H, W] in range [0, 1] or [0, 255]
        output_path: Path to save the video
        fps: Frames per second for the output video
        audio: Optional audio tensor of shape [C, samples] or [samples, C] in range [-1, 1]
        audio_sample_rate: Sample rate for the audio (required if audio is provided)
        video_format: Explicit layout of ``video_tensor``, either ``"CFHW"`` or ``"FCHW"``.
            When ``None`` (default), the layout is auto-detected using a heuristic that only
            works when ``shape[1] > 3`` — the ambiguous ``[C=3, F=3, H, W]`` / ``[F=3, C=3, H, W]``
            case requires passing this argument explicitly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [F, H, W, C] uint8 numpy array
    video_np = _prepare_video_array(video_tensor, video_format=video_format)
    _, height, width, _ = video_np.shape

    with av.open(str(output_path), mode="w") as container:
        # Setup video stream
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "18"}

        # Setup audio stream if needed
        if audio is not None:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate must be provided when audio is given")
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "stereo"
            audio_stream.time_base = Fraction(1, audio_sample_rate)

        # Write video frames
        for frame_array in video_np:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        # Write audio if provided
        if audio is not None:
            _write_audio(container, audio_stream, audio, audio_sample_rate)


def _prepare_video_array(
    video_tensor: torch.Tensor,
    video_format: VideoFormat | None = None,
) -> np.ndarray:
    """Convert video tensor to [F, H, W, C] uint8 numpy array.
    If ``video_format`` is provided, it is trusted. Otherwise, the layout is auto-detected
    using a heuristic that only fires when ``shape[0] == 3 and shape[1] > 3`` (CFHW). The
    ambiguous ``[C=3, F=3, H, W]`` / ``[F=3, C=3, H, W]`` case cannot be disambiguated and
    defaults to the FCHW interpretation — callers must pass ``video_format`` explicitly for
    3-frame CFHW tensors.
    """
    if video_format == "CFHW":
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]
    elif video_format is None and video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Normalize to [0, 255] uint8
    if video_tensor.max() <= 1.0:
        video_tensor = video_tensor * 255

    # [F, C, H, W] -> [F, H, W, C]
    return video_tensor.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


def _write_audio(
    container: av.container.Container,
    audio_stream: av.audio.AudioStream,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Write audio tensor to container as stereo AAC."""
    audio = audio.cpu().float()

    # Normalize to [samples, 2] stereo format
    if audio.ndim == 1:
        audio = audio.unsqueeze(1).repeat(1, 2)  # Mono -> stereo
    elif audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T  # [2, samples] -> [samples, 2]
    if audio.shape[1] == 1:
        audio = audio.repeat(1, 2)  # Mono -> stereo

    # Convert to int16 interleaved: [samples, 2] -> [1, samples*2]
    audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16)
    audio_interleaved = audio_int16.contiguous().view(1, -1).numpy()

    # Create audio frame
    frame = av.AudioFrame.from_ndarray(audio_interleaved, format="s16", layout="stereo")
    frame.sample_rate = sample_rate

    # Resample to encoder format and write
    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.codec_context.format,
        layout=audio_stream.codec_context.layout,
        rate=sample_rate,
    )

    pts = 0
    for resampled_frame in resampler.resample(frame):
        resampled_frame.pts = pts
        pts += resampled_frame.samples
        for packet in audio_stream.encode(resampled_frame):
            container.mux(packet)

    for packet in audio_stream.encode():
        container.mux(packet)
