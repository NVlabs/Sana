"""Media helpers for writing local visual artifacts."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path

import torch
import torchvision

# Cannot use get_logger here: logging.py imports this module; the root logger matches the framework format.
logger = logging.getLogger()


def _video_tensor_to_uint8_frames(
    tensor: torch.Tensor,
    *,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] = (-1, 1),
):
    """Convert a ``[B, C, T, H, W]`` tensor into ``[T, H, W, C]`` uint8 frames."""
    tensor = tensor.clamp(min=value_range[0], max=value_range[1])
    tensor = torch.stack(
        [
            torchvision.utils.make_grid(
                frame,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range,
            )
            for frame in tensor.unbind(2)
        ],
        dim=1,
    ).permute(1, 2, 3, 0)
    return (tensor * 255).type(torch.uint8).cpu().numpy()


def save_image(
    tensor: torch.Tensor,
    path: str | Path,
    *,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] = (-1, 1),
) -> None:
    """Write an image tensor to a file (format from the path suffix, e.g. ``.png``).

    Args:
        tensor: Image tensor with shape ``[B, C, H, W]`` and values expected in
            ``[-1, 1]`` by default. Batch items are arranged with
            ``torchvision.utils.make_grid`` using ``nrow`` -- the same grid and
            normalization semantics as :func:`save_video` frames. Single images
            must be unsqueezed to ``B=1`` by the caller.
        path: Local output path.
        nrow: Number of batch items per grid row.
        normalize: Passed to ``make_grid``.
        value_range: Clamp and normalization range; always passed explicitly so
            brightness stays comparable across calls (never per-tensor min-max).
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected an image tensor [B, C, H, W], got shape {tuple(tensor.shape)}")
    tensor = tensor.clamp(min=value_range[0], max=value_range[1])
    torchvision.utils.save_image(
        tensor,
        str(path),
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
    )


def save_video(
    tensor: torch.Tensor,
    path: str | Path,
    *,
    fps: int = 16,
    nrow: int = 8,
    crf: int = 18,
    audio: str | Path | None = None,
    audio_tensor: torch.Tensor | None = None,
    audio_sample_rate: int = 16000,
    normalize: bool = True,
    value_range: tuple[float, float] = (-1, 1),
) -> None:
    """Write a video tensor to an mp4 file with ffmpeg.

    Args:
        tensor: Video tensor with shape ``[B, C, T, H, W]`` and values expected
            in ``[-1, 1]`` by default. Frames are arranged with
            ``torchvision.utils.make_grid`` using ``nrow``.
        path: Local output path.
        fps: Output frame rate.
        nrow: Number of batch items per grid row.
        crf: x264 quality setting passed through to ffmpeg.
        audio: Optional path to an audio file to mux into the output.
        audio_tensor: Optional audio tensor with shape ``[T]`` or ``[C, T]``.
        audio_sample_rate: Sample rate for ``audio_tensor``.
        normalize: Passed to ``make_grid``.
        value_range: Clamp and normalization range passed to ``make_grid``.
    """
    if audio is not None and audio_tensor is not None:
        raise ValueError("Only one of audio and audio_tensor can be provided.")

    path = str(path)
    frames = _video_tensor_to_uint8_frames(
        tensor,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
    )
    frame_count, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames with 3 channels, got {channels}")
    del frame_count

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]

    audio_bytes = None
    audio_read_fd = None
    audio_write_fd = None
    pass_fds = ()
    if audio_tensor is not None:
        audio_tensor = audio_tensor.detach().cpu().float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim != 2:
            raise ValueError(f"audio_tensor should have shape [T] or [C, T], got {tuple(audio_tensor.shape)}")

        if audio_tensor.shape[0] > audio_tensor.shape[1] and audio_tensor.shape[1] <= 8:
            audio_tensor = audio_tensor.transpose(0, 1)

        # Raw f32le PCM straight into the ffmpeg BINARY -- the same
        # self-contained conda binary that already encodes the video leg, so
        # the audio leg runs on any node too. torchcodec built a WAV here
        # before, but it dlopens FFmpeg's shared libs from the process library
        # path, which only some node images can satisfy (15509560: all 16
        # group mains crashed on that import and the other 48 ranks gloo-timed
        # out 30 minutes later). ffmpeg reads the sample layout from the flags
        # instead, so no container is needed; validated on a zero-system-ffmpeg
        # node by tests/toy_ffmpeg_rawpcm.sbatch (job 15535781).
        audio_bytes = audio_tensor.transpose(0, 1).contiguous().numpy().tobytes()
        audio_read_fd, audio_write_fd = os.pipe()
        pass_fds = (audio_read_fd,)
        cmd += [
            "-f", "f32le",
            "-ar", str(audio_sample_rate),
            "-ac", str(audio_tensor.shape[0]),
            "-i", f"pipe:{audio_read_fd}",
            "-map", "0:v", "-map", "1:a", "-af", "apad", "-shortest",
        ]
    elif audio:
        cmd += [
            "-i", str(audio), "-map", "0:v", "-map", "1:a",
            "-af", "apad", "-shortest",
        ]

    cmd += [
        "-vcodec",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-acodec",
        "aac",
        path,
    ]

    audio_writer = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
        )
        if audio_read_fd is not None:
            os.close(audio_read_fd)
            audio_read_fd = None
        if audio_write_fd is not None:

            def _write_audio():
                try:
                    with os.fdopen(audio_write_fd, "wb") as f:
                        f.write(audio_bytes)
                except BrokenPipeError:
                    logger.warning("FFmpeg audio pipe closed early when saving %s.", path)

            audio_writer = threading.Thread(target=_write_audio)
            audio_writer.start()
            audio_write_fd = None
        _, stderr = proc.communicate(input=frames.tobytes())
        if audio_writer is not None:
            audio_writer.join()
        if proc.returncode != 0:
            logger.warning("FFmpeg failed when saving %s:\n%s", path, stderr.decode())
    finally:
        if audio_read_fd is not None:
            os.close(audio_read_fd)
        if audio_write_fd is not None:
            os.close(audio_write_fd)
