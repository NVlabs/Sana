# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Progressive MP4 writer for the chunk-pipelined Sana-WM streaming inference.

Spawns a single ``ffmpeg`` subprocess that consumes raw RGB frames via stdin
and emits a growing H.264 MP4 on disk. Designed for the streaming pipeline
where the orchestrator pushes one (T, H, W, 3) uint8 chunk per AR block as
decoding completes.

The writer keeps no buffering of its own — frames are forwarded byte-for-byte
to ffmpeg. Concurrency model: the orchestrator drives ``write_chunk`` from the
host thread after a CUDA-stream ``synchronize`` ensures the decoded pixels
have been copied to CPU. ``close`` blocks until ffmpeg has flushed and
finalized the MOOV atom so the resulting file is immediately playable.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import IO

import numpy as np


def resolve_ffmpeg_exe() -> str:
    """Return an ffmpeg executable path, falling back to imageio-ffmpeg."""
    env_path = os.environ.get("SANA_WM_FFMPEG_BIN", "").strip()
    if env_path:
        return env_path
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH and imageio_ffmpeg is unavailable. "
            "Install ffmpeg or set SANA_WM_FFMPEG_BIN."
        ) from exc


class StreamingMp4Writer:
    """Pipe raw RGB chunks to a single ffmpeg process; produce H.264 MP4.

    Args:
        output_path: Destination ``.mp4`` path. Parent directory is created.
        height: Pixel height of every frame.
        width: Pixel width of every frame.
        fps: Output frame rate.
        crf: ffmpeg constant rate factor (lower = higher quality; 18 is
            near-lossless visually).
        preset: ffmpeg libx264 preset (``slow``/``medium``/``fast``/...).
        extra_args: Optional list of extra ffmpeg CLI arguments inserted
            between ``-i pipe:0`` and the encoder flags.
        loglevel: ffmpeg log level (default ``warning``).
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        height: int,
        width: int,
        fps: int = 16,
        crf: int = 18,
        preset: str = "medium",
        encoder: str = "libx264",
        extra_args: list[str] | None = None,
        loglevel: str = "warning",
    ) -> None:
        self._path = Path(output_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._H = int(height)
        self._W = int(width)
        self._fps = int(fps)
        self._closed = False
        self._frames_written = 0

        encoder = str(encoder).strip().lower()
        if encoder in {"x264", "cpu", "libx264"}:
            codec_args = [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(int(crf)),
            ]
        elif encoder in {"nvenc", "h264_nvenc"}:
            # Use NVIDIA's hardware H.264 encoder. The raw RGB upload and
            # yuv420 conversion remain part of the ffmpeg process, but entropy
            # coding no longer competes with the Python process on CPU cores.
            codec_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-cq",
                str(int(crf)),
                "-b:v",
                "0",
            ]
        else:
            raise ValueError(f"Unsupported streaming MP4 encoder: {encoder!r}.")

        cmd: list[str] = [
            resolve_ffmpeg_exe(),
            "-y",
            "-loglevel",
            loglevel,
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._W}x{self._H}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            *(extra_args or []),
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-r",
            str(self._fps),
            str(self._path),
        ]
        self._cmd = cmd
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    @property
    def output_path(self) -> Path:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def ffmpeg_command(self) -> str:
        return shlex.join(self._cmd)

    def write_chunk(self, frames_uint8: np.ndarray) -> None:
        """Append a chunk of frames.

        Args:
            frames_uint8: ``(T, H, W, 3)`` uint8 RGB tensor with ``H == height``
                and ``W == width`` as configured at construction time.
        """
        if self._closed:
            raise RuntimeError("write_chunk called after close().")
        if frames_uint8.dtype != np.uint8:
            raise ValueError(f"frames must be uint8; got dtype {frames_uint8.dtype}.")
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3:
            raise ValueError(f"frames must have shape (T,H,W,3); got {frames_uint8.shape}.")
        if frames_uint8.shape[1] != self._H or frames_uint8.shape[2] != self._W:
            raise ValueError(f"frame H,W = {frames_uint8.shape[1:3]} but writer expects {(self._H, self._W)}.")
        if not frames_uint8.flags["C_CONTIGUOUS"]:
            frames_uint8 = np.ascontiguousarray(frames_uint8)

        stdin: IO[bytes] | None = self._proc.stdin
        if stdin is None:
            raise RuntimeError("ffmpeg stdin is None; subprocess failed to start.")
        try:
            stdin.write(frames_uint8.tobytes())
        except BrokenPipeError as exc:
            stderr_blob = b""
            if self._proc.stderr is not None:
                stderr_blob = self._proc.stderr.read() or b""
            raise RuntimeError(
                f"ffmpeg stdin BrokenPipeError; ffmpeg likely exited.\n"
                f"command: {self.ffmpeg_command}\n"
                f"stderr:\n{stderr_blob.decode(errors='replace')}"
            ) from exc
        self._frames_written += int(frames_uint8.shape[0])

    def close(self) -> Path:
        """Flush stdin, wait for ffmpeg to finalize, and return the output path.

        Idempotent — calling twice is a no-op on the second call. Raises
        ``RuntimeError`` if ffmpeg returned a non-zero exit code, including
        the captured stderr for diagnostics.
        """
        if self._closed:
            return self._path
        self._closed = True
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr_blob = b""
        if self._proc.stderr is not None:
            stderr_blob = self._proc.stderr.read() or b""
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {rc}\n"
                f"command: {self.ffmpeg_command}\n"
                f"stderr:\n{stderr_blob.decode(errors='replace')}"
            )
        return self._path

    def __enter__(self) -> StreamingMp4Writer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.close()
            else:
                try:
                    if self._proc.stdin is not None:
                        self._proc.stdin.close()
                except Exception:
                    pass
                self._proc.terminate()
        finally:
            pass
