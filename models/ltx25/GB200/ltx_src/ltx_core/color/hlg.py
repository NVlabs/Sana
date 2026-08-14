"""In-memory scene-linear HDR frames → BT.2020 / HLG / 10-bit HEVC MP4.
HLG = Hybrid Log-Gamma (ITU-R BT.2100, OETF per ARIB STD-B67). Output is HEVC Main10,
yuv420p10le, tagged BT.2020 primaries + arib-std-b67 transfer + bt2020nc matrix, limited
range.
Per frame: scene-linear RGB → primaries → BT.2020 → map linear light to HLG scene
signal (diffuse white at ``white_signal``, highlights rolled toward 1.0) → HLG OETF →
planar YUV420P10 → PyAV/libx265.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from queue import Queue
from typing import Any

import av
import numpy as np
import torch

from ltx_core.color.audio_mux import prepare_audio_stream, validate_audio_waveform, write_audio
from ltx_core.color.primaries import Primaries
from ltx_core.color.yuv import ColorRange, ColorSpace, rgb_to_yuv420p10
from ltx_core.types import Audio

logger = logging.getLogger(__name__)

# FFmpeg AVColor* tags not covered by ColorSpace / ColorRange.
_AV_COLOR_PRIMARIES_BT2020 = 9
_AV_COLOR_TRC_ARIB_STD_B67 = 18  # HLG

# ARIB STD-B67 (HLG) OETF constants.
HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073


def hlg_inverse_oetf(v: float) -> float:
    """Inverse HLG OETF for a scalar signal value → scene-linear."""
    return (v * v) / 3.0 if v <= 0.5 else (math.exp((v - HLG_C) / HLG_A) + HLG_B) / 12.0


@dataclass
class HlgGpuConverter:
    """GPU linear-HDR ``[F,H,W,3]`` → planar uint16 YUV420P10 (BT.2020 / HLG / limited)."""

    prim_mat: torch.Tensor  # [3, 3] on device
    white_x: float
    roll_k: float

    @classmethod
    def create(
        cls,
        primaries: Primaries = Primaries.REC709,
        white_signal: float = 0.75,
        rolloff_k: float | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> HlgGpuConverter:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        white_x = hlg_inverse_oetf(white_signal)
        roll_k = rolloff_k if rolloff_k is not None else white_x / (1.0 - white_x)
        prim = primaries.matrix_to_rec2020.to(device=device, dtype=dtype)
        return cls(prim_mat=prim, white_x=float(white_x), roll_k=float(roll_k))

    def to_hlg_signal_(self, rgb_linear_chw: torch.Tensor) -> torch.Tensor:
        """``(*, 3, H, W)`` scene-linear → HLG signal ``[0, 1]``."""
        lin = torch.nan_to_num(
            (rgb_linear_chw.movedim(-3, -1) @ self.prim_mat.T).clamp(min=0.0),
            nan=0.0,
            neginf=0.0,
        )
        x = torch.where(
            lin <= 1.0,
            lin * self.white_x,
            1.0 - (1.0 - self.white_x) * torch.exp(-self.roll_k * (lin - 1.0)),
        )
        sig = torch.where(
            x <= 1.0 / 12.0,
            torch.sqrt((3.0 * x).clamp(min=0.0)),
            HLG_A * torch.log((12.0 * x - HLG_B).clamp(min=1e-12)) + HLG_C,
        )
        return sig.clamp(0.0, 1.0).movedim(-1, -3)

    def __call__(self, frames_fhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[F,H,W,3]`` scene-linear → planar uint16 Y, U, V on the same device."""
        if frames_fhwc.ndim != 4 or frames_fhwc.shape[-1] != 3:
            raise ValueError(f"expected [F,H,W,3], got {tuple(frames_fhwc.shape)}")
        x = frames_fhwc.to(device=self.prim_mat.device, dtype=self.prim_mat.dtype)
        hlg = self.to_hlg_signal_(x.movedim(-1, -3).contiguous())
        return rgb_to_yuv420p10(hlg, ColorSpace.BT_2020_NCL, ColorRange.MPEG)


def _resolve_x265_thread_count(thread_count: int) -> int:
    """Clamp libx265 threads; ``0`` (auto) can hang on 100+ CPU hosts."""
    if thread_count > 0:
        return thread_count
    return max(1, min(os.cpu_count() or 8, 16))


@dataclass
class HlgPyAVEncoder:
    """Open PyAV libx265 HLG encoder."""

    container: av.container.OutputContainer
    stream: av.video.stream.VideoStream

    @classmethod
    def open(
        cls,
        out_path: str | Path,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 12,
        preset: str = "ultrafast",
        thread_count: int = 0,
    ) -> HlgPyAVEncoder:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        threads = _resolve_x265_thread_count(thread_count)
        container = av.open(str(out_path), mode="w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx265", rate=Fraction(fps).limit_denominator(1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p10le"
        stream.codec_tag = "hvc1"
        stream.options = {
            "crf": str(crf),
            "preset": preset,
            "x265-params": (
                "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:"
                f"repeat-headers=1:info=0:pools={threads}:frame-threads=4"
            ),
        }
        ctx = stream.codec_context
        ctx.thread_count = threads
        ctx.thread_type = "FRAME"
        ctx.color_primaries = _AV_COLOR_PRIMARIES_BT2020
        ctx.color_trc = _AV_COLOR_TRC_ARIB_STD_B67
        ctx.colorspace = ColorSpace.BT_2020_NCL.av_colorspace
        ctx.color_range = ColorRange.MPEG.av_color_range
        return cls(container=container, stream=stream)

    def encode_yuv420p10(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        """Encode one planar uint16 Y/U/V frame (already BT.2020 HLG limited)."""
        frame = av.VideoFrame(self.stream.width, self.stream.height, "yuv420p10le")
        for plane, src in zip(frame.planes, (y, u, v), strict=True):
            dest = np.frombuffer(plane, dtype=np.uint16).reshape(plane.height, plane.line_size // 2)
            dest[:, : src.shape[1]] = src
        frame.colorspace = ColorSpace.BT_2020_NCL.av_colorspace
        frame.color_range = ColorRange.MPEG.av_color_range
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def flush(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)

    def close(self) -> None:
        self.flush()
        self.container.close()


def _hlg_encode_planes_threaded(
    encoder: HlgPyAVEncoder,
    plane_chunks: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]],
    progress_total: int | None = None,
    *,
    audio: Audio | None = None,
    audio_stream: av.audio.AudioStream | None = None,
) -> int:
    """libx265-encode planar uint16 chunks on a background thread (1-slot queue)."""
    chunk_queue: Queue[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = Queue(maxsize=1)
    encoder_error: list[BaseException] = []
    n_frames = [0]

    def encoder_worker() -> None:
        error: BaseException | None = None
        while True:
            item = chunk_queue.get()
            if item is None:
                break
            if error is not None:
                continue
            try:
                y_b, u_b, v_b = item
                for i in range(y_b.shape[0]):
                    encoder.encode_yuv420p10(y_b[i], u_b[i], v_b[i])
                    n_frames[0] += 1
            except Exception as e:
                error = e
        if error is None:
            try:
                encoder.flush()
                if audio is not None and audio_stream is not None:
                    write_audio(encoder.container, audio_stream, audio)
                encoder.container.close()
            except Exception as e:
                error = e
        else:
            with contextlib.suppress(Exception):
                encoder.container.close()
        if error is not None:
            encoder_error.append(error)

    encoder_thread = threading.Thread(target=encoder_worker, name="hlg-encoder")
    encoder_thread.start()
    try:
        if progress_total is not None:
            from tqdm import tqdm  # noqa: PLC0415

            plane_chunks = tqdm(plane_chunks, total=progress_total, desc="HLG encode")
        for item in plane_chunks:
            chunk_queue.put(item)
    finally:
        chunk_queue.put(None)
        encoder_thread.join()

    if encoder_error:
        raise encoder_error[0]
    return n_frames[0]


def _as_fhwc_torch(frame: torch.Tensor | np.ndarray) -> torch.Tensor:
    t = frame.detach() if isinstance(frame, torch.Tensor) else torch.as_tensor(np.asarray(frame))
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.shape[-1] != 3 and t.shape[1] == 3:
        t = t.movedim(1, -1)
    return t


def encode_linear_hdr_frames_to_hlg_mp4(  # noqa: PLR0913
    frames: Iterable[Any],
    out_path: str | Path,
    fps: float = 24.0,
    primaries: Primaries = Primaries.REC709,
    white_signal: float = 0.75,
    rolloff_k: float | None = None,
    crf: int = 12,
    preset: str = "ultrafast",
    progress: bool = False,
    progress_total: int | None = None,
    thread_count: int = 0,
    audio: Audio | None = None,
    device: torch.device | None = None,
) -> int:
    """Encode scene-linear HDR frames to BT.2020/HLG/10-bit HEVC mp4 via PyAV.
    Consumes ``HxWx3`` / ``FxHxWx3`` chunks one at a time. Conversion runs on
    ``device``; the encoder thread only muxes planar uint16 into libx265.
    """
    converter = HlgGpuConverter.create(primaries, white_signal, rolloff_k, device=device)

    it = iter(frames)
    try:
        first = next(it)
    except StopIteration as e:
        raise ValueError("No HDR frames to encode.") from e

    first_t = _as_fhwc_torch(first)
    # ``first_t`` is [F,H,W,C]; open() takes (width, height).
    height, width = int(first_t.shape[1]), int(first_t.shape[2])
    encoder = HlgPyAVEncoder.open(
        out_path,
        width,
        height,
        fps,
        crf=crf,
        preset=preset,
        thread_count=thread_count,
    )

    audio_stream: av.audio.AudioStream | None = None
    if audio is not None:
        validate_audio_waveform(audio)
        audio_stream = prepare_audio_stream(encoder.container, audio.sampling_rate)

    def plane_chunks() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        y, u, v = converter(first_t)
        yield y.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy()
        for chunk in it:
            y, u, v = converter(_as_fhwc_torch(chunk))
            yield y.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy()

    try:
        n = _hlg_encode_planes_threaded(
            encoder,
            plane_chunks(),
            progress_total=progress_total if progress or progress_total is not None else None,
            audio=audio,
            audio_stream=audio_stream,
        )
    except Exception:
        Path(out_path).unlink(missing_ok=True)
        raise
    if progress:
        logger.info("  %d frames", n)
    return n
