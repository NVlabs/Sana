"""Unchanged pinned SDR encoding functions; imports exclude unrelated HDR decoding.
Source: LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/encode.py.
Only the SDR color_space=None call is used by this experiment.
"""
from __future__ import annotations
import logging
import threading
from collections.abc import Generator, Iterator
from pathlib import Path
from queue import Queue
import av
import numpy as np
import torch
from tqdm import tqdm
from ltx_core.color.audio_mux import prepare_audio_stream as _prepare_audio_stream
from ltx_core.color.audio_mux import validate_audio_waveform as _validate_audio_waveform
from ltx_core.color.audio_mux import write_audio as _write_audio
from ltx_core.color.yuv import FrameConverter, PixelFormat, yuv420p_bt709_converter_
from ltx_core.types import Audio
logger = logging.getLogger(__name__)

def encode_video(
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: Audio | None,
    output_path: str,
    video_chunks_number: int,
    frame_converter: FrameConverter = yuv420p_bt709_converter_,
    crf: int = 19,
    preset: str = "veryfast",
    thread_count: int = 0,
    *,
    color_space: HDRColorSpace | None = None,
) -> Path | None:
    """Encode RGB frames to video; SDR H.264 or HDR EXR+HLG when ``color_space`` is set.
    Args:
        video: RGB frames as a ``(F, H, W, C)`` float ``[0, 1]`` tensor, or an iterator of
            such per-chunk tensors (e.g. the VAE decoder output). An empty iterator raises.
            For HDR this is the compressed working-space signal from VAE decode.
        fps: Output frame rate.
        audio: Audio track to mux, or None for a video-only file. Waveform must be stereo
            ``(2, N)`` or ``(N, 2)``.
        output_path: Destination path. Parent directories are created if missing.
            Partial SDR output is removed if encoding fails. HDR writes EXR under
            ``<stem>_exr/`` and an HLG master to this path.
        video_chunks_number: Number of chunks yielded by ``video``, for the progress bar
            (SDR path only).
        frame_converter: Float-to-pixel converter (default YUV420p BT.709); SDR only.
        crf: libx264 constant rate factor; lower is higher quality (0-51); SDR only.
        preset: libx264 speed/compression preset; SDR only.
        thread_count: libx264 thread count (0 = auto); SDR only.
        color_space: HDR colour space for the encode sink. ``None`` (default) is SDR.
            When set, writes half EXR frames + HLG master.
    Returns:
        EXR output directory when HDR, else ``None``.
    Raises:
        ValueError: On an empty ``video`` or a non-stereo ``audio.waveform``.
    """
    if audio is not None:
        _validate_audio_waveform(audio)

    if isinstance(video, torch.Tensor):
        video = iter([video])

    if color_space is not None:
        return _encode_hdr_video_outputs(
            video,
            Path(output_path),
            fps=float(fps),
            color_space=color_space,
            audio=audio,
        )

    def convert(chunk: torch.Tensor) -> torch.Tensor:
        return frame_converter(chunk.movedim(-1, -3))

    first_raw_chunk = next(video, None)
    if first_raw_chunk is None:
        raise ValueError("video is empty; expected at least one frame chunk.")
    first_chunk = convert(first_raw_chunk)

    if frame_converter.pixel_format == PixelFormat.RGB24:
        height, width = first_chunk.shape[-3], first_chunk.shape[-2]
    else:
        height = first_chunk.shape[-2] * 2 // 3
        width = first_chunk.shape[-1]

    # PyAV/FFmpeg won't create missing parent directories; do it so a nested
    # --output-path (e.g. /tmp/run/clip.mp4) doesn't fail at mux time.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(output_path, mode="w")
    success = False
    try:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(crf), "preset": preset})
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = thread_count
        stream.codec_context.thread_type = "FRAME"
        if frame_converter.color_space is not None:
            stream.codec_context.colorspace = frame_converter.color_space.av_colorspace
        if frame_converter.color_range is not None:
            stream.codec_context.color_range = frame_converter.color_range.av_color_range

        if audio is not None:
            audio_stream = _prepare_audio_stream(container, audio.sampling_rate)

        av_format = frame_converter.pixel_format.av_format

        def cpu_chunks() -> Generator[np.ndarray, None, None]:
            yield first_chunk.to("cpu").numpy()
            for chunk in video:
                yield convert(chunk).to("cpu").numpy()

        _encode_chunks_threaded(
            container=container,
            stream=stream,
            av_format=av_format,
            chunks=cpu_chunks(),
            progress_total=video_chunks_number,
        )

        if audio is not None:
            _write_audio(container, audio_stream, audio)
        success = True
    finally:
        container.close()
        if not success:
            Path(output_path).unlink(missing_ok=True)
    logger.info(f"Video saved to {output_path}")
    return None

def _encode_chunks_threaded(
    container: av.container.Container,
    stream: av.video.stream.VideoStream,
    av_format: str,
    chunks: Iterator[np.ndarray],
    progress_total: int,
) -> None:
    """Run libx264 frame.encode + container.mux on a background thread while
    the caller produces numpy chunks on the current thread. The 1-slot queue
    lets the producer get one chunk ahead (so the next VAE/gather chunk
    overlaps with libx264 encoding the previous chunk) without buffering more
    than one chunk in CPU memory.
    """
    chunk_queue: Queue[np.ndarray | None] = Queue(maxsize=1)
    encoder_error: list[BaseException] = []

    def encoder_worker() -> None:
        error: BaseException | None = None
        while True:
            arr = chunk_queue.get()
            if arr is None:
                break
            if error is not None:
                continue
            try:
                for frame_array in arr:
                    frame = av.VideoFrame.from_ndarray(frame_array, format=av_format)
                    for packet in stream.encode(frame):
                        container.mux(packet)
            except Exception as e:
                error = e
        if error is None:
            try:
                for packet in stream.encode():
                    container.mux(packet)
            except Exception as e:
                error = e
        if error is not None:
            encoder_error.append(error)

    encoder_thread = threading.Thread(target=encoder_worker, name="h264-encoder")
    encoder_thread.start()
    try:
        for arr in tqdm(chunks, total=progress_total):
            chunk_queue.put(arr)
    finally:
        chunk_queue.put(None)
        encoder_thread.join()

    if encoder_error:
        raise encoder_error[0]
