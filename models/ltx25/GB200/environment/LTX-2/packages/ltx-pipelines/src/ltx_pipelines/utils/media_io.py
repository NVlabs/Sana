import enum
import logging
import math
import threading
from collections.abc import Generator, Iterator
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from queue import Queue

import av
import numpy as np
import OpenImageIO
import torch
from einops import rearrange
from PIL import ExifTags, Image, ImageCms
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

from ltx_core.hdr import LogC3
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.color_conversion import FrameConverter, PixelFormat, yuv420p_bt709_converter_
from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF

logger = logging.getLogger(__name__)
_ORIENTATION_EXIF_KEY = next(key for key, value in ExifTags.TAGS.items() if value == "Orientation")
_ORIENTATION_TO_ROTATION = {3: 180, 6: 270, 8: 90}
_SRGB_PROFILE = ImageCms.createProfile("sRGB")


class ResizeMode(enum.Enum):
    """How to fit a conditioning video to the target resolution."""

    CENTER_CROP = "center_crop"
    REFLECT_PAD = "reflect_pad"


def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image tensor with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Tensor with shape (F (optional), H, W, C) F = 1 if input is 3D, otherwise input shape[0]
    """
    height, width = image.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    # rearrange and remove batch dimension
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    # preserve input dimensions
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Resize tensor preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        latent: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D input or (1, C, F, height, width) for 4D input
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape

    scale = max(height / src_h, width / src_w)
    # Use ceil to avoid floating-point rounding causing new_h/new_w to be
    # slightly smaller than target, which would result in negative crop offsets.
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def normalize_images(images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (images / 127.5 - 1.0).to(device=device, dtype=dtype)


def to_vae_range(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] to [-1, 1] (VAE input convention)."""
    return torch.clamp(x, 0.0, 1.0) * 2.0 - 1.0


def from_vae_range(z: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] (VAE output convention) to [0, 1]."""
    return torch.clamp((z + 1.0) / 2.0, 0.0, 1.0)


def load_image_and_preprocess(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    crf: int = DEFAULT_IMAGE_CRF,
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    image = decode_image(image_path=image_path)
    image = preprocess(image=image, crf=crf)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    image = normalize_images(image, device, dtype)
    return image


def video_preprocess(
    frames: Generator[torch.Tensor],
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Preprocesses a video frame generator for conditioning.
    Args:
        frames: Generator of video frames as tensors of shape (1, H, W, C), dtype uint8.
        height: Target height in pixels.
        width: Target width in pixels.
        dtype: Target dtype for the output tensor.
        device: Target device for the output tensor.
    Returns:
        Tensor of shape (1, C, F, height, width) with values in [-1, 1].
    """
    result: torch.Tensor | None = None
    for f in frames:
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_images(frame, device, dtype)
        result = frame if result is None else torch.cat([result, frame], dim=2)
    if result is None:
        raise ValueError("video_preprocess received an empty frame generator; no frames were decoded from the source.")
    return result


def align_resolution(
    width: int,
    height: int,
    resize_mode: ResizeMode,
    divisor: int = 64,
) -> tuple[int, int, int, int]:
    """Compute aligned generation dimensions and crop-back size.
    Args:
        width: Source video width (need not be aligned).
        height: Source video height (need not be aligned).
        resize_mode: CENTER_CROP rounds down; REFLECT_PAD rounds up.
        divisor: Alignment divisor (default 64 for two-stage pipelines).
    Returns:
        ``(gen_width, gen_height, crop_width, crop_height)`` where
        ``gen_*`` are multiples of *divisor* and ``crop_*`` are the
        original dimensions to trim back to after decoding.  When no
        cropping is needed ``crop_*`` equals ``gen_*``.
    """
    if resize_mode is ResizeMode.REFLECT_PAD:
        gen_w = ((width + divisor - 1) // divisor) * divisor
        gen_h = ((height + divisor - 1) // divisor) * divisor
    else:
        gen_w = (width // divisor) * divisor
        gen_h = (height // divisor) * divisor

    crop_w = width if gen_w != width else gen_w
    crop_h = height if gen_h != height else gen_h
    return gen_w, gen_h, crop_w, crop_h


def resize_and_reflect_pad(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize tensor to fit within target, then reflect-pad to exact dimensions.
    Unlike resize_and_center_crop which stretches and crops, this preserves the
    original aspect ratio and pads the shorter dimension with reflected pixels.
    When the target is already >= the source in both dimensions, interpolation
    is skipped entirely to preserve original pixels.
    Args:
        tensor: Input with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D or (1, C, F, height, width) for 4D
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape

    if height >= src_h and width >= src_w:
        new_h, new_w = src_h, src_w
    else:
        scale = min(height / src_h, width / src_w)
        new_h = round(src_h * scale)
        new_w = round(src_w * scale)
        tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_bottom = height - new_h
    pad_right = width - new_w
    if pad_bottom > 0 or pad_right > 0:
        pad_mode = "reflect" if pad_bottom < new_h and pad_right < new_w else "replicate"
        tensor = torch.nn.functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode=pad_mode)

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def load_video_conditioning_hdr(
    video_path: str,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
    hdr_transform: str = "logc3",
    resize_mode: ResizeMode = ResizeMode.CENTER_CROP,
) -> Iterator[torch.Tensor]:
    """Load a video and yield preprocessed frames for HDR IC-LoRA conditioning.
    Decodes through the standard path and applies the LDR compression that
    matches training. Callers are responsible for providing Rec.709 SDR
    input — the HDR IC-LoRA was trained on that color space.
    Args:
        hdr_transform: LDR-compression name (currently only ``logc3``).
        resize_mode: How to fit the video to the target resolution.
    Yields:
        Per-frame tensors of shape ``(1, C, 1, height, width)``.
    """
    if hdr_transform != "logc3":
        raise ValueError(f"Unsupported HDR transform: {hdr_transform}")

    resize_fn = resize_and_reflect_pad if resize_mode is ResizeMode.REFLECT_PAD else resize_and_center_crop

    for f in decode_video_by_frame(path=video_path, frame_cap=frame_cap, device=device):
        frame = resize_fn(f.to(torch.float32), height, width)
        ldr = (frame / 255.0).clamp(0.0, 1.0)
        compressed = LogC3().compress_ldr(ldr)
        yield to_vae_range(compressed).to(device=device, dtype=dtype)


def decode_image(image_path: str) -> np.ndarray:
    """Load an image as an oriented, sRGB, three-channel uint8 RGB array."""
    with Image.open(image_path) as source_image:
        image = source_image
        orientation = image.getexif().get(_ORIENTATION_EXIF_KEY)
        if orientation in _ORIENTATION_TO_ROTATION:
            image = image.rotate(_ORIENTATION_TO_ROTATION[orientation], expand=True)

        icc_profile = image.info.get("icc_profile")

        if image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode == "LA":
            image = image.convert("L")

        if icc_profile:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                destination_profile = ImageCms.ImageCmsProfile(_SRGB_PROFILE)
                image = ImageCms.profileToProfile(
                    image,
                    source_profile,
                    destination_profile,
                    outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError, ValueError) as error:
                logger.warning("Failed to convert image to sRGB: %s", error)
                image = image.convert("RGB")
        else:
            image = image.convert("RGB")

        return np.array(image, dtype=np.uint8)


def _validate_audio_waveform(audio: Audio) -> None:
    """Raise ValueError if the waveform is empty or not stereo ``(2, N)`` / ``(N, 2)``."""
    samples = audio.waveform
    if samples.numel() == 0:
        raise ValueError("audio.waveform is empty; pass audio=None for no audio.")
    if samples.ndim != 2 or 2 not in samples.shape:
        raise ValueError(f"audio.waveform must be stereo (2, N) or (N, 2); got shape {tuple(samples.shape)}.")


def _normalize_audio_waveform(samples: torch.Tensor) -> torch.Tensor:
    """Transpose a validated stereo waveform to channel-last ``(N, 2)``."""
    return samples.T if samples.shape[1] != 2 else samples


def _write_audio(container: av.container.Container, audio_stream: av.audio.AudioStream, audio: Audio) -> None:
    samples = _normalize_audio_waveform(audio.waveform)

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

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
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
) -> None:
    """Encode RGB frames to an H.264 file, optionally muxing an audio track.
    Args:
        video: RGB frames as a ``(F, H, W, C)`` float ``[0, 1]`` tensor, or an iterator of
            such per-chunk tensors (e.g. the VAE decoder output). An empty iterator raises.
        fps: Output frame rate.
        audio: Audio track to mux, or None for a video-only file. Waveform must be stereo
            ``(2, N)`` or ``(N, 2)``.
        output_path: Destination path. Parent directories are created if missing.
            Partial output is removed if encoding fails.
        video_chunks_number: Number of chunks yielded by ``video``, for the progress bar.
        frame_converter: Float-to-pixel converter (default YUV420p BT.709).
        crf: libx264 constant rate factor; lower is higher quality (0-51).
        preset: libx264 speed/compression preset.
        thread_count: libx264 thread count (0 = auto).
    Raises:
        ValueError: On an empty ``video`` or a non-stereo ``audio.waveform``.
    """
    if audio is not None:
        _validate_audio_waveform(audio)

    if isinstance(video, torch.Tensor):
        video = iter([video])
    elif not hasattr(video, "__next__"):
        # the distributed VAE decoder returns a list of chunks: an iterable, but
        # not an iterator, and next() rejects it
        video = iter(video)

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


def encode_audio(audio: Audio, output_path: str) -> None:
    """Save an audio waveform as a 16-bit PCM ``.wav`` file at the source sampling rate.
    Reuses :func:`_write_audio` (the same muxing path used by :func:`encode_video`);
    the only difference is a PCM (``pcm_s16le``) stream in a WAV container instead of
    the AAC stream used for muxed video.
    """
    _validate_audio_waveform(audio)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(output_path, mode="w")
    audio_stream = container.add_stream("pcm_s16le", rate=audio.sampling_rate)
    audio_stream.codec_context.sample_rate = audio.sampling_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio.sampling_rate)
    try:
        _write_audio(container, audio_stream, audio)
    finally:
        container.close()
    logger.info(f"Audio saved to {output_path}")


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


_INT_FORMAT_MAX: dict[str, float] = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
}


def _audio_frame_to_float(frame: av.AudioFrame) -> np.ndarray:
    """Convert an audio frame to a float32 ndarray with values in [-1, 1] and shape (channels, samples)."""
    fmt = frame.format.name
    arr = frame.to_ndarray().astype(np.float32)
    if fmt in _INT_FORMAT_MAX:
        arr = arr / _INT_FORMAT_MAX[fmt]
    if not frame.format.is_planar:
        # Interleaved formats have shape (1, samples * channels) — reshape to (channels, samples).
        channels = len(frame.layout.channels)
        arr = arr.reshape(-1, channels).T
    return arr


def get_videostream_fps(path: str) -> float:
    """Read video stream FPS."""
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        return float(video_stream.average_rate)
    finally:
        container.close()


def get_videostream_metadata(path: str) -> VideoPixelShape:
    """Read video stream metadata as a VideoPixelShape with batch=1.
    If frame count is missing in the container, decodes the stream to count frames.
    Args:
        path: Path to the video file.
    Returns:
        VideoPixelShape with batch=1, frames, height, width, and fps populated from the stream.
    """
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        fps = float(video_stream.average_rate)
        num_frames = video_stream.frames or 0
        if num_frames == 0:
            num_frames = sum(1 for _ in container.decode(video_stream))
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height
        return VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=fps)
    finally:
        container.close()


def decode_audio_from_file(
    path: str, device: torch.device, start_time: float = 0.0, max_duration: float | None = None
) -> Audio | None:
    """Decodes audio from a file, optionally seeking to a start time and limiting duration.
    Args:
        path: Path to the audio/video file containing an audio stream.
        device: Device to place the resulting tensor on.
        start_time: Start time in seconds to begin reading audio from.
        max_duration: Maximum audio duration in seconds. If None, reads to end of stream.
    Returns:
        An Audio object with waveform of shape (1, channels, samples), or None if no audio stream.
    """
    container = av.open(path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        return None

    sample_rate = audio_stream.rate
    start_pts = int(start_time / audio_stream.time_base)
    end_time = start_time + max_duration if max_duration else audio_stream.duration * audio_stream.time_base
    container.seek(start_pts, stream=audio_stream)

    samples = []
    first_frame_time = None
    for frame in container.decode(audio=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * audio_stream.time_base)
        frame_end = frame_time + frame.samples / frame.sample_rate
        if frame_end < start_time:
            continue
        if frame_time > end_time:
            break
        if first_frame_time is None:
            first_frame_time = frame_time
        samples.append(_audio_frame_to_float(frame))

    container.close()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Trim samples that fall outside the requested [start_time, start_time + max_duration] window.
    # Audio codecs decode in fixed-size frames whose boundaries may not align with the requested
    # time range, so the first frame can start before start_time and the last frame can end after
    # start_time + max_duration.
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    if max_duration is not None:
        max_samples = round(max_duration * sample_rate)
        audio = audio[..., :max_samples]

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)

    return Audio(waveform=waveform, sampling_rate=sample_rate)


def decode_video_by_frame(
    path: str,
    device: DeviceLikeType,
    starting_frame: int = 0,
    frame_cap: int | None = None,
) -> Generator[torch.Tensor]:
    """Decodes video from a file by sequential frame index, without relying on pts.
    Args:
        path: Path to the video file.
        device: Device to place the resulting tensors on.
        starting_frame: Number of leading frames to skip (default 0).
        frame_cap: Maximum number of frames to yield. If None, no frame limit (default None).
    Yields:
        Frames as tensors of shape (1, H, W, C), dtype uint8.
    """
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for index, frame in enumerate(container.decode(video_stream)):
            if index < starting_frame:
                continue
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            if frame_cap is not None:
                frame_cap -= 1
                if frame_cap == 0:
                    break
    finally:
        container.close()


def decode_video_from_file(
    path: str,
    device: DeviceLikeType,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> Generator[torch.Tensor]:
    """Decodes video from a file using presentation timestamps for time-based trimming.
    If a frame with no pts is encountered, falls back to :func:`decode_video_by_frame`
    using FPS-derived frame indices.
    Args:
        path: Path to the video file.
        device: Device to place the resulting tensors on.
        start_time: Start time in seconds (default 0.0).
        max_duration: Maximum duration in seconds to decode. If None, reads to end of
            stream (default None).
    Yields:
        Frames as tensors of shape (1, H, W, C), dtype uint8.
    """
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        time_base = float(video_stream.time_base)

        if start_time > 0:
            container.seek(int(start_time / time_base), stream=video_stream)

        end_time = start_time + max_duration if max_duration is not None else None

        for frame in container.decode(video_stream):
            # PyAV may leave pts unset when the demuxer does not expose per-frame
            # timestamps (e.g. some raw/elementary streams, stripped or missing
            # metadata, or certain remux paths). Without pts we cannot map frames to
            # wall-clock time, so we fall back to sequential frame indices using the
            # stream's average frame rate.
            if frame.pts is None:
                fps = float(video_stream.average_rate)
                starting_frame = round(start_time * fps)
                frame_cap = round(max_duration * fps) if max_duration is not None else None
                yield from decode_video_by_frame(
                    path=path, device=device, starting_frame=starting_frame, frame_cap=frame_cap
                )
                return
            frame_time = frame.pts * time_base
            if frame_time < start_time:
                continue
            if end_time is not None and frame_time >= end_time:
                break
            yield torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: float = DEFAULT_IMAGE_CRF) -> np.array:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array


def save_exr_tensor(tensor: torch.Tensor, file_path: str | Path, half: bool = False) -> None:
    """Save a single tensor frame as EXR with linear sRGB colorspace metadata.
    Args:
        tensor: ``[H, W, C]`` or ``[C, H, W]`` float tensor.
        file_path: Output path (e.g. ``frame_0000.exr``).
        half: Force float16 output with ZIP compression.
    """
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    use_half = half or tensor.dtype in (torch.float16, torch.half)
    img_np = np.ascontiguousarray(tensor.cpu().numpy().astype(np.float32))
    file_path = str(file_path)

    h, w = img_np.shape[:2]
    fmt = OpenImageIO.HALF if use_half else OpenImageIO.FLOAT
    spec = OpenImageIO.ImageSpec(w, h, 3, fmt)
    spec.channelnames = ("R", "G", "B")
    spec.attribute("compression", "zip")
    spec.attribute("chromaticities", "float[8]", (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290))
    spec.attribute("colorSpace", "sRGB")

    out = OpenImageIO.ImageOutput.create(file_path)
    if out is None:
        raise RuntimeError(
            f"Failed to create EXR writer for '{file_path}'. Ensure OpenImageIO is built with OpenEXR support."
        )
    try:
        if not out.open(file_path, spec):
            raise RuntimeError(f"Failed to open EXR file '{file_path}': {out.geterror()}")
        if not out.write_image(img_np):
            raise RuntimeError(f"Failed to write EXR image '{file_path}': {out.geterror()}")
    finally:
        out.close()


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear -> sRGB OETF per IEC 61966-2-1. Input assumed in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def encode_exr_sequence_to_mp4(exr_dir: Path, output_mp4: Path, frame_rate: float) -> None:
    """Convert a linear EXR frame sequence to sRGB and encode to H.264 .mp4 via PyAV.
    Exposure is fixed at EV=0 (no gain). Each EXR frame is clamped to [0, 1],
    passed through the sRGB OETF, quantised to 8-bit BGR, and fed to a libx264
    stream (crf 18, yuv420p). ``frame_rate`` is the original source video's
    frame rate so playback matches the input timing.
    """
    import os  # noqa: PLC0415

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2  # noqa: PLC0415

    exr_files = sorted(exr_dir.glob("frame_*.exr"))
    if not exr_files:
        raise FileNotFoundError(f"No EXR frames found in {exr_dir}")

    container = av.open(str(output_mp4), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(frame_rate).limit_denominator(1000))
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "movflags": "+faststart"}

    try:
        for i, exr_path in enumerate(exr_files):
            hdr = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            sdr = _linear_to_srgb(np.maximum(hdr, 0.0))
            bgr8 = (sdr * 255.0 + 0.5).astype(np.uint8)

            if i == 0:
                stream.height = bgr8.shape[0]
                stream.width = bgr8.shape[1]

            frame = av.VideoFrame.from_ndarray(bgr8, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
