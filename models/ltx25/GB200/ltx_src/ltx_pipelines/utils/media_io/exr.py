"""OpenEXR read/write and HDR EXR conditioning loaders."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import OpenImageIO
import torch

from ltx_core.color.primaries import Primaries
from ltx_pipelines.utils.media_io.color_config import HDRColorSpace
from ltx_pipelines.utils.media_io.range_map import to_vae_range
from ltx_pipelines.utils.media_io.resize import ResizeMode, resize_and_center_crop, resize_and_reflect_pad

logger = logging.getLogger(__name__)


def is_exr_dir(path: str | Path) -> bool:
    """Return True if *path* is a directory that directly contains ``*.exr`` files."""
    p = Path(path)
    return p.is_dir() and any(p.glob("*.exr"))


def read_exr(path: str | Path) -> torch.Tensor:
    """Read a scene-linear EXR frame as a float32 ``[H, W, 3]`` RGB tensor.
    Uses OpenImageIO (the same backend as :func:`save_exr_tensor`), which returns
    channels in file order (R, G, B). Alpha and extra channels are dropped; a
    single-channel EXR is broadcast to three channels. Values are returned
    unmodified (unbounded scene-linear, so highlights > 1.0 survive).
    """
    buf = OpenImageIO.ImageBuf(str(path))
    pixels = buf.get_pixels(OpenImageIO.FLOAT)
    if pixels is None or buf.has_error:
        raise RuntimeError(f"Failed to read EXR '{path}': {buf.geterror()}")
    arr = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32))
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = np.repeat(arr, 3, axis=-1) if arr.shape[-1] == 1 else arr[..., :3]
    return torch.from_numpy(np.ascontiguousarray(arr))


def _exr_paths_for_conditioning(path: str | Path) -> list[Path]:
    """Resolve a still ``.exr`` or an EXR folder to a sorted frame list."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".exr":
        return [p]
    if p.is_dir():
        exr_files = sorted(p.glob("*.exr"))
        if not exr_files:
            raise RuntimeError(f"No EXR frames found in {path}")
        return exr_files
    raise ValueError(f"EXR conditioning path must be a .exr file or a directory of EXRs; got {path!r}")


def load_exr_conditioning_hdr(
    path: str | Path,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    frame_cap: int,
    frame_start: int = 0,
    color_space: HDRColorSpace,
    resize_mode: ResizeMode = ResizeMode.REFLECT_PAD,
) -> Iterator[torch.Tensor]:
    """Load EXR still(s) as HDR conditioning for the VAE.
    ``path`` may be a single ``.exr`` or a directory of ``*.exr`` frames (sorted
    lexicographically). Colour handling follows ``color_space``:
    * ``SRGB_LINEAR`` / ``ACESCG`` — compress via :meth:`HDRTransfer.to_working_space`
      (ACEScct) from the matching source primaries.
    * ``ACESCCT`` — already working-space log codes; clamp then :func:`to_vae_range`
      (no transfer).
    Yields:
        Per-frame tensors of shape ``(1, C, 1, height, width)`` in VAE range.
    """
    exr_files = _exr_paths_for_conditioning(path)[frame_start:]
    if not exr_files:
        raise RuntimeError(f"No EXR frames left in {path!r} after frame_start={frame_start}")

    resize_fn = resize_and_reflect_pad if resize_mode is ResizeMode.REFLECT_PAD else resize_and_center_crop

    count = 0
    for fp in exr_files:
        if count >= frame_cap:
            break
        pixels = read_exr(fp).to(device=device, dtype=torch.float32)  # [H, W, 3]
        frame = resize_fn(pixels, height, width)  # [1, 3, 1, H, W]
        if color_space.is_log_working:
            # Already working-space codes; clamp file noise into the legal range.
            working = frame.clamp(0.0, 1.0)
        else:
            working = color_space.transfer.to_working_space(frame, source_primaries=color_space.source_primaries)
        yield to_vae_range(working).to(device=device, dtype=dtype)
        count += 1

    if count < frame_cap:
        logger.warning(
            "load_exr_conditioning_hdr: requested %d frames from '%s' (frame_start=%d) "
            "but only %d were available — conditioning is shorter than the generation length.",
            frame_cap,
            path,
            frame_start,
            count,
        )


def load_exr_folder_conditioning_hdr(
    exr_dir: str | Path,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
    color_space: HDRColorSpace,
    resize_mode: ResizeMode = ResizeMode.REFLECT_PAD,
    frame_start: int = 0,
) -> Iterator[torch.Tensor]:
    """Load a folder of EXR frames as HDR conditioning. See :func:`load_exr_conditioning_hdr`."""
    return load_exr_conditioning_hdr(
        exr_dir,
        height,
        width,
        dtype,
        device,
        frame_cap=frame_cap,
        frame_start=frame_start,
        color_space=color_space,
        resize_mode=resize_mode,
    )


def load_exr_image_conditioning_hdr(
    exr_path: str | Path,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    color_space: HDRColorSpace,
    resize_mode: ResizeMode = ResizeMode.CENTER_CROP,
) -> torch.Tensor:
    """Load a single EXR still as HDR image conditioning for the VAE.
    Thin wrapper over :func:`load_exr_conditioning_hdr` (default center-crop).
    Returns:
        Tensor ``(1, C, 1, height, width)`` in VAE range ``[-1, 1]``.
    """
    return next(
        load_exr_conditioning_hdr(
            exr_path,
            height,
            width,
            dtype,
            device,
            frame_cap=1,
            color_space=color_space,
            resize_mode=resize_mode,
        )
    )


def save_exr_tensor(
    tensor: torch.Tensor,
    file_path: str | Path,
    half: bool = True,
    primaries: Primaries = Primaries.REC709,
    color_space: str = "sRGB",
) -> None:
    """Save a single tensor frame as EXR, tagged with its colour space.
    Args:
        tensor: ``[H, W, C]`` or ``[C, H, W]`` float tensor.
        file_path: Output path (e.g. ``frame_0000.exr``).
        half: Write float16 with ZIP compression (default True).
        primaries: Colour gamut for the ``chromaticities`` tag.
        color_space: Value for the ``colorSpace`` tag, describing the transfer
            encoding (e.g. ``"sRGB"``/``"ACEScg"`` for linear, ``"ACEScct"``/
            ``"LogC3"`` for log).
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
    spec.attribute("chromaticities", "float[8]", primaries.exr_chromaticities)
    spec.attribute("colorSpace", color_space)

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
