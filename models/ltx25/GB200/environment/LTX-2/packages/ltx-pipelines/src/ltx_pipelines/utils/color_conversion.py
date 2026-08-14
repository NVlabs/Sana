"""Color space conversion utilities for video encoding.
Provides GPU-accelerated RGB to YUV420 conversion that runs between the
VAE decoder (which yields float RGB chunks) and ``encode_video``, bypassing
pyav's CPU-side libswscale conversion. The ``FrameConverter`` also carries
the codec metadata (pixel format, colour space, colour range) that
``encode_video`` needs to tag the output stream.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


class ColorSpace(enum.Enum):
    """YUV color space standard."""

    BT_709 = "bt709"
    BT_2020_NCL = "bt2020ncl"

    @property
    def av_colorspace(self) -> int:
        """FFmpeg ``AVCOL_SPC_*`` constant for ``codec_context.colorspace``."""
        return _AV_COLORSPACE[self]


class ColorRange(enum.Enum):
    """YUV color range."""

    MPEG = "mpeg"
    JPEG = "jpeg"

    @property
    def av_color_range(self) -> int:
        """FFmpeg ``AVCOL_RANGE_*`` constant for ``codec_context.color_range``."""
        return _AV_COLOR_RANGE[self]


class PixelFormat(enum.Enum):
    """Pixel format for video frames."""

    RGB24 = "rgb24"
    YUV420P = "yuv420p"

    @property
    def av_format(self) -> str:
        """PyAV format string for ``VideoFrame.from_ndarray``."""
        return self.value


_AV_COLORSPACE = {
    ColorSpace.BT_709: 1,  # AVCOL_SPC_BT709
    ColorSpace.BT_2020_NCL: 9,  # AVCOL_SPC_BT2020_NCL
}

_AV_COLOR_RANGE = {
    ColorRange.MPEG: 1,  # AVCOL_RANGE_MPEG (limited)
    ColorRange.JPEG: 2,  # AVCOL_RANGE_JPEG (full)
}

# BT.709 RGB->YUV matrix (row-major: each row produces one of Y, U, V)
_BT709_MATRIX = torch.tensor(
    [
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5],
        [0.5, -0.4542, -0.0458],
    ],
    dtype=torch.float32,
)

# BT.2020 NCL RGB->YUV matrix
_KR_2020 = 0.2627
_KG_2020 = 0.6780
_KB_2020 = 0.0593
_BT2020_MATRIX = torch.tensor(
    [
        [_KR_2020, _KG_2020, _KB_2020],
        [-_KR_2020 / 1.8814, -_KG_2020 / 1.8814, 0.5],
        [0.5, -_KG_2020 / 1.4746, -_KB_2020 / 1.4746],
    ],
    dtype=torch.float32,
)

_COLOR_SPACE_MATRICES = {
    ColorSpace.BT_709: _BT709_MATRIX,
    ColorSpace.BT_2020_NCL: _BT2020_MATRIX,
}


@dataclass(frozen=True)
class FrameConverter:
    """Converts ``[*, C, H, W]`` float ``[0, 1]`` frames to uint8.
    Carries encoding metadata so ``encode_video`` can derive pixel format,
    color space, and color range from the converter itself.
    The ``fn_`` callable **may mutate its input** (PyTorch trailing-underscore
    convention).  Callers that need to keep the original ``frames`` afterwards
    must pass ``frames.clone()``.  Inside ``encode_video``'s per-chunk
    generator each chunk is consumed once, so direct passthrough is safe.
    """

    pixel_format: PixelFormat
    fn_: Callable[[torch.Tensor], torch.Tensor] = field(repr=False)
    color_space: ColorSpace | None = None
    color_range: ColorRange | None = None

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return self.fn_(frames)


def rgb_to_yuv(image: torch.Tensor, color_space: ColorSpace) -> torch.Tensor:
    """Convert an RGB image to YUV.
    The image data is assumed to be in the range of ``[0, 1]``.
    Uses a single matrix multiply for better memory locality.
    Args:
        image: RGB image with shape ``(*, 3, H, W)``.
        color_space: Color space standard for the conversion matrix.
    Returns:
        YUV image with shape ``(*, 3, H, W)``.
    """
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    mat = _COLOR_SPACE_MATRICES[color_space].to(device=image.device, dtype=image.dtype)
    # [*, 3, H, W] -> [*, H, W, 3] @ [3, 3]^T -> [*, H, W, 3] -> [*, 3, H, W]
    pixels = image.movedim(-3, -1)  # [*, H, W, 3]
    yuv = pixels @ mat.T  # [*, H, W, 3]
    return yuv.movedim(-1, -3)  # [*, 3, H, W]


def apply_color_range_(y: torch.Tensor, uv: torch.Tensor, color_range: ColorRange) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale Y and UV planes to the specified color range, in-place.
    Args:
        y: Luma plane in ``[0, 1]``.
        uv: Chroma planes centered at 0.
        color_range: Target color range.
    Returns:
        Scaled ``(Y, UV)`` tensors (modified in-place).
    """
    if color_range == ColorRange.MPEG:
        y.mul_(219).add_(16)
        uv.mul_(224).add_(128)
    elif color_range == ColorRange.JPEG:
        y.mul_(255)
        uv.add_(0.5).mul_(255)
    else:
        raise ValueError(f"Unsupported color range: {color_range}")
    return y, uv


def rgb_to_yuv420(
    image: torch.Tensor, color_space: ColorSpace, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an RGB image to YUV 4:2:0 with chroma subsampling.
    Chroma is subsampled by averaging 2x2 pixel blocks (chroma siting
    ``(128, 128)``).
    Args:
        image: RGB image with shape ``(*, 3, H, W)`` in ``[0, 1]``.
            H and W must be divisible by 2.
        color_space: Color space standard.
        color_range: Color range for the output.
    Returns:
        ``(Y, UV)`` where Y has shape ``(*, 1, H, W)`` and UV has shape
        ``(*, 2, H//2, W//2)``.
    """
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")
    if image.shape[-2] % 2 != 0 or image.shape[-1] % 2 != 0:
        raise ValueError(f"Input H and W must be divisible by 2. Got {image.shape}")

    yuv = rgb_to_yuv(image, color_space)
    y = yuv[..., :1, :, :]
    # Subsample chroma: average 2x2 blocks via avg_pool2d (contiguous, fused kernel)
    uv_full = yuv[..., 1:3, :, :].contiguous()
    # Flatten leading dims for avg_pool2d which expects [N, C, H, W]
    lead = uv_full.shape[:-3]
    uv_flat = uv_full.reshape(-1, 2, uv_full.shape[-2], uv_full.shape[-1])
    uv = torch.nn.functional.avg_pool2d(uv_flat, kernel_size=2, stride=2)
    uv = uv.reshape(*lead, 2, uv.shape[-2], uv.shape[-1])

    return apply_color_range_(y, uv, color_range)


def pack_i420(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Pack Y and UV planes into I420 layout for pyav.
    I420 packs the three planes into a single 2D array of height ``H * 3 // 2``
    and width ``W``.  The Y plane occupies the first ``H`` rows.  The UV tensor
    ``(*, 2, H//2, W//2)`` is reshaped to ``(*, H//2, W)`` -- U rows packed
    two-by-two followed by V rows packed two-by-two -- and appended below.
    Args:
        y: Luma with shape ``(*, 1, H, W)``.
        uv: Chroma with shape ``(*, 2, H//2, W//2)``.
    Returns:
        Packed tensor with shape ``(*, H*3//2, W)`` uint8.
    """
    y_plane = y[..., 0, :, :]  # [*, H, W]
    uv_packed = uv.reshape(*uv.shape[:-3], uv.shape[-2], uv.shape[-1] * 2)  # [*, H//2, W]
    packed = torch.cat([y_plane, uv_packed], dim=-2)  # [*, H*3//2, W]
    return packed.clamp_(0, 255).to(torch.uint8)


def _rgb_uint8_fn_(frames: torch.Tensor) -> torch.Tensor:
    """In-place: mutates ``frames`` via ``clamp_`` + ``mul_``, returns a uint8 view."""
    return frames.clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8).movedim(-3, -1)


rgb_uint8_converter_ = FrameConverter(pixel_format=PixelFormat.RGB24, fn_=_rgb_uint8_fn_)
"""``(*, 3, H, W)`` float ``[0, 1]`` to ``(*, H, W, 3)`` uint8.  Mutates input."""


def _yuv420p_bt709_fn_(frames: torch.Tensor) -> torch.Tensor:
    y, uv = rgb_to_yuv420(frames, ColorSpace.BT_709, ColorRange.MPEG)
    return pack_i420(y, uv)


yuv420p_bt709_converter_ = FrameConverter(
    pixel_format=PixelFormat.YUV420P,
    fn_=_yuv420p_bt709_fn_,
    color_space=ColorSpace.BT_709,
    color_range=ColorRange.MPEG,
)
"""``(*, 3, H, W)`` float ``[0, 1]`` to ``(*, H*3//2, W)`` uint8 YUV420p BT.709 MPEG."""
