"""GPU RGB→YUV420 packing for PyAV video encode (SDR 8-bit and HDR 10-bit).
Shared by ltx-pipelines and ltx-trainer. Runs between the VAE decoder (float RGB
chunks) and encode / HLG paths, bypassing pyav's CPU-side libswscale.
``FrameConverter`` also carries codec metadata (pixel format, colour space,
colour range).
"""

from __future__ import annotations

import enum

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

_KR_2020, _KG_2020, _KB_2020 = 0.2627, 0.6780, 0.0593
_COLOR_SPACE_MATRICES = {
    ColorSpace.BT_709: torch.tensor(
        [
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5],
            [0.5, -0.4542, -0.0458],
        ],
        dtype=torch.float32,
    ),
    ColorSpace.BT_2020_NCL: torch.tensor(
        [
            [_KR_2020, _KG_2020, _KB_2020],
            [-_KR_2020 / 1.8814, -_KG_2020 / 1.8814, 0.5],
            [0.5, -_KG_2020 / 1.4746, -_KB_2020 / 1.4746],
        ],
        dtype=torch.float32,
    ),
}


def _require_rgb_chw(image: torch.Tensor) -> None:
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")


def rgb_to_yuv(image: torch.Tensor, color_space: ColorSpace) -> torch.Tensor:
    """RGB ``[0, 1]`` ``(*, 3, H, W)`` → YUV ``(*, 3, H, W)``."""
    _require_rgb_chw(image)
    mat = _COLOR_SPACE_MATRICES[color_space].to(device=image.device, dtype=image.dtype)
    pixels = image.movedim(-3, -1)
    return (pixels @ mat.T).movedim(-1, -3)


def apply_color_range_(
    y: torch.Tensor,
    uv: torch.Tensor,
    color_range: ColorRange,
    *,
    bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale Y/UV to 8- or 10-bit code levels in-place.
    MPEG uses ``(219 * E' + 16) * 2^(bits-8)`` for Y and
    ``(224 * E' + 128) * 2^(bits-8)`` for Cb/Cr.
    """
    if bits not in (8, 10):
        raise ValueError(f"Unsupported bit depth: {bits}")
    factor = 1 << (bits - 8)
    if color_range == ColorRange.MPEG:
        y.mul_(219 * factor).add_(16 * factor)
        uv.mul_(224 * factor).add_(128 * factor)
    elif color_range == ColorRange.JPEG:
        max_code = (1 << bits) - 1
        y.mul_(max_code)
        uv.add_(0.5).mul_(max_code)
    else:
        raise ValueError(f"Unsupported color range: {color_range}")
    return y, uv


def apply_color_range_10bit_(
    y: torch.Tensor, uv: torch.Tensor, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale Y/UV to 10-bit code levels in-place."""
    return apply_color_range_(y, uv, color_range, bits=10)


def _rgb_to_yuv420_unit(image: torch.Tensor, color_space: ColorSpace) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB ``[0,1]`` → Y ``[0,1]`` + UV centered at 0, 4:2:0 (no code levels)."""
    _require_rgb_chw(image)
    if image.shape[-2] % 2 != 0 or image.shape[-1] % 2 != 0:
        raise ValueError(f"Input H and W must be divisible by 2. Got {image.shape}")

    yuv = rgb_to_yuv(image, color_space)
    y = yuv[..., :1, :, :]
    uv_full = yuv[..., 1:3, :, :].contiguous()
    lead = uv_full.shape[:-3]
    uv_flat = uv_full.reshape(-1, 2, uv_full.shape[-2], uv_full.shape[-1])
    uv = torch.nn.functional.avg_pool2d(uv_flat, kernel_size=2, stride=2)
    return y, uv.reshape(*lead, 2, uv.shape[-2], uv.shape[-1])


def rgb_to_yuv420(
    image: torch.Tensor, color_space: ColorSpace, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB ``(*, 3, H, W)`` → 8-bit-level Y / UV 4:2:0."""
    y, uv = _rgb_to_yuv420_unit(image, color_space)
    return apply_color_range_(y, uv, color_range)


def rgb_to_yuv420p10(
    image: torch.Tensor, color_space: ColorSpace, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB float ``[0, 1]`` ``(*, 3, H, W)`` → planar uint16 Y, U, V (yuv420p10)."""
    y, uv = _rgb_to_yuv420_unit(image, color_space)
    apply_color_range_(y, uv, color_range, bits=10)
    return yuv420_planes_u16(y, uv)


def pack_i420(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Pack Y + UV into I420 ``(*, H*3//2, W)`` uint8 for PyAV."""
    y_plane = y[..., 0, :, :]
    uv_packed = uv.reshape(*uv.shape[:-3], uv.shape[-2], uv.shape[-1] * 2)
    return torch.cat([y_plane, uv_packed], dim=-2).clamp_(0, 255).to(torch.uint8)


def yuv420_planes_u16(y: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split 10-bit Y/UV code levels into planar ``uint16`` Y, U, V."""
    y16 = y[..., 0, :, :].round_().clamp_(0, 1023).to(torch.uint16)
    u16 = uv[..., 0, :, :].round_().clamp_(0, 1023).to(torch.uint16)
    v16 = uv[..., 1, :, :].round_().clamp_(0, 1023).to(torch.uint16)
    return y16, u16, v16


def _rgb_uint8_fn_(frames: torch.Tensor) -> torch.Tensor:
    return frames.clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8).movedim(-3, -1)


def _yuv420p_bt709_fn_(frames: torch.Tensor) -> torch.Tensor:
    y, uv = rgb_to_yuv420(frames, ColorSpace.BT_709, ColorRange.MPEG)
    return pack_i420(y, uv)


class FrameConverter(enum.Enum):
    """Converts ``[*, C, H, W]`` float ``[0, 1]`` frames for encode.
    Carries encoding metadata so the caller can tag the output stream.
    ``__call__`` **may mutate** its input (aliases keep the trailing-underscore
    convention: ``rgb_uint8_converter_``, ``yuv420p_bt709_converter_``).
    HLG / 10-bit packing stays in ``hlg`` — not a case here.
    """

    RGB24 = "rgb24"
    YUV420P_BT709 = "yuv420p_bt709"

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        match self:
            case FrameConverter.RGB24:
                return _rgb_uint8_fn_(frames)
            case FrameConverter.YUV420P_BT709:
                return _yuv420p_bt709_fn_(frames)

    @property
    def pixel_format(self) -> PixelFormat:
        match self:
            case FrameConverter.RGB24:
                return PixelFormat.RGB24
            case FrameConverter.YUV420P_BT709:
                return PixelFormat.YUV420P

    @property
    def color_space(self) -> ColorSpace | None:
        match self:
            case FrameConverter.RGB24:
                return None
            case FrameConverter.YUV420P_BT709:
                return ColorSpace.BT_709

    @property
    def color_range(self) -> ColorRange | None:
        match self:
            case FrameConverter.RGB24:
                return None
            case FrameConverter.YUV420P_BT709:
                return ColorRange.MPEG


rgb_uint8_converter_ = FrameConverter.RGB24
yuv420p_bt709_converter_ = FrameConverter.YUV420P_BT709
