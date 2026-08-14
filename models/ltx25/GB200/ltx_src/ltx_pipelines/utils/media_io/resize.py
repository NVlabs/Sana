"""Spatial resize helpers for conditioning and encode."""

from __future__ import annotations

import enum
import math

import torch
from einops import rearrange


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
    height, width = image.shape[-3], image.shape[-2]
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
