import math

import torch
import torch.nn.functional as F


def chunked_interpolate(x, scale_factor, mode="nearest"):
    """
    Interpolate large tensors by chunking along the channel dimension. https://discuss.pytorch.org/t/error-using-f-interpolate-for-large-3d-input/207859
    Only supports 'nearest' interpolation mode.

    Args:
        x (torch.Tensor): Input tensor (B, C, D, H, W)
        scale_factor: Tuple of scaling factors (d, h, w)

    Returns:
        torch.Tensor: Interpolated tensor
    """
    assert (
        mode == "nearest"
    ), "Only the nearest mode is supported"  # actually other modes are theoretically supported but not tested
    if len(x.shape) != 5:
        raise ValueError("Expected 5D input tensor (B, C, D, H, W)")

    # Calculate max chunk size to avoid int32 overflow. num_elements < max_int32
    # Max int32 is 2^31 - 1
    max_elements_per_chunk = 2**31 - 1

    # Calculate output spatial dimensions
    out_d = math.ceil(x.shape[2] * scale_factor[0])
    out_h = math.ceil(x.shape[3] * scale_factor[1])
    out_w = math.ceil(x.shape[4] * scale_factor[2])

    # Calculate max channels per chunk to stay under limit
    elements_per_channel = out_d * out_h * out_w
    max_channels = max_elements_per_chunk // (x.shape[0] * elements_per_channel)

    # Use smaller of max channels or input channels
    chunk_size = min(max_channels, x.shape[1])

    # Ensure at least 1 channel per chunk
    chunk_size = max(1, chunk_size)

    chunks = []
    for i in range(0, x.shape[1], chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, x.shape[1])

        chunk = x[:, start_idx:end_idx, :, :, :]

        interpolated_chunk = F.interpolate(chunk, scale_factor=scale_factor, mode="nearest")

        chunks.append(interpolated_chunk)

    if not chunks:
        raise ValueError(f"No chunks were generated. Input shape: {x.shape}")

    # Concatenate chunks along channel dimension
    return torch.cat(chunks, dim=1)


def pixel_shuffle_3d(x, upscale_factor):
    """
    3D pixelshuffle operation.
    """
    B, C, T, H, W = x.shape
    r = upscale_factor
    assert C % (r * r * r) == 0, "channel number must be a multiple of the cube of the upsampling factor"

    C_new = C // (r * r * r)
    x = x.view(B, C_new, r, r, r, T, H, W)

    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)

    y = x.reshape(B, C_new, T * r, H * r, W * r)
    return y


def pixel_unshuffle_3d(x, downsample_factor):
    """
    3D pixel unshuffle operation.
    """
    B, C, T, H, W = x.shape

    r = downsample_factor
    assert T % r == 0, f"time dimension must be a multiple of the downsampling factor, got shape {x.shape}"
    assert H % r == 0, f"height dimension must be a multiple of the downsampling factor, got shape {x.shape}"
    assert W % r == 0, f"width dimension must be a multiple of the downsampling factor, got shape {x.shape}"
    T_new = T // r
    H_new = H // r
    W_new = W // r
    C_new = C * (r * r * r)

    x = x.view(B, C, T_new, r, H_new, r, W_new, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    y = x.reshape(B, C_new, T_new, H_new, W_new)
    return y


def ceil_to_divisible(n: int, dividend: int) -> int:
    return math.ceil(dividend / (dividend // n))
