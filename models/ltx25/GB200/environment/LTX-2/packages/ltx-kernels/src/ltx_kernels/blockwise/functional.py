"""Blockwise FP8/FP6 functional ops.
The CUDA kernels live in the ``ops_cpp`` extension.
"""

from typing import Optional, Tuple

import torch
from ops_cpp import fp6_pack, fp6_unpack
from ops_cpp import rms_norm_rope as rms_norm_rope_cuda
from ops_cpp import rms_norm_split_rope as rms_norm_split_rope_cuda

from ltx_kernels.blockwise.triton_ops import (
    blockwise_dequantize_triton,
    blockwise_quantize_adanorm_triton,  # noqa: F401  (re-exported for consumers)
    blockwise_quantize_rms_fma_triton,  # noqa: F401  (re-exported for consumers)
    blockwise_quantize_triton,  # noqa: F401  (used by linear.BlockwiseGemmLinearFunc)
    gated_attention_triton,  # noqa: F401  (re-exported for consumers)
)

# Quantization scale constants for different precisions
FP8_SCALE_MAX = 448.0
FP6_SCALE_MAX = 0.1172


@torch.library.custom_op("q8_kernels_ops::rms_norm_rope", mutates_args=())
def _rms_norm_rope_cuda(
    x: torch.Tensor,
    weights: Optional[torch.Tensor],
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    out_16bit: bool,
) -> torch.Tensor:
    return rms_norm_rope_cuda(x, weights, cos_freqs, sin_freqs, out_16bit)


@torch.library.register_fake("q8_kernels_ops::rms_norm_rope")
def _rms_norm_rope_cuda_fake(
    x: torch.Tensor,
    weights: Optional[torch.Tensor],
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    out_16bit: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    if out_16bit:
        return out.to(torch.bfloat16)
    else:
        return out.to(torch.float8_e4m3fn)


@torch.library.custom_op("q8_kernels_ops::rms_norm_split_rope", mutates_args=())
def _rms_norm_split_rope_cuda(
    x: torch.Tensor,
    sin_freqs: torch.Tensor,
    cos_freqs: torch.Tensor,
    weights: torch.Tensor,
    out_fp8: bool,
) -> torch.Tensor:
    return rms_norm_split_rope_cuda(x, sin_freqs, cos_freqs, weights, out_fp8)


@torch.library.register_fake("q8_kernels_ops::rms_norm_split_rope")
def _rms_norm_split_rope_cuda_fake(
    x: torch.Tensor,
    sin_freqs: torch.Tensor,
    cos_freqs: torch.Tensor,
    weights: torch.Tensor,
    out_fp8: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    if out_fp8:
        return out.to(torch.float8_e4m3fn)
    else:
        return out.to(torch.bfloat16)


class RMSNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, cos_freqs, sin_freqs, out_16bit):
        return torch.ops.q8_kernels_ops.rms_norm_rope(
            x, weights, cos_freqs, sin_freqs, out_16bit
        )


class RMSNormSplitRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sin_freqs, cos_freqs, weights, out_fp8):
        return torch.ops.q8_kernels_ops.rms_norm_split_rope(
            x, sin_freqs, cos_freqs, weights, out_fp8
        )


def rms_norm_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    out_fp8: bool = False,
) -> torch.Tensor:
    # weights=None flows through to the kernel's no-affine path (nullptr); see
    # rms_norm_rope.cpp, which dispatches the norm_affine template on has_value().
    return RMSNormRope.apply(x, weights, cos_freqs, sin_freqs, not out_fp8)


def rms_norm_split_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    out_fp8: bool = False,
) -> torch.Tensor:
    """
    Apply RMS normalization followed by split RoPE (Rotary Position Embedding).
    Args:
        x: Input tensor of shape [b, s, h] where h can be 2048, 4096, or 8192
        cos_freqs: Cos frequencies of shape [b, n, s, d] where n=32 and n*d = h/2
        sin_freqs: Sin frequencies of shape [b, n, s, d] where n=32
        weights: Optional RMS norm weights of shape [h]
        out_fp8: If True, output in float8_e4m3fn, otherwise bfloat16
    Returns:
        Output tensor of shape [b, s, h] with dtype based on out_fp8 parameter
    """
    if weights is None:
        weights = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    return RMSNormSplitRope.apply(x, sin_freqs, cos_freqs, weights, out_fp8)


def blockwise_quantize_weights(w: torch.Tensor, block_size=128, scale_max=FP8_SCALE_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.view(w.shape[0]//block_size, block_size, w.shape[1]//block_size, block_size).transpose(1, 2).contiguous()
    w_absmax = w.float().abs().view(w.shape[0], w.shape[1], block_size*block_size).max(dim=-1, keepdim=False).values
    w_scales = scale_max/w_absmax
    w_quant = (w.float()*w_scales[:, :, None, None].float()).to(torch.float8_e4m3fn)
    w_quant = w_quant.transpose(1, 2).contiguous()
    w_quant = w_quant.view(w.shape[0]*block_size, -1)
    return w_quant, (1/w_scales).contiguous()


def blockwise_quantize_torch(x: torch.Tensor, block_size=128, scale_max=FP8_SCALE_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, h = x.shape
    x = x.view(-1, x.shape[-1]//block_size, block_size)
    x_absmax = x.float().abs().max(dim=-1, keepdim=True).values
    x_scales = scale_max/x_absmax
    x_scaled = (x * x_scales).to(torch.float8_e4m3fn)
    return x_scaled.view(b, n, h), 1/x_scales.view(b*n, h//block_size).t().contiguous().float()


# Precision-specific convenience functions

def fp8_blockwise_quantize_weights_torch(x: torch.Tensor, block_size=128) -> tuple[torch.Tensor, torch.Tensor]:
    return blockwise_quantize_weights(x, block_size, FP8_SCALE_MAX)


def fp6_blockwise_quantize_weights_torch(x: torch.Tensor, block_size=128) -> tuple[torch.Tensor, torch.Tensor]:
    return blockwise_quantize_weights(x, block_size, FP6_SCALE_MAX)


def blockwise_dequantize(x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x_fp8, scales = x
    return blockwise_dequantize_triton(x_fp8, scales)


# FP6 Pack/Unpack operations
@torch.library.custom_op("q8_kernels_ops::fp6_pack", mutates_args=())
def _fp6_pack(x: torch.Tensor) -> torch.Tensor:
    """Pack 8-bit tensor to 6-bit by dropping e_1 and e_2 bits.
    Args:
        x: Input tensor of shape [m, n] with dtype uint8 or float8_e4m3fn
    Returns:
        Packed tensor of shape [m, n*3/4] with dtype uint8
    """
    return fp6_pack(x)


@torch.library.register_fake("q8_kernels_ops::fp6_pack")
def _fp6_pack_fake(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    n_packed = (n * 3) // 4
    return torch.empty(m, n_packed, dtype=torch.uint8, device=x.device)


@torch.library.custom_op("q8_kernels_ops::fp6_unpack", mutates_args=())
def _fp6_unpack(x: torch.Tensor, original_n: int) -> torch.Tensor:
    """Unpack 6-bit tensor back to 8-bit (with e_1 and e_2 set to 0).
    Args:
        x: Packed tensor of shape [m, n_packed] with dtype uint8
        original_n: Original dimension size (unpacked)
    Returns:
        Unpacked tensor of shape [m, original_n] with dtype uint8
    """
    return fp6_unpack(x, original_n)


@torch.library.register_fake("q8_kernels_ops::fp6_unpack")
def _fp6_unpack_fake(x: torch.Tensor, original_n: int) -> torch.Tensor:
    m = x.shape[0]
    return torch.empty(m, original_n, dtype=torch.uint8, device=x.device)


# Convenience wrapper functions
def fp6_pack_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pack 8-bit tensor to 6-bit by dropping e_1 and e_2 bits.
    This function packs FP8 weights into a more memory-efficient FP6 format
    by dropping the two highest exponent bits (e_1 and e_2). This achieves
    25% memory reduction (4 bytes -> 3 bytes per 4 elements).
    Args:
        x: Input tensor of shape [m, n] with dtype uint8 or float8_e4m3fn.
           n must be divisible by 8.
    Returns:
        Packed tensor of shape [m, n*3/4] with dtype uint8
    Example:
        >>> w_fp8 = torch.randn(1024, 4096, dtype=torch.float8_e4m3fn, device='cuda')
        >>> w_fp8_uint8 = w_fp8.view(torch.uint8)
        >>> w_packed = fp6_pack_tensor(w_fp8_uint8)
        >>> print(w_packed.shape)  # [1024, 3072]
    """
    return torch.ops.q8_kernels_ops.fp6_pack(x)


def fp6_unpack_tensor(x: torch.Tensor, original_n: int) -> torch.Tensor:
    """Unpack 6-bit tensor back to 8-bit (with e_1 and e_2 set to 0).
    This function unpacks FP6 weights back to FP8 format. Note that the
    two dropped exponent bits (e_1 and e_2) are restored as 0.
    Args:
        x: Packed tensor of shape [m, n_packed] with dtype uint8
        original_n: Original dimension size before packing (must be divisible by 8)
    Returns:
        Unpacked tensor of shape [m, original_n] with dtype uint8
    Example:
        >>> w_packed = torch.randint(0, 256, (1024, 3072), dtype=torch.uint8, device='cuda')
        >>> w_unpacked = fp6_unpack_tensor(w_packed, 4096)
        >>> w_fp8 = w_unpacked.view(torch.float8_e4m3fn)
        >>> print(w_fp8.shape)  # [1024, 4096]
    """
    return torch.ops.q8_kernels_ops.fp6_unpack(x, original_n)
