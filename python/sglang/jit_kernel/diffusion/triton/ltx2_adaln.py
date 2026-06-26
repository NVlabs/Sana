from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_rms_norm_modulate_kernel(
    out_ptr,
    x_ptr,
    scale_ptr,
    shift_ptr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_c: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_c: tl.constexpr,
    scale_stride_b: tl.constexpr,
    scale_stride_t: tl.constexpr,
    scale_stride_c: tl.constexpr,
    shift_stride_b: tl.constexpr,
    shift_stride_t: tl.constexpr,
    shift_stride_c: tl.constexpr,
    eps: tl.constexpr,
    SCALE_SEQ_LEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // seq_len
    token = row - batch * seq_len
    cols = tl.arange(0, BLOCK_C)
    mask = cols < hidden_size

    x_offsets = batch * x_stride_b + token * x_stride_t + cols * x_stride_c
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    x_masked = tl.where(mask, x, 0.0)
    mean_square = tl.sum(x_masked * x_masked, axis=0) / hidden_size
    rstd = tl.rsqrt(mean_square + eps)

    scale_token = 0
    if SCALE_SEQ_LEN != 1:
        scale_token = token
    scale_offsets = (
        batch * scale_stride_b + scale_token * scale_stride_t + cols * scale_stride_c
    )
    shift_offsets = (
        batch * shift_stride_b + scale_token * shift_stride_t + cols * shift_stride_c
    )
    scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + shift_offsets, mask=mask, other=0.0).to(tl.float32)

    # Match the eager bf16 chain closely:
    #   F.rms_norm(...) -> bf16
    #   (1 + scale) -> bf16
    #   norm * scale -> bf16
    norm = (x * rstd).to(tl.bfloat16).to(tl.float32)
    scale = (scale + 1.0).to(tl.bfloat16).to(tl.float32)
    y = (norm * scale).to(tl.bfloat16).to(tl.float32) + shift

    out_offsets = batch * out_stride_b + token * out_stride_t + cols * out_stride_c
    tl.store(out_ptr + out_offsets, y, mask=mask)


def ltx2_rms_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused bf16 RMSNorm + AdaLN scale/shift for LTX2 hidden states."""
    if os.environ.get("SGLANG_LTX2_FUSED_RMS_ADALN", "1") == "0":
        raise RuntimeError("SGLANG_LTX2_FUSED_RMS_ADALN disabled")
    if (
        x.ndim != 3
        or scale.ndim != 3
        or shift.ndim != 3
        or x.dtype != torch.bfloat16
        or scale.dtype != torch.bfloat16
        or shift.dtype != torch.bfloat16
        or not x.is_cuda
        or not scale.is_cuda
        or not shift.is_cuda
        or x.stride(-1) != 1
        or scale.stride(-1) != 1
        or shift.stride(-1) != 1
    ):
        raise RuntimeError("unsupported LTX2 fused RMS AdaLN input")

    batch, seq_len, hidden_size = x.shape
    if scale.shape != shift.shape:
        raise RuntimeError("scale/shift shape mismatch")
    if scale.shape[0] != batch or scale.shape[2] != hidden_size:
        raise RuntimeError("scale/shift shape incompatible with x")
    if scale.shape[1] not in (1, seq_len):
        raise RuntimeError("scale/shift sequence dimension must be 1 or seq_len")

    block_c = triton.next_power_of_2(hidden_size)
    if block_c > 8192:
        raise RuntimeError("hidden size too large for LTX2 fused RMS AdaLN")

    out = torch.empty_like(x)
    grid = (batch * seq_len,)
    _ltx2_rms_norm_modulate_kernel[grid](
        out,
        x,
        scale,
        shift,
        seq_len,
        hidden_size,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        shift.stride(0),
        shift.stride(1),
        shift.stride(2),
        float(eps),
        SCALE_SEQ_LEN=scale.shape[1],
        BLOCK_C=block_c,
        num_warps=8,
    )
    return out
