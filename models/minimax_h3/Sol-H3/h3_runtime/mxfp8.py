"""Blackwell MXFP8 linear layers with fused activation producers.

Weights and activations use E4M3 values with one E8M0 scale per 32 values
along K in cuBLASLt's ``SWIZZLE_32_4_4`` layout.  The producer variants emit
that representation directly from the existing RMSNorm/modulation and SwiGLU
kernels, avoiding an intermediate BF16 activation.

The swizzled layout and exponent selection follow the Apache-2.0 MiniMax-H3
implementation at yujincheng08/minimax-h3-deploy@bae3c859.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm


_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu


@triton.jit
def _mx_e8m0_from_amax(amax):
    bits = amax.to(tl.int32, bitcast=True)
    e0 = ((bits >> 23) & 0xFF) - 135
    threshold = (((e0 + 135) << 23) | 0x600000).to(tl.float32, bitcast=True)
    exponent = e0 + (amax > threshold).to(tl.int32)
    exponent = tl.maximum(exponent, -127)
    inverse = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
    return exponent + 127, inverse


@triton.jit
def _mx_scale_offsets(row, column, column_blocks):
    tile = (row // 128) * column_blocks + (column // 4)
    return (
        tile * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (column % 4)
    )


@triton.jit
def _store_mx_row(
    quantized_ptr,
    scale_ptr,
    value,
    row,
    columns,
    mask,
    num_columns,
    num_groups,
    column_blocks,
    BLOCK: tl.constexpr,
):
    values = tl.reshape(value, [BLOCK // 32, 32])
    amax = tl.max(tl.abs(values), axis=1)
    scale_byte, inverse = _mx_e8m0_from_amax(amax)
    quantized = tl.reshape(values * inverse[:, None], [BLOCK])
    tl.store(
        quantized_ptr + row.to(tl.int64) * num_columns + columns,
        quantized.to(tl.float8e4nv),
        mask=mask,
    )
    group = tl.arange(0, BLOCK // 32)
    tl.store(
        scale_ptr + _mx_scale_offsets(row, group, column_blocks),
        scale_byte.to(tl.uint8),
        mask=group < num_groups,
    )


@triton.jit
def _mxfp8_quant_kernel(
    input_ptr,
    quantized_ptr,
    scale_ptr,
    rows,
    k,
    num_groups,
    column_blocks,
    input_row_stride,
    BLOCK_ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row_program = tl.program_id(0)
    group_program = tl.program_id(1)
    row = row_program * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    group = group_program * GROUPS + tl.arange(0, GROUPS)
    column = group_program * (GROUPS * 32) + tl.arange(0, GROUPS * 32)
    row_mask = row < rows
    mask = row_mask[:, None] & (column < k)[None, :]
    value = tl.load(
        input_ptr
        + row[:, None].to(tl.int64) * input_row_stride
        + column[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.reshape(value, [BLOCK_ROWS, GROUPS, 32])
    amax = tl.max(tl.abs(values), axis=2)
    scale_byte, inverse = _mx_e8m0_from_amax(amax)
    quantized = tl.reshape(values * inverse[:, :, None], [BLOCK_ROWS, GROUPS * 32])
    tl.store(
        quantized_ptr + row[:, None].to(tl.int64) * k + column[None, :],
        quantized.to(tl.float8e4nv),
        mask=mask,
    )
    scale_mask = row_mask[:, None] & (group < num_groups)[None, :]
    tl.store(
        scale_ptr + _mx_scale_offsets(row[:, None], group[None, :], column_blocks),
        scale_byte.to(tl.uint8),
        mask=scale_mask,
    )


@triton.jit
def _rmsnorm_modulate_mxfp8_kernel(
    quantized_ptr,
    output_scale_ptr,
    input_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    index_ptr,
    num_columns,
    num_indices,
    num_groups,
    column_blocks,
    eps,
    row_stride,
    table_row_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < num_columns
    offset = row.to(tl.int64) * row_stride + columns
    table_offset = tl.load(index_ptr + (row % num_indices)) * table_row_stride + columns

    value = tl.load(input_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(value * value, axis=0) / num_columns
    normed = value * tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    normed = normed * weight

    scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    output = (normed * (1.0 + scale) + shift).to(tl.bfloat16).to(tl.float32)
    _store_mx_row(
        quantized_ptr,
        output_scale_ptr,
        output,
        row,
        columns,
        mask,
        num_columns,
        num_groups,
        column_blocks,
        BLOCK,
    )


@triton.jit
def _residual_gate_rmsnorm_modulate_mxfp8_kernel(
    hidden_output_ptr,
    quantized_ptr,
    output_scale_ptr,
    residual_ptr,
    branch_ptr,
    weight_ptr,
    gate_ptr,
    scale_ptr,
    shift_ptr,
    index_ptr,
    num_columns,
    num_indices,
    num_groups,
    column_blocks,
    eps,
    row_stride,
    table_row_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < num_columns
    offset = row.to(tl.int64) * row_stride + columns
    table_row = tl.load(index_ptr + (row % num_indices))
    table_offset = table_row * table_row_stride + columns

    residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)

    hidden = residual + gate * branch
    tl.store(
        hidden_output_ptr + offset,
        hidden.to(hidden_output_ptr.dtype.element_ty),
        mask=mask,
    )

    variance = tl.sum(hidden * hidden, axis=0) / num_columns
    normed = hidden * tl.math.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    normed = normed * weight

    scale = tl.load(scale_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + table_offset, mask=mask, other=0.0).to(tl.float32)
    output = (normed * (1.0 + scale) + shift).to(tl.bfloat16).to(tl.float32)
    _store_mx_row(
        quantized_ptr,
        output_scale_ptr,
        output,
        row,
        columns,
        mask,
        num_columns,
        num_groups,
        column_blocks,
        BLOCK,
    )


@triton.jit
def _swiglu_mxfp8_kernel(
    quantized_ptr,
    output_scale_ptr,
    input_ptr,
    num_columns,
    num_groups,
    column_blocks,
    input_row_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < num_columns
    base = input_ptr + row.to(tl.int64) * input_row_stride
    value = tl.load(base + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(base + num_columns + columns, mask=mask, other=0.0).to(tl.float32)
    output = (value * (gate * tl.sigmoid(gate))).to(tl.bfloat16).to(tl.float32)
    _store_mx_row(
        quantized_ptr,
        output_scale_ptr,
        output,
        row,
        columns,
        mask,
        num_columns,
        num_groups,
        column_blocks,
        BLOCK,
    )


class MXActivation:
    """Prequantized activation accepted by :class:`MXFP8Linear`."""

    __slots__ = ("q", "s", "shape", "dtype", "device")

    def __init__(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.q = quantized
        self.s = scale
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = quantized.device

    def dequantize(self) -> torch.Tensor:
        return mxfp8_dequantize_swizzled(self.q, self.s).to(self.dtype).view(self.shape)


def _scale_numel(rows: int, k: int) -> int:
    num_groups = k // 32
    return -(-rows // 128) * 128 * (-(-num_groups // 4) * 4)


def _allocate(rows: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = torch.empty(rows, k, dtype=_E4M3, device=device)
    scale = torch.zeros(_scale_numel(rows, k), dtype=torch.uint8, device=device)
    return quantized, scale


def _check_k(k: int) -> None:
    if k % 32 != 0:
        raise ValueError(f"MXFP8 requires K % 32 == 0, got K={k}")


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4


def _row_addressable(table: torch.Tensor) -> torch.Tensor:
    return table if table.stride(-1) == 1 else table.contiguous()


def mxfp8_quantize_swizzled(input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-major BF16 ``[rows, K]`` tensor for block-scaled GEMM."""
    if input_.ndim != 2 or input_.stride(-1) != 1 or input_.dtype != torch.bfloat16:
        raise ValueError("expected a row-major BF16 [rows, K] tensor")
    rows, k = input_.shape
    _check_k(k)
    quantized, scale = _allocate(rows, k, input_.device)
    if rows == 0:
        return quantized, scale
    num_groups = k // 32
    column_blocks = -(-num_groups // 4)
    block_rows, groups = 32, 8
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(num_groups, groups))
    _mxfp8_quant_kernel[grid](
        input_,
        quantized,
        scale,
        rows,
        k,
        num_groups,
        column_blocks,
        input_.stride(0),
        BLOCK_ROWS=block_rows,
        GROUPS=groups,
        num_warps=4,
    )
    return quantized, scale


def mxfp8_dequantize_swizzled(
    quantized: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Slow FP32 inverse used by validation tests."""
    rows, k = quantized.shape
    num_groups = k // 32
    column_blocks = -(-num_groups // 4)
    row = torch.arange(rows, device=quantized.device)[:, None]
    column = torch.arange(num_groups, device=quantized.device)[None, :]
    tile = (row // 128) * column_blocks + (column // 4)
    offsets = (
        tile * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (column % 4)
    )
    exponent = scale[offsets].to(torch.int32) - 127
    factors = torch.ldexp(torch.ones((), device=quantized.device), exponent)
    values = quantized.to(torch.float32).view(rows, num_groups, 32)
    return (values * factors[:, :, None].to(torch.float32)).view(rows, k)


def fused_rmsnorm_modulate_mxfp8(
    input_: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> MXActivation:
    """Fused RMSNorm/modulation producing MXFP8 instead of BF16."""
    columns = input_.shape[-1]
    _check_k(columns)
    flat = input_.reshape(-1, columns).contiguous()
    rows = flat.shape[0]
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    quantized, output_scale = _allocate(rows, columns, input_.device)
    num_groups = columns // 32
    block = _next_power_of_two(columns)
    _rmsnorm_modulate_mxfp8_kernel[(rows,)](
        quantized,
        output_scale,
        flat,
        weight,
        scale,
        shift,
        index,
        columns,
        index.numel(),
        num_groups,
        -(-num_groups // 4),
        eps,
        flat.stride(0),
        scale.stride(0),
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return MXActivation(quantized, output_scale, input_.shape, input_.dtype)


def fused_residual_gate_rmsnorm_modulate_mxfp8(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, MXActivation]:
    """Fused residual update producing a BF16 residual and MXFP8 norm output."""
    columns = residual.shape[-1]
    _check_k(columns)
    residual_flat = residual.reshape(-1, columns).contiguous()
    branch_flat = branch.reshape(-1, columns).contiguous()
    rows = residual_flat.shape[0]
    gate = _row_addressable(gate)
    scale = _row_addressable(scale)
    shift = _row_addressable(shift)
    hidden = torch.empty_like(residual_flat)
    quantized, output_scale = _allocate(rows, columns, residual.device)
    num_groups = columns // 32
    block = _next_power_of_two(columns)
    _residual_gate_rmsnorm_modulate_mxfp8_kernel[(rows,)](
        hidden,
        quantized,
        output_scale,
        residual_flat,
        branch_flat,
        weight,
        gate,
        scale,
        shift,
        index,
        columns,
        index.numel(),
        num_groups,
        -(-num_groups // 4),
        eps,
        residual_flat.stride(0),
        gate.stride(0),
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    activation = MXActivation(
        quantized, output_scale, residual.shape, residual.dtype
    )
    return hidden.view_as(residual), activation


def fused_swiglu_mxfp8(input_: torch.Tensor) -> MXActivation:
    """Fused SwiGLU producing MXFP8 instead of BF16."""
    columns = input_.shape[-1] // 2
    _check_k(columns)
    flat = input_.reshape(-1, input_.shape[-1]).contiguous()
    rows = flat.shape[0]
    quantized, output_scale = _allocate(rows, columns, input_.device)
    num_groups = columns // 32
    block = _next_power_of_two(columns)
    _swiglu_mxfp8_kernel[(rows,)](
        quantized,
        output_scale,
        flat,
        columns,
        num_groups,
        -(-num_groups // 4),
        flat.stride(0),
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return MXActivation(
        quantized, output_scale, (*input_.shape[:-1], columns), input_.dtype
    )


class MXFP8Linear(torch.nn.Module):
    """Bias-free block-scaled MXFP8 linear for Blackwell GPUs."""

    layout = "MXFP8Swizzled"

    def __init__(self, source: torch.nn.Linear) -> None:
        super().__init__()
        if source.bias is not None:
            raise ValueError("MXFP8Linear expects a bias-free linear")
        weight = source.weight.detach()
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"MXFP8Linear expects a BF16 weight, got {weight.dtype}")
        if weight.shape[0] % 16 != 0:
            raise ValueError(f"MXFP8 requires N % 16 == 0, got N={weight.shape[0]}")
        self.in_features = source.in_features
        self.out_features = source.out_features
        quantized, scale = mxfp8_quantize_swizzled(weight.contiguous())
        self.register_buffer("weight", quantized, persistent=False)
        self.register_buffer("weight_scale", scale.view(_E8M0), persistent=False)

    @property
    def storage_bytes(self) -> int:
        return self.weight.nbytes + self.weight_scale.nbytes

    def forward(self, input_: torch.Tensor | MXActivation) -> torch.Tensor:
        if isinstance(input_, MXActivation):
            activation = input_.q
            scale = input_.s
            leading_shape = input_.shape[:-1]
        else:
            leading_shape = input_.shape[:-1]
            flat = input_.reshape(-1, input_.shape[-1]).contiguous()
            activation, scale = mxfp8_quantize_swizzled(flat)
        output = scaled_mm(
            activation,
            self.weight.t(),
            scale_a=scale.view(_E8M0),
            scale_recipe_a=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            scale_b=self.weight_scale,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )
        return output.view(*leading_shape, self.out_features)


__all__ = [
    "MXActivation",
    "MXFP8Linear",
    "fused_residual_gate_rmsnorm_modulate_mxfp8",
    "fused_rmsnorm_modulate_mxfp8",
    "fused_swiglu_mxfp8",
    "mxfp8_dequantize_swizzled",
    "mxfp8_quantize_swizzled",
]
