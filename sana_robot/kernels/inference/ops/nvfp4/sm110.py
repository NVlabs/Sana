# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.dispatch import SM110
from sana_robot.kernels.inference.ops.nvfp4 import sm120
from sana_robot.kernels.inference.ops.nvfp4.build import load_extension
from sana_robot.kernels.inference.ops.nvfp4.sm120 import scale_bytes

WORD_ROWS = 64
WORD_WARPS = 4
AMAX_BLOCK = 4096
AMAX_WARPS = 8
AMAX_PARTS = 128
(
    ONE_SM_128X128,
    ONE_SM_128X256,
    ONE_SM_128X64,
    ONE_SM_128X192,
    TWO_SM_256X128,
    TWO_SM_256X256,
    SWAPPED_ONE_SM_128X64,
    SWAPPED_ONE_SM_128X128,
) = range(8)


@triton.jit
def _e2m1x2(even, odd):
    """Return float32 pairs as bytes of two E2M1 codes, ``even`` in the low nibble, by the conversion instruction of sm_110a: round to nearest even, saturating at 6. The instruction keeps the sign of a zero, so a negative zero must come in as a positive one."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b8 b0, b1, b2, b3;
        cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
        mov.b32 $0, {b0, b1, b2, b3};
        }""",
        "=r,f,f,f,f,f,f,f,f",
        [even, odd],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _e2m1x2_decode(packed):
    """Return bytes of two E2M1 codes as int32 words that hold the two values in fp16, the low nibble's in the low half, by the conversion instruction of sm_110a: the value keeps the code's sign, a negative zero for the code 8."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b8 b0, b1, b2, b3;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 $0, b0;
        cvt.rn.f16x2.e2m1x2 $1, b1;
        cvt.rn.f16x2.e2m1x2 $2, b2;
        cvt.rn.f16x2.e2m1x2 $3, b3;
        }""",
        "=r,=r,=r,=r,r",
        [packed],
        dtype=tl.int32,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _block_error(packed, total, even, odd):
    """Return the squared error of every block row of ``_e2m1x2`` bytes against x: each byte decoded to its two signed values times the block's total scale, minus the elements, squared and summed over the row's 8 words. It equals the error of ``sm120._nvfp4_quant_kernel``, which compares magnitudes, since a value carries its element's sign or is zero."""
    words = _e2m1x2_decode(packed)
    value_even = (words & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    value_odd = (words >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    error_even = value_even * total[:, None] - even
    error_odd = value_odd * total[:, None] - odd
    return tl.sum(error_even * error_even + error_odd * error_odd, axis=1)


@triton.jit
def _amax_parts_kernel(x_ptr, part_ptr, numel, chunk, BLOCK: tl.constexpr):
    """part[p] = the largest absolute value of the elements [p * chunk, (p + 1) * chunk) of a flat tensor; ``chunk`` is a multiple of BLOCK."""
    pid = tl.program_id(0).to(tl.int64)
    start = pid * chunk
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for offset in range(0, chunk, BLOCK):
        idx = start + offset + tl.arange(0, BLOCK)
        acc = tl.maximum(acc, tl.abs(tl.load(x_ptr + idx, mask=idx < numel, other=0.0).to(tl.float32)))
    tl.store(part_ptr + pid, tl.max(acc, axis=0))


@triton.jit
def _scale_offsets(rows, j, pid_k, col_tiles):
    """Return the byte of the block scale of row r and block 4 * pid_k + j in the tiled layout: (r % 32) * 16 + (r % 128) // 32 * 4 + j of tile (r // 128, pid_k) of 512 bytes."""
    return ((rows // 128).to(tl.int64) * col_tiles + pid_k) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4 + j


@triton.jit
def _nvfp4_quant_words_kernel(
    xw_ptr,
    q_ptr,
    s_ptr,
    part_ptr,
    n_parts,
    w_scale_ptr,
    alpha_ptr,
    M,
    M_pad,
    K,
    col_tiles,
    HAS_W_SCALE: tl.constexpr,
    BM: tl.constexpr,
    PARTS: tl.constexpr,
    FOUR_OVER_SIX: tl.constexpr,
):
    """BM rows by 64 columns of bf16 x [M, K] as NVFP4, the bytes of ``sm120._nvfp4_quant_kernel``: ``xw_ptr`` is x as int32 words [M, K / 2], element 2c in the low half of word c, and a program covers 4 * BM block rows of 8 words, one 16-element scale block each. The largest absolute value of x is the maximum of the ``n_parts`` partial maxima at ``part_ptr``. The grid covers M_pad rows, M up to the next multiple of 128: a program past M only writes zero scales for its rows, and the rows past M of the program that straddles M load as 0, so their scales are 0. With FOUR_OVER_SIX the tensor scale is amax / (6 * 256) and every block also tries 1.5 times its block scale and keeps the one with the lower squared error, the first at a tie; each choice takes the signs of its own conversion."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    block = tl.arange(0, 4 * BM)
    rows = pid_m * BM + block // 4
    j = block % 4
    if pid_m * BM >= M:
        tl.store(s_ptr + _scale_offsets(rows, j, pid_k, col_tiles), tl.zeros((4 * BM,), dtype=tl.uint8))
    else:
        parts = tl.arange(0, PARTS)
        amax = tl.max(tl.load(part_ptr + parts, mask=parts < n_parts, other=0.0), axis=0)
        if FOUR_OVER_SIX:
            scale = tl.maximum(tl.math.div_rn(amax, 1536.0), 1e-12)
        else:
            scale = tl.maximum(tl.math.div_rn(amax, 2688.0), 1e-12)
        if pid_m == 0 and pid_k == 0:
            if HAS_W_SCALE:
                tl.store(alpha_ptr, scale * tl.load(w_scale_ptr))
            else:
                tl.store(alpha_ptr, scale)
        live = rows < M
        offsets = rows[:, None].to(tl.int64) * (K // 2) + pid_k * 32 + j[:, None] * 8 + tl.arange(0, 8)[None, :]
        words = tl.load(xw_ptr + offsets, mask=live[:, None], other=0)
        even = (words << 16).to(tl.float32, bitcast=True)
        odd = (words & -65536).to(tl.float32, bitcast=True)
        block_amax = tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), axis=1)
        ratio = tl.math.div_rn(tl.math.div_rn(block_amax, 6.0), scale)
        block_scale = tl.minimum(ratio, 448.0).to(tl.float8e4nv)
        total = block_scale.to(tl.float32) * scale
        inverse = tl.where(total > 0, tl.math.div_rn(tl.full(total.shape, 1.0, tl.float32), total), 0.0)
        packed = _e2m1x2(even * inverse[:, None] + 0.0, odd * inverse[:, None] + 0.0)
        scale_byte = block_scale.to(tl.uint8, bitcast=True)
        if FOUR_OVER_SIX:
            block_scale_4 = tl.minimum(ratio * 1.5, 448.0).to(tl.float8e4nv)
            total_4 = block_scale_4.to(tl.float32) * scale
            inverse_4 = tl.where(total_4 > 0, tl.math.div_rn(tl.full(total_4.shape, 1.0, tl.float32), total_4), 0.0)
            packed_4 = _e2m1x2(even * inverse_4[:, None] + 0.0, odd * inverse_4[:, None] + 0.0)
            four = _block_error(packed_4, total_4, even, odd) < _block_error(packed, total, even, odd)
            packed = tl.where(four[:, None], packed_4, packed)
            scale_byte = tl.where(four, block_scale_4.to(tl.uint8, bitcast=True), scale_byte)
        tl.store(q_ptr + offsets, packed, mask=live[:, None])
        tl.store(s_ptr + _scale_offsets(rows, j, pid_k, col_tiles), scale_byte)


def amax_parts(x: torch.Tensor):
    """Return up to ``AMAX_PARTS`` partial maxima of the absolute values of a contiguous tensor as a float32 tensor on its device, and their count; their maximum is the largest absolute value of ``x``."""
    numel = x.numel()
    chunk = triton.cdiv(numel, AMAX_BLOCK * AMAX_PARTS) * AMAX_BLOCK
    n = triton.cdiv(numel, chunk)
    parts = torch.empty((n,), device=x.device, dtype=torch.float32)
    _amax_parts_kernel[(n,)](x.view(-1), parts, numel, chunk, BLOCK=AMAX_BLOCK, num_warps=AMAX_WARPS)
    return (parts, n)


def _quantize_words(x, parts, n_parts, weight_scale, q, s, alpha, four_over_six: bool = False) -> None:
    """Quantize bf16 ``x`` [M, K] into the given buffers: every byte of the rows of ``q`` and of ``s`` is written, the scales of the padding rows as 0."""
    (m, k) = x.shape
    m_pad = triton.cdiv(m, 128) * 128
    _nvfp4_quant_words_kernel[(m_pad // WORD_ROWS, k // 64)](
        x.view(torch.int32),
        q,
        s,
        parts,
        n_parts,
        weight_scale if weight_scale is not None else alpha,
        alpha,
        m,
        m_pad,
        k,
        triton.cdiv(k // 16, 4),
        HAS_W_SCALE=weight_scale is not None,
        BM=WORD_ROWS,
        PARTS=max(16, triton.next_power_of_2(n_parts)),
        FOUR_OVER_SIX=four_over_six,
        num_warps=WORD_WARPS,
    )


def quantize(
    x: torch.Tensor,
    amax: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    four_over_six: bool = False,
):
    """Return a contiguous tensor [M, K] as NVFP4, the bytes of ``sm120.quantize``: the packed E2M1 codes [M, K / 2] in uint8, the E4M3 block scales in the tiled layout in uint8 and a float32 [1] tensor scale, multiplied by ``weight_scale`` when one is given. ``four_over_six`` gives every block the better of two block scales (``_nvfp4_quant_words_kernel``). A bf16 tensor takes the kernel written for Jetson Thor, any other the sm120 kernel."""
    if x.dtype is not torch.bfloat16:
        return sm120.quantize(x, amax, weight_scale, four_over_six)
    (m, k) = x.shape
    if amax is None:
        (parts, n_parts) = amax_parts(x)
    else:
        (parts, n_parts) = (amax.reshape(1).to(torch.float32), 1)
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.empty(scale_bytes(m, k), device=x.device, dtype=torch.uint8)
    alpha = torch.empty((1,), device=x.device, dtype=torch.float32)
    _quantize_words(x, parts, n_parts, weight_scale, q, s, alpha, four_over_six)
    return (q, s, alpha)


def quantize_weight(weight: torch.Tensor):
    """Return a contiguous weight [N, K] as NVFP4 with Four Over Six block scales (``quantize``), as ``sm120.quantize_weight`` does. A weight is quantized once, at the install, so the second block scale costs nothing at inference; the tokens of every call keep the plain block scales."""
    return quantize(weight, four_over_six=True)


def swiglu(gate_up: torch.Tensor):
    """The Jetson Thor form of ``sm120.swiglu``: the same kernel."""
    return sm120.swiglu(gate_up)


def supports(out_features: int, in_features: int) -> bool:
    """Return whether the kernels serve a weight [out_features, in_features] on Jetson Thor: the sizes of the sm120 kernels."""
    return sm120.supports(out_features, in_features)


def load() -> None:
    """Compile the matrix product for sm110, or load its cached build."""
    load_extension(SM110)


def select_config(m: int, n: int, k: int) -> int:
    """Return the tile configuration of an [M, K] x [N, K] product on Jetson Thor: the swapped 128 x 128 tile (the weight rows on the kernel's M) up to 384 rows and for the widest outputs; above, wide outputs take the 128 x 192 tile, and narrow ones the two-SM 256 x 256 tile for a wide K and the two-SM 256 x 128 tile otherwise."""
    if m <= 384 or n >= 16384:
        return SWAPPED_ONE_SM_128X128
    if n >= 8192:
        return ONE_SM_128X192
    return TWO_SM_256X256 if k >= 8192 else TWO_SM_256X128


@torch.library.custom_op("sana_robot::nvfp4_gemm_sm110", mutates_args=())
def _nvfp4_gemm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None,
    config: int,
) -> torch.Tensor:
    """Return ``alpha * (A @ B^T) + bias`` [M, N] in bf16 for packed NVFP4 operands with their tiled block scales, on the compiled kernels of sm110."""
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    load_extension(SM110).gemm(out, a, b, a_scales, b_scales, alpha, bias, config)
    return out


@_nvfp4_gemm_op.register_fake
def _nvfp4_gemm_layout(a, b, a_scales, b_scales, alpha, bias, config):
    """Return an empty tensor laid out as the output of ``sana_robot::nvfp4_gemm_sm110``, for tracing."""
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def nvfp4_gemm(a, a_scales, b, b_scales, alpha, bias=None) -> torch.Tensor:
    """Return ``alpha * (A @ B^T) + bias`` [M, N] in bf16: ``a`` [M, K / 2] and ``b`` [N, K / 2] are packed E2M1 codes in uint8 with their block scales in the tiled layout (``quantize``), ``alpha`` is a float32 [1] tensor on the device, the product of the two tensor scales, and ``bias`` [N] is bf16."""
    (m, n, k) = (a.shape[0], b.shape[0], a.shape[1] * 2)
    return torch.ops.sana_robot.nvfp4_gemm_sm110(a, b, a_scales, b_scales, alpha, bias, int(select_config(m, n, k)))


def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_scales: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    amax: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``x @ W^T + bias`` [M, N] in bf16 through NVFP4 for contiguous ``x`` [M, K] and a weight quantized by ``quantize`` (packed codes, block scales, tensor scale): ``x`` is quantized here, with ``amax`` when the caller knows its largest absolute value, and the product applies both tensor scales and the bias."""
    (packed, scales, alpha) = quantize(x, amax, weight_scale)
    return nvfp4_gemm(packed, scales, weight, block_scales, alpha, bias)
