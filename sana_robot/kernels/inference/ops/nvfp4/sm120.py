# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sana_robot.kernels.inference.ops.dispatch import SM120
from sana_robot.kernels.inference.ops.fp8.portable import absolute_maximum
from sana_robot.kernels.inference.ops.nvfp4.build import load_extension

ROWS = 32
(
    COOPERATIVE_128X128,
    PINGPONG_128X128,
    PINGPONG_128X64,
    COOPERATIVE_128X256,
    COOPERATIVE_128X128_K256,
    SWAPPED_PINGPONG_128X64,
    SWAPPED_PINGPONG_128X32,
    SWAPPED_COOPERATIVE_128X128,
) = range(8)
_RTX_PRO_6000 = "RTX PRO 6000"
_SPLIT_HEADS_WIDE = (256, 512)
_SPLIT_HEADS_WIDEST = (256, 384)
_NARROW_SWAPPED_MAX_M = 576
_DEVICE = {}


@triton.jit
def _e2m1_code(magnitude):
    """Return the E2M1 code 0..7 nearest to a magnitude that is already divided by its scales, ties to even."""
    return (
        (magnitude > 0.25).to(tl.int32)
        + (magnitude >= 0.75).to(tl.int32)
        + (magnitude > 1.25).to(tl.int32)
        + (magnitude >= 1.75).to(tl.int32)
        + (magnitude > 2.5).to(tl.int32)
        + (magnitude >= 3.5).to(tl.int32)
        + (magnitude > 5.0).to(tl.int32)
    )


@triton.jit
def _e2m1_value(code):
    """Return the magnitude of an E2M1 code 0..7: 0, 0.5, 1, 1.5, 2, 3, 4 or 6."""
    return (
        0.5 * tl.minimum(code, 4).to(tl.float32)
        + tl.minimum(tl.maximum(code - 4, 0), 2).to(tl.float32)
        + 2.0 * (code == 7).to(tl.float32)
    )


@triton.jit
def _nvfp4_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    amax_ptr,
    w_scale_ptr,
    alpha_ptr,
    M,
    K,
    col_tiles,
    HAS_W_SCALE: tl.constexpr,
    BM: tl.constexpr,
    FOUR_OVER_SIX: tl.constexpr,
):
    """BM rows by 64 columns of x [M, K] as NVFP4: the tensor scale is amax / (6 * 448), every 16 columns of a row share one E4M3 block scale, which puts the block's largest magnitude on the code of 6, and an element is the E2M1 code nearest to x / (block scale * tensor scale), ties to even. With FOUR_OVER_SIX the tensor scale is amax / (6 * 256) and every block also tries 1.5 times its block scale, which puts its largest magnitude on the code of 4, and keeps the scale with the lower squared error, the first one at a tie. The codes are packed two per byte, element 2j in the low nibble; the block scale of row r and block 4 * pid_k + j goes to byte (r % 32) * 16 + (r % 128) // 32 * 4 + j of tile (r // 128, pid_k) of 512 bytes; the first program writes alpha, the tensor scale times the weight's when one is given."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_k * 64 + tl.arange(0, 64)
    live = rows < M
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * K + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    if FOUR_OVER_SIX:
        scale = tl.maximum(tl.math.div_rn(tl.load(amax_ptr), 1536.0), 1e-12)
    else:
        scale = tl.maximum(tl.math.div_rn(tl.load(amax_ptr), 2688.0), 1e-12)
    if pid_m == 0 and pid_k == 0:
        if HAS_W_SCALE:
            tl.store(alpha_ptr, scale * tl.load(w_scale_ptr))
        else:
            tl.store(alpha_ptr, scale)
    blocks = tl.reshape(x, (BM, 4, 16))
    block_amax = tl.max(tl.abs(blocks), axis=2)
    ratio = tl.math.div_rn(tl.math.div_rn(block_amax, 6.0), scale)
    block_scale = tl.minimum(ratio, 448.0).to(tl.float8e4nv)
    total = block_scale.to(tl.float32) * scale
    inverse = tl.where(total > 0, tl.math.div_rn(tl.full(total.shape, 1.0, tl.float32), total), 0.0)
    v = blocks * inverse[:, :, None]
    code = _e2m1_code(tl.abs(v))
    scale_byte = block_scale.to(tl.uint8, bitcast=True)
    if FOUR_OVER_SIX:
        magnitude = tl.abs(blocks)
        error = _e2m1_value(code) * total[:, :, None] - magnitude
        error = tl.sum(error * error, axis=2)
        block_scale_4 = tl.minimum(ratio * 1.5, 448.0).to(tl.float8e4nv)
        total_4 = block_scale_4.to(tl.float32) * scale
        inverse_4 = tl.where(total_4 > 0, tl.math.div_rn(tl.full(total_4.shape, 1.0, tl.float32), total_4), 0.0)
        code_4 = _e2m1_code(magnitude * inverse_4[:, :, None])
        error_4 = _e2m1_value(code_4) * total_4[:, :, None] - magnitude
        error_4 = tl.sum(error_4 * error_4, axis=2)
        four = error_4 < error
        code = tl.where(four[:, :, None], code_4, code)
        scale_byte = tl.where(four, block_scale_4.to(tl.uint8, bitcast=True), scale_byte)
    code = code | ((v < 0).to(tl.int32) << 3)
    even, odd = tl.split(tl.reshape(code, (BM, 32, 2)))
    packed = (even | (odd << 4)).to(tl.uint8)
    tl.store(
        q_ptr + rows[:, None].to(tl.int64) * (K // 2) + pid_k * 32 + tl.arange(0, 32)[None, :],
        packed,
        mask=live[:, None],
    )
    j = tl.arange(0, 4)
    offset = ((rows // 128).to(tl.int64) * col_tiles + pid_k) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4
    tl.store(s_ptr + offset[:, None] + j[None, :], scale_byte, mask=live[:, None])


@triton.jit
def _swiglu_amax_kernel(acc_ptr, out_ptr, amax_ptr, F, BLOCK: tl.constexpr):
    """One block of one row of the packed [rows, 2F] gate and up projections: ``silu(gate) * up`` in float32, rounded to bf16, and the largest absolute value of the rounded block merged into ``amax_ptr`` by an atomic maximum."""
    row = tl.program_id(0).to(tl.int64)
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < F
    g = tl.load(acc_ptr + row * 2 * F + idx, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(acc_ptr + row * 2 * F + F + idx, mask=mask, other=0.0).to(tl.float32)
    h = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    tl.store(out_ptr + row * F + idx, h, mask=mask)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(h.to(tl.float32)), axis=0))


def scale_bytes(rows: int, k: int) -> int:
    """Return the bytes of the block scales of a [rows, K] operand in the tiled layout: the rows padded to a multiple of 128, the K / 16 blocks of a row to a multiple of 4."""
    return triton.cdiv(rows, 128) * 128 * triton.cdiv(k // 16, 4) * 4


def quantize(
    x: torch.Tensor,
    amax: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    four_over_six: bool = False,
):
    """Return a contiguous tensor [M, K] as NVFP4: the packed E2M1 codes [M, K / 2] in uint8, the E4M3 block scales in the tiled layout in uint8 (``scale_bytes``) and a float32 [1] tensor scale, multiplied by ``weight_scale`` when one is given. ``amax`` is the largest absolute value of ``x`` when the caller knows it; it is computed here otherwise. ``four_over_six`` gives every block the better of two block scales (``_nvfp4_quant_kernel``)."""
    (m, k) = x.shape
    if amax is None:
        amax = absolute_maximum(x)
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.zeros(scale_bytes(m, k), device=x.device, dtype=torch.uint8)
    alpha = torch.empty((1,), device=x.device, dtype=torch.float32)
    _nvfp4_quant_kernel[(triton.cdiv(m, ROWS), k // 64)](
        x,
        q,
        s,
        amax,
        weight_scale if weight_scale is not None else alpha,
        alpha,
        m,
        k,
        triton.cdiv(k // 16, 4),
        HAS_W_SCALE=weight_scale is not None,
        BM=ROWS,
        FOUR_OVER_SIX=four_over_six,
        num_warps=4,
    )
    return (q, s, alpha)


def quantize_weight(weight: torch.Tensor):
    """Return a contiguous weight [N, K] as NVFP4 with Four Over Six block scales (``quantize``). A weight is quantized once, at the install, so the second block scale costs nothing at inference; the tokens of every call keep the plain block scales."""
    return quantize(weight, four_over_six=True)


def swiglu(gate_up: torch.Tensor):
    """Return ``silu(gate) * up`` [M, F] in bf16 for the contiguous packed projections ``gate_up`` [M, 2F] in bf16 (gate columns, then up columns), and the largest absolute value of the result as a float32 scalar on the device."""
    (m, twof) = gate_up.shape
    out = torch.empty((m, twof // 2), device=gate_up.device, dtype=torch.bfloat16)
    peak = torch.zeros((), device=gate_up.device, dtype=torch.float32)
    _swiglu_amax_kernel[(m, triton.cdiv(twof // 2, 2048))](gate_up, out, peak, twof // 2, BLOCK=2048, num_warps=8)
    return (out, peak)


def supports(out_features: int, in_features: int) -> bool:
    """Return whether the kernels serve a weight [out_features, in_features]: the input features a multiple of 64, the columns one quantizer program covers, and the output features a multiple of 8."""
    return in_features > 0 and in_features % 64 == 0 and out_features > 0 and out_features % 8 == 0


def load() -> None:
    """Compile the matrix product for sm120, or load its cached build."""
    load_extension(SM120)


def select_config(m: int, n: int, k: int) -> int:
    """Return the tile configuration of an [M, K] x [N, K] product on the RTX 5090: wide outputs take the 128 x 128 pingpong tile; narrow ones a 64-wide N tile up to 256 rows, the swapped 128 x 64 tile (the tokens on the kernel's N) up to 384 rows and the 128 x 128 cooperative tile above."""
    if n >= 8192:
        return PINGPONG_128X128
    if m <= 256:
        return PINGPONG_128X64
    if m <= 384:
        return SWAPPED_PINGPONG_128X64
    return COOPERATIVE_128X128


def is_rtx_pro_6000(device: torch.device) -> bool:
    """Return whether a device is an RTX PRO 6000, which has its own tile plan."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    entry = _DEVICE.get(index)
    if entry is None:
        entry = _DEVICE[index] = (tuple(torch.cuda.get_device_capability(index)), torch.cuda.get_device_name(index))
    (capability, name) = entry
    return capability == (12, 0) and _RTX_PRO_6000 in name


def _plan_rtx_pro_6000(m: int, n: int, k: int) -> tuple[int, int, int]:
    """Return the plan of the RTX PRO 6000: a wide output (N >= 8192) whose rows are a head of whole 128-row tiles from ``_SPLIT_HEADS_*`` and a tail of up to 64 rows runs the head on the 128 x 128 pingpong tile and the tail on a swapped narrow tile; a narrow output of 257 to 576 rows takes the swapped 128 x 64 tile; everything else follows ``select_config``."""
    if n >= 8192:
        head, tail = 128 * (m // 128), m % 128
        if 0 < tail <= 64 and head in (_SPLIT_HEADS_WIDEST if n >= 16384 else _SPLIT_HEADS_WIDE):
            return PINGPONG_128X128, head, (SWAPPED_PINGPONG_128X32 if tail <= 32 else SWAPPED_PINGPONG_128X64)
    elif 256 < m <= _NARROW_SWAPPED_MAX_M:
        return SWAPPED_PINGPONG_128X64, 0, -1
    return select_config(m, n, k), 0, -1


def select_plan(m: int, n: int, k: int, device: torch.device) -> tuple[int, int, int]:
    """Return (config, split_rows, tail_config) of an [M, K] x [N, K] product on a device. With ``split_rows`` above 0, a multiple of 128, the rows before it run ``config`` and the rows from it on run ``tail_config``."""
    if is_rtx_pro_6000(device):
        return _plan_rtx_pro_6000(m, n, k)
    return select_config(m, n, k), 0, -1


@torch.library.custom_op("sana_robot::nvfp4_gemm", mutates_args=())
def _nvfp4_gemm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None,
    config: int,
    split_rows: int = 0,
    tail_config: int = -1,
) -> torch.Tensor:
    """Return ``alpha * (A @ B^T) + bias`` [M, N] in bf16 for packed NVFP4 operands with their tiled block scales, on the compiled kernels of sm120; a split plan writes the two row ranges of one output."""
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    extension = load_extension(SM120)
    if split_rows > 0:
        cut = scale_bytes(split_rows, a.shape[1] * 2)
        extension.gemm(out[:split_rows], a[:split_rows], b, a_scales[:cut], b_scales, alpha, bias, config)
        extension.gemm(out[split_rows:], a[split_rows:], b, a_scales[cut:], b_scales, alpha, bias, tail_config)
    else:
        extension.gemm(out, a, b, a_scales, b_scales, alpha, bias, config)
    return out


@_nvfp4_gemm_op.register_fake
def _nvfp4_gemm_layout(a, b, a_scales, b_scales, alpha, bias, config, split_rows=0, tail_config=-1):
    """Return an empty tensor laid out as the output of ``sana_robot::nvfp4_gemm``, for tracing."""
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


def nvfp4_gemm(a, a_scales, b, b_scales, alpha, bias=None) -> torch.Tensor:
    """Return ``alpha * (A @ B^T) + bias`` [M, N] in bf16: ``a`` [M, K / 2] and ``b`` [N, K / 2] are packed E2M1 codes in uint8 with their block scales in the tiled layout (``quantize``), ``alpha`` is a float32 [1] tensor on the device, the product of the two tensor scales, and ``bias`` [N] is bf16."""
    (m, n, k) = (a.shape[0], b.shape[0], a.shape[1] * 2)
    (config, split_rows, tail_config) = select_plan(m, n, k, a.device)
    return torch.ops.sana_robot.nvfp4_gemm(
        a, b, a_scales, b_scales, alpha, bias, int(config), int(split_rows), int(tail_config)
    )


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
