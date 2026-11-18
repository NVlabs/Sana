from typing import *

import torch
from triton import jit
from triton import language as tl
from triton import next_power_of_2
from torch.library import triton_op, wrap_triton

def get_tma_aligned_size(n, element_size):
    num_elems_tma = 16 // element_size
    return ((n + num_elems_tma - 1) // num_elems_tma) * num_elems_tma 

@jit
def _quantize(x, scale_max: tl.constexpr, NUM_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    x = tl.reshape(x, (NUM_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(x)
    x_abs = tl.broadcast_to(x_abs, (NUM_BLOCKS, BLOCK_SIZE))
    x_absmax = tl.max(x_abs, axis=1)[:, None]
    x_scales = scale_max / x_absmax
    x_quant = (x_scales * x).to(tl.float8e4nv)
    x_out_scales = 1.0/x_scales
    return x_quant, x_out_scales

@jit
def _kernel(X, OUT, SCALES, HDIM, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * HDIM
    out_ptr = OUT + row_idx * HDIM

    h_offset = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + h_offset, mask=h_offset < HDIM).to(tl.float32)
    x_scale = 127.0 / tl.max(tl.abs(x))
    x_scaled = x * x_scale
    x_scaled += (0.5 * tl.where(x_scaled >= 0, 1, -1)).to(tl.int8)

    tl.store(out_ptr + h_offset, x_scaled, mask=h_offset < HDIM)
    tl.store(SCALES + row_idx, 1 / x_scale)

def run_quantize_kernel(x: torch.Tensor, out_dtype: Optional[torch.dtype] = None):
    x_shape_orig = x.shape
    x = x.view(-1, x_shape_orig[-1])
    out = torch.empty(x_shape_orig, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.shape[0], dtype=torch.float, device=x.device)

    BLOCK_SIZE = next_power_of_2(x_shape_orig[-1])
    grid = (x.shape[0],)
    _kernel[grid](x, out, scales, x_shape_orig[-1], BLOCK_SIZE, num_warps=4)

    return out.view(x_shape_orig), scales.view(x_shape_orig[:-1])

@jit
def _block_quant_norm_kernel(
    X,
    Norm_Scale,
    Norm_Shift,
    Out_scales,
    X_out,
    norm_scale_batch_stride: int,
    norm_shift_batch_stride: int,
    norm_scale_token_stride: int,
    norm_shift_token_stride: int,
    seqlen: int,
    H: tl.constexpr,
    BROADCAST_SEQLEN: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TMA_ALIGNED_MN: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0)
    batch_id =  token_idx // seqlen

    x_ptr = X + token_idx*H + tl.arange(0, H)
    if BROADCAST_SEQLEN:
        norm_scale_ptr = Norm_Scale + batch_id*norm_scale_batch_stride + tl.arange(0, H)
        norm_shift_ptr = Norm_Shift + batch_id*norm_shift_batch_stride + tl.arange(0, H)
    else:
        norm_scale_ptr = Norm_Scale + batch_id*norm_scale_batch_stride + (token_idx % seqlen) * norm_scale_token_stride + tl.arange(0, H)
        norm_shift_ptr = Norm_Shift + batch_id*norm_shift_batch_stride + (token_idx % seqlen) * norm_shift_token_stride + tl.arange(0, H)

    norm_scales = tl.load(norm_scale_ptr)
    norm_shift = tl.load(norm_shift_ptr)
    x = tl.load(x_ptr)

    x_sqr = x*x
    x_norm = tl.sum(x_sqr) / H
    x_norm = tl.rsqrt(x_norm + 0.00001)
    x = (x * x_norm).to(tl.bfloat16)
    x = x * (1.0+norm_scales) + norm_shift

    o_quant, o_scales = _quantize(x, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
    x_out_ptr = X_out + token_idx*H + BLOCK_SIZE*tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    tl.store(x_out_ptr, o_quant)
    out_scales_ptr = Out_scales + token_idx + TMA_ALIGNED_MN*tl.arange(0, NUM_BLOCKS)[:, None]
    tl.store(out_scales_ptr, o_scales)
    
@jit
def _gelu(x):
    return x * tl.sigmoid(1.702*x)

@jit
def _block_quant_kernel(
    X,
    Out_scales,
    X_out,
    seqlen: int,
    H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TMA_ALIGNED_MN: tl.constexpr,
    USE_GELU: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    batch_id, seq_id = tl.program_id(0), tl.program_id(1)
    token_idx = batch_id*seqlen + seq_id
    x_ptr = X + token_idx*H + BLOCK_SIZE*tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    x = tl.load(x_ptr).to(tl.float32)
    if USE_GELU:
        x = _gelu(x)

    o_quant, o_scales = _quantize(x, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
    x_out_ptr = X_out + token_idx*H + BLOCK_SIZE*tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    tl.store(x_out_ptr, o_quant)
    out_scales_ptr = Out_scales + token_idx + TMA_ALIGNED_MN*tl.arange(0, NUM_BLOCKS)[:, None]
    tl.store(out_scales_ptr, o_scales)

@triton_op("blockwise::quantize", mutates_args=())
def _quant_blockwise_tma_aligned_func(x: torch.Tensor, use_gelu: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h = x.shape
    num_rows = b*s
    block_size = 128
    num_blocks = h // block_size
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
    tma_aligned_mn = get_tma_aligned_size(num_rows, 4)
    scales = torch.empty_strided((num_rows, num_blocks), (1, tma_aligned_mn), dtype=torch.float, device=x.device)
    scale_max: float = 448.0
    wrap_triton(_block_quant_kernel)[(b, s)](
        x,
        scales,
        out,
        H=h,
        seqlen=s,
        BLOCK_SIZE=128,
        NUM_BLOCKS=num_blocks,
        TMA_ALIGNED_MN=tma_aligned_mn,
        USE_GELU=use_gelu,
        SCALE_MAX=scale_max
    )

    return out.view(b, s, h), scales

def run_quant_blockwise_tma_aligned(x):
    return _quant_blockwise_tma_aligned_func(x, False)

def run_quant_blockwise_gelu_tma_aligned(x):
    return _quant_blockwise_tma_aligned_func(x, True)

@triton_op("blockwise::adanorm", mutates_args=())
def run_quant_blockwise_norm_tma_aligned(x: torch.Tensor, w: Optional[torch.Tensor], norm_scale: torch.Tensor, norm_shift: torch.Tensor, out_dtype: torch.dtype, hd_scale: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h = x.shape
    is_broadcast = (norm_scale.shape[1] == 1)
    num_rows = b*s
    block_size = 128
    num_blocks = h // block_size
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
    tma_aligned_mn = get_tma_aligned_size(num_rows, 4)
    scales = torch.empty_strided((num_rows, num_blocks), (1, tma_aligned_mn), dtype=torch.float, device=x.device)
    norm_scale_batch_stride = norm_scale.stride(0)
    norm_shift_batch_stride = norm_shift.stride(0)

    scale_max: float = 448.0
    wrap_triton(_block_quant_norm_kernel)[(num_rows, 1, 1)](
        x,
        norm_scale,
        norm_shift,
        scales,
        out,
        norm_scale_batch_stride=norm_scale_batch_stride,
        norm_shift_batch_stride=norm_shift_batch_stride,
        norm_scale_token_stride=norm_scale.stride(1),
        norm_shift_token_stride=norm_shift.stride(1),
        H=h,
        seqlen=s,
        BROADCAST_SEQLEN=is_broadcast,
        BLOCK_SIZE=128,
        NUM_BLOCKS=num_blocks,
        TMA_ALIGNED_MN=tma_aligned_mn,
        SCALE_MAX=scale_max
    )

    return out.view(b, s, h), scales

@jit
def _quant_rms_sum_mult_kernel(
    X,
    Y,
    Z,
    Out,
    Out_scales,
    seqlen: int,
    z_batch_stride: int,
    z_token_stride: int,
    TMA_ALIGNED_MN: int,
    H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_Z_BROADCAST: tl.constexpr,
    QUANTIZE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0)
    batch_id =  token_idx // seqlen

    x_ptr = X + token_idx*H + tl.arange(0, H)
    y_ptr = Y + token_idx*H + tl.arange(0, H)
    if IS_Z_BROADCAST:
        z_ptr = Z + batch_id*z_batch_stride + tl.arange(0, H)
    else:
        z_ptr = Z + batch_id*z_batch_stride + (token_idx % seqlen) * z_token_stride + tl.arange(0, H)
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    z = tl.load(z_ptr)

    o = x + y*z

    tl.store(x_ptr, o)

    o_sqr = o * o
    o_norm = tl.sum(o_sqr, axis=0) / H
    o_inv = tl.rsqrt(o_norm)
    o *= o_inv

    if QUANTIZE:
        o_quant, o_scales = _quantize(o, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
        x_out_ptr = Out + token_idx*H + BLOCK_SIZE*tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
        tl.store(x_out_ptr, o_quant)
        out_scales_ptr = Out_scales + token_idx + TMA_ALIGNED_MN*tl.arange(0, NUM_BLOCKS)[:, None]
        tl.store(out_scales_ptr, o_scales)
    else:
        x_out_ptr = Out + token_idx*H + tl.arange(0, H)
        tl.store(x_out_ptr, o)

@triton_op("blockwise::quant_rms_fma", mutates_args=("x", ))
def run_quant_blockwise_rms_fma(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h = x.shape
    scale_max: float = 448.0
    is_z_broadcast = (z.shape[1] == 1)
    num_rows = b*s
    block_size = 128
    num_blocks = h // block_size
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
    tma_aligned_mn = get_tma_aligned_size(num_rows, 4)
    scales = torch.empty_strided((num_rows, num_blocks), (1, tma_aligned_mn), dtype=torch.float, device=x.device)
    z_batch_stride = z.stride(0)

    wrap_triton(_quant_rms_sum_mult_kernel)[(num_rows, 1, 1)](
        x,
        y,
        z,
        out,
        scales,
        seqlen=s,
        z_batch_stride=z_batch_stride,
        z_token_stride=z.stride(1),
        TMA_ALIGNED_MN=tma_aligned_mn,
        H=h,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=128,
        IS_Z_BROADCAST=is_z_broadcast,
        QUANTIZE=True,
        SCALE_MAX=scale_max
    )

    return out.view(b, s, h), scales


@triton_op("blockwise::rms_fma", mutates_args=("x", ))
def run_rms_fma(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    b, s, h = x.shape
    is_z_broadcast = (z.shape[1] == 1)
    num_rows = b*s
    block_size = 128
    num_blocks = h // block_size
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.bfloat16)
    z_batch_stride = z.stride(0)
    
    wrap_triton(_quant_rms_sum_mult_kernel)[(num_rows, 1, 1)](
        x,
        y,
        z,
        out,
        None,
        seqlen=s,
        z_batch_stride=z_batch_stride,
        z_token_stride=z.stride(1),
        TMA_ALIGNED_MN=0,
        H=h,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=128,
        IS_Z_BROADCAST=is_z_broadcast,
        QUANTIZE=False,
        SCALE_MAX=448.0
    )

    return out.view(b, s, h)


@jit
def _gated_attention_kernel(
    X,
    Gate_Logits,
    Out,
    Out_scales,
    H: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    DIM_HEAD: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TMA_ALIGNED_MN: tl.constexpr,
    QUANTIZE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    token_idx = tl.program_id(0)

    # Load gate logits for all heads: shape [NUM_HEADS]
    gate_logits = tl.load(Gate_Logits + token_idx * NUM_HEADS + tl.arange(0, NUM_HEADS)).to(tl.float32)
    # 2*sigmoid so that zero-init gives identity (2 * 0.5 = 1.0)
    gates = 2.0 * tl.sigmoid(gate_logits)                           # [NUM_HEADS]
    gates = tl.broadcast_to(gates[:, None], (NUM_HEADS, DIM_HEAD))  # [NUM_HEADS, DIM_HEAD]

    # Load x viewed as [NUM_HEADS, DIM_HEAD]
    offsets = tl.arange(0, NUM_HEADS)[:, None] * DIM_HEAD + tl.arange(0, DIM_HEAD)[None, :]
    x = tl.load(X + token_idx * H + offsets).to(tl.float32)

    # Apply per-head gating
    gated = x * gates  # [NUM_HEADS, DIM_HEAD]

    if QUANTIZE:
        o_quant, o_scales = _quantize(gated, SCALE_MAX, NUM_BLOCKS, BLOCK_SIZE)
        out_ptr = Out + token_idx * H + BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
        tl.store(out_ptr, o_quant)
        scales_ptr = Out_scales + token_idx + TMA_ALIGNED_MN * tl.arange(0, NUM_BLOCKS)[:, None]
        tl.store(scales_ptr, o_scales)
    else:
        out_ptr = Out + token_idx * H + offsets
        tl.store(out_ptr, gated)


@triton_op("blockwise::gated_attention", mutates_args=())
def run_gated_attention(x: torch.Tensor, gate_logits: torch.Tensor, quantize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, h = x.shape
    num_heads = gate_logits.shape[-1]
    dim_head = h // num_heads
    block_size = 128
    num_blocks = h // block_size
    num_rows = b * t
    scale_max: float = 448.0

    if quantize:
        out = torch.empty((num_rows, h), device=x.device, dtype=torch.float8_e4m3fn)
        tma_aligned_mn = get_tma_aligned_size(num_rows, 4)
        scales = torch.empty_strided((num_rows, num_blocks), (1, tma_aligned_mn), dtype=torch.float, device=x.device)
    else:
        out = torch.empty((num_rows, h), device=x.device, dtype=torch.bfloat16)
        tma_aligned_mn = 0
        scales = None

    wrap_triton(_gated_attention_kernel)[(num_rows,)](
        x,
        gate_logits,
        out,
        scales,
        H=h,
        NUM_HEADS=num_heads,
        DIM_HEAD=dim_head,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=128,
        TMA_ALIGNED_MN=tma_aligned_mn,
        QUANTIZE=quantize,
        SCALE_MAX=scale_max,
    )

    return out.view(b, t, h), scales


@jit
def _blockwise_dequantize_kernel(
    X,
    Scales,
    Out,
    scales_row_stride: int,
    scales_col_stride: int,
    H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    # Load per-block scales [NUM_BLOCKS] respecting the actual tensor strides
    scales = tl.load(
        Scales + token_idx * scales_row_stride + scales_col_stride * tl.arange(0, NUM_BLOCKS)
    ).to(tl.float32)
    scales = tl.broadcast_to(scales[:, None], (NUM_BLOCKS, BLOCK_SIZE))

    # Load fp8 input viewed as [NUM_BLOCKS, BLOCK_SIZE]
    x_offsets = BLOCK_SIZE * tl.arange(0, NUM_BLOCKS)[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]
    x = tl.load(X + token_idx * H + x_offsets).to(tl.float32)

    # Dequantize and store as bf16
    tl.store(Out + token_idx * H + x_offsets, (x * scales).to(tl.bfloat16))


@triton_op("blockwise::dequantize", mutates_args=())
def run_blockwise_dequantize(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    b, s, h = x.shape
    block_size = 128
    num_blocks = h // block_size
    num_rows = b * s
    out = torch.empty((num_rows, h), device=x.device, dtype=torch.bfloat16)

    wrap_triton(_blockwise_dequantize_kernel)[(num_rows,)](
        x,
        scales,
        out,
        scales_row_stride=scales.stride(0),
        scales_col_stride=scales.stride(1),
        H=h,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=128,
    )

    return out.view(b, s, h)


rowwise_int_quantize_triton = run_quantize_kernel
blockwise_quantize_adanorm_triton = run_quant_blockwise_norm_tma_aligned
blockwise_quantize_triton = run_quant_blockwise_tma_aligned
blockwise_quantize_gelu_triton = run_quant_blockwise_gelu_tma_aligned
blockwise_quantize_rms_fma_triton = run_quant_blockwise_rms_fma
rms_fma_triton = run_rms_fma
gated_attention_triton = run_gated_attention
blockwise_dequantize_triton = run_blockwise_dequantize

# # Precision-specific convenience functions for Triton kernels
# def fp8_blockwise_quantize_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return run_quant_blockwise_tma_aligned(x, scale_max=FP8_SCALE_MAX)

# def fp7_blockwise_quantize_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return run_quant_blockwise_tma_aligned(x, scale_max=FP7_SCALE_MAX)

# def fp6_blockwise_quantize_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     return run_quant_blockwise_tma_aligned(x, scale_max=FP6_SCALE_MAX)