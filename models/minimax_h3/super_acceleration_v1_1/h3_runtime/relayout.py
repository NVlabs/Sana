"""The two layout changes an Ulysses all-to-all needs, each as a single kernel.

The benchmark that motivated this file is unambiguous: carrying q, k and v in one collective
instead of three measured **0.953x** — a 4.7% regression — while the same idea in SGLang's runtime
is a win. The difference is not the collective. It is what happens on either side of it.

`all_to_all_single` scatters along dimension 0, so the rank that owns each head group has to lead.
Expressed in PyTorch that is

    torch.stack((q, k, v), dim=1)                       full copy #1
      .reshape(rows, 3, world, heads_local, head_dim)
      .permute(2, 0, 1, 3, 4).contiguous()              full copy #2

two full passes over 3 x rows x heads x head_dim bfloat16 before a single byte moves between GPUs.
The measured pre-collective copies were 42 ms/step against a 341 ms attention, and packing made that
worse rather than better because the destination-major permute is a nastier stride pattern than the
three separate ones it replaced.

SGLang does not pay either copy. `pack_qkv_destination_major` reads q, k and v *through their own
strides* and writes the destination-major buffer directly, so there is no stack, no intermediate,
and no `.contiguous()` — one pass instead of two, and the input is allowed to stay a strided view of
the fused QKV projection. `usp_merge_heads` does the same for the return trip. Both are ported here.

Bit-exactness: the plain relayout kernels only move elements and remain bit-identical to the
PyTorch permutes they replace. The optional SOL/BSA wire path explicitly applies group-scaled
int8 after the existing BF16 QK-normalization/RoPE boundary; its error and end-to-end PSNR are
validated separately.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


QKV_RECORDS_PER_PROGRAM = 4
QKV_PACK_WARPS = 1


@triton.jit
def _encode_ue5m3_int8(values, GROUPS: tl.constexpr):
    """Return biased int8 values and one UE5M3 code per group of 32."""
    values = tl.reshape(values.to(tl.float32), GROUPS, 32)
    maximum = tl.max(tl.abs(values), axis=1)
    target_scale = maximum * (1.0 / 127.0)
    scale_bits = target_scale.to(tl.int32, bitcast=True)
    exponent = ((scale_bits >> 23) & 255) - 127
    mantissa_bits = scale_bits & 0x7FFFFF
    # Round the positive scale upward to the next 3-bit mantissa so the group
    # maximum remains representable.  Eight mantissa steps carry into the
    # exponent.  The unsigned 5-bit exponent is biased by 15 on the wire.
    mantissa = (mantissa_bits + 0xFFFFF) >> 20
    carry = mantissa == 8
    exponent = exponent + carry.to(tl.int32)
    mantissa = tl.where(carry, 0, mantissa)
    below = exponent < -15
    above = exponent > 16
    exponent = tl.maximum(-15, tl.minimum(16, exponent))
    mantissa = tl.where(below, 0, tl.where(above, 7, mantissa))
    encoded_scale = ((exponent + 15) << 3) | mantissa
    decoded_bits = ((exponent + 127) << 23) | (mantissa << 20)
    decoded_scale = decoded_bits.to(tl.float32, bitcast=True)
    inverse_scale = libdevice.fast_dividef(1.0, decoded_scale)
    scaled = values * inverse_scale[:, None]
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    quantized = tl.maximum(-127.0, tl.minimum(127.0, rounded)).to(tl.int32)
    return tl.reshape(quantized + 127, GROUPS * 32), encoded_scale


@triton.jit
def _pack_qkv_kernel(
    out_ptr, q_ptr, k_ptr, v_ptr,
    total_elements, rows, heads_local, head_dim,
    stride_q_row, stride_q_head,
    stride_k_row, stride_k_head,
    stride_v_row, stride_v_head,
    BLOCK: tl.constexpr,
):
    """One thread per (destination, row, local head, dim) element of the output.

    The output index is decomposed rather than the input index: that way the *stores* are perfectly
    coalesced along `dim`, and the loads gather across `global_head`, which is the direction the
    input is contiguous in anyway.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    dim = offsets % head_dim
    head_slot = offsets // head_dim
    local_head = head_slot % heads_local
    row_slot = head_slot // heads_local
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * heads_local + local_head

    q = tl.load(q_ptr + row * stride_q_row + global_head * stride_q_head + dim, mask=mask)
    k = tl.load(k_ptr + row * stride_k_row + global_head * stride_k_head + dim, mask=mask)
    v = tl.load(v_ptr + row * stride_v_row + global_head * stride_v_head + dim, mask=mask)

    base = head_slot * (3 * head_dim) + dim
    tl.store(out_ptr + base, q, mask=mask)
    tl.store(out_ptr + base + head_dim, k, mask=mask)
    tl.store(out_ptr + base + 2 * head_dim, v, mask=mask)


@triton.jit
def _qknorm_rope_pack_qkv_kernel(
    out_ptr, q_ptr, k_ptr, v_ptr, q_weight_ptr, k_weight_ptr, cos_ptr, sin_ptr,
    rows, heads, heads_local, head_dim, rotary_dim, half_dim, q_eps, k_eps,
    stride_q_row, stride_q_head,
    stride_k_row, stride_k_head,
    stride_v_row, stride_v_head,
    WIRE_INT8: tl.constexpr,
    RECORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse both QK RMSNorm/partial-RoPE operations into destination-major QKV packing.

    One program owns one or more consecutive ``(row, global_head)`` pairs. It
    reads the projection outputs, performs the arithmetic used by
    ``_qknorm_partial_rope_kernel``, and writes directly into the buffer
    consumed by ``all_to_all_single``. The normalized Q/K intermediates
    therefore never make a round trip through HBM and the standalone pack
    launch disappears.
    """
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    record_valid = record < rows * heads
    row = record // heads
    global_head = record - row * heads
    destination = global_head // heads_local
    local_head = global_head - destination * heads_local

    cols = tl.arange(0, BLOCK)
    mask = record_valid[:, None] & (cols[None, :] < head_dim)
    in_rotary = cols[None, :] < rotary_dim
    in_first_half = cols[None, :] < half_dim
    partner = tl.where(in_first_half, cols[None, :] + half_dim, cols[None, :] - half_dim)

    q_base = row[:, None] * stride_q_row + global_head[:, None] * stride_q_head
    k_base = row[:, None] * stride_k_row + global_head[:, None] * stride_k_head
    v_base = row[:, None] * stride_v_row + global_head[:, None] * stride_v_head

    q = tl.load(q_ptr + q_base + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + k_base + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    q_inv_rms = tl.math.rsqrt(tl.sum(q * q, axis=1) / head_dim + q_eps)
    k_inv_rms = tl.math.rsqrt(tl.sum(k * k, axis=1) / head_dim + k_eps)

    q_weight = tl.load(q_weight_ptr + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    q_normed = q * q_inv_rms[:, None] * q_weight
    k_normed = k * k_inv_rms[:, None] * k_weight

    partner_mask = record_valid[:, None] & in_rotary
    q_partner = tl.load(q_ptr + q_base + partner, mask=partner_mask, other=0.0).to(tl.float32)
    k_partner = tl.load(k_ptr + k_base + partner, mask=partner_mask, other=0.0).to(tl.float32)
    q_partner_weight = tl.load(q_weight_ptr + partner, mask=partner_mask, other=0.0).to(tl.float32)
    k_partner_weight = tl.load(k_weight_ptr + partner, mask=partner_mask, other=0.0).to(tl.float32)
    q_partner_normed = q_partner * q_inv_rms[:, None] * q_partner_weight
    k_partner_normed = k_partner * k_inv_rms[:, None] * k_partner_weight
    q_rotated = tl.where(in_first_half, -q_partner_normed, q_partner_normed)
    k_rotated = tl.where(in_first_half, -k_partner_normed, k_partner_normed)

    cos = tl.load(cos_ptr + row[:, None] * rotary_dim + cols[None, :],
                  mask=partner_mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + row[:, None] * rotary_dim + cols[None, :],
                  mask=partner_mask, other=0.0).to(tl.float32)
    q_out = tl.where(in_rotary, q_normed * cos + q_rotated * sin, q_normed)
    k_out = tl.where(in_rotary, k_normed * cos + k_rotated * sin, k_normed)
    v_out = tl.load(v_ptr + v_base + cols[None, :], mask=mask, other=0.0)

    head_slot = (destination * rows + row) * heads_local + local_head
    # Preserve the existing BF16 arithmetic boundary before quantization.
    q_bf16 = q_out.to(q_ptr.dtype.element_ty)
    k_bf16 = k_out.to(k_ptr.dtype.element_ty)
    v_bf16 = v_out.to(v_ptr.dtype.element_ty)
    if WIRE_INT8:
        q_store, q_scale_code = _encode_ue5m3_int8(q_bf16, GROUPS=RECORDS * 4)
        k_store, k_scale_code = _encode_ue5m3_int8(k_bf16, GROUPS=RECORDS * 4)
        v_store, v_scale_code = _encode_ue5m3_int8(v_bf16, GROUPS=RECORDS * 4)
        q_store = tl.reshape(q_store, RECORDS, 128)
        k_store = tl.reshape(k_store, RECORDS, 128)
        v_store = tl.reshape(v_store, RECORDS, 128)
        q_scale_code = tl.reshape(q_scale_code, RECORDS, 4)
        k_scale_code = tl.reshape(k_scale_code, RECORDS, 4)
        v_scale_code = tl.reshape(v_scale_code, RECORDS, 4)
        out_base = head_slot[:, None] * 400
        tl.store(out_ptr + out_base + cols[None, :], q_store, mask=mask)
        tl.store(out_ptr + out_base + 128 + cols[None, :], k_store, mask=mask)
        tl.store(out_ptr + out_base + 256 + cols[None, :], v_store, mask=mask)
        groups4 = tl.arange(0, 4)
        scale_mask = record_valid[:, None]
        tl.store(out_ptr + out_base + 384 + groups4[None, :], q_scale_code,
                 mask=scale_mask)
        tl.store(out_ptr + out_base + 388 + groups4[None, :], k_scale_code,
                 mask=scale_mask)
        tl.store(out_ptr + out_base + 392 + groups4[None, :], v_scale_code,
                 mask=scale_mask)
        # Never transmit uninitialized padding bytes.
        tl.store(out_ptr + out_base + 396 + groups4[None, :], 0, mask=scale_mask)
    else:
        out_base = head_slot[:, None] * (3 * head_dim)
        tl.store(out_ptr + out_base + cols[None, :], q_bf16, mask=mask)
        tl.store(out_ptr + out_base + head_dim + cols[None, :], k_bf16, mask=mask)
        tl.store(out_ptr + out_base + 2 * head_dim + cols[None, :], v_bf16, mask=mask)


def can_pack_qkv(q, k, v) -> bool:
    return (
        q.is_cuda and q.ndim == 3
        and q.shape == k.shape == v.shape
        and q.dtype == k.dtype == v.dtype
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and not torch.compiler.is_compiling()
    )


def pack_qkv_destination_major(q, k, v, world: int) -> torch.Tensor:
    """`(rows, heads, head_dim)` x3 -> `(world, rows, heads_local, 3 * head_dim)` contiguous.

    q, k and v may be arbitrary strided views as long as the head dimension is contiguous, which is
    what lets the fused QKV projection's output be consumed without materialising anything.
    """
    rows, heads, head_dim = q.shape
    if heads % world:
        raise ValueError(f"heads ({heads}) must divide the Ulysses degree ({world})")
    heads_local = heads // world

    out = torch.empty((world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device)
    total = rows * heads * head_dim
    if total == 0:
        return out

    block = 1024
    _pack_qkv_kernel[(triton.cdiv(total, block),)](
        out, q, k, v,
        total, rows, heads_local, head_dim,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        BLOCK=block, num_warps=8,
    )
    return out


def can_qknorm_rope_pack(q, k, v, cos, sin, world: int) -> bool:
    return (
        can_pack_qkv(q, k, v)
        and q.shape[1] % world == 0
        and cos.is_cuda and sin.is_cuda
        and cos.shape == sin.shape
        and cos.ndim == 2
        and cos.shape[0] == q.shape[0]
        and cos.is_contiguous() and sin.is_contiguous()
        and cos.shape[-1] <= q.shape[-1]
        and cos.shape[-1] % 2 == 0
    )


def qknorm_rope_pack_qkv_destination_major(
    q, k, v, q_weight, k_weight, cos, sin, q_eps: float, k_eps: float, world: int,
    wire_dtype: str = "bf16",
) -> torch.Tensor:
    """Q/K RMSNorm + partial RoPE + QKV destination-major pack in one kernel.

    Inputs are `(rows, heads, head_dim)` and the output is the exact packed layout expected by the
    Ulysses all-to-all. This has the same arithmetic as the already-enabled fused QKNorm/RoPE path;
    it only removes its materialized Q/K outputs and the following pack pass. ``int8_qkv`` keeps
    that BF16 arithmetic boundary, then encodes aligned group-scaled transport records.
    """
    if not can_qknorm_rope_pack(q, k, v, cos, sin, world):
        raise ValueError(
            "combined qknorm/rope/pack requires CUDA 3D QKV tensors, contiguous 2D rotary "
            "tables matching rows, an even rotary width, and heads divisible by world"
        )
    rows, heads, head_dim = q.shape
    heads_local = heads // world
    if wire_dtype not in {"bf16", "int8_qkv"}:
        raise ValueError(f"unsupported QKV wire dtype {wire_dtype!r}")
    if wire_dtype == "int8_qkv" and head_dim != 128:
        raise ValueError("int8 QKV transport requires head_dim=128")
    shape = ((world, rows, heads_local, 400) if wire_dtype == "int8_qkv"
             else (world, rows, heads_local, 3 * head_dim))
    dtype = torch.uint8 if wire_dtype == "int8_qkv" else q.dtype
    out = torch.empty(shape, dtype=dtype, device=q.device)
    if out.numel() == 0:
        return out
    block = triton.next_power_of_2(head_dim)
    records = (
        int(os.environ.get("H3_QKV_RECORDS_PER_PROGRAM", str(QKV_RECORDS_PER_PROGRAM)))
        if wire_dtype == "int8_qkv"
        else 1
    )
    if records not in {1, 2, 4}:
        raise ValueError("H3_QKV_RECORDS_PER_PROGRAM must be 1, 2, or 4")
    warps = (
        int(os.environ.get("H3_QKV_PACK_WARPS", str(QKV_PACK_WARPS)))
        if wire_dtype == "int8_qkv"
        else 4
    )
    if warps not in {1, 2, 4, 8}:
        raise ValueError("H3_QKV_PACK_WARPS must be 1, 2, 4, or 8")
    _qknorm_rope_pack_qkv_kernel[(triton.cdiv(rows * heads, records),)](
        out, q, k, v, q_weight, k_weight, cos, sin,
        rows, heads, heads_local, head_dim, cos.shape[-1], cos.shape[-1] // 2, q_eps, k_eps,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        WIRE_INT8=wire_dtype == "int8_qkv",
        RECORDS=records, BLOCK=block, num_warps=warps,
    )
    return out


def pack_qkv_reference(q, k, v, world: int) -> torch.Tensor:
    """What the kernel replaces, kept so the test compares against the real thing."""
    rows, heads, head_dim = q.shape
    heads_local = heads // world
    out = torch.empty((world, rows, heads_local, 3 * head_dim), dtype=q.dtype, device=q.device)
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.reshape(rows, world, heads_local, head_dim).permute(1, 0, 2, 3)
        out[..., index * head_dim : (index + 1) * head_dim].copy_(shards)
    return out


@triton.jit
def _merge_heads_kernel(
    out_ptr, x_ptr, total_elements, world, rows, inner, BLOCK: tl.constexpr,
):
    """`(world, rows, heads_local, head_dim)` -> `(rows, world, heads_local, head_dim)`.

    Both sides are contiguous, so this is a leading-dimension transpose of blocks of
    `inner = heads_local * head_dim` elements. The output index is the one decomposed, which keeps
    the stores linear and puts the gather on the load side.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    tail = offsets % inner
    slot = offsets // inner          # slot = row * world + source_rank
    source = slot % world
    row = slot // world
    src = (source * rows + row) * inner + tail
    tl.store(out_ptr + offsets, tl.load(x_ptr + src, mask=mask), mask=mask)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """`(world, rows, heads_local, head_dim)` -> `(rows, world * heads_local, head_dim)`.

    Bit-exact replacement for `x.permute(1, 0, 2, 3).contiguous().reshape(rows, -1, head_dim)` on
    the Ulysses output path.
    """
    world, rows, heads_local, head_dim = x.shape
    if not x.is_contiguous() or torch.compiler.is_compiling():
        return x.permute(1, 0, 2, 3).contiguous().reshape(rows, world * heads_local, head_dim)

    out = torch.empty((rows, world, heads_local, head_dim), dtype=x.dtype, device=x.device)
    total = out.numel()
    if total == 0:
        return out.reshape(rows, world * heads_local, head_dim)

    block = 1024
    _merge_heads_kernel[(triton.cdiv(total, block),)](
        out, x, total, world, rows, heads_local * head_dim, BLOCK=block, num_warps=8,
    )
    return out.reshape(rows, world * heads_local, head_dim)
