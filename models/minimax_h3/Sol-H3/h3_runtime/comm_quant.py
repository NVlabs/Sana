"""Aligned block-int8 transport for the MiniMax-H3 Ulysses QKV exchange."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .quant_kernels import _encode_ue5m3_int8


VECTOR = 128
# One aligned record carries Q, K and V values plus twelve group-32
# finite-positive UE5M3 scale codes.  The remaining four bytes keep NCCL splits
# and vector payloads aligned.
QKV_PACKET = 400
# One output record carries one 128-wide attention head, four group-32
# finite-positive UE5M3 scale codes, and twelve zero padding bytes.  B300
# measurements show that keeping every record 16-byte aligned is materially
# faster in NCCL all-to-all than the smaller, unaligned 132-byte layout.
OUTPUT_PACKET = 144
FP8_OUTPUT_PACKET = 128
VALUE_BIAS = 127
TRIPLES_PER_PROGRAM = 8
QKV_DECODE_WARPS = 4
OUTPUT_RECORDS_PER_PROGRAM = 16


@triton.jit
def _encode_output_kernel(input_ptr, packet_ptr, records,
                          RECORDS: tl.constexpr, PACKET: tl.constexpr,
                          VALUE_OFFSET: tl.constexpr):
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    values = tl.load(
        input_ptr + record[:, None] * 128 + columns[None, :], mask=valid, other=0.0
    )
    stored, scale_codes = _encode_ue5m3_int8(values, GROUPS=RECORDS * 4)
    stored = tl.reshape(stored, RECORDS, 128)
    scale_codes = tl.reshape(scale_codes, RECORDS, 4)
    base = record[:, None] * PACKET
    tl.store(packet_ptr + base + columns[None, :], stored, mask=valid)
    groups = tl.arange(0, 4)
    tl.store(
        packet_ptr + base + 128 + groups[None, :], scale_codes,
        mask=record[:, None] < records,
    )
    # Initialize padding so collectives never carry allocator contents.
    padding = tl.arange(0, 16)
    tl.store(
        packet_ptr + base + 132 + padding[None, :], 0,
        mask=(record[:, None] < records) & (padding[None, :] < PACKET - 132),
    )


@triton.jit
def _decode_merge_output_kernel(packet_ptr, output_ptr, rows, heads_local, world,
                                output_records, RECORDS: tl.constexpr,
                                PACKET: tl.constexpr, VALUE_OFFSET: tl.constexpr):
    """Decode rank-major packets directly into row-major global-head order."""
    output_record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    valid = output_record[:, None] < output_records
    global_head = output_record % (heads_local * world)
    row = output_record // (heads_local * world)
    source = global_head // heads_local
    local_head = global_head - source * heads_local
    source_record = (source * rows + row) * heads_local + local_head

    columns = tl.arange(0, 128)
    base = source_record[:, None] * PACKET
    stored = tl.load(
        packet_ptr + base + columns[None, :], mask=valid, other=VALUE_OFFSET
    ).to(tl.int32)
    group = columns // 32
    code = tl.load(
        packet_ptr + base + 128 + group[None, :], mask=valid, other=0
    ).to(tl.int32)
    scale_bits = ((((code >> 3) - 15 + 127) << 23) | ((code & 7) << 20))
    scale = scale_bits.to(tl.float32, bitcast=True)
    decoded = (stored - VALUE_OFFSET).to(tl.float32) * scale
    tl.store(
        output_ptr + output_record[:, None] * 128 + columns[None, :], decoded, mask=valid
    )


def quantize_output(x: torch.Tensor) -> torch.Tensor:
    """Encode contiguous ``[..., 128]`` BF16 attention output for transport."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("output transport requires a contiguous CUDA BF16 tensor")
    if x.ndim != 3 or x.shape[-1] != VECTOR:
        raise ValueError(f"expected [tokens, heads, {VECTOR}], got {tuple(x.shape)}")
    records = x.numel() // VECTOR
    packet = torch.empty((*x.shape[:-1], OUTPUT_PACKET), dtype=torch.uint8, device=x.device)
    if records:
        _encode_output_kernel[(triton.cdiv(records, OUTPUT_RECORDS_PER_PROGRAM),)](
            x, packet, records, RECORDS=OUTPUT_RECORDS_PER_PROGRAM,
            PACKET=OUTPUT_PACKET, VALUE_OFFSET=VALUE_BIAS, num_warps=8,
        )
    return packet


def dequantize_merge_output(packet: torch.Tensor, world: int) -> torch.Tensor:
    """Decode packed rank-major output records and merge global heads."""
    if packet.dtype != torch.uint8 or not packet.is_cuda or not packet.is_contiguous():
        raise ValueError("output transport requires contiguous CUDA uint8 packets")
    if packet.ndim != 4 or packet.shape[0] != world or packet.shape[-1] != OUTPUT_PACKET:
        raise ValueError(
            f"expected [{world}, rows, heads_local, {OUTPUT_PACKET}], got {tuple(packet.shape)}"
        )
    _, rows, heads_local, _ = packet.shape
    output = torch.empty(
        (rows, world * heads_local, VECTOR), dtype=torch.bfloat16, device=packet.device
    )
    records = rows * world * heads_local
    if records:
        _decode_merge_output_kernel[(triton.cdiv(records, OUTPUT_RECORDS_PER_PROGRAM),)](
            packet, output, rows, heads_local, world, records,
            RECORDS=OUTPUT_RECORDS_PER_PROGRAM, PACKET=OUTPUT_PACKET,
            VALUE_OFFSET=VALUE_BIAS, num_warps=8,
        )
    return output


@triton.jit
def _encode_output_fp8_kernel(input_ptr, packet_ptr, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    # Direct E4M3FN conversion maps overflow to NaN. Saturation keeps every
    # transmitted attention value finite without carrying a separate scale.
    values = tl.maximum(-448.0, tl.minimum(448.0, values))
    encoded = values.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(packet_ptr + offsets, encoded, mask=valid)


@triton.jit
def _decode_merge_output_fp8_kernel(
    packet_ptr,
    output_ptr,
    elements,
    world,
    rows,
    inner,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < elements
    tail = offsets % inner
    slot = offsets // inner
    source = slot % world
    row = slot // world
    source_offset = (source * rows + row) * inner + tail
    encoded = tl.load(packet_ptr + source_offset, mask=valid, other=0)
    decoded = encoded.to(tl.float8e4nv, bitcast=True)
    tl.store(output_ptr + offsets, decoded, mask=valid)


def quantize_output_fp8(x: torch.Tensor) -> torch.Tensor:
    """Encode a contiguous BF16 attention output as saturated raw E4M3 bytes."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("FP8 output transport requires a contiguous CUDA BF16 tensor")
    if x.ndim != 3 or x.shape[-1] != VECTOR:
        raise ValueError(f"expected [tokens, heads, {VECTOR}], got {tuple(x.shape)}")
    packet = torch.empty_like(x, dtype=torch.uint8)
    if x.numel():
        block = 1024
        _encode_output_fp8_kernel[(triton.cdiv(x.numel(), block),)](
            x, packet, x.numel(), BLOCK=block, num_warps=8
        )
    return packet


def dequantize_merge_output_fp8(packet: torch.Tensor, world: int) -> torch.Tensor:
    """Decode raw E4M3 bytes while merging rank-major heads."""
    if packet.dtype != torch.uint8 or not packet.is_cuda or not packet.is_contiguous():
        raise ValueError("FP8 output transport requires contiguous CUDA uint8 packets")
    if packet.ndim != 4 or packet.shape[0] != world or packet.shape[-1] != VECTOR:
        raise ValueError(
            f"expected [{world}, rows, heads_local, {VECTOR}], got {tuple(packet.shape)}"
        )
    _, rows, heads_local, _ = packet.shape
    output = torch.empty(
        (rows, world * heads_local, VECTOR), dtype=torch.bfloat16, device=packet.device
    )
    if output.numel():
        block = 1024
        _decode_merge_output_fp8_kernel[(triton.cdiv(output.numel(), block),)](
            packet,
            output,
            output.numel(),
            world,
            rows,
            heads_local * VECTOR,
            BLOCK=block,
            num_warps=8,
        )
    return output


@triton.jit
def _decode_qkv_padded_kernel(packet_ptr, output_ptr, source_triples, output_triples,
                              TRIPLES: tl.constexpr, PACKET: tl.constexpr,
                              VALUE_OFFSET: tl.constexpr):
    triple = tl.program_id(0) * TRIPLES + tl.arange(0, TRIPLES)
    columns = tl.arange(0, 128)
    output_valid = triple[:, None] < output_triples
    source_valid = triple[:, None] < source_triples
    packet_base = triple[:, None] * PACKET

    q_int = tl.load(packet_ptr + packet_base + columns[None, :],
                    mask=source_valid, other=VALUE_OFFSET).to(tl.int32)
    k_int = tl.load(packet_ptr + packet_base + 128 + columns[None, :],
                    mask=source_valid, other=VALUE_OFFSET).to(tl.int32)
    v_int = tl.load(packet_ptr + packet_base + 256 + columns[None, :],
                    mask=source_valid, other=VALUE_OFFSET).to(tl.int32)
    group = columns // 32
    q_code = tl.load(packet_ptr + packet_base + 384 + group[None, :],
                     mask=source_valid, other=0).to(tl.int32)
    k_code = tl.load(packet_ptr + packet_base + 388 + group[None, :],
                     mask=source_valid, other=0).to(tl.int32)
    v_code = tl.load(packet_ptr + packet_base + 392 + group[None, :],
                     mask=source_valid, other=0).to(tl.int32)

    q_scale_bits = ((((q_code >> 3) - 15 + 127) << 23)
                    | ((q_code & 7) << 20))
    k_scale_bits = ((((k_code >> 3) - 15 + 127) << 23)
                    | ((k_code & 7) << 20))
    v_scale_bits = ((((v_code >> 3) - 15 + 127) << 23)
                    | ((v_code & 7) << 20))
    q_scale = q_scale_bits.to(tl.float32, bitcast=True)
    k_scale = k_scale_bits.to(tl.float32, bitcast=True)
    v_scale = v_scale_bits.to(tl.float32, bitcast=True)
    output_base = triple[:, None] * 384
    tl.store(output_ptr + output_base + columns[None, :],
             (q_int - VALUE_OFFSET).to(tl.float32) * q_scale, mask=output_valid)
    tl.store(output_ptr + output_base + 128 + columns[None, :],
             (k_int - VALUE_OFFSET).to(tl.float32) * k_scale, mask=output_valid)
    tl.store(output_ptr + output_base + 256 + columns[None, :],
             (v_int - VALUE_OFFSET).to(tl.float32) * v_scale, mask=output_valid)


def dequantize_qkv_padded(packet: torch.Tensor, tokens: int, heads: int,
                          capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode ``[token, head, 400]`` directly into the padded BSA allocation."""
    if packet.dtype != torch.uint8 or not packet.is_contiguous():
        raise ValueError("int8 QKV transport requires contiguous uint8 packets")
    if packet.ndim != 3 or packet.shape != (tokens, heads, QKV_PACKET):
        raise ValueError(
            f"expected packet shape {(tokens, heads, QKV_PACKET)}, got {tuple(packet.shape)}"
        )
    if not 0 < tokens <= capacity:
        raise ValueError(f"tokens must be in [1, {capacity}], got {tokens}")

    source_triples = tokens * heads
    output_triples = capacity * heads
    packed = torch.empty((1, capacity, heads, 3 * VECTOR),
                         dtype=torch.bfloat16, device=packet.device)
    triples = int(os.environ.get("H3_QKV_DECODE_TRIPLES", str(TRIPLES_PER_PROGRAM)))
    if triples not in {4, 8, 16, 32}:
        raise ValueError("H3_QKV_DECODE_TRIPLES must be 4, 8, 16, or 32")
    warps = int(os.environ.get("H3_QKV_DECODE_WARPS", str(QKV_DECODE_WARPS)))
    if warps not in {4, 8}:
        raise ValueError("H3_QKV_DECODE_WARPS must be 4 or 8")
    _decode_qkv_padded_kernel[(triton.cdiv(output_triples, triples),)](
        packet, packed, source_triples, output_triples,
        TRIPLES=triples, PACKET=QKV_PACKET, VALUE_OFFSET=VALUE_BIAS,
        num_warps=warps,
    )
    return packed.split(VECTOR, dim=-1)
