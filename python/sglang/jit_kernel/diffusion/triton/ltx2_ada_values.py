import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_ada_values3_kernel(
    temb_ptr,
    table_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    total_params: tl.constexpr,
    start_index: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden

    temb_row = temb_ptr + row * total_params * hidden
    p0: tl.constexpr = start_index
    p1: tl.constexpr = start_index + 1
    p2: tl.constexpr = start_index + 2

    table0 = tl.load(
        table_ptr + p0 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table1 = tl.load(
        table_ptr + p1 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table2 = tl.load(
        table_ptr + p2 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    temb0 = tl.load(
        temb_row + (p0 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb1 = tl.load(
        temb_row + (p1 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb2 = tl.load(
        temb_row + (p2 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    base = row * hidden + cols
    tl.store(out0_ptr + base, (table0 + temb0).to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + base, (table1 + temb1).to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + base, (table2 + temb2).to(tl.bfloat16), mask=mask)


def ltx2_ada_values3(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    start_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if timestep.ndim != 3:
        raise ValueError("timestep must have shape [B,S,total_params*D]")
    if not timestep.is_cuda or timestep.dtype != torch.bfloat16:
        raise ValueError("timestep must be a CUDA bfloat16 tensor")
    if not timestep.is_contiguous():
        raise ValueError("timestep must be contiguous")
    if scale_shift_table.ndim != 2:
        raise ValueError("scale_shift_table must have shape [P,D]")
    if (
        not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError(
            "scale_shift_table must be CUDA, bf16/fp32, last-dim contiguous"
        )

    total_params = int(scale_shift_table.shape[0])
    hidden = int(scale_shift_table.shape[1])
    if start_index < 0 or start_index + 3 > total_params:
        raise ValueError("start_index must select 3 valid Ada parameters")
    if hidden <= 0 or timestep.shape[-1] != total_params * hidden:
        raise ValueError("timestep last dim must equal total_params * hidden")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("hidden size is outside the supported LTX2 fast-path range")

    batch, seq, _ = timestep.shape
    rows = int(batch * seq)
    out0 = torch.empty(
        (batch, seq, hidden), device=timestep.device, dtype=timestep.dtype
    )
    out1 = torch.empty_like(out0)
    out2 = torch.empty_like(out0)
    _ltx2_ada_values3_kernel[(rows,)](
        timestep,
        scale_shift_table,
        out0,
        out1,
        out2,
        rows,
        hidden,
        total_params,
        int(start_index),
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return out0, out1, out2


@triton.jit
def _ltx2_ada_values9_kernel(
    temb_ptr,
    table_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
    out5_ptr,
    out6_ptr,
    out7_ptr,
    out8_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    total_params: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden
    temb_row = temb_ptr + row * total_params * hidden
    base = row * hidden + cols

    table0 = tl.load(
        table_ptr + 0 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb0 = tl.load(
        temb_row + (0 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table1 = tl.load(
        table_ptr + 1 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb1 = tl.load(
        temb_row + (1 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table2 = tl.load(
        table_ptr + 2 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb2 = tl.load(
        temb_row + (2 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table3 = tl.load(
        table_ptr + 3 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb3 = tl.load(
        temb_row + (3 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table4 = tl.load(
        table_ptr + 4 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb4 = tl.load(
        temb_row + (4 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table5 = tl.load(
        table_ptr + 5 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb5 = tl.load(
        temb_row + (5 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table6 = tl.load(
        table_ptr + 6 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb6 = tl.load(
        temb_row + (6 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table7 = tl.load(
        table_ptr + 7 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb7 = tl.load(
        temb_row + (7 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table8 = tl.load(
        table_ptr + 8 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb8 = tl.load(
        temb_row + (8 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(out0_ptr + base, (table0 + temb0).to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + base, (table1 + temb1).to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + base, (table2 + temb2).to(tl.bfloat16), mask=mask)
    tl.store(out3_ptr + base, (table3 + temb3).to(tl.bfloat16), mask=mask)
    tl.store(out4_ptr + base, (table4 + temb4).to(tl.bfloat16), mask=mask)
    tl.store(out5_ptr + base, (table5 + temb5).to(tl.bfloat16), mask=mask)
    tl.store(out6_ptr + base, (table6 + temb6).to(tl.bfloat16), mask=mask)
    tl.store(out7_ptr + base, (table7 + temb7).to(tl.bfloat16), mask=mask)
    tl.store(out8_ptr + base, (table8 + temb8).to(tl.bfloat16), mask=mask)


def ltx2_ada_values9(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if timestep.ndim != 3:
        raise ValueError("timestep must have shape [B,S,9*D]")
    if not timestep.is_cuda or timestep.dtype != torch.bfloat16:
        raise ValueError("timestep must be a CUDA bfloat16 tensor")
    if not timestep.is_contiguous():
        raise ValueError("timestep must be contiguous")
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[0] != 9:
        raise ValueError("scale_shift_table must have shape [9,D]")
    if (
        not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError(
            "scale_shift_table must be CUDA, bf16/fp32, last-dim contiguous"
        )

    total_params = int(scale_shift_table.shape[0])
    hidden = int(scale_shift_table.shape[1])
    if hidden <= 0 or timestep.shape[-1] != total_params * hidden:
        raise ValueError("timestep last dim must equal 9 * hidden")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("hidden size is outside the supported LTX2 fast-path range")

    batch, seq, _ = timestep.shape
    rows = int(batch * seq)
    outs = tuple(
        torch.empty((batch, seq, hidden), device=timestep.device, dtype=timestep.dtype)
        for _ in range(9)
    )
    _ltx2_ada_values9_kernel[(rows,)](
        timestep,
        scale_shift_table,
        *outs,
        rows,
        hidden,
        total_params,
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return outs


@triton.jit
def _ltx2_rmsnorm_ada_scale_shift_kernel(
    x_ptr,
    temb_ptr,
    table_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    total_params: tl.constexpr,
    shift_index: tl.constexpr,
    scale_index: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden

    base = row * hidden + cols
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden
    normed = x * tl.rsqrt(var + eps)

    temb_row = temb_ptr + row * total_params * hidden
    shift_table = tl.load(
        table_ptr + shift_index * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    scale_table = tl.load(
        table_ptr + scale_index * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    shift = (
        shift_table
        + tl.load(
            temb_row + (shift_index * hidden + cols),
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    ).to(tl.bfloat16)
    scale = (
        scale_table
        + tl.load(
            temb_row + (scale_index * hidden + cols),
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    ).to(tl.bfloat16)

    tl.store(out_ptr + base, normed * (1.0 + scale) + shift, mask=mask)


def ltx2_rmsnorm_ada_scale_shift(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    shift_index: int,
    scale_index: int,
    eps: float,
) -> torch.Tensor:
    if x.ndim != 3 or timestep.ndim != 3:
        raise ValueError("x and timestep must have shape [B,S,D] / [B,S,P*D]")
    if (
        not x.is_cuda
        or not timestep.is_cuda
        or x.dtype != torch.bfloat16
        or timestep.dtype != x.dtype
    ):
        raise ValueError("x/timestep must be CUDA bfloat16 tensors")
    if not x.is_contiguous() or not timestep.is_contiguous():
        raise ValueError("x/timestep must be contiguous")
    if scale_shift_table.ndim != 2 or not scale_shift_table.is_cuda:
        raise ValueError("scale_shift_table must have shape [P,D] on CUDA")
    if (
        scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError("scale_shift_table must be bf16/fp32 and last-dim contiguous")
    batch, seq, hidden = x.shape
    total_params = int(scale_shift_table.shape[0])
    if scale_shift_table.shape[1] != hidden or timestep.shape != (
        batch,
        seq,
        total_params * hidden,
    ):
        raise ValueError("shape mismatch between x, table and timestep")
    if (
        min(shift_index, scale_index) < 0
        or max(shift_index, scale_index) >= total_params
    ):
        raise ValueError("Ada parameter index out of range")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("hidden size is outside the supported LTX2 fast-path range")

    out = torch.empty_like(x)
    rows = int(batch * seq)
    _ltx2_rmsnorm_ada_scale_shift_kernel[(rows,)](
        x,
        timestep,
        scale_shift_table,
        out,
        rows,
        hidden,
        total_params,
        int(shift_index),
        int(scale_index),
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        eps,
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return out


@triton.jit
def _ltx2_ada_values_indices3_kernel(
    temb_ptr,
    table_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    total_params: tl.constexpr,
    index0: tl.constexpr,
    index1: tl.constexpr,
    index2: tl.constexpr,
    table_stride_p: tl.constexpr,
    table_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden
    temb_row = temb_ptr + row * total_params * hidden
    base = row * hidden + cols

    table0 = tl.load(
        table_ptr + index0 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table1 = tl.load(
        table_ptr + index1 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    table2 = tl.load(
        table_ptr + index2 * table_stride_p + cols * table_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb0 = tl.load(
        temb_row + (index0 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb1 = tl.load(
        temb_row + (index1 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    temb2 = tl.load(
        temb_row + (index2 * hidden + cols),
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)

    tl.store(out0_ptr + base, (table0 + temb0).to(tl.bfloat16), mask=mask)
    tl.store(out1_ptr + base, (table1 + temb1).to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + base, (table2 + temb2).to(tl.bfloat16), mask=mask)


def ltx2_ada_values_indices3(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    index0: int,
    index1: int,
    index2: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if timestep.ndim != 3:
        raise ValueError("timestep must have shape [B,S,total_params*D]")
    if not timestep.is_cuda or timestep.dtype != torch.bfloat16:
        raise ValueError("timestep must be a CUDA bfloat16 tensor")
    if not timestep.is_contiguous():
        raise ValueError("timestep must be contiguous")
    if scale_shift_table.ndim != 2:
        raise ValueError("scale_shift_table must have shape [P,D]")
    if (
        not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError(
            "scale_shift_table must be CUDA, bf16/fp32, last-dim contiguous"
        )

    total_params = int(scale_shift_table.shape[0])
    hidden = int(scale_shift_table.shape[1])
    indices = (int(index0), int(index1), int(index2))
    if min(indices) < 0 or max(indices) >= total_params:
        raise ValueError("Ada parameter index out of range")
    if hidden <= 0 or timestep.shape[-1] != total_params * hidden:
        raise ValueError("timestep last dim must equal total_params * hidden")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("hidden size is outside the supported LTX2 fast-path range")

    batch, seq, _ = timestep.shape
    rows = int(batch * seq)
    out0 = torch.empty(
        (batch, seq, hidden), device=timestep.device, dtype=timestep.dtype
    )
    out1 = torch.empty_like(out0)
    out2 = torch.empty_like(out0)
    _ltx2_ada_values_indices3_kernel[(rows,)](
        timestep,
        scale_shift_table,
        out0,
        out1,
        out2,
        rows,
        hidden,
        total_params,
        indices[0],
        indices[1],
        indices[2],
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return out0, out1, out2


def ltx2_ada_values9_packed(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if timestep.ndim != 3:
        raise ValueError("timestep must have shape [B,S,9*D]")
    if not timestep.is_cuda or timestep.dtype != torch.bfloat16:
        raise ValueError("timestep must be a CUDA bfloat16 tensor")
    if not timestep.is_contiguous():
        raise ValueError("timestep must be contiguous")
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[0] != 9:
        raise ValueError("scale_shift_table must have shape [9,D]")
    if (
        not scale_shift_table.is_cuda
        or scale_shift_table.dtype not in (torch.bfloat16, torch.float32)
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError(
            "scale_shift_table must be CUDA, bf16/fp32, last-dim contiguous"
        )

    total_params = int(scale_shift_table.shape[0])
    hidden = int(scale_shift_table.shape[1])
    if hidden <= 0 or timestep.shape[-1] != total_params * hidden:
        raise ValueError("timestep last dim must equal 9 * hidden")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("hidden size is outside the supported LTX2 fast-path range")

    batch, seq, _ = timestep.shape
    rows = int(batch * seq)
    packed = torch.empty(
        (9, batch, seq, hidden), device=timestep.device, dtype=timestep.dtype
    )
    outs = tuple(packed[i] for i in range(9))
    _ltx2_ada_values9_kernel[(rows,)](
        timestep,
        scale_shift_table,
        *outs,
        rows,
        hidden,
        total_params,
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=4 if hidden >= 4096 else 8,
    )
    return outs
