import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from sglang.jit_kernel.diffusion.cutedsl.common.norm_fusion import (
    apply_norm_cta,
    tensor_slice_for_bsfd,
)
from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
    to_fake_cute_args,
)
from sglang.jit_kernel.diffusion.cutedsl.utils import WARP_SIZE

_COMPILE_CACHE = {}


class NormAdaScaleShift:
    @classmethod
    def make_hash_key(cls, *inputs):
        def _sig(val):
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(
        self,
        D: int,
        norm_type: str,
        shift_index: int,
        scale_index: int,
    ):
        self.D = D
        self.norm_type = norm_type
        self.shift_index = shift_index
        self.scale_index = scale_index
        self.num_warps = self.D // 256
        self.num_threads = self.num_warps * WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mY,
        mX,
        mTimestep,
        mTable,
        eps: cutlass.Float32 = cutlass.Float32(1e-5),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        B, S, _ = mX.shape
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        tiled_copy = cute.make_tiled_copy_tv(
            atom_copy,
            cute.make_layout(self.num_threads),
            cute.make_layout(8),
        )
        self.kernel(mY, mX, mTimestep, mTable, tiled_copy, eps).launch(
            grid=[B * S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mY, mX, mTimestep, mTable, tiled_copy: cute.TiledCopy, eps):
        _, S, _ = mX.shape
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = cutlass.Int32(bid)
        bidx = cutlass.Int32(bid // S)
        bidy = cutlass.Int32(bid % S)
        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def copy_if(src, dst):
            if cutlass.const_expr(
                isinstance(src, cute.Tensor) and isinstance(dst, cute.Tensor)
            ):
                cute.autovec_copy(src, dst)

        @cute.jit
        def norm(x):
            return apply_norm_cta(
                self.norm_type, self.num_warps, tidx, x, 1, 0, self.D, eps
            )

        @cute.jit
        def table_slice(param_index: cutlass.Constexpr):
            g_tile = cute.local_tile(
                mTable,
                tiler=(1, self.D),
                coord=(param_index, 0),
            )
            return g_tile[0, None]

        @cute.jit
        def timestep_slice(param_index: cutlass.Constexpr):
            g_tile = cute.local_tile(
                mTimestep,
                tiler=(1, 1, self.D),
                coord=(row, param_index, 0),
            )
            return g_tile[0, 0, None]

        tXgX, tXrX = tensor_slice_for_bsfd(mX, thr_copy, bidx, bidy, S, self.D)
        tYgY, tYrY = tensor_slice_for_bsfd(mY, thr_copy, bidx, bidy, S, self.D)
        copy_if(tXgX, tXrX)

        tNrN = cute.make_rmem_tensor_like(tXrX, tXrX.element_type)
        tNrN.store(tXrX.load())
        tNrN = norm(tNrN)

        tSHgTable = thr_copy.partition_S(table_slice(self.shift_index))
        tSHgTimestep = thr_copy.partition_S(timestep_slice(self.shift_index))
        tSCgTable = thr_copy.partition_S(table_slice(self.scale_index))
        tSCgTimestep = thr_copy.partition_S(timestep_slice(self.scale_index))
        tSHrTable = cute.make_fragment_like(tSHgTable, tSHgTable.element_type)
        tSHrTimestep = cute.make_fragment_like(tSHgTimestep, tSHgTimestep.element_type)
        tSCrTable = cute.make_fragment_like(tSCgTable, tSCgTable.element_type)
        tSCrTimestep = cute.make_fragment_like(tSCgTimestep, tSCgTimestep.element_type)
        copy_if(tSHgTable, tSHrTable)
        copy_if(tSHgTimestep, tSHrTimestep)
        copy_if(tSCgTable, tSCrTable)
        copy_if(tSCgTimestep, tSCrTimestep)

        tSHr = cute.make_fragment_like(tXrX)
        tSCr = cute.make_fragment_like(tXrX)
        tSHr.store(
            (
                tSHrTable.load().to(tSHr.element_type)
                + tSHrTimestep.load().to(tSHr.element_type)
            ).to(tSHr.element_type)
        )
        tSCr.store(
            (
                tSCrTable.load().to(tSCr.element_type)
                + tSCrTimestep.load().to(tSCr.element_type)
            ).to(tSCr.element_type)
        )

        value = tNrN.load()
        value = value * (1 + tSCr.load())
        value = value + tSHr.load()
        tYrY.store(value.to(tYrY.element_type))
        copy_if(tYrY, tYgY)


def _validate_inputs(
    x: torch.Tensor,
    timestep: torch.Tensor,
    scale_shift_table: torch.Tensor,
    shift_index: int,
    scale_index: int,
) -> None:
    if x.ndim != 3 or timestep.ndim != 3:
        raise ValueError("x and timestep must have shape [B,S,D] / [B,S,P*D]")
    if not x.is_cuda or not timestep.is_cuda or not scale_shift_table.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("x dtype must be fp16, bf16 or fp32")
    if timestep.dtype != x.dtype:
        raise ValueError("timestep dtype must match x dtype")
    if scale_shift_table.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("scale_shift_table dtype must be fp16, bf16 or fp32")
    if (
        x.stride(-1) != 1
        or timestep.stride(-1) != 1
        or scale_shift_table.stride(-1) != 1
    ):
        raise ValueError("inputs must be last-dim contiguous")
    batch, seq, hidden = x.shape
    total_params = int(scale_shift_table.shape[0])
    if scale_shift_table.shape != (total_params, hidden):
        raise ValueError("scale_shift_table must have shape [P,D]")
    if timestep.shape != (batch, seq, total_params * hidden):
        raise ValueError("shape mismatch between x, timestep and scale_shift_table")
    if (
        min(shift_index, scale_index) < 0
        or max(shift_index, scale_index) >= total_params
    ):
        raise ValueError("Ada parameter index out of range")
    if hidden % 256 != 0 or hidden > 8192:
        raise ValueError("D must be a multiple of 256 and <= 8192")


@torch.library.custom_op("sglang::fused_norm_ada_scale_shift", mutates_args=())
def fused_norm_ada_scale_shift(
    x: torch.Tensor,
    timestep: torch.Tensor,
    scale_shift_table: torch.Tensor,
    shift_index: int,
    scale_index: int,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_inputs(x, timestep, scale_shift_table, shift_index, scale_index)
    if norm_type not in ("layer", "rms"):
        raise ValueError('norm_type must be one of "layer" and "rms"')

    y = torch.empty_like(x)
    batch, seq, hidden = x.shape
    total_params = int(scale_shift_table.shape[0])
    timestep_3d = timestep.reshape(batch * seq, total_params, hidden)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    hash_key = NormAdaScaleShift.make_hash_key(
        norm_type,
        int(shift_index),
        int(scale_index),
        y,
        x,
        timestep_3d,
        scale_shift_table,
    )
    compiled_fn = _COMPILE_CACHE.get(hash_key)
    if compiled_fn is None:
        kernel = NormAdaScaleShift(
            x.shape[-1], norm_type, int(shift_index), int(scale_index)
        )
        fake_sig_args = [
            to_fake_cute_args(t) for t in (y, x, timestep_3d, scale_shift_table)
        ]
        compiled_fn = cute.compile(kernel, *fake_sig_args, options="--enable-tvm-ffi")
        _COMPILE_CACHE[hash_key] = compiled_fn

    compiled_fn(y, x, timestep_3d, scale_shift_table, eps, stream)
    return y


_CUTE_INT32_INDEX_LIMIT = 2_000_000_000


def fused_norm_ada_scale_shift_chunked(
    x: torch.Tensor,
    timestep: torch.Tensor,
    scale_shift_table: torch.Tensor,
    shift_index: int,
    scale_index: int,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_inputs(x, timestep, scale_shift_table, shift_index, scale_index)
    batch, seq, hidden = x.shape
    total_params = int(scale_shift_table.shape[0])
    max_seq = max(1, _CUTE_INT32_INDEX_LIMIT // (batch * total_params * hidden))
    if seq <= max_seq:
        return fused_norm_ada_scale_shift(
            x,
            timestep,
            scale_shift_table,
            shift_index,
            scale_index,
            norm_type,
            eps,
        )

    chunks = []
    for start in range(0, seq, max_seq):
        end = min(seq, start + max_seq)
        chunks.append(
            fused_norm_ada_scale_shift(
                x[:, start:end, :],
                timestep[:, start:end, :],
                scale_shift_table,
                shift_index,
                scale_index,
                norm_type,
                eps,
            )
        )
    return torch.cat(chunks, dim=1)


@fused_norm_ada_scale_shift.register_fake
def _fused_norm_ada_scale_shift_fake(
    x, timestep, scale_shift_table, shift_index, scale_index, norm_type, eps=1e-5
):
    return x.new_empty(x.shape)


class ScaleResidualNormAdaScaleShift:
    @classmethod
    def make_hash_key(cls, *inputs):
        def _sig(val):
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(
        self,
        D: int,
        norm_type: str,
        shift_index: int,
        scale_index: int,
    ):
        self.D = D
        self.norm_type = norm_type
        self.shift_index = shift_index
        self.scale_index = scale_index
        self.num_warps = self.D // 256
        self.num_threads = self.num_warps * WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mY,
        mResOut,
        mResidual,
        mX,
        mGate,
        mTimestep,
        mTable,
        eps: cutlass.Float32 = cutlass.Float32(1e-5),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        B, S, _ = mX.shape
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        tiled_copy = cute.make_tiled_copy_tv(
            atom_copy,
            cute.make_layout(self.num_threads),
            cute.make_layout(8),
        )
        self.kernel(
            mY,
            mResOut,
            mResidual,
            mX,
            mGate,
            mTimestep,
            mTable,
            tiled_copy,
            eps,
        ).launch(
            grid=[B * S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mY,
        mResOut,
        mResidual,
        mX,
        mGate,
        mTimestep,
        mTable,
        tiled_copy: cute.TiledCopy,
        eps,
    ):
        _, S, _ = mX.shape
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = cutlass.Int32(bid)
        bidx = cutlass.Int32(bid // S)
        bidy = cutlass.Int32(bid % S)
        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def copy_if(src, dst):
            if cutlass.const_expr(
                isinstance(src, cute.Tensor) and isinstance(dst, cute.Tensor)
            ):
                cute.autovec_copy(src, dst)

        @cute.jit
        def norm(x):
            return apply_norm_cta(
                self.norm_type, self.num_warps, tidx, x, 1, 0, self.D, eps
            )

        @cute.jit
        def table_slice(param_index: cutlass.Constexpr):
            g_tile = cute.local_tile(
                mTable,
                tiler=(1, self.D),
                coord=(param_index, 0),
            )
            return g_tile[0, None]

        @cute.jit
        def timestep_slice(param_index: cutlass.Constexpr):
            g_tile = cute.local_tile(
                mTimestep,
                tiler=(1, 1, self.D),
                coord=(row, param_index, 0),
            )
            return g_tile[0, 0, None]

        tRgR, tRrR = tensor_slice_for_bsfd(mResidual, thr_copy, bidx, bidy, S, self.D)
        tXgX, tXrX = tensor_slice_for_bsfd(mX, thr_copy, bidx, bidy, S, self.D)
        tGgG, tGrG = tensor_slice_for_bsfd(mGate, thr_copy, bidx, bidy, S, self.D)
        tROgRO, tROrRO = tensor_slice_for_bsfd(mResOut, thr_copy, bidx, bidy, S, self.D)
        tYgY, tYrY = tensor_slice_for_bsfd(mY, thr_copy, bidx, bidy, S, self.D)
        copy_if(tRgR, tRrR)
        copy_if(tXgX, tXrX)
        copy_if(tGgG, tGrG)

        value = tXrX.load() * tGrG.load()
        value = value + tRrR.load()
        tROrRO.store(value.to(tROrRO.element_type))
        copy_if(tROrRO, tROgRO)

        tNrN = cute.make_rmem_tensor_like(tXrX, tXrX.element_type)
        tNrN.store(value.to(tNrN.element_type))
        tNrN = norm(tNrN)

        tSHgTable = thr_copy.partition_S(table_slice(self.shift_index))
        tSHgTimestep = thr_copy.partition_S(timestep_slice(self.shift_index))
        tSCgTable = thr_copy.partition_S(table_slice(self.scale_index))
        tSCgTimestep = thr_copy.partition_S(timestep_slice(self.scale_index))
        tSHrTable = cute.make_fragment_like(tSHgTable, tSHgTable.element_type)
        tSHrTimestep = cute.make_fragment_like(tSHgTimestep, tSHgTimestep.element_type)
        tSCrTable = cute.make_fragment_like(tSCgTable, tSCgTable.element_type)
        tSCrTimestep = cute.make_fragment_like(tSCgTimestep, tSCgTimestep.element_type)
        copy_if(tSHgTable, tSHrTable)
        copy_if(tSHgTimestep, tSHrTimestep)
        copy_if(tSCgTable, tSCrTable)
        copy_if(tSCgTimestep, tSCrTimestep)

        tSHr = cute.make_fragment_like(tXrX)
        tSCr = cute.make_fragment_like(tXrX)
        tSHr.store(
            (
                tSHrTable.load().to(tSHr.element_type)
                + tSHrTimestep.load().to(tSHr.element_type)
            ).to(tSHr.element_type)
        )
        tSCr.store(
            (
                tSCrTable.load().to(tSCr.element_type)
                + tSCrTimestep.load().to(tSCr.element_type)
            ).to(tSCr.element_type)
        )

        value = tNrN.load()
        value = value * (1 + tSCr.load())
        value = value + tSHr.load()
        tYrY.store(value.to(tYrY.element_type))
        copy_if(tYrY, tYgY)


def _validate_residual_inputs(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    timestep: torch.Tensor,
    scale_shift_table: torch.Tensor,
    shift_index: int,
    scale_index: int,
) -> None:
    _validate_inputs(x, timestep, scale_shift_table, shift_index, scale_index)
    if residual.shape != x.shape or residual.dtype != x.dtype or not residual.is_cuda:
        raise ValueError("residual must match x shape/dtype/device")
    if gate.shape != x.shape or gate.dtype != x.dtype or not gate.is_cuda:
        raise ValueError("gate must match x shape/dtype/device")
    if residual.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("residual/gate must be last-dim contiguous")


@torch.library.custom_op(
    "sglang::fused_scale_residual_norm_ada_scale_shift", mutates_args=()
)
def fused_scale_residual_norm_ada_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    timestep: torch.Tensor,
    scale_shift_table: torch.Tensor,
    shift_index: int,
    scale_index: int,
    norm_type: str,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_residual_inputs(
        residual, x, gate, timestep, scale_shift_table, shift_index, scale_index
    )
    if norm_type not in ("layer", "rms"):
        raise ValueError('norm_type must be one of "layer" and "rms"')

    y = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    batch, seq, hidden = x.shape
    total_params = int(scale_shift_table.shape[0])
    timestep_3d = timestep.reshape(batch * seq, total_params, hidden)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    hash_key = ScaleResidualNormAdaScaleShift.make_hash_key(
        norm_type,
        int(shift_index),
        int(scale_index),
        y,
        residual_out,
        residual,
        x,
        gate,
        timestep_3d,
        scale_shift_table,
    )
    compiled_fn = _COMPILE_CACHE.get(hash_key)
    if compiled_fn is None:
        kernel = ScaleResidualNormAdaScaleShift(
            x.shape[-1], norm_type, int(shift_index), int(scale_index)
        )
        fake_sig_args = [
            to_fake_cute_args(t)
            for t in (
                y,
                residual_out,
                residual,
                x,
                gate,
                timestep_3d,
                scale_shift_table,
            )
        ]
        compiled_fn = cute.compile(kernel, *fake_sig_args, options="--enable-tvm-ffi")
        _COMPILE_CACHE[hash_key] = compiled_fn

    compiled_fn(
        y, residual_out, residual, x, gate, timestep_3d, scale_shift_table, eps, stream
    )
    return y, residual_out


@fused_scale_residual_norm_ada_scale_shift.register_fake
def _fused_scale_residual_norm_ada_scale_shift_fake(
    residual,
    x,
    gate,
    timestep,
    scale_shift_table,
    shift_index,
    scale_index,
    norm_type,
    eps=1e-5,
):
    return x.new_empty(x.shape), residual.new_empty(residual.shape)


def fused_scale_residual_norm_ada_scale_shift_chunked(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    timestep: torch.Tensor,
    scale_shift_table: torch.Tensor,
    shift_index: int,
    scale_index: int,
    norm_type: str,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_residual_inputs(
        residual, x, gate, timestep, scale_shift_table, shift_index, scale_index
    )
    batch, seq, hidden = x.shape
    total_params = int(scale_shift_table.shape[0])
    max_seq = max(1, _CUTE_INT32_INDEX_LIMIT // (batch * total_params * hidden))
    if seq <= max_seq:
        return fused_scale_residual_norm_ada_scale_shift(
            residual,
            x,
            gate,
            timestep,
            scale_shift_table,
            shift_index,
            scale_index,
            norm_type,
            eps,
        )

    y_chunks = []
    residual_chunks = []
    for start in range(0, seq, max_seq):
        end = min(seq, start + max_seq)
        y_chunk, residual_chunk = fused_scale_residual_norm_ada_scale_shift(
            residual[:, start:end, :],
            x[:, start:end, :],
            gate[:, start:end, :],
            timestep[:, start:end, :],
            scale_shift_table,
            shift_index,
            scale_index,
            norm_type,
            eps,
        )
        y_chunks.append(y_chunk)
        residual_chunks.append(residual_chunk)
    return torch.cat(y_chunks, dim=1), torch.cat(residual_chunks, dim=1)
