"""Unified parallel process-group helpers and sharded tensor primitives.

This module keeps sequence/context parallel group state, all-to-all helpers,
autograd wrappers for sequence sharding, temporal-convolution context-parallel
utilities, distributed-forward input synchronization, and main-rank collectives.
"""

import contextlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union
import pickle

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class UnifiedParallelTopology:
    """One eagerly-created UP topology as seen by the current world rank."""

    size: int
    group: dist.ProcessGroup
    main_group: dist.ProcessGroup


_UNIFIED_PARALLEL_REGISTRY: dict[int, UnifiedParallelTopology] = {}
_ACTIVE_UNIFIED_PARALLEL_TOPOLOGY: Optional[UnifiedParallelTopology] = None

_SEQ_DATA_BUF = defaultdict(lambda: [None, None, None])
_SEQ_DATA_META_SHAPES = defaultdict()
_SEQ_DATA_META_DTYPES = defaultdict()
_SEQ_DATA_ASYNC_COMMS = defaultdict(list)


def slice_tensor(tensor, dim, start, end):
    indices = slice(start, end)
    return tensor[(slice(None),) * dim + (indices,)]


def init_unified_parallel(options, active_size):
    """Eagerly register UP sizes on every world rank and activate one.

    Sizes are de-duplicated and created in sorted order. For each size, every
    contiguous UP group is created first, followed by its group-main process
    group.
    """
    global _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY

    assert not _UNIFIED_PARALLEL_REGISTRY, "unified parallel is already initialized"
    assert dist.is_initialized()
    options = (options,) if isinstance(options, int) else tuple(options)
    assert options, "at least one unified-parallel size is required"
    assert all(type(size) is int and size > 0 for size in options)
    options = tuple(sorted(set(options)))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert all(world_size % size == 0 for size in options)
    assert active_size in options

    for size in options:
        local_group = None
        for start_rank in range(0, world_size, size):
            ranks = range(start_rank, start_rank + size)
            group = dist.new_group(ranks)
            if rank in ranks:
                local_group = group

        main_ranks = range(0, world_size, size)
        main_group = dist.new_group(main_ranks)
        assert local_group is not None
        _UNIFIED_PARALLEL_REGISTRY[size] = UnifiedParallelTopology(size, local_group, main_group)

    _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY = _UNIFIED_PARALLEL_REGISTRY[active_size]


@contextlib.contextmanager
def use_unified_parallel(size: Optional[int]):
    """Collectively switch active topology for the dynamic extent of the context."""
    global _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY

    target = None if size is None else _UNIFIED_PARALLEL_REGISTRY[size]
    previous = _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY
    dist.barrier(group=dist.group.WORLD)
    _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY = target
    try:
        yield
    finally:
        dist.barrier(group=dist.group.WORLD)
        _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY = previous


def get_unified_parallel_group():
    topology = _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY
    return topology.group if topology is not None else None


def _get_unified_parallel_main_group():
    topology = _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY
    return topology.main_group if topology is not None else None


def get_unified_parallel_rank():
    group = get_unified_parallel_group()
    return dist.get_rank(group) if group is not None else 0


def get_unified_parallel_world_size():
    group = get_unified_parallel_group()
    return dist.get_world_size(group) if group is not None else 1


def is_unified_parallel_initialized():
    return _ACTIVE_UNIFIED_PARALLEL_TOPOLOGY is not None


def pad_tensor(x: Tensor, dim: int, padding_size: int):
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def unpad_tensor(x: Tensor, dim: int, padding_size: int):
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(x: Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup, async_op: bool = False):
    """
    A function to do all-to-all on the first two dim
    """
    sp_world_size = dist.get_world_size(group)
    assert scatter_dim <= 1, "scatter_dim must be 0 or 1 when using all_to_all_single!"
    assert gather_dim <= 1, "gather_dim must be 0 or 1 when using all_to_all_single!"
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape([gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .transpose(0, 1)
            .reshape([gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
            .contiguous()
        )

    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x.contiguous(), group=group, async_op=async_op)

    if async_op:

        def wait():
            comm.wait()
            if scatter_dim == 0:
                return torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
            else:
                return output

        return wait

    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output


def all_to_all_tensor(
    x: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group, async_op)
    else:
        return _all_to_all(x, scatter_dim, gather_dim, group, async_op)


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        else:
            input_t = grad_output[0]
        # Return one gradient slot for each non-context forward argument.
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
        )


def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor,
    seq_dim: int,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
    async_op: bool = False,
):
    """
    A func to sync splited qkv tensor
    qkv_tensor: the tensor we want to do alltoall with. The last dim must
        be the projection_idx, which we will split into 3 part. After
        spliting, the gather idx will be projecttion_idx + 1
    seq_dim: gather_dim for all2all comm
    restore_shape: if True, output will has the same shape length as input
    """
    group = get_unified_parallel_group()
    if not group:
        return qkv_tensor
    world = get_unified_parallel_world_size()
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    if async_op:
        return SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
    else:
        qkv_tensor = SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)

        if restore_shape:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= world
            out_shape[-1] = qkv_proj_dim // world
            qkv_tensor = qkv_tensor.view(out_shape)

        # remove padding
        if unpadded_dim_size and unpadded_dim_size % world != 0:
            padding_size = qkv_tensor.size(seq_dim) - unpadded_dim_size
            qkv_tensor = unpad_tensor(qkv_tensor, seq_dim, padding_size)

        return qkv_tensor


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    async_op: bool = False,
    # group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    """
    group = get_unified_parallel_group()
    if not group:
        return x
    sp_world = get_unified_parallel_world_size()
    if async_op:
        return SeqAllToAll.apply(group, x, head_dim, seq_dim, async_op)
    else:
        x = SeqAllToAll.apply(group, x, head_dim, seq_dim, async_op)
        if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
            padding_size = x.size(seq_dim) - unpadded_dim_size
            x = unpad_tensor(x, seq_dim, padding_size)
        return x


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    """
    group = get_unified_parallel_group()
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_unified_parallel_world_size()
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


class Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, dim: int, scale_grad: bool) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        ctx.dim = dim
        ctx.scale_grad = scale_grad
        dim_size = local_input.shape[dim]
        if not ctx.group:
            return local_input
        return local_input.split(dim_size // seq_world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor, None]:
        if not ctx.group:
            return None, grad_output, None, None
        dim_size = list(grad_output.size())
        split_size = dim_size[0]
        dim_size[0] = dim_size[0] * ctx.seq_world_size
        output = torch.empty(dim_size, dtype=grad_output.dtype, device=torch.cuda.current_device())
        dist.all_gather_into_tensor(output, grad_output.contiguous(), group=ctx.group)
        if ctx.scale_grad:
            output = output / ctx.seq_world_size
        return (None, torch.cat(output.split(split_size), dim=ctx.dim), None, None)


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        dim: int,
        grad_scale: Optional[bool] = False,
    ) -> Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        seq_world_size = dist.get_world_size(group)
        ctx.seq_world_size = seq_world_size
        dim_size = list(local_input.size())
        split_size = dim_size[0]
        ctx.part_size = dim_size[dim]
        dim_size[0] = dim_size[0] * seq_world_size
        output = torch.empty(dim_size, dtype=local_input.dtype, device=torch.cuda.current_device())
        dist.all_gather_into_tensor(output, local_input.contiguous(), group=ctx.group)
        return torch.cat(output.split(split_size), dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        if ctx.grad_scale:
            grad_output = grad_output * ctx.seq_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.dim)[ctx.rank].contiguous(),
            None,
            None,
        )


def unpadding_tensor_for_sequence_parallel(x: Tensor, dim: int, unpadded_dim_size: int):
    """
    A func to remove the padding part of the tensor based on its original shape
    """
    group = get_unified_parallel_group()
    if group is None:
        return x
    sp_world = get_unified_parallel_world_size()
    if unpadded_dim_size % sp_world == 0:
        return x
    padding_size = sp_world - (unpadded_dim_size % sp_world)
    assert (padding_size + unpadded_dim_size) % sp_world == 0
    return unpad_tensor(x, dim=dim, padding_size=padding_size)


def gather_outputs(
    x: Tensor,
    gather_dim: int,
    padding_dim: Optional[int] = None,
    unpad_dim_size: Optional[int] = None,
    scale_grad=True,
):
    """
    A func to gather the outputs for the model result in sequence parallel
    """
    group = get_unified_parallel_group()
    if not group:
        return x
    x = Gather.apply(group, x, gather_dim, scale_grad)
    if padding_dim is not None:
        x = unpadding_tensor_for_sequence_parallel(x, padding_dim, unpad_dim_size)
    return x


def _conv_gather(input_, dim, frame_length):
    cp_size = get_unified_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_size == 1:
        return input_

    group = get_unified_parallel_group()
    cp_rank = get_unified_parallel_rank()

    split_size = [None for _ in range(cp_size)]
    dist.all_gather_object(split_size, input_.size(dim), group=group)

    max_input_size = list(input_.shape[:dim]) + [max(split_size)] + list(input_.shape[dim + 1 :])
    tensor_list = [torch.empty(size=max_input_size, dtype=input_.dtype, device=input_.device)
                   for _ in range(cp_size)]

    slc = [[slice(None)] * dim + [slice(0, size)] + \
           [slice(None)] * (len(input_.shape) - dim - 1) for size in split_size]
    tensor_list[cp_rank][tuple(slc[cp_rank])] = input_.contiguous()
    dist.all_gather(tensor_list, tensor_list[cp_rank], group=group)
    for rank in range(cp_size):
        tensor_list[rank] = tensor_list[rank][tuple(slc[rank])]

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim)
    output = slice_tensor(output, dim, 0, frame_length)

    # output = output.contiguous()
    return output


def _conv_split(input_, dim):
    cp_size = get_unified_parallel_world_size()
    # assert cp_world_size <= input_.size(dim)

    # Bypass the function if context parallel is 1
    if cp_size == 1:
        return input_

    cp_rank = get_unified_parallel_rank()

    padding_size = input_.size(dim) % cp_size
    split_size = [(input_.size(dim) // cp_size) + (rank < padding_size) for rank in range(cp_size)]
    idxs = [sum(split_size[:i]) for i in range(cp_size + 1)]

    if split_size[cp_rank] > 0: # padding required
        output = slice_tensor(input_, dim, idxs[cp_rank], idxs[cp_rank + 1])
    else:
        output = slice_tensor(input_, dim, 0, 1)

    # output = output.contiguous()
    return output


class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, frame_length, scale_grad):
        ctx.dim = dim
        ctx.scale_grad = scale_grad
        return _conv_gather(input_, dim, frame_length)

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = get_unified_parallel_world_size()
        if ctx.scale_grad:
            grad_output = grad_output * cp_size
        return _conv_split(grad_output, ctx.dim), None, None, None


class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, scale_grad):
        ctx.dim = dim
        ctx.frame_length = input_.size(dim)
        ctx.scale_grad = scale_grad
        return _conv_split(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = get_unified_parallel_world_size()
        if ctx.scale_grad:
            grad_output = grad_output / cp_size
        return _conv_gather(grad_output, ctx.dim, ctx.frame_length), None, None


def conv_gather_from_context_parallel_region(input_, dim, frame_length, scale_grad=True):
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, frame_length, scale_grad)


def conv_scatter_to_context_parallel_region(input_, dim, scale_grad=True):
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, scale_grad)


def get_neighbor_rank(offset):
    cp_size = get_unified_parallel_world_size()
    global_rank = dist.get_rank()
    group_index = global_rank // cp_size
    cp_rank = get_unified_parallel_rank()
    return group_index * cp_size + (cp_rank + offset) % cp_size


def _pass_from_previous_rank(input_, dim, kernel_size, padding_mode="first"):
    # Make sure the input is contiguous
    input_ = input_.contiguous()

    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    cp_size = get_unified_parallel_world_size()
    cp_rank = get_unified_parallel_rank()
    gpu_group = get_unified_parallel_group()

    if cp_size == 1:
        if padding_mode == "first":
            input_ = torch.cat([slice_tensor(input_, dim, 0, 1)] * (kernel_size - 1) + [input_], dim=dim)
        elif padding_mode == "zero":
            input_ = torch.cat([torch.zeros_like(slice_tensor(input_, dim, 0, 1))] * (kernel_size - 1) + [input_], dim=dim)
        elif padding_mode == "none":
            input_ = input_
        return input_

    send_dst = get_neighbor_rank(1)
    recv_src = get_neighbor_rank(-1)

    frame_length = input_.shape[dim]
    assert padding_mode != "none" or frame_length >= kernel_size - 1, (
        "padding_mode='none' requires frame_length >= kernel_size - 1 because no left padding is available"
    )
    memory_list = [input_.new_empty((*input_.shape[:dim], 1, *input_.shape[dim + 1:]))
                   for i in range(kernel_size - 1)]

    if cp_rank == 0:
        padding_tensor = None
        if padding_mode == "first":
            padding_tensor = slice_tensor(input_, dim, 0, 1)
        elif padding_mode == "zero":
            padding_tensor = torch.zeros_like(slice_tensor(input_, dim, 0, 1))

        for i in range(1, kernel_size):
            send_tensor = slice_tensor(input_, dim, frame_length - i, frame_length - i + 1).clone() \
                if (i <= frame_length) else padding_tensor.clone()
            recv_tensor = memory_list[kernel_size - 1 - i]

            send_op = dist.P2POp(dist.isend, send_tensor, send_dst, group=gpu_group)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_src, group=gpu_group)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

        if padding_tensor is not None:
            input_ = torch.cat([padding_tensor] * (kernel_size - 1) + [input_], dim=dim)
    else:
        for i in range(1, kernel_size):
            send_tensor = slice_tensor(input_, dim, frame_length - i, frame_length - i + 1).clone() \
                if (i <= frame_length) else memory_list[(kernel_size - 1) - (i - frame_length)].clone()
            recv_tensor = memory_list[kernel_size - 1 - i]

            send_op = dist.P2POp(dist.isend, send_tensor, send_dst, group=gpu_group)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_src, group=gpu_group)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

        input_ = torch.cat([*memory_list, input_], dim=dim)

    return input_


def _drop_from_previous_rank(input_, dim, kernel_size, padding_mode):
    # Make sure the input is contiguous
    input_ = input_.contiguous()

    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    cp_size = get_unified_parallel_world_size()
    cp_rank = get_unified_parallel_rank()
    gpu_group = get_unified_parallel_group()

    frame_length = input_.shape[dim]
    if cp_size > 1:
        send_dst = get_neighbor_rank(-1)
        recv_src = get_neighbor_rank(1)

        if cp_rank != cp_size - 1:
            input_ = input_.clone()

        for i in range(0, kernel_size - 1):
            send_tensor = slice_tensor(input_, dim, i, i + 1).clone()
            recv_tensor = input_.new_empty((*input_.shape[:dim], 1, *input_.shape[dim + 1:]))

            send_op = dist.P2POp(dist.isend, send_tensor, send_dst, group=gpu_group)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_src, group=gpu_group)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

            if cp_rank != cp_size - 1:
                slice_tensor(input_, dim, frame_length - kernel_size + 1 + i,
                             frame_length - kernel_size + 1 + (i + 1)).add_(recv_tensor)

    if padding_mode == "none" and (cp_size == 1 or cp_rank == 0):
        return input_

    if padding_mode == "first" and (cp_size == 1 or cp_rank == 0):
        pad_grad = slice_tensor(input_, dim, 0, kernel_size - 1).sum(dim=dim, keepdim=True)
        out = slice_tensor(input_, dim, kernel_size - 1, frame_length).clone()
        slice_tensor(out, dim, 0, 1).add_(pad_grad)
        return out

    return slice_tensor(input_, dim, kernel_size - 1, frame_length)


class _ConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size, padding_mode="first"):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        ctx.padding_mode = padding_mode
        return _pass_from_previous_rank(input_, dim, kernel_size, padding_mode)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size, ctx.padding_mode), None, None, None


def conv_pass_from_last_rank(input_, dim, kernel_size, padding_mode="first"):
    return _ConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size, padding_mode)


def _conv_all_to_all(local_input, scatter_dim, gather_dim, gather_size):
    cp_size = get_unified_parallel_world_size()

    if cp_size == 1:
        return local_input, None, None

    scatter_padding_size = local_input.size(scatter_dim) % cp_size
    scatter_split_size = [local_input.size(scatter_dim) // cp_size + (
        rank < scatter_padding_size) for rank in range(cp_size)]
    idxs = [sum(scatter_split_size[:i]) for i in range(cp_size + 1)]

    input_list = []
    for rank in range(cp_size):
        if scatter_split_size[rank] > 0:
            input_list.append(slice_tensor(local_input, scatter_dim, idxs[rank], idxs[rank + 1]))
        else:
            input_list.append(slice_tensor(local_input, scatter_dim, 0, 1).clone()) # padding
    input_list = [input_.contiguous() for input_ in input_list]

    group = get_unified_parallel_group()
    cp_rank = get_unified_parallel_rank()

    gather_split_size = [None for _ in range(cp_size)]
    dist.all_gather_object(gather_split_size, local_input.size(gather_dim), group=group)
    output_list = []
    for rank in range(cp_size):
        output_size = list(local_input.shape)
        output_size[gather_dim] = gather_split_size[rank]
        output_size[scatter_dim] = scatter_split_size[cp_rank] if scatter_split_size[cp_rank] > 0 else 1
        output_list.append(input_list[rank].new_empty(output_size))

    dist.all_to_all(output_list, input_list, group=group)
    output = torch.cat(output_list, dim=gather_dim)
    output = slice_tensor(output, gather_dim, 0, gather_size)

    if scatter_split_size[cp_rank] > 0:
        start, end = idxs[cp_rank], idxs[cp_rank + 1]
    else:
        start, end = 0, 1
    return output, start, end


class _AllToAllContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim, gather_size):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.bwd_gather_size = input_.size(ctx.scatter_dim)
        return _conv_all_to_all(input_, scatter_dim, gather_dim, gather_size)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_output = grad_output[0]
        grad_output, _, _ = _conv_all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.bwd_gather_size)
        return grad_output, None, None, None


def all_to_all_context_parallel_region(input_, scatter_dim, gather_dim, gather_size):
    return _AllToAllContextParallelRegion.apply(input_, scatter_dim, gather_dim, gather_size)


def all_to_all_split(input_list, output_list):
    # 初始化一个全0的矩阵
    num_ranks = len(input_list)
    input_split = np.zeros((num_ranks, num_ranks), dtype=int)

    # 用来记录每个rank处理的偏移量
    input_offset = [0] * num_ranks
    output_offset = [0] * num_ranks

    for i in range(num_ranks):
        # 当前rank i的input数据总量
        remaining_input = input_list[i]

        for j in range(num_ranks):
            # rank i发给rank j的数据量
            send_amount = min(remaining_input, output_list[j] - output_offset[j])
            input_split[i][j] = send_amount
            # 更新remaining_input和output_offset
            remaining_input -= send_amount
            output_offset[j] += send_amount

            if remaining_input == 0:
                break

    return input_split


def _conv_all_to_all_single(local_input, input_split, dim):
    cp_size = get_unified_parallel_world_size()

    if cp_size == 1:
        return local_input

    gpu_group = get_unified_parallel_group()
    cp_rank = get_unified_parallel_rank()

    output_split = input_split.transpose().tolist()
    input_split = input_split.tolist()
    output_length = sum(output_split[cp_rank])

    local_input = local_input.transpose(dim, 0).contiguous()
    output_tensor = local_input.new_empty((output_length, *local_input.shape[1:]))
    dist.all_to_all_single(output_tensor, local_input, output_split[cp_rank], input_split[cp_rank], group=gpu_group)
    output_tensor = output_tensor.transpose(0, dim).contiguous()
    return output_tensor


class _AllToAllSingleContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, input_split, dim):
        ctx.input_split = input_split
        ctx.dim = dim
        return _conv_all_to_all_single(input_, input_split, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_all_to_all_single(grad_output, ctx.input_split.transpose(), ctx.dim), None, None


def all_to_all_single_context_parallel_region(input_, dim, output_list=None, seq_length=None):
    cp_size = get_unified_parallel_world_size()
    group = get_unified_parallel_group()

    if not group:
        return input_

    split_size = [None for _ in range(cp_size)]
    dist.all_gather_object(split_size, input_.size(dim), group=group)
    seq_length = seq_length

    if output_list is None:
        padding_size = seq_length % cp_size
        output_list = [max(1, (seq_length // cp_size) + (rank < padding_size)) for rank in range(cp_size)]
    assert sum(output_list) == sum(split_size), f"all_to_all_single_context_parallel_region: sum(output_list)!="\
                                                f"sum(split_size), got {sum(output_list)} vs {sum(split_size)}"

    input_split = all_to_all_split(split_size, output_list)
    if np.array_equal(input_split, np.diag(np.diag(input_split))):
        return input_

    return _AllToAllSingleContextParallelRegion.apply(input_, input_split, dim)


class MetaTuple(tuple):
    def __new__(cls, values):
        return super().__new__(cls, values)

    def __init__(self, values):
        self._use_scatter = False
        self._scatter_dim = None
        self.values = values


def mark_sp(x: Tensor, scatter_dim: int):
    """
    Mark tensor to be used in sequence parallel
    """
    x._use_scatter = True
    x._scatter_dim = scatter_dim
    return x


def _construct_sync_buffer(shapes, dtypes, device):
    if isinstance(shapes, (torch.Size, MetaTuple)):
        if getattr(shapes, "_use_scatter", False):
            scatter_dim = shapes._scatter_dim
            sp_size = get_unified_parallel_world_size()
            shapes = list(shapes)
            shapes[scatter_dim] = math.ceil(shapes[scatter_dim] / sp_size)
            return mark_sp(torch.empty(shapes, dtype=dtypes, device=device), scatter_dim)
        return torch.empty(shapes, dtype=dtypes, device=device)

    if isinstance(shapes, list):
        buffer = [_construct_sync_buffer(sub_shape, dtypes[i], device) for i, sub_shape in enumerate(shapes)]
    elif isinstance(shapes, tuple):
        buffer = tuple(_construct_sync_buffer(sub_shape, dtypes[i], device) for i, sub_shape in enumerate(shapes))
    elif isinstance(shapes, Mapping):
        buffer = shapes.__class__(
            {key: _construct_sync_buffer(sub_shape, dtypes[key], device) for key, sub_shape in shapes.items()}
        )
    else:
        return shapes
    return buffer


def _traverse(data: Any, op: Callable) -> Union[None, List, Mapping, Any]:
    if isinstance(data, list):
        return [_traverse(sub_data, op) for sub_data in data]
    if isinstance(data, tuple):
        return tuple([_traverse(sub_data, op) for sub_data in data])
    elif isinstance(data, Mapping):
        return data.__class__({key: _traverse(sub_data, op) for key, sub_data in data.items()})
    elif isinstance(data, Tensor):
        return op(data)
    else:
        return data


def shape_with_meta(data: Tensor):
    shape = data.shape
    if getattr(data, "_use_scatter", False):
        shape = MetaTuple(shape)
        shape._use_scatter = True
        shape._scatter_dim = data._scatter_dim
    return shape


def _get_shapes(data):
    return _traverse(data, op=shape_with_meta)


def _get_dtypes(data):
    return _traverse(data, op=lambda x: x.dtype)


def _sync_data_in_group(data, shape, dtype, src, is_src, group, async_op, return_storage=False):
    comms = []
    storages = []
    if isinstance(data, (list, tuple)):
        for i, sub_shape in enumerate(shape):
            sub_comms, sub_storages = _sync_data_in_group(
                data[i], sub_shape, dtype[i], src, is_src, group, async_op, True
            )
            comms.extend(sub_comms)
            storages.extend(sub_storages)
    elif isinstance(data, Mapping):
        for key, sub_data in data.items():
            sub_comms, sub_storages = _sync_data_in_group(
                sub_data, shape[key], dtype[key], src, is_src, group, async_op, True
            )
            comms.extend(sub_comms)
            storages.extend(sub_storages)
    elif isinstance(data, Tensor):
        if getattr(shape, "_use_scatter", False):
            sp_size = get_unified_parallel_world_size()
            if is_src:
                scatter_dim = shape._scatter_dim
                # scatter will just use the tensor storage, so contiguous() is a must
                scatter_list = [
                    t.contiguous() for t in pad_tensor(data, scatter_dim, sp_size).chunk(sp_size, scatter_dim)
                ]
                data.set_(torch.empty_like(scatter_list[get_unified_parallel_rank()]))
            else:
                scatter_list = None
            data._unpad_shape = torch.Size(shape)
            if int(os.environ.get("DIST_ATTN_SYNC_SCATTER", 0)):
                torch.distributed.scatter(data, scatter_list, src=src, group=group, async_op=False)
                del scatter_list
            else:
                comms.append(torch.distributed.scatter(data, scatter_list, src=src, group=group, async_op=async_op))
                if is_src:
                    for r, data in enumerate(scatter_list):
                        if r != get_unified_parallel_rank():
                            storages.append(data.untyped_storage())
        else:
            data = data.contiguous()
            comms.append(torch.distributed.broadcast(data, src=src, group=group, async_op=async_op))
    if return_storage:
        return comms, storages
    return comms


class SPDistForward:
    """A forward tool to sync different result across sp group

    Args:
        module: a function or module to process users input
        sp_step: current training step to judge which rank to broadcast its result to all
        name: a distinct str to save meta and async comm
        comm_shape: if different ranks have different shape, mark this arg to True
        start_rank: which sp rank we start to loop
        device: the device for current rank, can be empty
    """

    def __init__(
        self,
        name: str,
        comm_shape: bool,
        start_rank: int = 0,
        device: torch.device = None,
    ):
        self.name = name
        self.comm_shape = comm_shape
        self.start_rank = start_rank
        if device:
            self.device = device
        else:
            self.device = torch.cuda.current_device()

    def _has_tensor(self, data):
        if isinstance(data, (list, tuple)):
            return any([self._has_tensor(sub_data) for sub_data in data])
        elif isinstance(data, Mapping):
            return any([self._has_tensor(sub_data) for sub_data in data.values()])
        elif isinstance(data, Tensor):
            return True
        else:
            return False

    def encode_bytes(self, data):
        if isinstance(data, list) and self._has_tensor(data):
            return [self.encode_bytes(sub_data) for sub_data in data]
        if isinstance(data, tuple) and self._has_tensor(data):
            return tuple([self.encode_bytes(sub_data) for sub_data in data])
        elif isinstance(data, Mapping) and self._has_tensor(data):
            return data.__class__({key: self.encode_bytes(sub_data) for key, sub_data in data.items()})
        elif isinstance(data, Tensor):
            assert data.dtype != torch.uint8, f"no support of uint8 dtype"
            return data
        else:
            return torch.tensor(list(pickle.dumps(data)), dtype=torch.uint8, device=self.device)

    def decode_bytes(self, data):
        if isinstance(data, list):
            return [self.decode_bytes(sub_data) for sub_data in data]
        if isinstance(data, tuple):
            return tuple([self.decode_bytes(sub_data) for sub_data in data])
        elif isinstance(data, Mapping):
            return data.__class__({key: self.decode_bytes(sub_data) for key, sub_data in data.items()})
        elif isinstance(data, Tensor) and data.dtype == torch.uint8:
            return pickle.loads(bytes(data.tolist()))
        else:
            return data

    def __call__(self, inputs) -> Any:
        group = get_unified_parallel_group()
        if not group:
            yield inputs
        else:
            inputs = self.encode_bytes(inputs)
            device = self.device
            sp_world = get_unified_parallel_world_size()
            sp_rank = get_unified_parallel_rank()
            for local_step in range(sp_world):
                local_step = (local_step + self.start_rank) % sp_world
                src_rank = dist.get_global_rank(group, local_step)
                is_src = sp_rank == local_step
                local_shapes = []
                local_dtypes = []
                if local_step == self.start_rank:
                    # we sync shape and dtype inside the group in the first step
                    local_result = inputs
                    _SEQ_DATA_BUF[self.name][-1] = local_result
                    local_shapes = _get_shapes(local_result)
                    local_dtypes = _get_dtypes(local_result)
                    if self.comm_shape:
                        group_shapes_lists = [None] * sp_world
                        group_dtypes_lists = [None] * sp_world
                        # dist.all_gather_object(group_shapes_lists, local_shapes, group=group)
                        dist.all_gather_object(group_shapes_lists, local_shapes, group=group)
                        _SEQ_DATA_META_SHAPES[self.name] = group_shapes_lists
                        dist.all_gather_object(group_dtypes_lists, local_dtypes, group=group)
                        _SEQ_DATA_META_DTYPES[self.name] = group_dtypes_lists
                    else:
                        _SEQ_DATA_META_SHAPES[self.name] = [local_shapes] * sp_world
                        _SEQ_DATA_META_DTYPES[self.name] = [local_dtypes] * sp_world

                shapes = _SEQ_DATA_META_SHAPES[self.name][local_step]
                dtypes = _SEQ_DATA_META_DTYPES[self.name][local_step]
                buf_id = local_step % 2
                if local_step == self.start_rank:
                    # sync data in the first step, async in other steps
                    sync_sp_data = local_result if is_src else _construct_sync_buffer(shapes, dtypes, device)
                    _sync_data_in_group(sync_sp_data, shapes, dtypes, src_rank, is_src, group, False)
                    _SEQ_DATA_BUF[self.name][buf_id] = sync_sp_data

                # wait for async comm ops
                if _SEQ_DATA_ASYNC_COMMS[self.name]:
                    for comm in _SEQ_DATA_ASYNC_COMMS[self.name]:
                        comm.wait()

                # before return the sync result, do async broadcast for next batch
                if local_step != (self.start_rank - 1) % sp_world:
                    next_buf_id = 1 - buf_id
                    shapes = _SEQ_DATA_META_SHAPES[self.name][local_step + 1]
                    dtypes = _SEQ_DATA_META_DTYPES[self.name][local_step + 1]
                    src_rank = dist.get_global_rank(group, local_step + 1)
                    is_src = sp_rank == local_step + 1
                    next_sync_data = (
                        _SEQ_DATA_BUF[self.name][-1] if is_src else _construct_sync_buffer(shapes, dtypes, device)
                    )
                    _SEQ_DATA_ASYNC_COMMS[self.name] = _sync_data_in_group(
                        next_sync_data, shapes, dtypes, src_rank, is_src, group, True
                    )
                    _SEQ_DATA_BUF[self.name][next_buf_id] = next_sync_data
                yield self.decode_bytes(_SEQ_DATA_BUF[self.name][buf_id])


def gather_main(tensor, gather_list, dst=0):
    main_group = _get_unified_parallel_main_group()
    if not dist.is_available() or not dist.is_initialized():
        assert len(gather_list) == 1, f"gather_list size must be 1 if dist is not initialized"
        gather_list[0] = tensor
        return
    elif main_group is None:
        dist.gather(tensor, gather_list=gather_list, dst=dst)
    elif main_group == dist.GroupMember.NON_GROUP_MEMBER:
        return None
    else:
        return dist.gather(tensor, gather_list=gather_list, dst=dst, group=main_group)


def gather_object_main(obj, dst=0):
    main_group = _get_unified_parallel_main_group()
    if not dist.is_available() or not dist.is_initialized():
        return [obj]
    elif main_group is None:
        object_list = [None for _ in range(dist.get_world_size())] if dist.get_rank() == dst else None
        dist.gather_object(obj, object_list, dst=dst)
        return object_list
    elif main_group == dist.GroupMember.NON_GROUP_MEMBER:
        return None
    object_list = (
        [None for _ in range(dist.get_world_size(group=main_group))]
        if dist.get_rank() == dst
        else None
    )
    dist.gather_object(obj, object_list, group=main_group, dst=dst)
    return object_list


def all_gather_main(tensor, gather_list):
    main_group = _get_unified_parallel_main_group()
    if not dist.is_available() or not dist.is_initialized():
        assert len(gather_list) == 1, f"gather_list size must be 1 if dist is not initialized"
        gather_list[0] = tensor
        return
    elif main_group is None:
        dist.all_gather(tensor_list=gather_list, tensor=tensor)
    elif main_group == dist.GroupMember.NON_GROUP_MEMBER:
        return None
    else:
        return dist.all_gather(tensor_list=gather_list, tensor=tensor, group=main_group)


def all_gather_object_main(obj):
    main_group = _get_unified_parallel_main_group()
    if not dist.is_available() or not dist.is_initialized():
        return [obj]
    elif main_group is None:
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(object_list=object_list, obj=obj)
        return object_list
    elif main_group == dist.GroupMember.NON_GROUP_MEMBER:
        return None
    else:
        object_list = [None for _ in range(dist.get_world_size(group=main_group))]
        dist.all_gather_object(object_list=object_list, obj=obj, group=main_group)
        return object_list
