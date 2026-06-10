"""Differentiable halo exchange for temporal convolutions under Context Parallel.

When the temporal sequence is sharded across CP ranks, causal convolutions
of kernel size K need K-1 frames of left context from the previous rank.
Bidirectional convolutions additionally need right context from the next rank.

This module provides ``cp_halo_exchange``, a differentiable primitive that
uses ``torch.distributed.batch_isend_irecv`` for P2P communication on the
CP process group.  Gradients flow back correctly through the same channel.

Safety with FSDP2: FSDP2 uses stream-to-stream synchronization (not
``recordStream``), so P2P ops on a separate CP group are inherently safe
and will not cause deadlocks.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import P2POp, ProcessGroup

from diffusion.distributed.context_parallel.config import get_cp_halo_impl


def _to_global_rank(group: ProcessGroup, local_rank: int) -> int:
    """Convert a group-local rank to its global rank."""
    return dist.get_global_rank(group, local_rank)


def _allgather_tensor(inp: Tensor, group: ProcessGroup) -> Tensor:
    """All-gather helper returning ``(world, *inp.shape)``."""
    world = dist.get_world_size(group)
    inp_contig = inp.contiguous()
    flat_out = torch.empty(
        (world * inp_contig.shape[0],) + tuple(inp_contig.shape[1:]),
        dtype=inp_contig.dtype,
        device=inp_contig.device,
    )
    dist.all_gather_into_tensor(flat_out, inp_contig, group=group)
    return flat_out.reshape((world,) + tuple(inp_contig.shape))


class _CPHaloExchange(torch.autograd.Function):
    """Differentiable halo exchange via P2P send/recv.

    Forward:
        Rank r sends its first ``right_size`` slices to rank r-1 (as their
        right halo) and its last ``left_size`` slices to rank r+1 (as their
        left halo).  Conversely, it receives left halo from rank r-1 and
        right halo from rank r+1.

    Backward:
        Gradients are routed back via the reverse P2P direction.
    """

    @staticmethod
    def forward(
        ctx: object,
        x: Tensor,
        left_size: int,
        right_size: int,
        dim: int,
        group: ProcessGroup,
    ) -> Tensor:
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)

        ctx.left_size = left_size
        ctx.right_size = right_size
        ctx.dim = dim
        ctx.group = group
        ctx.rank = rank
        ctx.world = world

        T = x.shape[dim]

        left_recv = torch.zeros_like(x.narrow(dim, 0, left_size)) if left_size > 0 else None
        right_recv = torch.zeros_like(x.narrow(dim, 0, right_size)) if right_size > 0 else None

        halo_impl = get_cp_halo_impl()
        if halo_impl == "p2p":
            ops: list[P2POp] = []

            if left_size > 0:
                if rank > 0:
                    peer = _to_global_rank(group, rank - 1)
                    ops.append(P2POp(dist.irecv, left_recv, peer, group=group))
                if rank < world - 1:
                    send_buf = x.narrow(dim, T - left_size, left_size).contiguous()
                    peer = _to_global_rank(group, rank + 1)
                    ops.append(P2POp(dist.isend, send_buf, peer, group=group))

            if right_size > 0:
                if rank < world - 1:
                    peer = _to_global_rank(group, rank + 1)
                    ops.append(P2POp(dist.irecv, right_recv, peer, group=group))
                if rank > 0:
                    send_buf = x.narrow(dim, 0, right_size).contiguous()
                    peer = _to_global_rank(group, rank - 1)
                    ops.append(P2POp(dist.isend, send_buf, peer, group=group))

            if ops:
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
        else:
            # Deterministic collective path: all ranks always participate in
            # the same collectives, which is safer under FSDP2 overlap.
            if left_size > 0:
                send_left = x.narrow(dim, T - left_size, left_size).contiguous()
                gathered_left = _allgather_tensor(send_left, group)
                left_recv = gathered_left[rank - 1].clone() if rank > 0 else torch.zeros_like(send_left)
            if right_size > 0:
                send_right = x.narrow(dim, 0, right_size).contiguous()
                gathered_right = _allgather_tensor(send_right, group)
                right_recv = gathered_right[rank + 1].clone() if rank < world - 1 else torch.zeros_like(send_right)

        parts: list[Tensor] = []
        if left_size > 0:
            parts.append(left_recv)
        parts.append(x)
        if right_size > 0:
            parts.append(right_recv)

        out = torch.cat(parts, dim=dim)
        return out

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        left_size = ctx.left_size
        right_size = ctx.right_size
        dim = ctx.dim
        group = ctx.group
        rank = ctx.rank
        world = ctx.world

        T_with_halo = grad_output.shape[dim]
        T_local = T_with_halo - left_size - right_size

        grad_left = grad_output.narrow(dim, 0, left_size) if left_size > 0 else None
        grad_local = grad_output.narrow(dim, left_size, T_local)
        grad_right = grad_output.narrow(dim, left_size + T_local, right_size) if right_size > 0 else None

        recv_from_left = (
            torch.zeros_like(grad_local.narrow(dim, T_local - left_size, left_size)) if left_size > 0 else None
        )
        recv_from_right = torch.zeros_like(grad_local.narrow(dim, 0, right_size)) if right_size > 0 else None

        halo_impl = get_cp_halo_impl()
        if halo_impl == "p2p":
            ops: list[P2POp] = []

            if left_size > 0:
                if rank > 0:
                    peer = _to_global_rank(group, rank - 1)
                    ops.append(P2POp(dist.isend, grad_left.contiguous(), peer, group=group))
                if rank < world - 1:
                    peer = _to_global_rank(group, rank + 1)
                    ops.append(P2POp(dist.irecv, recv_from_left, peer, group=group))

            if right_size > 0:
                if rank < world - 1:
                    peer = _to_global_rank(group, rank + 1)
                    ops.append(P2POp(dist.isend, grad_right.contiguous(), peer, group=group))
                if rank > 0:
                    peer = _to_global_rank(group, rank - 1)
                    ops.append(P2POp(dist.irecv, recv_from_right, peer, group=group))

            if ops:
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
        else:
            # Collective gradient routing mirrors forward neighbor selection.
            if left_size > 0:
                gathered_grad_left = _allgather_tensor(grad_left.contiguous(), group)
                recv_from_left = (
                    gathered_grad_left[rank + 1].clone() if rank < world - 1 else torch.zeros_like(recv_from_left)
                )
            if right_size > 0:
                gathered_grad_right = _allgather_tensor(grad_right.contiguous(), group)
                recv_from_right = (
                    gathered_grad_right[rank - 1].clone() if rank > 0 else torch.zeros_like(recv_from_right)
                )

        grad_x = grad_local.clone()
        if left_size > 0 and recv_from_left is not None:
            grad_x.narrow(dim, T_local - left_size, left_size).add_(recv_from_left)
        if right_size > 0 and recv_from_right is not None:
            grad_x.narrow(dim, 0, right_size).add_(recv_from_right)

        return grad_x, None, None, None, None


def cp_halo_exchange(
    x: Tensor,
    left_size: int,
    right_size: int,
    dim: int,
    group: ProcessGroup,
) -> Tensor:
    """Exchange halo regions between CP ranks along the given dimension.

    Args:
        x: Local tensor shard.
        left_size: Number of slices to receive from the left neighbor
            (appended before ``x`` along ``dim``). Rank 0 gets zero-padding.
        right_size: Number of slices to receive from the right neighbor
            (appended after ``x`` along ``dim``). Last rank gets zero-padding.
        dim: Dimension along which to exchange halos.
        group: CP process group.

    Returns:
        Tensor with shape ``x.shape[dim] + left_size + right_size`` along
        ``dim``, where boundary halos are filled from neighbors (or zeros
        for edge ranks).
    """
    if left_size == 0 and right_size == 0:
        return x
    return _CPHaloExchange.apply(x, left_size, right_size, dim, group)
