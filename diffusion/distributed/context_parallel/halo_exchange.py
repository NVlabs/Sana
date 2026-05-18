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

from diffusion.distributed.context_parallel.config import (
    get_cp_halo_impl,
    get_cp_halo_trace,
    get_cp_halo_trace_limit,
)

_CP_HALO_TRACE_COUNTER = 0


def _halo_trace(msg: str, group: ProcessGroup | None = None) -> None:
    """Env-gated trace printer for halo exchange.

    Enable with:
        CP_HALO_TRACE=1
    Optional:
        CP_HALO_TRACE_LIMIT=<int> (default: 400)
    """
    if not get_cp_halo_trace():
        return

    global _CP_HALO_TRACE_COUNTER
    _CP_HALO_TRACE_COUNTER += 1

    limit = get_cp_halo_trace_limit()
    if _CP_HALO_TRACE_COUNTER > limit:
        return

    global_rank = -1
    local_rank = -1
    local_world = -1
    if dist.is_initialized():
        global_rank = dist.get_rank()
        if group is not None:
            local_rank = dist.get_rank(group)
            local_world = dist.get_world_size(group)
    print(
        f"[CP-HALO] n={_CP_HALO_TRACE_COUNTER} grank={global_rank} " f"crank={local_rank}/{local_world} {msg}",
        flush=True,
    )


def _to_global_rank(group: ProcessGroup, local_rank: int) -> int:
    """Convert a group-local rank to its global rank."""
    return dist.get_global_rank(group, local_rank)


def _allgather_tensor(inp: Tensor, group: ProcessGroup) -> Tensor:
    """All-gather helper returning ``(world, *inp.shape)``."""
    world = dist.get_world_size(group)
    out = torch.empty((world,) + tuple(inp.shape), dtype=inp.dtype, device=inp.device)
    dist.all_gather_into_tensor(out, inp.contiguous(), group=group)
    return out


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
        _halo_trace(
            f"forward enter left={left_size} right={right_size} dim={dim} "
            f"xshape={list(x.shape)} rank={rank}/{world}",
            group,
        )

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
                _halo_trace(f"forward p2p submit ops={len(ops)}", group)
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                _halo_trace("forward p2p done", group)
        else:
            # Deterministic collective path: all ranks always participate in
            # the same collectives, which is safer under FSDP2 overlap.
            if left_size > 0:
                send_left = x.narrow(dim, T - left_size, left_size).contiguous()
                _halo_trace("forward collective gather_left submit", group)
                gathered_left = _allgather_tensor(send_left, group)
                _halo_trace("forward collective gather_left done", group)
                left_recv = gathered_left[rank - 1].clone() if rank > 0 else torch.zeros_like(send_left)
            if right_size > 0:
                send_right = x.narrow(dim, 0, right_size).contiguous()
                _halo_trace("forward collective gather_right submit", group)
                gathered_right = _allgather_tensor(send_right, group)
                _halo_trace("forward collective gather_right done", group)
                right_recv = gathered_right[rank + 1].clone() if rank < world - 1 else torch.zeros_like(send_right)

        parts: list[Tensor] = []
        if left_size > 0:
            parts.append(left_recv)
        parts.append(x)
        if right_size > 0:
            parts.append(right_recv)

        out = torch.cat(parts, dim=dim)
        _halo_trace(f"forward exit outshape={list(out.shape)}", group)
        return out

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        left_size = ctx.left_size
        right_size = ctx.right_size
        dim = ctx.dim
        group = ctx.group
        rank = ctx.rank
        world = ctx.world
        _halo_trace(
            f"backward enter left={left_size} right={right_size} dim={dim} "
            f"gshape={list(grad_output.shape)} rank={rank}/{world}",
            group,
        )

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
                _halo_trace(f"backward p2p submit ops={len(ops)}", group)
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                _halo_trace("backward p2p done", group)
        else:
            # Collective gradient routing mirrors forward neighbor selection.
            if left_size > 0:
                _halo_trace("backward collective gather_grad_left submit", group)
                gathered_grad_left = _allgather_tensor(grad_left.contiguous(), group)
                _halo_trace("backward collective gather_grad_left done", group)
                recv_from_left = (
                    gathered_grad_left[rank + 1].clone() if rank < world - 1 else torch.zeros_like(recv_from_left)
                )
            if right_size > 0:
                _halo_trace("backward collective gather_grad_right submit", group)
                gathered_grad_right = _allgather_tensor(grad_right.contiguous(), group)
                _halo_trace("backward collective gather_grad_right done", group)
                recv_from_right = (
                    gathered_grad_right[rank - 1].clone() if rank > 0 else torch.zeros_like(recv_from_right)
                )

        grad_x = grad_local.clone()
        if left_size > 0 and recv_from_left is not None:
            grad_x.narrow(dim, T_local - left_size, left_size).add_(recv_from_left)
        if right_size > 0 and recv_from_right is not None:
            grad_x.narrow(dim, 0, right_size).add_(recv_from_right)

        _halo_trace(f"backward exit grad_x_shape={list(grad_x.shape)}", group)
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
