"""Distributed GDN scan with Context Parallel state correction.

Each GPU runs a local scan on its T/P frames, then corrects the results
using the true initial state obtained via all-gather + merge.

Communication is O(P * D^2) via all-gather -- symmetric collective that
avoids cross-communicator deadlocks when FSDP and Ulysses SP operate on
other NCCL process groups concurrently.

Algorithm:

    1. Local scan with S_init=0  -->  S_local[t]
    2. Cumulative transition products  -->  W_cum[t]
    3. Extract chunk composites: h_ext = S_local[-1], M = W_cum[-1]
    4. All-gather (h_ext, M) across P ranks
    5. Merge: compose predecessors to get S_init
    6. Correction: S_corrected[t] = f(S_init, W_cum[t]) + S_local[t]

The KV state uses right-multiply: S = S_prev @ W + U
The Z  state uses left-multiply:  S = W @ S_prev + U
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


class CpFrameGdnScanResult(NamedTuple):
    """Return type of :func:`cp_frame_gdn_scan` when ``truncate_to_active`` is set.

    Carries the per-position corrected scan outputs (same shape as the
    legacy 2-tuple return) plus the terminal recurrence state at logical
    global position ``truncate_to_active - 1`` (identical on every rank,
    broadcast from the owning rank).

    Note: ``NamedTuple`` iterates ALL fields when unpacked. Callers that
    use the legacy 2-tuple unpacking (``S_kv, S_z = cp_frame_gdn_scan(...)``)
    MUST NOT pass ``truncate_to_active``; instead use the default
    ``truncate_to_active=None`` path which returns a plain 2-tuple.
    """

    S_kv_all: Tensor  # (BH, T_local, D, D)
    S_z_all: Tensor  # (BH, T_local, D)
    terminal_state_kv: Tensor  # (BH, D, D), same on every rank
    terminal_state_z: Tensor  # (BH, D), same on every rank


from diffusion.distributed.context_parallel.config import (
    get_cp_allgather_impl,
    get_cp_scan_backend,
)


# ---------------------------------------------------------------------------
# Local scan backends
# ---------------------------------------------------------------------------


@torch.compile(dynamic=True)
def _pytorch_scan_compiled(
    W_kv: Tensor,
    U_kv: Tensor,
    W_z: Tensor,
    U_z: Tensor,
    S_init_kv: Tensor | None = None,
    S_init_z: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compiled local scan for CP.

    ``torch.compile`` traces through the loop and generates an efficient
    fused kernel with automatic backward differentiation. All computation
    is done in FP32 for numerical stability.
    """
    orig_dtype = W_kv.dtype
    W_kv, U_kv = W_kv.float(), U_kv.float()
    W_z, U_z = W_z.float(), U_z.float()

    BH, T, D, _ = W_kv.shape
    if S_init_kv is not None:
        S_kv = S_init_kv.float()
    else:
        S_kv = torch.zeros(BH, D, D, device=W_kv.device, dtype=torch.float32)

    if S_init_z is not None:
        S_z = S_init_z.float()
    else:
        S_z = torch.zeros(BH, D, device=U_z.device, dtype=torch.float32)

    S_kv_all = torch.empty_like(U_kv)
    S_z_all = torch.empty_like(U_z)
    for t in range(T):
        S_kv = torch.matmul(S_kv, W_kv[:, t]) + U_kv[:, t]
        S_z = torch.bmm(W_z[:, t], S_z.unsqueeze(-1)).squeeze(-1) + U_z[:, t]
        S_kv_all[:, t] = S_kv
        S_z_all[:, t] = S_z
    return S_kv_all.to(orig_dtype), S_z_all.to(orig_dtype)


class _PyTorchScan:
    """Wrapper that mimics the ``autograd.Function`` ``.apply()`` interface
    while delegating to the compiled scan function.

    ``torch.compile`` handles backward differentiation automatically, so
    a custom ``autograd.Function`` is no longer needed.
    """

    @staticmethod
    def apply(
        W_kv: Tensor,
        U_kv: Tensor,
        W_z: Tensor,
        U_z: Tensor,
        S_init_kv: Tensor | None = None,
        S_init_z: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return _pytorch_scan_compiled(W_kv, U_kv, W_z, U_z, S_init_kv, S_init_z)


def _get_local_scan_cls(device_is_cuda: bool) -> type:
    """Select the local scan implementation based on config.

    Args:
        device_is_cuda: Whether the tensors reside on a CUDA device.

    Returns:
        ``_PyTorchScan`` or ``FrameGDNScan`` (Triton) autograd Function class.
    """
    backend = get_cp_scan_backend()
    if backend == "triton":
        from diffusion.model.ops.frame_gdn.scan_triton import FrameGDNScan

        return FrameGDNScan
    return _PyTorchScan


# Keep backward-compatible alias.
get_local_scan_cls = _get_local_scan_cls


# ---------------------------------------------------------------------------
# Cumulative matrix products
# ---------------------------------------------------------------------------


@torch.compile(dynamic=True)
def _cumulative_matmul_right(W: Tensor) -> Tensor:
    """Cumulative right-multiply: W_cum[t] = W[0] @ W[1] @ ... @ W[t].

    Args:
        W: ``(BH, T, D, D)``

    Returns:
        W_cum: ``(BH, T, D, D)`` where ``W_cum[:, t]`` is the cumulative
        product of transition matrices up to and including step *t*.
    """
    slices: list[Tensor] = [W[:, 0]]
    for t in range(1, W.shape[1]):
        slices.append(torch.matmul(slices[-1], W[:, t]))
    return torch.stack(slices, dim=1)


@torch.compile(dynamic=True)
def _cumulative_matmul_left(W: Tensor) -> Tensor:
    """Cumulative left-multiply: W_cum[t] = W[t] @ ... @ W[1] @ W[0].

    For the Z state with left-multiply convention.

    Args:
        W: ``(BH, T, D, D)``

    Returns:
        W_cum: ``(BH, T, D, D)``
    """
    slices: list[Tensor] = [W[:, 0]]
    for t in range(1, W.shape[1]):
        slices.append(torch.matmul(W[:, t], slices[-1]))
    return torch.stack(slices, dim=1)


# ---------------------------------------------------------------------------
# All-gather helpers
# ---------------------------------------------------------------------------


from diffusion.distributed.context_parallel.halo_exchange import _to_global_rank


def _allgather(tensor: Tensor, group: ProcessGroup) -> Tensor:
    """All-gather a tensor across the group, returning ``(P, *shape)``."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    tensor_contig = tensor.contiguous()
    out = torch.empty((world,) + tensor_contig.shape, dtype=tensor_contig.dtype, device=tensor_contig.device)
    impl = get_cp_allgather_impl()

    if impl == "collective":
        # ``all_gather_into_tensor`` concatenates rank inputs along dim 0.
        # Reshape back to the stacked ``(world, *shape)`` contract used by
        # this module. This works for both Gloo and NCCL.
        flat_out = torch.empty(
            (world * tensor_contig.shape[0],) + tuple(tensor_contig.shape[1:]),
            dtype=tensor_contig.dtype,
            device=tensor_contig.device,
        )
        dist.all_gather_into_tensor(flat_out, tensor_contig, group=group)
        out = flat_out.reshape((world,) + tuple(tensor_contig.shape))
    elif impl == "list":
        # Conservative fallback for communicator behavior checks.
        gathered = [torch.empty_like(tensor_contig) for _ in range(world)]
        dist.all_gather(gathered, tensor_contig, group=group)
        out = torch.stack(gathered, dim=0)
    else:
        # FSDP2-oriented P2P implementation.
        ops = []
        out[rank].copy_(tensor_contig)
        for i in range(world):
            if i != rank:
                peer = _to_global_rank(group, i)
                ops.append(dist.P2POp(dist.isend, tensor_contig, peer, group=group))
                ops.append(dist.P2POp(dist.irecv, out[i], peer, group=group))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    return out


# ---------------------------------------------------------------------------
# All-gather + merge autograd Function
# ---------------------------------------------------------------------------


class _CPAllGatherMerge(torch.autograd.Function):
    """Differentiable all-gather + exclusive prefix merge for CP.

    Each rank contributes its chunk composite ``(h_ext, M)`` for both the
    KV and Z scans.  All composites are all-gathered, then each rank
    locally computes the exclusive prefix composition of all preceding
    chunks to obtain the correct initial state ``S_init``.

    Handles two multiply conventions simultaneously:
      - KV (right-multiply): S_final = S_init @ M + h_ext
      - Z  (left-multiply):  S_final = M @ S_init + h_ext

    Args (forward):
        h_ext_kv: ``(BH, D, D)`` -- KV input composite (local scan final state).
        M_kv:     ``(BH, D, D)`` -- KV transition composite (cumulative product).
        h_ext_z:  ``(BH, D)``    -- Z input composite.
        M_z:      ``(BH, D, D)`` -- Z transition composite.
        group:    CP process group.
        reverse:  If True, state flows from rank P-1 to rank 0.

    Returns:
        S_init_kv: ``(BH, D, D)`` -- correct KV initial state for this rank.
        S_init_z:  ``(BH, D)``    -- correct Z initial state for this rank.
    """

    @staticmethod
    def forward(
        ctx: Any,
        h_ext_kv: Tensor,
        M_kv: Tensor,
        h_ext_z: Tensor,
        M_z: Tensor,
        group: ProcessGroup,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)

        # All-gather composites from all ranks.
        h_all_kv = _allgather(h_ext_kv, group)  # (P, BH, D, D)
        M_all_kv = _allgather(M_kv, group)  # (P, BH, D, D)
        h_all_z = _allgather(h_ext_z, group)  # (P, BH, D)
        M_all_z = _allgather(M_z, group)  # (P, BH, D, D)

        if reverse:
            logical_rank = world - 1 - rank
            h_all_kv = h_all_kv.flip(0)
            M_all_kv = M_all_kv.flip(0)
            h_all_z = h_all_z.flip(0)
            M_all_z = M_all_z.flip(0)
        else:
            logical_rank = rank

        # Exclusive prefix composition: compose chunks 0..logical_rank-1.
        S_init_kv, S_init_z = _exclusive_prefix_compose(
            h_all_kv,
            M_all_kv,
            h_all_z,
            M_all_z,
            logical_rank,
        )

        ctx.save_for_backward(M_kv, M_z, S_init_kv, S_init_z)
        ctx.group = group
        ctx.reverse = reverse
        ctx.world = world
        ctx.rank = rank
        return S_init_kv, S_init_z

    @staticmethod
    def backward(
        ctx: Any,
        dS_init_kv: Tensor,
        dS_init_z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, None, None]:
        M_kv, M_z, S_init_kv, S_init_z = ctx.saved_tensors
        group = ctx.group
        reverse = ctx.reverse
        world = ctx.world
        rank = ctx.rank

        compute_dtype = torch.float32
        dS_init_kv = dS_init_kv.to(compute_dtype)
        dS_init_z = dS_init_z.to(compute_dtype)
        M_kv = M_kv.to(compute_dtype)
        M_z = M_z.to(compute_dtype)
        S_init_kv = S_init_kv.to(compute_dtype)
        S_init_z = S_init_z.to(compute_dtype)

        if reverse:
            logical_rank = world - 1 - rank
        else:
            logical_rank = rank

        # All-gather dS_init and M from all ranks.
        dS_all_kv = _allgather(dS_init_kv, group)  # (P, BH, D, D)
        dS_all_z = _allgather(dS_init_z, group)  # (P, BH, D)
        M_all_kv = _allgather(M_kv, group)  # (P, BH, D, D)
        M_all_z = _allgather(M_z, group)  # (P, BH, D, D)

        if reverse:
            dS_all_kv = dS_all_kv.flip(0)
            dS_all_z = dS_all_z.flip(0)
            M_all_kv = M_all_kv.flip(0)
            M_all_z = M_all_z.flip(0)

        # Compute dS_final: gradient flowing into this rank's composite
        # from all successor ranks.
        #
        # In the forward, rank j > logical_rank uses our (h, M) through:
        #   S_init(j) = ... @ M[logical_rank] + h[logical_rank] ...
        #
        # The backward sweep accumulates:
        #   sent[r] = dS_init[r] + sent[r+1] @ M[r]^T   (KV, right-multiply)
        #   sent[r] = dS_init[r] + M[r]^T @ sent[r+1]   (Z, left-multiply)
        # and dS_final[logical_rank] = sent[logical_rank + 1].
        if logical_rank >= world - 1:
            dS_final_kv = torch.zeros_like(dS_init_kv)
            dS_final_z = torch.zeros_like(dS_init_z)
        else:
            sent_kv = dS_all_kv[world - 1].clone()
            sent_z = dS_all_z[world - 1].clone()
            for r in range(world - 2, logical_rank, -1):
                sent_kv = dS_all_kv[r] + torch.matmul(
                    sent_kv,
                    M_all_kv[r].transpose(-1, -2),
                )
                sent_z = dS_all_z[r] + torch.bmm(
                    M_all_z[r].transpose(-1, -2),
                    sent_z.unsqueeze(-1),
                ).squeeze(-1)
            dS_final_kv = sent_kv
            dS_final_z = sent_z

        # Gradients w.r.t. this rank's composites.
        # Forward: S_final = S_init @ M + h_ext  (KV)
        #          S_final = M @ S_init + h_ext   (Z)
        dh_ext_kv = dS_final_kv
        dM_kv = torch.matmul(S_init_kv.transpose(-1, -2), dS_final_kv)

        dh_ext_z = dS_final_z
        dM_z = torch.bmm(dS_final_z.unsqueeze(-1), S_init_z.unsqueeze(-2))

        return dh_ext_kv, dM_kv, dh_ext_z, dM_z, None, None


class _BroadcastFromLastRank(torch.autograd.Function):
    """Autograd-aware broadcast of a tensor from the LAST CP rank.

    Forward:
        Every rank emits the value of ``tensor`` from the last rank (the
        non-last ranks' input value is DROPPED). Equivalent to
        ``dist.broadcast(tensor, src=last_rank)`` but autograd-tracked.

    Backward:
        Gradient flowing into the broadcasted output on EVERY rank is
        summed (all-reduced) and accumulated into the last rank's source
        tensor. Non-last ranks receive zero gradient (they didn't
        contribute to the forward).

    This is mathematically equivalent to a "scatter" of the source value
    to every rank with a "sum" gradient back to the source.
    """

    @staticmethod
    def forward(ctx: Any, tensor: Tensor, group: ProcessGroup) -> Tensor:
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)
        last_rank_global = _to_global_rank(group, world - 1)
        ctx.group = group
        ctx.world = world
        ctx.rank = rank
        ctx.last_rank_global = last_rank_global

        out = tensor.detach().clone().contiguous()
        # Single broadcast: out becomes the last rank's value on every rank.
        if world > 1:
            dist.broadcast(out, src=last_rank_global, group=group)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, None]:
        group = ctx.group
        world = ctx.world
        rank = ctx.rank

        if world <= 1:
            return grad_out, None

        # Sum gradients across ranks. The result is the total gradient flowing
        # into the source (last rank's) terminal state. Only the last rank
        # returns this sum; non-last ranks return zeros (their input was
        # dropped in the forward).
        summed = grad_out.contiguous().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=group)
        if rank == world - 1:
            return summed, None
        return torch.zeros_like(grad_out), None


def _exclusive_prefix_compose(
    h_all_kv: Tensor,
    M_all_kv: Tensor,
    h_all_z: Tensor,
    M_all_z: Tensor,
    logical_rank: int,
) -> tuple[Tensor, Tensor]:
    """Compose chunks 0, 1, ..., logical_rank-1 to get S_init.

    For logical_rank == 0, returns zeros (first rank starts from zero state).

    KV (right-multiply): h = h @ M[j] + h_ext[j]  for j = 0..rank-1
    Z  (left-multiply):  h = M[j] @ h + h_ext[j]  for j = 0..rank-1
    """
    if logical_rank == 0:
        return torch.zeros_like(h_all_kv[0]), torch.zeros_like(h_all_z[0])

    S_kv = torch.zeros_like(h_all_kv[0])
    S_z = torch.zeros_like(h_all_z[0])
    for j in range(logical_rank):
        S_kv = torch.matmul(S_kv, M_all_kv[j]) + h_all_kv[j]
        S_z = torch.bmm(M_all_z[j], S_z.unsqueeze(-1)).squeeze(-1) + h_all_z[j]
    return S_kv, S_z


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cp_frame_gdn_scan(
    W_kv: Tensor,
    U_kv: Tensor,
    W_z: Tensor,
    U_z: Tensor,
    group: ProcessGroup,
    reverse: bool = False,
    truncate_to_active: int | None = None,
) -> tuple[Tensor, Tensor] | CpFrameGdnScanResult:
    """Distributed GDN scan across CP ranks with state correction.

    Produces the same result as running the scan on the globally concatenated
    (W, U) sequence, but each GPU only touches its local T_local frames plus
    O(P * D^2) communication via all-gather.

    Algorithm:
        1. Local scan with S_init = 0
        2. Cumulative transition products
        3. Extract chunk composites (h_ext, M)
        4. All-gather + merge to get S_init
        5. Correct all local states

    Args:
        W_kv: ``(BH, T_local, D, D)`` -- local KV transition matrices.
        U_kv: ``(BH, T_local, D, D)`` -- local KV input matrices.
        W_z:  ``(BH, T_local, D, D)`` -- local Z transition matrices.
        U_z:  ``(BH, T_local, D)``    -- local Z input vectors.
        group: CP process group.
        reverse: If True, the scan direction is reversed (for backward
            recurrence in BidirectionalGDN).
        truncate_to_active: When set to an integer
            ``K_active`` (logical valid global cond length), the scan
            internally masks ``(W, U)`` at positions ``>= K_active`` so
            those positions do NOT contribute to state propagation
            (``W = I``, ``U = 0``). The scan also extracts the terminal
            state at global position ``K_active - 1`` and broadcasts it
            to every CP rank. Return shape changes to
            :class:`CpFrameGdnScanResult` (NamedTuple of 4 fields).

            Constraints (forward direction only, ``reverse=False``):
            ``1 <= K_active <= T_local * cp_size``.
            ``reverse=True`` is not supported (no AR rollout consumer).
            ``cp_size=1`` is supported (the mask still applies; terminal
            state is extracted locally without communication).

    Returns:
        When ``truncate_to_active is None`` (default, backward-compatible
        path): plain 2-tuple
        ``(S_kv_all: (BH, T_local, D, D), S_z_all: (BH, T_local, D))``.

        When ``truncate_to_active`` is set: :class:`CpFrameGdnScanResult`
        with fields ``S_kv_all``, ``S_z_all``, ``terminal_state_kv``
        ``(BH, D, D)``, ``terminal_state_z`` ``(BH, D)``. Terminal state
        is identical on every CP rank.
    """
    # Handle truncate_to_active by masking W/U at padded positions so state
    # propagation stops at position ``K_active - 1``.
    if truncate_to_active is not None:
        if reverse:
            raise NotImplementedError(
                "cp_frame_gdn_scan: truncate_to_active is only supported for "
                "reverse=False (no AR rollout consumer needs reverse trunc)."
            )
        T_local_in = W_kv.shape[1]
        D_dim = W_kv.shape[-1]
        cp_world = dist.get_world_size(group)
        cp_rank_in = dist.get_rank(group)
        T_global = T_local_in * cp_world
        K_active = int(truncate_to_active)
        if K_active < 1 or K_active > T_global:
            raise ValueError(f"truncate_to_active={K_active} must satisfy 1 <= K_active " f"<= T_global={T_global}")

        # Build per-rank position mask: positions >= K_active should have
        # W=I and U=0. The mask is ``valid[local_t] = (rank * T_local + local_t < K_active)``.
        local_positions = torch.arange(T_local_in, device=W_kv.device)
        global_positions = cp_rank_in * T_local_in + local_positions
        valid_mask = (global_positions < K_active).to(W_kv.dtype)  # (T_local,) 0/1
        valid_kv = valid_mask.view(1, T_local_in, 1, 1)  # (1, T_local, 1, 1)
        valid_z_W = valid_mask.view(1, T_local_in, 1, 1)
        valid_z_U = valid_mask.view(1, T_local_in, 1)

        eye_kv = torch.eye(D_dim, device=W_kv.device, dtype=W_kv.dtype).view(1, 1, D_dim, D_dim)
        eye_z = torch.eye(D_dim, device=W_z.device, dtype=W_z.dtype).view(1, 1, D_dim, D_dim)

        # W -> I, U -> 0 at padded positions.
        W_kv = valid_kv * W_kv + (1.0 - valid_kv) * eye_kv
        U_kv = valid_kv * U_kv  # 0 at padded.
        W_z = valid_z_W * W_z + (1.0 - valid_z_W) * eye_z
        U_z = valid_z_U * U_z

    # --- Step 1: cumulative transition products ---
    W_kv_cum = _cumulative_matmul_right(W_kv)  # (BH, T_local, D, D)
    W_z_cum = _cumulative_matmul_left(W_z)  # (BH, T_local, D, D)

    # --- Step 2: local scan with S_init = 0 to get h_ext ---
    local_scan = _get_local_scan_cls(W_kv.is_cuda)
    S_kv_local, S_z_local = local_scan.apply(W_kv, U_kv, W_z, U_z)

    # --- Step 3: extract chunk composites ---
    h_ext_kv = S_kv_local[:, -1]  # (BH, D, D)
    M_kv = W_kv_cum[:, -1]  # (BH, D, D)
    h_ext_z = S_z_local[:, -1]  # (BH, D)
    M_z = W_z_cum[:, -1]  # (BH, D, D)

    # --- Step 4: all-gather + merge to get correct S_init ---
    S_init_kv, S_init_z = _CPAllGatherMerge.apply(
        h_ext_kv,
        M_kv,
        h_ext_z,
        M_z,
        group,
        reverse,
    )

    # --- Step 5: additive correction (replaces full rescan) ---
    # By linearity of the recurrence s[t] = s[t-1] @ W[t] + U[t]:
    #   S_corrected[t] = S_zero[t] + S_init @ W_cum[t]
    # This is a parallel matmul instead of a sequential scan.
    # KV (right-multiply convention): S[t] = S[t-1] @ W[t] + U[t]
    S_kv_corrected = S_kv_local + torch.matmul(S_init_kv.unsqueeze(1), W_kv_cum)
    # Z (left-multiply convention): S[t] = W[t] @ S[t-1] + U[t]
    # W_z_cum[t] = W[t] @ ... @ W[0], so correction = W_z_cum[t] @ S_init_z
    T_local = W_z_cum.shape[1]
    S_z_corrected = S_z_local + torch.bmm(
        W_z_cum.reshape(-1, W_z_cum.shape[2], W_z_cum.shape[3]),
        S_init_z.unsqueeze(1).expand(-1, T_local, -1).reshape(-1, W_z_cum.shape[3], 1),
    ).reshape(S_z_local.shape)

    if truncate_to_active is None:
        return S_kv_corrected, S_z_corrected

    # Extract terminal state at global position ``K_active - 1`` and
    # broadcast to all ranks. Because padded positions have W=I, U=0, the
    # recurrence state stays constant after the active prefix.
    terminal_kv_local = S_kv_corrected[:, -1].contiguous()  # (BH, D, D)
    terminal_z_local = S_z_corrected[:, -1].contiguous()  # (BH, D)
    terminal_kv = _BroadcastFromLastRank.apply(terminal_kv_local, group)
    terminal_z = _BroadcastFromLastRank.apply(terminal_z_local, group)

    return CpFrameGdnScanResult(
        S_kv_all=S_kv_corrected,
        S_z_all=S_z_corrected,
        terminal_state_kv=terminal_kv,
        terminal_state_z=terminal_z,
    )
