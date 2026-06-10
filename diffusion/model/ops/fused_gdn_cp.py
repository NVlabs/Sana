"""Context-parallel wrappers for fused Triton GDN kernels.

The fused non-CP kernels use a left-multiply recurrence for the KV state,
while ``cp_frame_gdn_scan`` uses a right-multiply recurrence. This module
adapts between those state conventions so context-parallel training can keep
the Triton fused prep/output kernels and replace only the middle scan with the
distributed CP scan.

* :func:`phase_a` returns ``I_P_kv`` and ``A`` representing the raw factor
  ``(I - k_rot * beta * k_rot.T)`` and the input
  ``A_t = (v * beta) @ k_rot.T``. **Decay is NOT folded in.**
* :func:`phase_b_triton` applies decay inside the kernel:
  ``M_t = decay_t * (I - P_t) @ M_{t-1} + A_t``.
* :func:`_build_transition_matrices` returns
  ``W_kv = decay_f * (I - k_rot*beta @ k_rot.T)`` with decay pre-folded. The
  downstream eager scan uses ``S_t = S_{t-1} @ W_t + U_t`` (right-multiply),
  so the right-multiply state is the transpose of Phase B's left-multiply
  ``M_t``.

Mapping (KV):
    Phase B (left-multiply, decay outside I_P_kv):
        M_t = decay_t * (I - P_t) @ M_{t-1} + A_t
    cp_frame_gdn_scan (right-multiply):
        S_t = S_{t-1} @ W_t + U_t   with   S_t = M_t.T
    Therefore:
        W_t = (decay_t * (I - P_t)).T
        U_t = A_t.T

Mapping (Z):
    Phase B and cp_frame_gdn_scan both use a left-multiply Z recurrence
    (``z_t = decay_t * (I - P_z) @ z_{t-1} + B_t`` vs
    ``S_t = W_t @ S_{t-1} + U_t``); the only difference is again that decay
    is folded into ``W_z`` for the eager / cp_scan path but applied inside
    the Triton kernel for Phase B. Hence
        W_z = decay_t * I_P_z   (with no transpose)
        U_z = B_z

Precision contract:
    Phase B uses fp32 recurrence state. The adapter therefore promotes
    per-frame transitions and decay to fp32 before multiplying, transposing,
    and slicing. Phase C consumes fp32 state as well.

The padded ``BLOCK_D`` slice ``[head_dim:BLOCK_D, :BLOCK_D]`` and
``[:BLOCK_D, head_dim:BLOCK_D]`` is structurally inert in both fused Phase B
and the eager scan (Phase A writes zeros into those tiles via masked
``tl.store``). The adapter slices the active ``D x D`` sub-block before
transposing so garbage in the padded region cannot poison the recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributed import ProcessGroup


def _resolve_use_checkpoint(use_checkpoint: bool | None, *leaves: Tensor | None) -> bool:
    """Resolve the ``use_checkpoint`` argument for the fused-CP entry points.

    Args:
        use_checkpoint: Explicit override (``True``/``False``) or ``None`` to
            auto-detect from autograd state. ``None`` resolves to ``True``
            iff :func:`torch.is_grad_enabled` AND any non-None leaf has
            ``requires_grad=True`` -- i.e., training mode with autograd
            actually active. This mirrors reference CamCtrl SFT practice
            (gradient_checkpointing on during training, off during eval).
        *leaves: The input tensors that participate in autograd. ``None``
            entries (e.g. optional norm weights) are ignored.

    Returns:
        The effective ``use_checkpoint`` flag.
    """
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled() and any((t is not None) and t.requires_grad for t in leaves)
    return bool(use_checkpoint)


__all__ = [
    "CpFusedGdnRawResult",
    "CpFusedTransitionBundle",
    "_CpFusedGdnOutput",
    "_CpFusedGdnPrep",
    "cp_fused_cam_gdn_num_autograd",
    "cp_fused_gdn_chunkwise_raw_autograd",
    "cp_scan_states_to_phase_c_states",
    "phase_a_to_cp_scan_transitions",
]


@dataclass(frozen=True)
class CpFusedTransitionBundle:
    """Transitions in :func:`cp_frame_gdn_scan` convention plus adapter metadata.

    Attributes:
        W_kv: ``(BH, T_local, D, D)`` -- right-multiply KV transition,
            equal to ``(decay_t * (I - P_t)).T`` from Phase A. Always the
            active ``D x D`` sub-block (NOT padded to ``BLOCK_D``).
        U_kv: ``(BH, T_local, D, D)`` -- right-multiply KV input, equal
            to ``A_t.T``.
        W_z:  ``(BH, T_local, D, D)`` -- left-multiply Z transition,
            equal to ``decay_t * I_P_z`` (no transpose). Size-0 placeholder
            when ``skip_z=True``.
        U_z:  ``(BH, T_local, D)``   -- left-multiply Z input, equal to
            ``B_z``. Size-0 placeholder when ``skip_z=True``.
        block_d: Padded head dimension (``triton.next_power_of_2(D)``).
            Required to re-pad scan outputs for Phase C consumption.
        head_dim: Active head dimension ``D``.
    """

    W_kv: Tensor
    U_kv: Tensor
    W_z: Tensor
    U_z: Tensor
    block_d: int
    head_dim: int


def phase_a_to_cp_scan_transitions(
    I_P_kv: Tensor,
    A_kv: Tensor,
    I_P_z: Tensor,
    B_z: Tensor,
    decay: Tensor,
    *,
    head_dim: int,
    skip_z: bool = False,
) -> CpFusedTransitionBundle:
    """Map fused Phase A tensors to :func:`cp_frame_gdn_scan` transitions.

    See module docstring for the convention derivation.

    Args:
        I_P_kv: ``(BH, T_local, BLOCK_D, BLOCK_D)`` -- raw
            ``(I - k_rot * beta * k_rot.T)`` from
            ``_phase_a_kv_kernel``. Decay is NOT folded in. May be bf16
            (Phase A inter-phase bridge at ``dot_precision=0``) or fp32
            (``dot_precision>=1``); the adapter always promotes to fp32
            before multiplying.
        A_kv:   ``(BH, T_local, BLOCK_D, BLOCK_D)`` -- raw
            ``(v * beta) @ k_rot.T`` from ``_phase_a_kv_kernel``. Same
            dtype contract as ``I_P_kv``.
        I_P_z:  ``(BH, T_local, BLOCK_D, BLOCK_D)`` -- raw
            ``(I - k * beta * k.T)`` for the Z stream. Ignored when
            ``skip_z=True`` (kernel allocates a 1-element placeholder).
        B_z:    ``(BH, T_local, BLOCK_D)``        -- raw ``k * beta``
            summed over heads-S for the Z stream. Same placeholder
            convention. Always fp32 in the fused Phase A path.
        decay:  Either ``(BH, T_local)`` or broadcastable to that shape.
            Reshaped + cast to float32 internally so the per-frame
            ``decay_t`` scalar can be multiplied against the
            ``(BLOCK_D, BLOCK_D)`` tile in a single broadcast.
        head_dim: Active head dimension ``D`` (Phase A pads to
            ``BLOCK_D = next_pow2(D)``; we slice the active sub-block here
            to defensively isolate the recurrence from any garbage in
            padded tiles).
        skip_z: When True, return size-0 placeholders for ``W_z`` / ``U_z``.
            Used by the camera-branch numerator-only scan.

            CONTRACT: A ``skip_z=True`` bundle must not be passed directly
            to :func:`cp_frame_gdn_scan`; callers must provide valid dummy Z
            tensors or use a numerator-only scan path.

    Returns:
        :class:`CpFusedTransitionBundle` whose tensor fields live in the
        :func:`cp_frame_gdn_scan` recurrence convention. All tensor fields
        are fp32 regardless of input dtype (see "Precision contract" in
        the module docstring).
    """
    if I_P_kv.ndim != 4:
        raise ValueError(
            f"phase_a_to_cp_scan_transitions: expected I_P_kv with 4 dims "
            f"(BH, T, BLOCK_D, BLOCK_D), got shape {tuple(I_P_kv.shape)}"
        )
    BH, T_local, BLOCK_D, BLOCK_D2 = I_P_kv.shape
    if BLOCK_D != BLOCK_D2:
        raise ValueError(
            f"phase_a_to_cp_scan_transitions: I_P_kv last two dims must be " f"square, got {(BLOCK_D, BLOCK_D2)}"
        )
    if head_dim < 1 or head_dim > BLOCK_D:
        raise ValueError(
            f"phase_a_to_cp_scan_transitions: head_dim={head_dim} must " f"satisfy 1 <= head_dim <= BLOCK_D={BLOCK_D}"
        )
    if A_kv.shape != I_P_kv.shape:
        raise ValueError(
            f"phase_a_to_cp_scan_transitions: A_kv shape {tuple(A_kv.shape)} " f"!= I_P_kv shape {tuple(I_P_kv.shape)}"
        )

    # Match Phase B's fp32 recurrence state before multiplying by decay.
    I_P_kv_f32 = I_P_kv.to(torch.float32) if I_P_kv.dtype != torch.float32 else I_P_kv
    A_kv_f32 = A_kv.to(torch.float32) if A_kv.dtype != torch.float32 else A_kv
    # Reshape decay to (BH, T_local, 1, 1) so the broadcast multiplies each
    # (BLOCK_D, BLOCK_D) tile by its scalar decay_t.
    decay_view = decay.reshape(BH, T_local).to(torch.float32).view(BH, T_local, 1, 1)

    # Left-multiply form (Phase B convention) on the active D x D slice.
    W_kv_left = decay_view * I_P_kv_f32[..., :head_dim, :head_dim]
    # cp_frame_gdn_scan uses right-multiply, so S_t = M_t.T. Therefore
    # transpose every transition / input pair.
    W_kv = W_kv_left.transpose(-1, -2).contiguous()
    U_kv = A_kv_f32[..., :head_dim, :head_dim].transpose(-1, -2).contiguous()

    if skip_z:
        # NUM_ONLY camera-branch callers do not consume Z. We materialise
        # size-0 placeholders rather than fake (BH, T_local, D, D) tensors
        # so any accidental downstream read crashes loudly with a shape
        # mismatch instead of silently producing wrong numbers.
        #
        # CONTRACT: callers MUST NOT hand a `skip_z=True` bundle directly
        # to `cp_frame_gdn_scan`; see the function docstring under `skip_z`.
        W_z = torch.empty(0, device=I_P_kv.device, dtype=torch.float32)
        U_z = torch.empty(0, device=I_P_kv.device, dtype=torch.float32)
    else:
        # Z is already left-multiply in both conventions; just slice + fold
        # decay in (same as W_z = decay_f * I_P_z in _build_transition_matrices).
        I_P_z_f32 = I_P_z.to(torch.float32) if I_P_z.dtype != torch.float32 else I_P_z
        B_z_f32 = B_z.to(torch.float32) if B_z.dtype != torch.float32 else B_z
        W_z = (decay_view * I_P_z_f32[..., :head_dim, :head_dim]).contiguous()
        U_z = B_z_f32[..., :head_dim].contiguous()

    return CpFusedTransitionBundle(
        W_kv=W_kv,
        U_kv=U_kv,
        W_z=W_z,
        U_z=U_z,
        block_d=BLOCK_D,
        head_dim=head_dim,
    )


def cp_scan_states_to_phase_c_states(
    S_kv: Tensor,
    S_z: Tensor,
    *,
    block_d: int,
) -> tuple[Tensor, Tensor]:
    """Re-pad and re-transpose cp_frame_gdn_scan output for Phase C consumption.

    Phase C (:func:`phase_c` in ``fused_gdn_chunkwise.py``) expects the
    state ``M_t`` in its native left-multiply convention, padded to
    ``BLOCK_D``. This inverts the operations performed by
    :func:`phase_a_to_cp_scan_transitions` on the state side.

    Args:
        S_kv: ``(BH, T_local, head_dim, head_dim)`` -- corrected KV
            recurrence state in :func:`cp_frame_gdn_scan` (right-multiply)
            convention.
        S_z:  ``(BH, T_local, head_dim)`` -- corrected Z state.
        block_d: Padded head dimension used by the Triton kernels.

    Returns:
        ``(M_kv_padded, M_z_padded)`` where
        ``M_kv_padded`` has shape ``(BH, T_local, block_d, block_d)`` and
        contents ``S_kv.T`` over the active ``head_dim`` slice with zeros
        in the padded tile, and
        ``M_z_padded`` has shape ``(BH, T_local, block_d)`` with the
        ``head_dim`` slice populated and zeros in the pad.
    """
    if S_kv.ndim != 4:
        raise ValueError(
            f"cp_scan_states_to_phase_c_states: expected S_kv with 4 dims, " f"got shape {tuple(S_kv.shape)}"
        )
    BH, T_local, head_dim, head_dim2 = S_kv.shape
    if head_dim != head_dim2:
        raise ValueError(
            f"cp_scan_states_to_phase_c_states: S_kv last two dims must be " f"square, got {(head_dim, head_dim2)}"
        )
    if head_dim > block_d:
        raise ValueError(f"cp_scan_states_to_phase_c_states: head_dim={head_dim} must " f"be <= block_d={block_d}")

    M_kv_padded = torch.zeros(BH, T_local, block_d, block_d, device=S_kv.device, dtype=S_kv.dtype)
    # Inverse transpose (right-multiply S -> left-multiply M).
    M_kv_padded[..., :head_dim, :head_dim] = S_kv.transpose(-1, -2)
    M_z_padded = torch.zeros(BH, T_local, block_d, device=S_z.device, dtype=S_z.dtype)
    M_z_padded[..., :head_dim] = S_z
    return M_kv_padded, M_z_padded


@dataclass(frozen=True)
class CpFusedGdnRawResult:
    """Raw numerator/denominator output of the fused GDN CP scan.

    Returned by :func:`cp_fused_gdn_chunkwise_raw_autograd`. Carries the
    ``(num, den)`` pair plus optional terminal-state fields when the caller
    requested ``truncate_to_active``.

    Attributes:
        num: ``(B, N_local, H, D)`` -- raw numerator before output gate /
            projection / final divide. dtype matches Phase C output:
            bf16 at ``dot_precision=0``, fp32 at ``dot_precision>=1``.
        den: ``(B, H, N_local)`` -- raw denominator. Same dtype contract
            as ``num``.
        terminal_state_kv: ``(BH, D, D)`` fp32, present only when
            ``truncate_to_active`` was set on the call; ``None`` otherwise.
            Identical on every CP rank.
        terminal_state_z:  ``(BH, D)``    fp32, same condition.
    """

    num: Tensor
    den: Tensor
    terminal_state_kv: Tensor | None = None
    terminal_state_z: Tensor | None = None


class _CpFusedGdnPrep(torch.autograd.Function):
    """RMSNorm + Phase A + transition adapter as a single autograd Function.

    Forward composes:

      1. Full-channel RMSNorm on Q and K channels of ``qkv``. V is not
         normalized.
      2. :func:`phase_a` on the normalized ``qkv_normed`` with identity
         ``inv_rms`` / norm-weight (so the Phase A kernel does no further
         norm). Returns ``(I_P_kv, A, I_P_z, B_z)``.
      3. :func:`phase_a_to_cp_scan_transitions` adapts to the
         :func:`cp_frame_gdn_scan` convention. Returns
         ``CpFusedTransitionBundle(W_kv, U_kv, W_z, U_z, ...)``.

    Backward composes:

      1. Inverse adapter VJP: ``(dW_kv, dU_kv, dW_z, dU_z) -> (dI_P_kv,
         dA, dI_P_z, dB_z, ddecay)``. ``dI_P_*`` / ``dA`` are padded back
         to ``(BH, F, BLOCK_D, BLOCK_D)`` and ``dB_z`` to ``(BH, F,
         BLOCK_D)`` for Phase A backward kernels.
      2. :func:`phase_a_kv_bwd` on ``(dA, -dI_P_kv)`` (since the kernel
         takes ``dP_kv`` and ``I_P_kv = I - P_kv``, ``dP_kv = -dI_P_kv``).
         Returns ``(dK_kv_bhfsd, dV_bhfsd, dbeta_kv_bhfs)``.
      3. :func:`phase_a_z_bwd` analogous -> ``(dK_z_bhfsd, dbeta_z_bhfs)``.
      4. :func:`fused_rope_unrope_bwd` combines the K-channel grads
         coming out of ``phase_a_*_bwd`` (with ``dQ_kv``/``dQ_z`` zero
         since Q has no Phase A grad) and unrope+relu-masks them ->
         ``(_dQ_zero_via_relu, dK_normed_bhfsd_from_phase_a)``.
      5. Add the ``dqkv_normed`` upstream grad contributions:
         - Q: ``dQ_normed_total = dqkv_normed[:, :, 0]_bhfsd`` (no Phase A path).
         - K: ``dK_normed_total = dK_normed_bhfsd_from_phase_a + dqkv_normed[:, :, 1]_bhfsd``.
         - V: ``dV_total_bnhd = dV_from_phase_a_kv_bwd_bnhd + dqkv_normed[:, :, 2]_bnhd``.
      6. Per-channel RMSNorm VJP for Q and K -> ``dQ_raw``, ``dK_raw``,
         ``dq_norm_w``, ``dk_norm_w``.
      7. Stack ``dqkv = stack([dQ_raw, dK_raw, dV_total], dim=2)``.

    The decay grad is accumulated entirely inside step 1 (since
    ``W_kv = decay * I_P_kv`` and ``W_z = decay * I_P_z``); the Phase A
    backward kernels do not contribute to ``ddecay``.

    The qkv_normed output is differentiable: ``_CpFusedGdnOutput`` consumes
    it and its backward produces ``dqkv_normed`` which is summed with the
    Phase A chain above to yield the full ``dqkv``.
    """

    @staticmethod
    def forward(
        ctx,
        qkv: Tensor,
        beta: Tensor,
        decay: Tensor,
        q_norm_weight: Tensor | None,
        k_norm_weight: Tensor | None,
        rope_cos: Tensor,
        rope_sin: Tensor,
        F: int,
        S: int,
        k_scale: float,
        norm_eps: float = 1e-5,
        dot_precision: int = 0,
    ):
        from diffusion.model.ops.fused_gdn_chunkwise import phase_a

        B, N, three, H, D = qkv.shape
        if three != 3:
            raise ValueError(f"_CpFusedGdnPrep.forward: qkv dim 2 must equal 3, got {three}")
        if N != F * S:
            raise ValueError(f"_CpFusedGdnPrep.forward: N={N} must equal F*S={F * S}")
        C = H * D
        device = qkv.device
        fp32 = torch.float32
        dtype = qkv.dtype

        # Track missing norm weights so backward returns None for those slots.
        ctx.q_nw_was_none = q_norm_weight is None
        ctx.k_nw_was_none = k_norm_weight is None

        # When both weights are None, q_norm/k_norm are identity modules.
        skip_rmsnorm = ctx.q_nw_was_none and ctx.k_nw_was_none
        ctx.skip_rmsnorm = skip_rmsnorm
        if q_norm_weight is None:
            q_norm_weight = torch.ones(C, device=device, dtype=fp32)
        if k_norm_weight is None:
            k_norm_weight = torch.ones(C, device=device, dtype=fp32)

        # 1. Full-channel RMSNorm on Q and K.
        q_raw_v = qkv[:, :, 0]  # view, same dtype as qkv
        k_raw_v = qkv[:, :, 1]
        if skip_rmsnorm:
            # Identity contract: qkv_normed === qkv. Save ones for
            # q_inv_rms / k_inv_rms so the backward RMSNorm-VJP
            # bookkeeping is well-defined but the VJP degenerates to
            # the identity.
            q_inv_rms = torch.ones(B, N, device=device, dtype=fp32)
            k_inv_rms = torch.ones(B, N, device=device, dtype=fp32)
            qkv_normed = qkv
        else:
            q_inv_rms = torch.rsqrt((q_raw_v.float().pow(2)).sum(dim=(-2, -1)) / C + norm_eps)
            k_inv_rms = torch.rsqrt((k_raw_v.float().pow(2)).sum(dim=(-2, -1)) / C + norm_eps)
            q_nw_hd = q_norm_weight.reshape(H, D)
            k_nw_hd = k_norm_weight.reshape(H, D)
            qkv_normed = qkv.clone()
            qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(dtype)
            qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(dtype)

        # 2. phase_a with identity inv_rms / norm_w so the kernel does no re-norm.
        dummy_inv = torch.ones(B, N, device=device, dtype=fp32)
        dummy_nw = torch.ones(C, device=device, dtype=fp32)
        I_P_kv, A, I_P_z, B_z = phase_a(
            qkv_normed,
            beta,
            dummy_inv,
            dummy_inv,
            dummy_nw,
            dummy_nw,
            rope_cos,
            rope_sin,
            F=F,
            S=S,
            k_scale=k_scale,
            norm_eps=norm_eps,
            dot_precision=dot_precision,
        )

        # 3. Adapter: Phase A layout -> cp_frame_gdn_scan convention.
        bundle = phase_a_to_cp_scan_transitions(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            head_dim=D,
            skip_z=False,
        )

        # Save for backward.
        ctx.save_for_backward(
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            q_inv_rms,
            k_inv_rms,
            rope_cos,
            rope_sin,
            I_P_kv,
            A,
            I_P_z,
            B_z,
        )
        ctx.shape = (B, N, H, D, F, S, C)
        ctx.k_scale = float(k_scale)
        ctx.dot_precision = int(dot_precision)

        return bundle.W_kv, bundle.U_kv, bundle.W_z, bundle.U_z, qkv_normed

    @staticmethod
    def backward(ctx, dW_kv, dU_kv, dW_z, dU_z, dqkv_normed):
        # Inline imports keep top-of-file lightweight.
        from diffusion.model.ops.fused_gdn_chunkwise_bwd import (
            _resolve_bwd_block_s,
            fused_rope_relu_fwd,
            fused_rope_unrope_bwd,
            phase_a_kv_bwd,
            phase_a_z_bwd,
        )

        (
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            q_inv_rms,
            k_inv_rms,
            rope_cos,
            rope_sin,
            I_P_kv,
            A_kv,
            I_P_z,
            B_z,
        ) = ctx.saved_tensors
        B, N, H, D, F, S, C = ctx.shape
        k_scale = ctx.k_scale
        dot_precision = ctx.dot_precision
        BLOCK_S = _resolve_bwd_block_s()
        device = qkv.device
        fp32 = torch.float32
        dtype = qkv.dtype
        BH = B * H

        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)

        # 1. Inverse adapter VJP.
        # Forward adapter with full-pad slicing applied:
        #   W_kv_left[bh, f, :D, :D] = decay[bh, f] * I_P_kv_active[bh, f, :D, :D]
        #   W_kv[bh, f, d1, d2]      = W_kv_left[bh, f, d2, d1]   (transpose)
        #   U_kv[bh, f, d1, d2]      = A_kv_active[bh, f, d2, d1] (transpose)
        #   W_z[bh, f, d1, d2]       = decay[bh, f] * I_P_z_active[bh, f, d1, d2]
        #   U_z[bh, f, d]            = B_z_active[bh, f, d]
        # All cast to fp32 first; B_z is already fp32.
        BLOCK_D = I_P_kv.shape[-1]
        decay_f = decay.reshape(BH, F).to(fp32)
        decay_view = decay_f.view(BH, F, 1, 1)

        # Active D x D slice in fp32.
        I_P_kv_active = I_P_kv[..., :D, :D].to(fp32)
        I_P_z_active = I_P_z[..., :D, :D].to(fp32)

        # Inputs to inverse VJP in fp32.
        dW_kv_f = dW_kv.to(fp32)
        dU_kv_f = dU_kv.to(fp32)
        dW_z_f = dW_z.to(fp32)
        dU_z_f = dU_z.to(fp32)

        # Convert W_kv = (decay * I_P_kv_active).T; its VJP for I_P_kv_active is
        # decay * dW_kv.T. Same logic for U_kv (just transpose, no decay).
        dI_P_kv_active = decay_view * dW_kv_f.transpose(-1, -2)
        dA_kv_active = dU_kv_f.transpose(-1, -2)

        # W_z and U_z are not transposed (per adapter).
        dI_P_z_active = decay_view * dW_z_f
        dB_z_active = dU_z_f  # (BH, F, D)

        # ddecay contributions from W_kv and W_z (note W_kv = (decay * I_P_kv).T,
        # so d/d(decay) = sum_{d1,d2}( dW_kv[d1,d2] * I_P_kv[d2,d1] )
        #               = sum_{d1,d2}( dW_kv * I_P_kv.T )
        # which is identical to sum( dI_P_kv_active * I_P_kv_active ) / decay
        # but the cleaner formulation is direct:
        ddecay_from_kv = (dW_kv_f * I_P_kv_active.transpose(-1, -2)).sum(dim=(-1, -2))
        ddecay_from_z = (dW_z_f * I_P_z_active).sum(dim=(-1, -2))
        ddecay = (ddecay_from_kv + ddecay_from_z).reshape(B, H, F)

        # Pad active grads back to BLOCK_D shape for the Triton kernels.
        # (phase_a_kv_bwd / phase_a_z_bwd pad internally if needed, but we
        # pass D x D which they will pad. The kernels accept either D x D or
        # BLOCK_D x BLOCK_D inputs; see the `pad_DxD` helpers in those
        # driver functions.)
        # Sign flip: kernel expects dP (where P = I - I_P), so dP = -dI_P.
        dP_kv = (-dI_P_kv_active).contiguous()
        dP_z = (-dI_P_z_active).contiguous()
        dA_kv_for_kernel = dA_kv_active.contiguous()
        dB_z_for_kernel = dB_z_active.contiguous()

        # 2. Reconstruct qkv_normed for the rope/relu recomputation.
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        skip_rmsnorm = getattr(ctx, "skip_rmsnorm", False)
        if skip_rmsnorm:
            qkv_normed = qkv
        else:
            qkv_normed = qkv.clone()
            qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(dtype)
            qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(dtype)

        # 3. Recompute Q/K post-relu masks in BHFSD layout.
        def bnhd_to_bhfsd(x):
            return x.permute(0, 2, 1, 3).reshape(B, H, F, S, D).reshape(BH, F, S, D).contiguous()

        def bhfsd_to_bnhd(x):
            return x.reshape(BH, F * S, D).reshape(B, H, N, D).permute(0, 2, 1, 3).contiguous()

        Q_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 0])
        K_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 1])
        V_bhfsd = bnhd_to_bhfsd(qkv[:, :, 2])

        # fused_rope_relu_fwd returns (Q_post_relu, K_post_relu, Q_for_num, K_kv).
        # We need: K_kv_bhfsd (post-rope, key chain for Phase A KV) and
        #          K_post_relu_bhfsd (no rope on K_z, key chain for Phase A Z).
        Q_post_relu_bhfsd, K_post_relu_bhfsd, _Q_for_num_bhfsd, K_kv_bhfsd = fused_rope_relu_fwd(
            Q_normed_bhfsd,
            K_normed_bhfsd,
            rope_cos,
            rope_sin,
            k_scale,
            F,
            S,
        )
        del Q_normed_bhfsd, K_normed_bhfsd

        # K for the Z stream is K_post_relu (no rope applied, see Phase A
        # Z kernel which does NOT apply rope to K_z).
        K_z_bhfsd = K_post_relu_bhfsd

        beta_bhfs = beta.reshape(BH, F, S).float()

        # 4. Phase-A KV backward.
        dK_kv_bhfsd, dV_bhfsd, dbeta_kv = phase_a_kv_bwd(
            K_kv_bhfsd.contiguous(),
            V_bhfsd.contiguous(),
            beta_bhfs,
            dA_kv_for_kernel,
            dP_kv,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # 5. Phase-A Z backward.
        dK_z_bhfsd, dbeta_z = phase_a_z_bwd(
            K_z_bhfsd.contiguous(),
            beta_bhfs,
            dB_z_for_kernel,
            dP_z,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # 6. Combine via fused_rope_unrope_bwd. Q has no Phase-A contribution.
        dQ_zero_kv = torch.zeros_like(dK_kv_bhfsd)
        dQ_zero_z = torch.zeros_like(dK_z_bhfsd)
        # Note: fused_rope_unrope_bwd internally multiplies dK channel by
        # k_scale (the K-side relu+scale flip). Q channel uses no scale.
        # Sanity: outputs are post-RMSNorm but pre-RMSNorm-VJP gradients.
        dQ_normed_via_relu_rope_bhfsd, dK_normed_from_phase_a_bhfsd = fused_rope_unrope_bwd(
            dQ_zero_kv,
            dK_kv_bhfsd,
            dQ_zero_z,
            dK_z_bhfsd,
            Q_post_relu_bhfsd,
            K_post_relu_bhfsd,
            rope_cos,
            rope_sin,
            k_scale,
            F,
            S,
        )
        del dQ_zero_kv, dQ_zero_z, dK_kv_bhfsd, dK_z_bhfsd
        del Q_post_relu_bhfsd, K_post_relu_bhfsd, K_kv_bhfsd

        # Q's Phase A grad is structurally zero because both inputs were zero.
        del dQ_normed_via_relu_rope_bhfsd

        # 7. Add upstream dqkv_normed contribution.
        # dqkv_normed shape: (B, N, 3, H, D), dtype = dtype.
        # Channel layout: 0 = Q, 1 = K, 2 = V.
        dqkv_normed_Q_bnhd = dqkv_normed[:, :, 0].contiguous()
        dqkv_normed_K_bnhd = dqkv_normed[:, :, 1].contiguous()
        dqkv_normed_V_bnhd = dqkv_normed[:, :, 2].contiguous()

        # Convert Q to BHFSD (for RMSNorm VJP we'll bring back to BNHD).
        # Q has no Phase A contribution, so dQ_normed_total_bnhd is just the
        # upstream Q channel.
        dQ_normed_total_bnhd = dqkv_normed_Q_bnhd.to(fp32)

        # K: add upstream to phase-A-chain K grad.
        dK_normed_from_phase_a_bnhd = bhfsd_to_bnhd(dK_normed_from_phase_a_bhfsd)
        del dK_normed_from_phase_a_bhfsd
        dK_normed_total_bnhd = dK_normed_from_phase_a_bnhd.to(fp32) + dqkv_normed_K_bnhd.to(fp32)
        del dK_normed_from_phase_a_bnhd

        # V: phase_a_kv_bwd's dV is in BHFSD; convert to BNHD then add upstream.
        dV_from_phase_a_bnhd = bhfsd_to_bnhd(dV_bhfsd)
        del dV_bhfsd
        dV_total_bnhd = dV_from_phase_a_bnhd.to(fp32) + dqkv_normed_V_bnhd.to(fp32)
        del dV_from_phase_a_bnhd

        # 8. RMSNorm VJP.
        # d/dx = inv_rms*w*d/dy - (inv_rms^3 / C) * x * sum(w*d/dy*x)
        if skip_rmsnorm:
            dQ_raw = dQ_normed_total_bnhd
            dK_raw = dK_normed_total_bnhd
            dq_nw = torch.zeros(C, device=device, dtype=fp32)
            dk_nw = torch.zeros(C, device=device, dtype=fp32)
            del dQ_normed_total_bnhd, dK_normed_total_bnhd
        else:
            q_raw_f = q_raw_v.float()
            q_irms = q_inv_rms[:, :, None, None]
            gw_q = dQ_normed_total_bnhd * q_nw_hd[None, None]
            dq_nw = (dQ_normed_total_bnhd * q_raw_f * q_irms).sum(dim=(0, 1)).reshape(-1)
            corr_q = (gw_q * q_raw_f).sum(dim=(-2, -1), keepdim=True)
            dQ_raw = q_irms * gw_q - (q_irms**3) / C * q_raw_f * corr_q
            del dQ_normed_total_bnhd, gw_q, corr_q, q_raw_f

            k_raw_f = k_raw_v.float()
            k_irms = k_inv_rms[:, :, None, None]
            gw_k = dK_normed_total_bnhd * k_nw_hd[None, None]
            dk_nw = (dK_normed_total_bnhd * k_raw_f * k_irms).sum(dim=(0, 1)).reshape(-1)
            corr_k = (gw_k * k_raw_f).sum(dim=(-2, -1), keepdim=True)
            dK_raw = k_irms * gw_k - (k_irms**3) / C * k_raw_f * corr_k
            del dK_normed_total_bnhd, gw_k, corr_k, k_raw_f

        # 9. Stack dqkv = [dQ_raw, dK_raw, dV_total] along the channel dim.
        dqkv = torch.stack(
            [dQ_raw.to(dtype), dK_raw.to(dtype), dV_total_bnhd.to(dtype)],
            dim=2,
        )

        # 10. Reshape dbeta / ddecay to match input shapes.
        dbeta_total = (dbeta_kv + dbeta_z).reshape(B, H, F, S)
        # ddecay was already reshaped to (B, H, F) above.

        # Preserve PyTorch's None-gradient contract for omitted norm weights.
        dq_nw_out = None if ctx.q_nw_was_none else dq_nw.to(q_norm_weight.dtype)
        dk_nw_out = None if ctx.k_nw_was_none else dk_nw.to(k_norm_weight.dtype)

        return (
            dqkv,
            dbeta_total.to(beta.dtype),
            ddecay.to(decay.dtype),
            dq_nw_out,
            dk_nw_out,
            None,  # rope_cos
            None,  # rope_sin
            None,  # F
            None,  # S
            None,  # k_scale
            None,  # norm_eps
            None,  # dot_precision
        )


class _CpFusedGdnOutput(torch.autograd.Function):
    """Inverse state adapter + Phase C as a single autograd Function.

    Forward composes one scan direction and does not accumulate reverse Phase C
    state.

      1. :func:`cp_scan_states_to_phase_c_states` re-pads + re-transposes the
         cp_frame_gdn_scan states ``(S_kv, S_z)`` to the BLOCK_D-padded
         left-multiply layout that Phase C expects.
      2. :func:`phase_c` with dummy ``q_inv_rms`` / ``q_norm_w`` (RMSNorm is
         already baked into ``qkv_normed`` from :class:`_CpFusedGdnPrep`).
         Returns raw ``(num, den)`` BEFORE the output divide (the caller
         is expected to fuse the divide downstream).

    Backward composes the Phase C VJP, inverse-state-adapter VJP, and
    Q-channel rope/relu VJP. K/V/decay handling lives in
    :class:`_CpFusedGdnPrep`.

      1. :func:`phase_c_bwd` on ``(Q_for_num_bhfsd, M_combined, dnum_bhfsd)``
         -> ``(dQ_kv_bhfsd, dM_C_active)`` where ``M_combined = M_kv_active``
         (CP single-direction has only forward contribution).
      2. Manual Z-chain VJP:
         ``dQ_z = (dden * z_active).to(dtype)`` and
         ``dz_C = (Q_for_den.float() * dden.float()).sum(dim=2)``
      3. Inverse-state-adapter VJP for ``(dS_kv, dS_z)``: the forward
         inverse adapter does ``M_kv_padded[..., :D, :D] = S_kv.T`` and
         ``M_z_padded[..., :D] = S_z``. Its VJP is
         ``dS_kv = dM_C_active.transpose(-1, -2)`` and
         ``dS_z  = dz_C_active``. ``phase_c_bwd`` already returns
         ``dM_C`` trimmed to the active ``D x D`` slice, so no explicit
         slice is needed here.
      4. Q-channel rope/relu VJP via :func:`fused_rope_unrope_bwd` with
         **zero** K-channel inputs (K does not flow through Phase C; only
         Q does). The returned ``dK_normed`` is structurally zero and is
         discarded.
      5. Assemble ``dqkv_normed``: Q channel = ``dQ_normed_bnhd``, K and V
         channels are zero.

    Returns 9 tensors matching the 9 forward inputs.

    Notes:
      * ``q_norm_weight`` is **NOT** taken as an input to Output's forward
        by design. Phase C consumes ``qkv_normed`` (which already has the Q RMSNorm scale
        baked in), so the kernel runs with ``dummy_nw = ones(C)``. The
        gradient for ``q_norm_weight`` flows back through ``dqkv_normed``
        into :class:`_CpFusedGdnPrep`'s backward, which owns the RMSNorm
        VJP.
      * ``z_active`` and ``M_kv_active`` correspond to the post-update
        state at each frame from the CP scan output (right-multiply
        convention transposed back to left-multiply). In the bidi
        reference these would be ``M_fwd_full[:, 1:]`` and
        ``z_fwd_full[:, 1:]`` (1-shifted to align with post-update at
        frame ``f``). CP's ``cp_frame_gdn_scan`` already emits the
        post-update state at each frame, so no shift is needed.
    """

    @staticmethod
    def forward(
        ctx,
        qkv_normed: Tensor,  # (B, N, 3, H, D) bf16, RMS-normed
        rope_cos: Tensor,  # (N, D) fp32, CP-local
        rope_sin: Tensor,  # (N, D) fp32, CP-local
        S_kv: Tensor,  # (BH, F, head_dim, head_dim) fp32
        S_z: Tensor,  # (BH, F, head_dim) fp32
        block_d: int,
        F: int,
        S: int,
        dot_precision: int = 0,
    ):
        from diffusion.model.ops.fused_gdn_chunkwise import phase_c

        B, N, three, H, D = qkv_normed.shape
        if three != 3:
            raise ValueError(f"_CpFusedGdnOutput.forward: qkv_normed dim 2 must equal 3, got {three}")
        if N != F * S:
            raise ValueError(f"_CpFusedGdnOutput.forward: N={N} must equal F*S={F * S}")
        BH = B * H
        if S_kv.shape != (BH, F, D, D):
            raise ValueError(
                f"_CpFusedGdnOutput.forward: S_kv shape {tuple(S_kv.shape)} != " f"(BH={BH}, F={F}, D={D}, D={D})"
            )
        if S_z.shape != (BH, F, D):
            raise ValueError(f"_CpFusedGdnOutput.forward: S_z shape {tuple(S_z.shape)} != " f"(BH={BH}, F={F}, D={D})")

        device = qkv_normed.device
        fp32 = torch.float32
        C = H * D

        # 1. Inverse state adapter: cp_scan output -> Phase C state layout.
        # (BH, F, head_dim, head_dim) right-multiply -> (BH, F, BLOCK_D, BLOCK_D)
        # left-multiply (transpose + pad).
        M_kv_padded, M_z_padded = cp_scan_states_to_phase_c_states(S_kv, S_z, block_d=block_d)

        # 2. Phase C with dummy norm; qkv_normed already carries RMSNorm.
        dummy_inv = torch.ones(B, N, device=device, dtype=fp32)
        dummy_nw = torch.ones(C, device=device, dtype=fp32)
        num, den = phase_c(
            qkv_normed,
            dummy_inv,
            dummy_nw,
            rope_cos,
            rope_sin,
            M_kv_padded,
            M_z_padded,
            F=F,
            S=S,
            dot_precision=dot_precision,
            accumulate=False,
        )

        # Save for backward. We keep M_kv_padded for phase_c_bwd's M input and
        # M_z_padded for the manual Z-chain VJP (we'll slice it to active D
        # there). We DON'T save num/den because Output's backward only needs
        # them via the divide VJP, which is done by the CALLER (Output returns
        # raw num/den; the divide VJP happens outside this Function in the
        # composition wrapper).
        ctx.save_for_backward(
            qkv_normed,
            rope_cos,
            rope_sin,
            M_kv_padded,
            M_z_padded,
        )
        ctx.shape = (B, N, H, D, F, S, C)
        ctx.block_d = int(block_d)
        ctx.dot_precision = int(dot_precision)

        return num, den

    @staticmethod
    def backward(ctx, dnum, dden):
        from diffusion.model.ops.fused_gdn_chunkwise_bwd import (
            _resolve_bwd_block_s,
            fused_rope_relu_fwd,
            fused_rope_unrope_bwd,
            phase_c_bwd,
        )

        (
            qkv_normed,
            rope_cos,
            rope_sin,
            M_kv_padded,
            M_z_padded,
        ) = ctx.saved_tensors
        B, N, H, D, F, S, C = ctx.shape
        dot_precision = ctx.dot_precision
        BLOCK_S = _resolve_bwd_block_s()
        device = qkv_normed.device
        fp32 = torch.float32
        dtype = qkv_normed.dtype
        BH = B * H

        # 1. Slice active M/z and recompute Q rope/relu intermediates. The CP
        # scan already returns post-update state aligned with each frame.
        M_kv_active = M_kv_padded[:, :, :D, :D].to(fp32).contiguous()  # (BH, F, D, D)
        z_active = M_z_padded[:, :, :D].to(fp32).contiguous()  # (BH, F, D)
        del M_kv_padded, M_z_padded

        # 2. Recompute Q_post_relu, Q_for_num, etc. from qkv_normed.
        # We need:
        #   - Q_for_num_bhfsd (post-rope, post-relu Q) for phase_c_bwd input
        #   - Q_for_den_bhfsd = Q_post_relu_bhfsd for the manual Z-chain VJP
        #   - Q_post_relu_bhfsd, K_post_relu_bhfsd for the relu-mask in
        #     fused_rope_unrope_bwd
        # K_kv/K_z are not used by Output's bwd (K does not enter Phase C);
        # we still receive them from fused_rope_relu_fwd but discard.
        def bnhd_to_bhfsd(x):
            return x.permute(0, 2, 1, 3).reshape(B, H, F, S, D).reshape(BH, F, S, D).contiguous()

        def bhfsd_to_bnhd(x):
            return x.reshape(BH, F * S, D).reshape(B, H, N, D).permute(0, 2, 1, 3).contiguous()

        Q_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 0])
        K_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 1])
        # k_scale: Output's forward doesn't take k_scale (it's only consumed by
        # Phase C internally and by fused_rope_relu_fwd). Phase C reads
        # k_scale=1.0 implicitly because the K stream there is gated by
        # `q_norm_w`, not k_norm_w. For the rope/relu recomputation of K
        # (whose output we'll discard), k_scale=1.0 is harmless.
        # However: fused_rope_unrope_bwd internally multiplies dK_normed by
        # k_scale; since dK_kv and dK_z inputs are zero, dK_normed=0 regardless.
        # So we can pass any k_scale here; use 1.0 for consistency with
        # phase_c's internal expectation that the Q channel is unscaled.
        k_scale_for_recompute = 1.0
        Q_post_relu_bhfsd, K_post_relu_bhfsd, Q_for_num_bhfsd, _K_kv_bhfsd_unused = fused_rope_relu_fwd(
            Q_normed_bhfsd,
            K_normed_bhfsd,
            rope_cos,
            rope_sin,
            k_scale_for_recompute,
            F,
            S,
        )
        del Q_normed_bhfsd, K_normed_bhfsd, _K_kv_bhfsd_unused
        Q_for_den_bhfsd = Q_post_relu_bhfsd  # alias: Phase C den path uses post-relu, pre-rope Q

        # 3. Phase-C backward. M_combined = M_kv_active for CP
        # single-direction, and phase_c_bwd returns active (BH, F, D, D).
        dnum_bhfsd = bnhd_to_bhfsd(dnum)
        dQ_kv_bhfsd, dM_C = phase_c_bwd(
            Q_for_num_bhfsd.contiguous(),
            M_kv_active,
            dnum_bhfsd,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )
        del dnum_bhfsd, M_kv_active

        # 4. Manual Z-chain VJP.
        # In bidi: z_combined = z_fwd_full[:, 1:] + z_rev_full[:, :F].
        # In CP single-direction: z_combined = z_active (no reverse term).
        dden_bhfs = dden.reshape(BH, F, S).contiguous()  # bf16 / fp32 same as den dtype
        # dQ_z is bf16 (or matches qkv_normed dtype). Cast at the end of unrope.
        # Use unsqueeze convention from the reference: dden (BH, F, S) -> (BH, F, S, 1);
        # z_active (BH, F, D) -> (BH, F, 1, D); broadcast to (BH, F, S, D).
        dQ_z_bhfsd = (dden_bhfs.float().unsqueeze(-1) * z_active.unsqueeze(2)).to(dtype)
        # dz_C: contribution to dM_z from Q_for_den.
        # Q_for_den_bhfsd is (BH, F, S, D), dden_bhfs is (BH, F, S).
        dz_C = (Q_for_den_bhfsd.float() * dden_bhfs.float().unsqueeze(-1)).sum(dim=2)  # (BH, F, D)

        # 5. Inverse-state-adapter VJP: (dM_C, dz_C) -> (dS_kv, dS_z).
        # Forward inverse adapter (cp_scan_states_to_phase_c_states):
        #     M_kv_padded[..., :D, :D] = S_kv.transpose(-1, -2)
        #     M_z_padded[..., :D] = S_z
        # phase_c_bwd already returned dM_C trimmed to the active D x D slice,
        # so the slice step is free. Just transpose for KV; identity for Z.
        dS_kv = dM_C.transpose(-1, -2).contiguous()  # (BH, F, D, D) fp32
        dS_z = dz_C.contiguous()  # (BH, F, D) fp32
        del dM_C, dz_C, z_active

        # 6. Q-channel rope/relu VJP. K does NOT enter Phase C, so
        # dK_kv = dK_z = 0. fused_rope_unrope_bwd accepts zero K inputs;
        # the kernel multiplies dK_normed by (K_relu > 0) * k_scale which is
        # zero anyway. We discard the returned dK_normed.
        dQ_kv_bhfsd_typed = dQ_kv_bhfsd.to(dtype) if dQ_kv_bhfsd.dtype != dtype else dQ_kv_bhfsd
        dK_kv_zero = torch.zeros_like(dQ_kv_bhfsd_typed)
        dK_z_zero = torch.zeros_like(dQ_z_bhfsd)
        dQ_normed_bhfsd, _dK_normed_bhfsd_zero = fused_rope_unrope_bwd(
            dQ_kv_bhfsd_typed,
            dK_kv_zero,
            dQ_z_bhfsd,
            dK_z_zero,
            Q_post_relu_bhfsd,
            K_post_relu_bhfsd,
            rope_cos,
            rope_sin,
            k_scale_for_recompute,
            F,
            S,
        )
        del dQ_kv_bhfsd, dQ_kv_bhfsd_typed, dQ_z_bhfsd, dK_kv_zero, dK_z_zero
        del Q_post_relu_bhfsd, K_post_relu_bhfsd, Q_for_num_bhfsd, Q_for_den_bhfsd
        del _dK_normed_bhfsd_zero  # K's grad from Output is structurally zero

        # Reshape BHFSD -> BNHD for the Q channel of dqkv_normed.
        dQ_normed_bnhd = bhfsd_to_bnhd(dQ_normed_bhfsd)
        del dQ_normed_bhfsd

        # 7. Assemble dqkv_normed: Q channel = dQ_normed, K = 0, V = 0.
        # K and V do not enter Phase C, so their grads from this Function are
        # structurally zero. Allocate with the same dtype/device as qkv_normed.
        dqkv_normed = torch.zeros_like(qkv_normed)
        dqkv_normed[:, :, 0] = dQ_normed_bnhd.to(dtype)
        # [:, :, 1] (K) and [:, :, 2] (V) remain zero.
        del dQ_normed_bnhd

        return (
            dqkv_normed,  # qkv_normed
            None,  # rope_cos
            None,  # rope_sin
            dS_kv,  # S_kv
            dS_z,  # S_z
            None,  # block_d
            None,  # F
            None,  # S
            None,  # dot_precision
        )


def cp_fused_gdn_chunkwise_raw_autograd(
    qkv: Tensor,
    beta: Tensor,
    decay: Tensor,
    q_norm_weight: Tensor | None,
    k_norm_weight: Tensor | None,
    rope_cos: Tensor,
    rope_sin: Tensor,
    *,
    F: int,
    S: int,
    group: ProcessGroup,
    k_scale: float = 1.0,
    norm_eps: float = 1e-5,
    eps: float = 1e-6,
    dot_precision: int = 0,
    reverse_rank_order: bool = False,
    truncate_to_active: int | None = None,
    use_checkpoint: bool | None = None,
    rope_cos_q: Tensor | None = None,
    rope_sin_q: Tensor | None = None,
) -> CpFusedGdnRawResult:
    """End-to-end differentiable CP fused GDN raw entry (num, den).

    Composes :class:`_CpFusedGdnPrep` -> :func:`cp_frame_gdn_scan` ->
    :class:`_CpFusedGdnOutput` with **full autograd** through
    ``qkv``, ``beta``, ``decay``, ``q_norm_weight``, ``k_norm_weight``.

    This is the reference training-path entry for the fused CP main branch.

    Pipeline:

      1. :class:`_CpFusedGdnPrep` fuses RMSNorm + :func:`phase_a` +
         :func:`phase_a_to_cp_scan_transitions` with a hand-written VJP
         that composes :func:`phase_a_kv_bwd` + :func:`phase_a_z_bwd` +
         :func:`fused_rope_unrope_bwd` + RMSNorm VJP.
      2. :func:`cp_frame_gdn_scan` is differentiable
         (:class:`FrameGDNScan` + :class:`_CPAllGatherMerge`).
      3. :class:`_CpFusedGdnOutput` fuses
         :func:`cp_scan_states_to_phase_c_states` + :func:`phase_c` with
         a hand-written VJP that composes :func:`phase_c_bwd` + manual
         Z-chain VJP + inverse-state-adapter VJP + Q-channel
         :func:`fused_rope_unrope_bwd`.

    The Q channel of ``qkv`` accumulates grads ONLY from Output's
    backward (RMSNorm VJP for Q lives in Prep). The K channel
    accumulates grads from BOTH Output's backward (added to the
    ``qkv_normed`` K channel) AND Prep's backward (via Phase A KV/Z).
    The V channel accumulates ONLY from Prep's backward (Phase A KV
    returns ``dV``; Phase C does not consume V).

    Args:
        qkv: ``(B, N, 3, H, D)`` bf16/fp32 local CP-rank slice. Channel
            0 = Q, 1 = K, 2 = V.
        beta: ``(B, H, F, S)`` bf16/fp32 per-token update gate.
        decay: ``(B, H, F)`` bf16/fp32 per-frame decay.
        q_norm_weight: ``(H*D,)`` fp32 or ``None`` (defaults to ones).
        k_norm_weight: ``(H*D,)`` fp32 or ``None`` (defaults to ones).
        rope_cos: ``(N, D)`` fp32 CP-local RoPE cosines.
        rope_sin: ``(N, D)`` fp32 CP-local RoPE sines.
        F: Local frame count (``N // S``).
        S: Spatial token count per frame.
        group: CP process group.
        k_scale: K scale factor used by Phase A (typically ``D ** -0.5``).
        norm_eps: RMSNorm epsilon. Default ``1e-5``.
        eps: Currently unused at this layer; reserved for the final
            ``num / (den + eps)`` divide done by the caller.
        dot_precision: 0 = TF32 bf16 bridge (default); 1 = TF32 fp32
            bridge; 2 = IEEE fp32 + fp32 bridge.
        reverse_rank_order: If True, :func:`cp_frame_gdn_scan` traverses
            the rank order in reverse. Used by BidirectionalGDN's
            backward recurrence consumer.
        truncate_to_active: terminal state at
            global position ``truncate_to_active - 1`` is broadcast to
            all CP ranks and returned in the result. When ``None``
            (default), the result's terminal-state fields are ``None``.
        use_checkpoint: When ``True``, wrap the Prep ->
            ``cp_frame_gdn_scan`` -> Output pipeline in
            ``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` so
            the saved tensors held by ``_CpFusedGdnPrep`` (13 tensors
            incl. ``qkv``/``decay``/``I_P_kv``/``A``/``I_P_z``/``B_z``)
            and ``_CpFusedGdnOutput`` (5 tensors incl.
            ``M_kv_padded``/``M_z_padded``) are discarded after forward
            and recomputed during backward. Trades ~10-20% extra backward
            compute for substantially lower forward peak memory. Mirrors
            the eager ``_forward_cp_scan`` path's ``grad_checkpoint``
            wrap around ``_build_transition_matrices``.

            When ``None`` (default), the flag auto-detects from autograd
            state: ``True`` iff :func:`torch.is_grad_enabled` AND any of
            ``qkv``/``beta``/``decay``/``q_norm_weight``/``k_norm_weight``
            has ``requires_grad=True``. This matches reference CamCtrl
            SFT practice (gradient_checkpointing default on during
            training, off during eval / no_grad). When ``True``/``False``,
            it is an explicit override.

    Returns:
        :class:`CpFusedGdnRawResult` with:
            ``num`` ``(B, N_local, H, D)`` raw numerator before output
            divide. Grad flows back to qkv/beta/decay/norm weights.
            ``den`` ``(B, H, N_local)`` raw denominator. Same.
            ``terminal_state_kv`` / ``terminal_state_z`` fp32, present
            only when ``truncate_to_active`` was set.

    Notes:
        * BLOCK_D is derived as ``triton.next_power_of_2(head_dim)``
          where ``head_dim = S_kv.shape[-1]``. For reference D=112 this
          yields BLOCK_D=128.
        * Numerator-only / camera-branch (``skip_z=True``) is NOT
          supported by this entry; :class:`_CpFusedGdnPrep` calls the
          adapter with ``skip_z=False``.
    """
    del eps  # API symmetry only; final divide is done by the caller.

    # Module-level local import to avoid a heavyweight top-level
    # dependency on the distributed/context_parallel subtree and the
    # triton package (which is a runtime-only dep of the fused kernels).
    import triton

    from diffusion.distributed.context_parallel.distributed_scan import (
        CpFrameGdnScanResult,
        cp_frame_gdn_scan,
    )

    use_checkpoint_resolved = _resolve_use_checkpoint(
        use_checkpoint,
        qkv,
        beta,
        decay,
        q_norm_weight,
        k_norm_weight,
    )

    rope_cos_q = rope_cos if rope_cos_q is None else rope_cos_q
    rope_sin_q = rope_sin if rope_sin_q is None else rope_sin_q

    def _inner_pipeline(
        qkv_in: Tensor,
        beta_in: Tensor,
        decay_in: Tensor,
        q_nw_in: Tensor | None,
        k_nw_in: Tensor | None,
        rope_cos_k_in: Tensor,
        rope_sin_k_in: Tensor,
        rope_cos_q_in: Tensor,
        rope_sin_q_in: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Composes Prep -> cp_frame_gdn_scan -> Output.

        Returns ``(num, den, terminal_state_kv, terminal_state_z)``;
        the terminal states are ``None`` when ``truncate_to_active`` is
        ``None`` and tensors otherwise.  ``torch.utils.checkpoint`` accepts
        ``None`` returns as long as the closure shape is consistent across
        forward and the recomputed forward in backward, which it is here
        (``truncate_to_active`` is captured from the outer scope).
        """
        # 1. Prep: RMSNorm + phase_a + adapter.
        W_kv, U_kv, W_z, U_z, qkv_normed = _CpFusedGdnPrep.apply(
            qkv_in,
            beta_in,
            decay_in,
            q_nw_in,
            k_nw_in,
            rope_cos_k_in,
            rope_sin_k_in,
            F,
            S,
            float(k_scale),
            float(norm_eps),
            int(dot_precision),
        )

        # 2. CP scan.
        if truncate_to_active is None:
            scan_result = cp_frame_gdn_scan(
                W_kv,
                U_kv,
                W_z,
                U_z,
                group=group,
                reverse=reverse_rank_order,
            )
            S_kv, S_z = scan_result
            terminal_state_kv_inner = None
            terminal_state_z_inner = None
        else:
            scan_result = cp_frame_gdn_scan(
                W_kv,
                U_kv,
                W_z,
                U_z,
                group=group,
                reverse=reverse_rank_order,
                truncate_to_active=int(truncate_to_active),
            )
            # Defensive type check so a scan API mismatch fails before
            # feeding invalid state downstream.
            if not isinstance(scan_result, CpFrameGdnScanResult):
                raise TypeError(
                    "cp_fused_gdn_chunkwise_raw_autograd: expected "
                    "CpFrameGdnScanResult from cp_frame_gdn_scan(truncate_to_active="
                    f"{truncate_to_active}), got {type(scan_result).__name__}"
                )
            S_kv = scan_result.S_kv_all
            S_z = scan_result.S_z_all
            terminal_state_kv_inner = scan_result.terminal_state_kv
            terminal_state_z_inner = scan_result.terminal_state_z

        # (3) BLOCK_D derivation: padded head dim for Phase C consumption.
        # S_kv shape: (BH, F, head_dim, head_dim).
        head_dim = S_kv.shape[-1]
        block_d = triton.next_power_of_2(head_dim)

        # (4) Output: inverse adapter + phase_c (autograd-aware).
        num_inner, den_inner = _CpFusedGdnOutput.apply(
            qkv_normed,
            rope_cos_q_in,
            rope_sin_q_in,
            S_kv,
            S_z,
            int(block_d),
            F,
            S,
            int(dot_precision),
        )
        return num_inner, den_inner, terminal_state_kv_inner, terminal_state_z_inner

    if use_checkpoint_resolved:
        from torch.utils.checkpoint import checkpoint as _grad_checkpoint

        num, den, terminal_state_kv, terminal_state_z = _grad_checkpoint(
            _inner_pipeline,
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            rope_cos_q,
            rope_sin_q,
            use_reentrant=False,
        )
    else:
        num, den, terminal_state_kv, terminal_state_z = _inner_pipeline(
            qkv,
            beta,
            decay,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            rope_cos_q,
            rope_sin_q,
        )

    return CpFusedGdnRawResult(
        num=num,
        den=den,
        terminal_state_kv=terminal_state_kv,
        terminal_state_z=terminal_state_z,
    )


def cp_fused_cam_gdn_num_autograd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    decay: Tensor,
    *,
    F: int,
    S: int,
    group: ProcessGroup,
    reverse_rank_order: bool = False,
    truncate_to_active: int | None = None,
    eps_recurrence: float = 0.0,
    use_checkpoint: bool | None = None,
) -> tuple[Tensor, Tensor | None]:
    """End-to-end differentiable CP camera-branch (num-only) **forward** scan.

    Composes pure-PyTorch transition build + the autograd-aware
    :func:`cp_frame_gdn_scan` + pure-PyTorch numerator output projection
    into a single autograd-correct path. The KV recurrence is
    ``M_t = decay_t * (I - k_rot*beta @ k_rot^T) @ M_{t-1} + (v*beta) @ k_rot^T``
    (camera num-only -- no Z denominator). All ops are vanilla PyTorch
    matmul/elementwise, so autograd flows back to ``q``/``k``/``v``/
    ``beta``/``decay`` natively without any custom VJP.

    The role of this wrapper relative to the main-branch fused entry
    (:func:`cp_fused_gdn_chunkwise_raw_autograd`) is more conservative:

    * The main branch reuses :func:`phase_a` / :func:`phase_c` Triton
      kernels with custom backward for forward speedup.
    * The camera branch uses **pure-PyTorch** transition build +
      output projection. The "fused" part of the camera path lives
      outside this function: it is the upstream
	      :func:`cam_prep_func_with_grad` Triton kernel which fuses
	      RMSNorm + ReLU + K-scale + UCPE-projmat + RoPE on the raw QKV.

    Args:
        q: ``(B, H, D, N)`` -- post-UCPE+RoPE rotated camera queries.
        k: ``(B, H, D, N)`` -- post-UCPE+RoPE rotated camera keys.
        v: ``(B, H, D, N)`` -- post-UCPE camera values.
        beta: ``(B, H, F, S)`` or ``(B, H, F)`` -- per-token update
            gate (camera-discounted). Reshaped internally to ``(B, H,
            F, 1, S)`` so the broadcast against ``(B, H, F, D, S)``
            frame tensors works.
        decay: ``(B, H, F)`` -- per-frame decay.
        F: Local frame count (``N // S``).
        S: Spatial token count per frame.
        group: CP process group.
        reverse_rank_order: Forwarded to :func:`cp_frame_gdn_scan`.
        truncate_to_active: When set, ``cp_frame_gdn_scan``
            masks padded positions and returns a terminal-state KV that
            is broadcast to all ranks. We surface it as the second
            tuple element so the caller can resume a local non-CP gen
            scan from that boundary state.
        eps_recurrence: API symmetry only; the camera num-only path performs
            no divide so this is currently unused.
        use_checkpoint: When ``True``, wrap the transition
            build -> ``cp_frame_gdn_scan`` -> output projection pipeline in
            ``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` so
            the saved intermediates (``k_rot_beta``, ``W_kv``, ``U_kv``,
            ``S_kv_all``, ``out_5d``) are discarded after forward and
            recomputed during backward. Trades ~10-20% extra backward
            compute for substantially lower forward peak memory.

            When ``None`` (default), auto-detect ``True`` iff
            :func:`torch.is_grad_enabled` AND any of ``q``/``k``/``v``/
            ``beta``/``decay`` has ``requires_grad=True``.  Matches
            reference CamCtrl SFT practice.

    Returns:
        ``(out_num, terminal_state_kv)`` where ``out_num`` has shape
        ``(B, H, D, N)`` (camera num-only output, no divide) and
        ``terminal_state_kv`` has shape ``(BH, D, D)`` when
        ``truncate_to_active`` was provided, else ``None``.
    """
    del eps_recurrence  # API symmetry

    from diffusion.distributed.context_parallel.distributed_scan import (
        CpFrameGdnScanResult,
        cp_frame_gdn_scan,
    )

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"cp_fused_cam_gdn_num_autograd: q/k/v shape mismatch -- "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.ndim != 4:
        raise ValueError(
            f"cp_fused_cam_gdn_num_autograd: expected q with 4 dims (B, H, D, N), got shape {tuple(q.shape)}"
        )
    B, H, D, N = q.shape
    if N != F * S:
        raise ValueError(f"cp_fused_cam_gdn_num_autograd: N={N} != F*S={F * S} (F={F}, S={S})")

    use_checkpoint_resolved = _resolve_use_checkpoint(
        use_checkpoint,
        q,
        k,
        v,
        beta,
        decay,
    )

    def _inner_pipeline(
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        beta_in: Tensor,
        decay_in: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Composes transition build -> cp_frame_gdn_scan -> output projection.

        Returns ``(out, terminal_state_kv)``; the terminal state is
        ``None`` when ``truncate_to_active`` is ``None``.
        """

        # 1. Reshape (B, H, D, N) -> frame layout (B, H, F, D, S).
        # Use ``view`` + ``permute`` to match the eager camera branch layout.
        def _to_frame(t: Tensor) -> Tensor:
            return t.view(B, H, D, F, S).permute(0, 1, 3, 2, 4).contiguous()

        q_f = _to_frame(q_in)
        k_f = _to_frame(k_in)
        v_f = _to_frame(v_in)
        if beta_in.ndim == 4:
            # beta is per-token (B, H, F, S) -- inject the D singleton at dim 3
            # so the frame broadcast (B, H, F, 1, S) works against (B, H, F, D, S).
            beta_f = beta_in.unsqueeze(3)
        elif beta_in.ndim == 3:
            # Per-frame (B, H, F) -> (B, H, F, 1, 1).
            beta_f = beta_in.view(B, H, F, 1, 1)
        else:
            raise ValueError(f"cp_fused_cam_gdn_num_autograd: beta.ndim must be 3 or 4, got {beta_in.ndim}")
        decay_f = decay_in.view(B, H, F, 1, 1)
        I = torch.eye(D, device=q_in.device, dtype=q_in.dtype).reshape(1, 1, 1, D, D)
        BH = B * H

        # 2. Build transitions (single-path: KV only, Z zeroed).
        # ``k_rot`` is used in both spots and v carries the input. Zero Z
        # matches the single-path numerator-only camera path.
        k_rot_beta = k_f * beta_f
        W_kv = decay_f * (I - torch.matmul(k_rot_beta, k_f.transpose(-1, -2)))
        U_kv = torch.matmul(v_f * beta_f, k_f.transpose(-1, -2))
        W_kv = W_kv.reshape(BH, F, D, D).contiguous()
        U_kv = U_kv.reshape(BH, F, D, D).contiguous()
        # Z is zeroed for the single-path numerator-only scan -- the
        # downstream output projection ignores the Z output and the scan's
        # backward returns zero gradients through the dummy Z slot. This
        # matches the eager numerator-only path.
        W_z = torch.zeros(BH, F, D, D, device=q_in.device, dtype=W_kv.dtype)
        U_z = torch.zeros(BH, F, D, device=q_in.device, dtype=W_kv.dtype)

        # 3. Distributed scan with autograd-aware all-gather merge.
        if truncate_to_active is None:
            scan_result = cp_frame_gdn_scan(
                W_kv,
                U_kv,
                W_z,
                U_z,
                group=group,
                reverse=reverse_rank_order,
            )
            S_kv_all, _ = scan_result  # discard zeroed Z output
            terminal_state_kv_inner = None
        else:
            scan_result = cp_frame_gdn_scan(
                W_kv,
                U_kv,
                W_z,
                U_z,
                group=group,
                reverse=reverse_rank_order,
                truncate_to_active=int(truncate_to_active),
            )
            if not isinstance(scan_result, CpFrameGdnScanResult):
                raise TypeError(
                    "cp_fused_cam_gdn_num_autograd: expected CpFrameGdnScanResult from "
                    f"cp_frame_gdn_scan(truncate_to_active={truncate_to_active}), "
                    f"got {type(scan_result).__name__}"
                )
            S_kv_all = scan_result.S_kv_all
            terminal_state_kv_inner = scan_result.terminal_state_kv

        # 4. Output projection: out[b,h,f] = S_kv[b,h,f] @ q_rot[b,h,f].
        # cp_frame_gdn_scan returns S_kv in right-multiply
        # convention (S_t = S_{t-1} @ W_t + U_t), so by the transpose
        # convention noted in the module docstring, M_t = S_t.T.
        S_kv_5d = S_kv_all.view(B, H, F, D, D)
        out_5d = torch.matmul(S_kv_5d, q_f)  # (B, H, F, D, S)
        # Permute back to (B, H, D, N).
        out_inner = out_5d.permute(0, 1, 3, 2, 4).reshape(B, H, D, N).contiguous()
        return out_inner, terminal_state_kv_inner

    if use_checkpoint_resolved:
        from torch.utils.checkpoint import checkpoint as _grad_checkpoint

        out, terminal_state_kv = _grad_checkpoint(
            _inner_pipeline,
            q,
            k,
            v,
            beta,
            decay,
            use_reentrant=False,
        )
    else:
        out, terminal_state_kv = _inner_pipeline(q, k, v, beta, decay)

    return out, terminal_state_kv
