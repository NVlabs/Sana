# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel wrappers around the fused Triton GDN Phase A/B/C kernels.

Provides state-convention adapters that translate between the fused-kernel
state layout and the recurrence convention expected by
:func:`cp_frame_gdn_scan`, plus higher-level wrappers for the main-branch
scan, the camera-branch numerator-only scan, and the bidirectional combine.

State-convention mapping (KV):
    Phase B uses a left-multiply recurrence with decay applied outside the
    factor ``I_P_kv``:
        M_t = decay_t * (I - P_t) @ M_{t-1} + A_t
    :func:`cp_frame_gdn_scan` uses a right-multiply recurrence with decay
    pre-folded:
        S_t = S_{t-1} @ W_t + U_t   with   S_t = M_t.T
    Therefore the adapter sets ``W_t = (decay_t * (I - P_t)).T`` and
    ``U_t = A_t.T``.

State-convention mapping (Z):
    Both Phase B and :func:`cp_frame_gdn_scan` use a left-multiply Z
    recurrence; decay is folded into ``W_z`` for the eager / cp_scan path
    but applied inside the Triton kernel for Phase B. Hence
    ``W_z = decay_t * I_P_z`` (no transpose) and ``U_z = B_z``.

Precision contract:
    Phase B promotes ``decay`` to fp32 internally and allocates ``M_fwd`` /
    ``z_fwd`` in fp32. To match the end-to-end fp32 state precision used by
    the eager reference under ``fp32_attention=True``, the adapter promotes
    ``I_P_kv``, ``A_kv``, ``I_P_z``, ``B_z``, and ``decay`` to fp32 before
    the multiply / transpose / slice. The returned bundle tensors are
    therefore always fp32 regardless of Phase A's bridge dtype.

The padded ``BLOCK_D`` slice ``[head_dim:BLOCK_D, :BLOCK_D]`` and
``[:BLOCK_D, head_dim:BLOCK_D]`` is structurally inert in both fused Phase B
and the eager scan (Phase A writes zeros into those tiles via masked
``tl.store``). The adapter slices the active ``D x D`` sub-block before
transposing so that any garbage that might land in the padded region during
a future CP exchange cannot poison the recurrence.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributed import ProcessGroup


def _should_force_no_checkpoint() -> bool:
    """Diagnostic-only knob to disable the gradient-checkpoint wrap.

    When ``CP_TRITON_BLOCK_FUSION_FORCE_NO_CHECKPOINT`` is set to a truthy
    value (``1``, ``true``, ``yes``, ``on``, case-insensitive), the fused-CP
    entry points skip the auto-baked
    ``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` wrap on the
    fused-forward pipeline even when ``use_checkpoint`` would otherwise
    resolve to ``True`` (training mode with autograd active). Production
    code paths should not set this env var.
    """
    return os.environ.get("CP_TRITON_BLOCK_FUSION_FORCE_NO_CHECKPOINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolve_use_checkpoint(use_checkpoint: bool | None, *leaves: Tensor | None) -> bool:
    """Resolve the ``use_checkpoint`` argument for the fused-CP entry points.

    Args:
        use_checkpoint: Explicit override (``True``/``False``) or ``None`` to
            auto-detect from autograd state. ``None`` resolves to ``True``
            iff :func:`torch.is_grad_enabled` AND any non-None leaf has
            ``requires_grad=True`` -- i.e., training mode with autograd
            actually active. This mirrors production CamCtrl SFT practice
            (gradient_checkpointing on during training, off during eval).
        *leaves: The input tensors that participate in autograd. ``None``
            entries (e.g. optional norm weights) are ignored.

    Returns:
        The effective ``use_checkpoint`` flag after applying the
        ``CP_TRITON_BLOCK_FUSION_FORCE_NO_CHECKPOINT`` diagnostic override.
    """
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled() and any((t is not None) and t.requires_grad for t in leaves)
    if use_checkpoint and _should_force_no_checkpoint():
        return False
    return bool(use_checkpoint)


__all__ = [
    "CpFusedGdnRawResult",
    "CpFusedTransitionBundle",
    "_CpFusedGdnOutput",
    "_CpFusedGdnPrep",
    "cp_fused_cam_gdn_num_autograd",
    "cp_fused_gdn_chunkwise_raw_autograd",
    "cp_fused_gdn_raw_ag",
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
            convention. Always fp32 in real Phase A (see
            ``fused_gdn_chunkwise.py:585``).
        decay:  Either ``(BH, T_local)`` or broadcastable to that shape.
            Reshaped + cast to float32 internally so the per-frame
            ``decay_t`` scalar can be multiplied against the
            ``(BLOCK_D, BLOCK_D)`` tile in a single broadcast.
        head_dim: Active head dimension ``D`` (Phase A pads to
            ``BLOCK_D = next_pow2(D)``; we slice the active sub-block here
            to defensively isolate the recurrence from any garbage in
            padded tiles -- see the perturbation unit test).
        skip_z: When True, return size-0 placeholders for ``W_z`` / ``U_z``.
            Used by the camera-branch numerator-only scan.

            CONTRACT: A ``skip_z=True`` bundle MUST NOT be passed directly
            to :func:`cp_frame_gdn_scan` -- the scan would fail with a
            shape mismatch on the size-0 ``W_z`` / ``U_z`` fields. The
            numerator-only camera-branch wrapper either supplies valid
            dummy Z tensors of shape ``(BH, T_local, head_dim, head_dim)``
            and ``(BH, T_local, head_dim)`` before calling the scan, or
            uses a NUM_ONLY scan path that does not consume Z.

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

    # Promote every per-frame input to fp32 before the multiply / transpose
    # / slice. Phase B operates internally on fp32 decay + fp32 M/z state,
    # so the CP path's transitions must also be fp32 to match the eager
    # reference under ``fp32_attention=True``.
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


# ---------------------------------------------------------------------------
# One-directional main-branch CP fused wrapper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CpFusedGdnRawResult:
    """Raw numerator/denominator output of the fused GDN CP scan.

    Returned by :func:`cp_fused_gdn_raw_ag`. Mirrors the
    ``(num, den)`` pair from :func:`fused_gdn_func_chunkwise`
    (``fused_gdn_chunkwise.py:1484``) but adds optional terminal-state
    fields when the caller requested ``truncate_to_active``.

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


def cp_fused_gdn_raw_ag(
    qkv: Tensor,
    beta: Tensor,
    decay: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    q_inv_rms: Tensor,
    k_inv_rms: Tensor,
    *,
    F: int,
    S: int,
    group: ProcessGroup,
    k_scale: float,
    norm_eps: float = 1e-5,
    eps: float = 1e-6,
    reverse_rank_order: bool = False,
    truncate_to_active: int | None = None,
    dot_precision: int = 0,
) -> CpFusedGdnRawResult:
    """One-directional GDN scan with fused Triton prep/output around an AG-CP scan.

    .. warning::

        **Forward-only / no-grad scaffolding.** This wrapper calls
        :func:`phase_a` and :func:`phase_c` directly (plain Python wrappers
        around Triton kernels — not :class:`torch.autograd.Function`
        subclasses), so the produced ``num``/``den`` are DETACHED from
        ``qkv``, ``beta``, ``q_norm_w``, ``k_norm_w``, ``rope_*``. Backward
        through any of those leaves WILL silently produce ``None`` grads
        and break training. The interior :class:`FrameGDNScan` *is*
        differentiable, but the chain is severed by Phase A / Phase C.

        For grad-enabled training paths, use the autograd-aware
        :func:`cp_fused_gdn_chunkwise_raw_autograd` instead. It composes
        :class:`_CpFusedGdnPrep` (``RMSNorm + phase_a + adapter`` with
        custom backward) and :class:`_CpFusedGdnOutput` (``inverse adapter
        + phase_c`` with custom backward) around the same
        :func:`cp_frame_gdn_scan` so gradients flow end-to-end back to
        ``qkv``/``beta``/``decay``/norm weights.

        This function is retained for inference / eval paths that
        explicitly do not require backward and want a single
        non-autograd-Function call.

    Composes the existing fused-non-CP pipeline (``fused_gdn_func_chunkwise``
    in ``fused_gdn_chunkwise.py:1484``) but REPLACES the local
    ``phase_b_triton`` step with the adapter + :func:`cp_frame_gdn_scan`
    + inverse adapter chain so the per-rank scan state is corrected via
    all-gather communication instead of recomputing the whole sequence
    on every rank.

    Pipeline (cf. ``fused_gdn_func_chunkwise:1510-1538``):

      1. :func:`phase_a` on the CP-local ``(qkv, beta)`` slice produces
         ``(I_P_kv, A_kv, I_P_z, B_z)`` on the local ``F`` frames.
      2. :func:`phase_a_to_cp_scan_transitions` maps these to
         :func:`cp_frame_gdn_scan` convention (W_kv/U_kv right-multiply,
         W_z/U_z left-multiply, both fp32 — see module docstring's
         "Precision contract" §1).
      3. :func:`cp_frame_gdn_scan` runs the corrected scan with comm
         contained inside its custom autograd Function.
         ``reverse=reverse_rank_order`` flips the rank traversal order;
         ``truncate_to_active`` activates terminal-state broadcast.
      4. :func:`cp_scan_states_to_phase_c_states` re-pads + re-transposes
         the corrected scan output to the BLOCK_D-padded left-multiply
         layout that Phase C expects.
      5. :func:`phase_c` projects ``(qkv, q_inv_rms, ...)`` against the
         scan states to produce ``(num, den)``.

    Args:
        qkv: ``(B, N_local, 3, H, D)`` -- standard SANA QKV tensor on the
            CP-local frame slice. Shape mirrors :func:`phase_a`.
        beta: ``(B, H, F, S)`` -- per-token update gate on the local
            slice. Reshaped internally for Phase A / cp_frame_gdn_scan.
        decay: ``(B, H, F)`` -- per-frame decay on the local slice.
        q_norm_weight: ``(H*D,)`` fp32 -- RMSNorm scale for Q.
        k_norm_weight: ``(H*D,)`` fp32 -- RMSNorm scale for K.
        rope_cos: ``(N_local, D)`` fp32 -- CP-local RoPE cosines.
        rope_sin: ``(N_local, D)`` fp32 -- CP-local RoPE sines.
        q_inv_rms: ``(B, N_local)`` fp32 -- pre-computed 1/RMS for Q via
            :func:`_precompute_inv_rms` or :func:`fused_qk_inv_rms`.
        k_inv_rms: ``(B, N_local)`` fp32 -- pre-computed 1/RMS for K.
        F: Local frame count (``N_local // S``).
        S: Spatial token count per frame.
        group: CP process group (must already be set via :func:`set_cp_group`
            for the inner :func:`cp_frame_gdn_scan` to dispatch correctly,
            but we additionally pass it explicitly here).
        k_scale: Key scale factor used by Phase A (typically ``D ** -0.5``).
        norm_eps: RMSNorm epsilon. Default ``1e-5`` mirrors fused_gdn defaults.
        eps: Unused at this layer; reserved for symmetry with future
            wrappers that perform the final ``num / (den + eps)`` divide.
        reverse_rank_order: If True, :func:`cp_frame_gdn_scan` traverses
            the rank order in reverse. Used by BidirectionalGDN's backward
            recurrence; the wrapper itself is symmetric — it just forwards
            the flag.
        truncate_to_active: Terminal state at global position
            ``truncate_to_active - 1`` is broadcast to all CP ranks and
            returned in ``CpFusedGdnRawResult``. When ``None`` (default),
            the result's terminal-state fields are ``None``.
        dot_precision: 0 = TF32 bf16 bridge (default); 1 = TF32 fp32
            bridge; 2 = IEEE fp32 + fp32 bridge. Forwarded to
            :func:`phase_a` and :func:`phase_c`.

    Returns:
        :class:`CpFusedGdnRawResult` carrying ``num`` (B, N_local, H, D),
        ``den`` (B, H, N_local), and optional terminal states.

    Notes:
        * All distributed comm lives inside :func:`cp_frame_gdn_scan`. No
          new autograd Function is introduced by this wrapper, and backward
          is NOT supported across the wrapper because :func:`phase_a` and
          :func:`phase_c` are non-autograd Triton wrappers. Use
          :func:`cp_fused_gdn_chunkwise_raw_autograd` for trainable paths.
        * The wrapper does not consume ``norm_eps`` for Phase A. Phase A
          reads its own ``NORM_EPS`` constant; the kwarg is accepted for
          API symmetry. ``q_inv_rms`` / ``k_inv_rms`` already encode the
          ``norm_eps`` from
          :func:`diffusion.model.ops.fused_gdn._precompute_inv_rms`.
        * ``skip_z=True`` Phase A path is not supported here; use the
          numerator-only camera-branch wrapper for that case.
    """
    del eps  # API symmetry only; final divide is done by the caller.
    # Guard against accidental trainable callers. Phase A / Phase C are
    # non-autograd Triton wrappers, so any leaf with ``requires_grad=True``
    # would silently get ``None`` grads at the boundary. Warn but do not
    # hard-raise: forward-only parity tests legitimately pass tensors with
    # ``requires_grad=True`` and discard the upstream chain.
    if any(
        t is not None and isinstance(t, torch.Tensor) and t.requires_grad
        for t in (qkv, beta, decay, q_norm_weight, k_norm_weight)
    ):
        warnings.warn(
            "cp_fused_gdn_raw_ag is forward-only scaffolding; grad-enabled "
            "inputs will silently lose gradients through phase_a/phase_c. "
            "Use cp_fused_gdn_chunkwise_raw_autograd for grad-enabled paths.",
            UserWarning,
            stacklevel=2,
        )
    # Local imports to avoid circular dependencies between this module
    # (model ops) and the distributed/context_parallel subtree, and to
    # keep the import cost paid only when the wrapper is actually used.
    from diffusion.distributed.context_parallel.distributed_scan import (
        CpFrameGdnScanResult,
        cp_frame_gdn_scan,
    )
    from diffusion.model.ops.fused_gdn_chunkwise import (
        phase_a,
        phase_c,
    )

    if qkv.ndim != 5:
        raise ValueError(f"cp_fused_gdn_raw_ag: expected qkv with 5 dims (B, N, 3, H, D), got shape {tuple(qkv.shape)}")
    B, N_local, three, H, D = qkv.shape
    if three != 3:
        raise ValueError(f"cp_fused_gdn_raw_ag: qkv dim 2 must equal 3, got {three}")
    if N_local != F * S:
        raise ValueError(f"cp_fused_gdn_raw_ag: N_local={N_local} must equal F*S={F * S} (F={F}, S={S})")

    # ── (1) Phase A: build raw (I-P_kv, A_kv, I-P_z, B_z) on the local slice ──
    # Same kwarg passing as fused_gdn_func_chunkwise:1511-1524.
    I_P_kv, A_kv, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )

    # ── (2) Adapter: Phase A layout → cp_frame_gdn_scan convention ──
    bundle = phase_a_to_cp_scan_transitions(
        I_P_kv,
        A_kv,
        I_P_z,
        B_z,
        decay,
        head_dim=D,
        skip_z=False,
    )

    # ── (3) Corrected CP scan via all-gather. comm is inside the inner
    #         autograd Function; no extra wrapping needed here. ──
    if truncate_to_active is not None:
        scan_result = cp_frame_gdn_scan(
            bundle.W_kv,
            bundle.U_kv,
            bundle.W_z,
            bundle.U_z,
            group=group,
            reverse=reverse_rank_order,
            truncate_to_active=truncate_to_active,
        )
        # When truncate_to_active is set, the scan returns a NamedTuple
        # with 4 fields. Defensive isinstance check so a future signature
        # change crashes loudly here instead of silently feeding garbage
        # downstream.
        if not isinstance(scan_result, CpFrameGdnScanResult):
            raise TypeError(
                "cp_fused_gdn_raw_ag: expected CpFrameGdnScanResult from "
                f"cp_frame_gdn_scan(truncate_to_active={truncate_to_active}), "
                f"got {type(scan_result).__name__}"
            )
        S_kv = scan_result.S_kv_all
        S_z = scan_result.S_z_all
        terminal_state_kv = scan_result.terminal_state_kv
        terminal_state_z = scan_result.terminal_state_z
    else:
        scan_result = cp_frame_gdn_scan(
            bundle.W_kv,
            bundle.U_kv,
            bundle.W_z,
            bundle.U_z,
            group=group,
            reverse=reverse_rank_order,
        )
        S_kv, S_z = scan_result
        terminal_state_kv = None
        terminal_state_z = None

    # ── (4) Inverse adapter: cp_scan output → Phase C's BLOCK_D-padded
    #         left-multiply state layout. ──
    M_kv_padded, M_z_padded = cp_scan_states_to_phase_c_states(
        S_kv,
        S_z,
        block_d=bundle.block_d,
    )

    # ── (5) Phase C: project (qkv, q_inv_rms, RoPE, ...) against M, z ──
    # Same kwarg passing as fused_gdn_func_chunkwise:1537-1539.
    num, den = phase_c(
        qkv,
        q_inv_rms,
        q_norm_weight,
        rope_cos,
        rope_sin,
        M_kv_padded,
        M_z_padded,
        F=F,
        S=S,
        dot_precision=dot_precision,
    )

    return CpFusedGdnRawResult(
        num=num,
        den=den,
        terminal_state_kv=terminal_state_kv,
        terminal_state_z=terminal_state_z,
    )


# ---------------------------------------------------------------------------
# _CpFusedGdnPrep autograd Function.
# ---------------------------------------------------------------------------


class _CpFusedGdnPrep(torch.autograd.Function):
    """RMSNorm + Phase A + transition adapter as a single autograd Function.

    Forward composes (matching ``FusedBiGDNChunkwiseFunction.forward:944-973``
    minus Phase B / Phase C):

      1. Full-channel RMSNorm on Q and K channels of ``qkv``. V is not
         normalized.
      2. :func:`phase_a` on the normalized ``qkv_normed`` with identity
         ``inv_rms`` / norm-weight (so the Phase A kernel does no further
         norm). Returns ``(I_P_kv, A, I_P_z, B_z)``.
      3. :func:`phase_a_to_cp_scan_transitions` adapts to the
         :func:`cp_frame_gdn_scan` convention. Returns
         ``CpFusedTransitionBundle(W_kv, U_kv, W_z, U_z, ...)``.

    Backward composes:

      1. Inverse adapter VJP: ``(dW_kv, dU_kv, dW_z, dU_z) → (dI_P_kv,
         dA, dI_P_z, dB_z, ddecay)``. ``dI_P_*`` / ``dA`` are padded back
         to ``(BH, F, BLOCK_D, BLOCK_D)`` and ``dB_z`` to ``(BH, F,
         BLOCK_D)`` for Phase Ā kernel consumption.
      2. :func:`phase_a_kv_bwd` on ``(dA, -dI_P_kv)`` (since the kernel
         takes ``dP_kv`` and ``I_P_kv = I - P_kv``, ``dP_kv = -dI_P_kv``).
         Returns ``(dK_kv_bhfsd, dV_bhfsd, dbeta_kv_bhfs)``.
      3. :func:`phase_a_z_bwd` analogous → ``(dK_z_bhfsd, dbeta_z_bhfs)``.
      4. :func:`fused_rope_unrope_bwd` combines the K-channel grads
         coming out of ``phase_a_*_bwd`` (with ``dQ_kv``/``dQ_z`` zero
         since Q has no Phase A grad) and unrope+relu-masks them →
         ``(_dQ_zero_via_relu, dK_normed_bhfsd_from_phase_a)``.
      5. Add the ``dqkv_normed`` upstream grad contributions:
         - Q: ``dQ_normed_total = dqkv_normed[:, :, 0]_bhfsd`` (no Phase A path).
         - K: ``dK_normed_total = dK_normed_bhfsd_from_phase_a + dqkv_normed[:, :, 1]_bhfsd``.
         - V: ``dV_total_bnhd = dV_from_phase_a_kv_bwd_bnhd + dqkv_normed[:, :, 2]_bnhd``.
      6. Per-channel RMSNorm VJP for Q and K → ``dQ_raw``, ``dK_raw``,
         ``dq_norm_w``, ``dk_norm_w``.
      7. Stack ``dqkv = stack([dQ_raw, dK_raw, dV_total], dim=2)``.

    The decay grad is accumulated entirely inside step 1 (since
    ``W_kv = decay * I_P_kv`` and ``W_z = decay * I_P_z``); the Phase
    Ā kernels do not contribute to ``ddecay``.

    The qkv_normed output is differentiable: ``_CpFusedGdnOutput``
    (sibling Function) consumes it and its backward produces
    ``dqkv_normed`` which is summed with the Phase Ā chain above to
    yield the full ``dqkv``.
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

        # Track whether caller passed ``None`` for the norm weights so the
        # backward can return ``None`` grads for those slots (PyTorch
        # custom autograd contract: non-Tensor inputs must get ``None``
        # grads).
        ctx.q_nw_was_none = q_norm_weight is None
        ctx.k_nw_was_none = k_norm_weight is None

        # Semantic gating contract: when both ``q_norm_weight`` and
        # ``k_norm_weight`` are ``None``, skip the RMSNorm pass entirely
        # (``qkv_normed === qkv``). The eager reference uses
        # ``nn.Identity`` for the norms when the model was constructed
        # with ``qk_norm=False`` -- this branch matches that behavior.
        # The backward mirrors this by skipping the RMSNorm VJP.
        skip_rmsnorm = ctx.q_nw_was_none and ctx.k_nw_was_none
        ctx.skip_rmsnorm = skip_rmsnorm
        if q_norm_weight is None:
            q_norm_weight = torch.ones(C, device=device, dtype=fp32)
        if k_norm_weight is None:
            k_norm_weight = torch.ones(C, device=device, dtype=fp32)

        # ──── 1. Full-channel RMSNorm on Q and K (mirror fused_gdn_chunkwise_bwd.py:944-954) ────
        q_raw_v = qkv[:, :, 0]  # view, same dtype as qkv
        k_raw_v = qkv[:, :, 1]
        if skip_rmsnorm:
            # Identity contract: qkv_normed === qkv. Save ones for
            # q_inv_rms / k_inv_rms so the backward RMSNorm-VJP
            # bookkeeping is well-defined but the VJP degenerates to
            # the identity (Section "RMSNorm VJP" below skips both
            # divide-by-rms scaling and the per-token correction).
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

        # ──── 2. phase_a with identity inv_rms / norm_w (so kernel does no re-norm) ────
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

        # ──── 3. Adapter: Phase A layout → cp_frame_gdn_scan convention ────
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
        ctx.norm_eps = float(norm_eps)
        ctx.dot_precision = int(dot_precision)

        return bundle.W_kv, bundle.U_kv, bundle.W_z, bundle.U_z, qkv_normed

    @staticmethod
    def backward(ctx, dW_kv, dU_kv, dW_z, dU_z, dqkv_normed):
        # Inline imports — keep top-of-file lightweight.
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
        norm_eps = ctx.norm_eps  # noqa: F841 — kept for documentation; RMSNorm VJP uses q_inv_rms directly
        dot_precision = ctx.dot_precision
        BLOCK_S = _resolve_bwd_block_s()
        device = qkv.device
        fp32 = torch.float32
        dtype = qkv.dtype
        BH = B * H

        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)

        # ──── 1. Inverse adapter VJP ────
        # Forward adapter (with full-pad slicing applied — see
        # phase_a_to_cp_scan_transitions:206-228):
        #   W_kv_left[bh, f, :D, :D] = decay[bh, f] * I_P_kv_active[bh, f, :D, :D]
        #   W_kv[bh, f, d1, d2]      = W_kv_left[bh, f, d2, d1]   (transpose)
        #   U_kv[bh, f, d1, d2]      = A_kv_active[bh, f, d2, d1] (transpose)
        #   W_z[bh, f, d1, d2]       = decay[bh, f] * I_P_z_active[bh, f, d1, d2]
        #   U_z[bh, f, d]            = B_z_active[bh, f, d]
        # All cast to fp32 first; B_z is already fp32.
        I_P_kv.shape[-1]
        decay_f = decay.reshape(BH, F).to(fp32)
        decay_view = decay_f.view(BH, F, 1, 1)

        # Active D×D slice in fp32.
        I_P_kv_active = I_P_kv[..., :D, :D].to(fp32)
        I_P_z_active = I_P_z[..., :D, :D].to(fp32)

        # Inputs to inverse VJP — fp32.
        dW_kv_f = dW_kv.to(fp32)
        dU_kv_f = dU_kv.to(fp32)
        dW_z_f = dW_z.to(fp32)
        dU_z_f = dU_z.to(fp32)

        # Convert W_kv = (decay * I_P_kv_active).T → its VJP for I_P_kv_active is
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
        # pass D×D which they will pad. The kernels accept either D×D or
        # BLOCK_D×BLOCK_D inputs — see the `pad_DxD` helpers in those
        # driver functions.)
        # Sign flip: kernel expects dP (where P = I - I_P), so dP = -dI_P.
        dP_kv = (-dI_P_kv_active).contiguous()
        dP_z = (-dI_P_z_active).contiguous()
        dA_kv_for_kernel = dA_kv_active.contiguous()
        dB_z_for_kernel = dB_z_active.contiguous()

        # ──── 2. Reconstruct qkv_normed for the rope/relu recomputation ────
        # Under the skip-RMSNorm gating, the saved q_inv_rms / k_inv_rms /
        # q_nw / k_nw are all ones, so the reconstruction is mathematically
        # equivalent to ``qkv_normed = qkv``; branch to avoid the unneeded
        # clone + cast roundtrip.
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        skip_rmsnorm = getattr(ctx, "skip_rmsnorm", False)
        if skip_rmsnorm:
            qkv_normed = qkv
        else:
            qkv_normed = qkv.clone()
            qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(dtype)
            qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(dtype)

        # ──── 3. Recompute Q/K post_relu (in BHFSD layout) for the
        # ── fused rope+relu (matches forward Phase A internal — used as
        # ── rope/relu masks in fused_rope_unrope_bwd). ────
        def bnhd_to_bhfsd(x):
            return x.permute(0, 2, 1, 3).reshape(B, H, F, S, D).reshape(BH, F, S, D).contiguous()

        def bhfsd_to_bnhd(x):
            return x.reshape(BH, F * S, D).reshape(B, H, N, D).permute(0, 2, 1, 3).contiguous()

        Q_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 0])
        K_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 1])
        V_bhfsd = bnhd_to_bhfsd(qkv[:, :, 2])

        # fused_rope_relu_fwd returns (Q_post_relu, K_post_relu, Q_for_num, K_kv).
        # We need: K_kv_bhfsd (post-rope, key chain for Phase Ā KV) and
        #          K_post_relu_bhfsd (no rope on K_z, key chain for Phase Ā Z).
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

        # ──── 4. Phase Ā KV: (dA, dP_kv) → (dK_kv_bhfsd, dV_bhfsd, dbeta_kv) ────
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

        # ──── 5. Phase Ā Z: (dB_z, dP_z) → (dK_z_bhfsd, dbeta_z) ────
        dK_z_bhfsd, dbeta_z = phase_a_z_bwd(
            K_z_bhfsd.contiguous(),
            beta_bhfs,
            dB_z_for_kernel,
            dP_z,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # ──── 6. Combine via fused_rope_unrope_bwd. Q has no Phase Ā
        # ── contribution → pass zeros for dQ_kv and dQ_z. ────
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

        # dQ_normed_via_relu_rope_bhfsd should be zero (since inputs were zero).
        # We still keep it around for the audit but discard the value — Q's
        # Phase A grad is structurally zero.
        del dQ_normed_via_relu_rope_bhfsd

        # ──── 7. Add upstream dqkv_normed contribution ────
        # dqkv_normed shape: (B, N, 3, H, D), dtype = dtype.
        # Channel layout: 0 = Q, 1 = K, 2 = V.
        dqkv_normed_Q_bnhd = dqkv_normed[:, :, 0].contiguous()
        dqkv_normed_K_bnhd = dqkv_normed[:, :, 1].contiguous()
        dqkv_normed_V_bnhd = dqkv_normed[:, :, 2].contiguous()

        # Convert Q to BHFSD (for RMSNorm VJP we'll bring back to BNHD).
        # Q has NO Phase Ā contribution, so dQ_normed_total_bnhd is just the
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

        # ──── 8. RMSNorm VJP ────
        # d/dx = inv_rms*w*d/dy - (inv_rms^3 / C) * x * Σ(w*d/dy*x)
        # When the caller passed both ``q_norm_weight=None`` and
        # ``k_norm_weight=None``, the forward skipped RMSNorm entirely
        # (qkv_normed === qkv). The VJP collapses to the identity and
        # there is no weight grad to compute; we still allocate zero
        # fp32 tensors so the None-vs-tensor decision in the return
        # tuple below can be driven by ``ctx.q_nw_was_none`` /
        # ``ctx.k_nw_was_none``.
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

        # ──── 9. Stack dqkv = [dQ_raw, dK_raw, dV_total] along the channel dim ────
        dqkv = torch.stack(
            [dQ_raw.to(dtype), dK_raw.to(dtype), dV_total_bnhd.to(dtype)],
            dim=2,
        )

        # ──── 10. Reshape dbeta / ddecay to match input shapes ────
        dbeta_total = (dbeta_kv + dbeta_z).reshape(B, H, F, S)
        # ddecay was already reshaped to (B, H, F) above.

        # When caller passed ``q_norm_weight=None`` / ``k_norm_weight=None``
        # for the forward, the corresponding grad slot must be ``None``
        # per the PyTorch autograd contract.
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


# ---------------------------------------------------------------------------
# _CpFusedGdnOutput autograd Function.
# ---------------------------------------------------------------------------


class _CpFusedGdnOutput(torch.autograd.Function):
    """Inverse state adapter + Phase C as a single autograd Function.

    Forward composes (mirroring ``FusedBiGDNChunkwiseFunction.forward:975-1003``
    SINGLE-direction — no reverse Phase C accumulation):

      1. :func:`cp_scan_states_to_phase_c_states` re-pads + re-transposes the
         cp_frame_gdn_scan states ``(S_kv, S_z)`` to the BLOCK_D-padded
         left-multiply layout that Phase C expects.
      2. :func:`phase_c` with dummy ``q_inv_rms`` / ``q_norm_w`` (RMSNorm is
         already baked into ``qkv_normed`` from :class:`_CpFusedGdnPrep` —
         mirrors lines ``fused_gdn_chunkwise_bwd.py:957-958, 975-987``).
         Returns raw ``(num, den)`` BEFORE the output divide (the caller
         is expected to fuse the divide downstream).

    Backward composes (mirroring ``FusedBiGDNChunkwiseFunction.backward:1156-1289``
    minus Phase B̄ / Phase Ā chains and minus the K/V/decay handling
    which live in :class:`_CpFusedGdnPrep`):

      1. :func:`phase_c_bwd` on ``(Q_for_num_bhfsd, M_combined, dnum_bhfsd)``
         → ``(dQ_kv_bhfsd, dM_C_active)`` where ``M_combined = M_kv_active``
         (CP single-direction has only forward contribution, unlike the
         bidi reference at line 1158 which adds reverse).
      2. Manual Z-chain VJP:
         ``dQ_z = (dden * z_active).to(dtype)`` (mirrors line 1188 with
         ``z_combined = z_active``) and
         ``dz_C = (Q_for_den.float() * dden.float()).sum(dim=2)``
         (mirrors line 1189).
      3. Inverse-state-adapter VJP for ``(dS_kv, dS_z)``: the forward
         inverse adapter does ``M_kv_padded[..., :D, :D] = S_kv.T`` and
         ``M_z_padded[..., :D] = S_z``. Its VJP is
         ``dS_kv = dM_C_active.transpose(-1, -2)`` and
         ``dS_z  = dz_C_active``. ``phase_c_bwd`` already returns
         ``dM_C`` trimmed to the active ``D×D`` slice, so no explicit
         slice is needed here.
      4. Q-channel rope/relu VJP via :func:`fused_rope_unrope_bwd` with
         **zero** K-channel inputs (K does not flow through Phase C; only
         Q does). The returned ``dK_normed`` is structurally zero and is
         discarded.
      5. Assemble ``dqkv_normed``: Q channel = ``dQ_normed_bnhd``, K and V
         channels are zero.

    Returns 9 tensors matching the 9 forward inputs.

    Notes:
      * ``q_norm_weight`` is NOT taken as an input to Output's forward.
        Phase C consumes ``qkv_normed`` (which already has the Q RMSNorm
        scale baked in), so the kernel runs with ``dummy_nw = ones(C)``.
        The gradient for ``q_norm_weight`` flows back through
        ``dqkv_normed`` into :class:`_CpFusedGdnPrep`'s backward, which
        owns the RMSNorm VJP.
      * ``z_active`` and ``M_kv_active`` correspond to the post-update
        state at each frame from the CP scan output (right-multiply
        convention transposed back to left-multiply). In the bidi
        reference these would be ``M_fwd_full[:, 1:]`` and
        ``z_fwd_full[:, 1:]`` (1-shifted to align with post-update at
        frame ``f``). CP's ``cp_frame_gdn_scan`` already emits the
        post-update state at each frame (see
        ``distributed_scan.py:721, :744``), so no shift is needed.
    """

    @staticmethod
    def forward(
        ctx,
        qkv_normed: Tensor,  # (B, N, 3, H, D) bf16 — from _CpFusedGdnPrep, RMS-normed
        rope_cos: Tensor,  # (N, D) fp32 — CP-local
        rope_sin: Tensor,  # (N, D) fp32 — CP-local
        S_kv: Tensor,  # (BH, F, head_dim, head_dim) fp32 — from cp_frame_gdn_scan
        S_z: Tensor,  # (BH, F, head_dim) fp32 — from cp_frame_gdn_scan
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

        # ──── 1. Inverse state adapter: cp_scan output → Phase C state layout ────
        # (BH, F, head_dim, head_dim) right-multiply → (BH, F, BLOCK_D, BLOCK_D)
        # left-multiply (transpose + pad). Same op as in cp_fused_gdn_raw_ag.
        M_kv_padded, M_z_padded = cp_scan_states_to_phase_c_states(S_kv, S_z, block_d=block_d)

        # ──── 2. Phase C with dummy norm: qkv_normed already carries the RMSNorm ────
        # Mirrors fused_gdn_chunkwise_bwd.py:957-958, 975-987 (single direction; no
        # accumulate=True second call for the reverse direction).
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
        qkv_normed.device
        fp32 = torch.float32
        dtype = qkv_normed.dtype
        BH = B * H

        # ──── 1. Slice the active M / z and recompute Q's rope/relu intermediates ────
        # Mirror lines 1096, 1100 of the bidi reference: slice padded BLOCK_D
        # → active D for the bwd math. The bidi reference also pads with a
        # leading zero frame (line 1106) — that is for shifting M_fwd_full
        # so post-update at frame f is at index f+1, then it slices
        # M_fwd_full[:, 1:] which equals our M_kv_active directly. For CP
        # single-direction we use M_kv_active as-is (see class docstring).
        M_kv_active = M_kv_padded[:, :, :D, :D].to(fp32).contiguous()  # (BH, F, D, D)
        z_active = M_z_padded[:, :, :D].to(fp32).contiguous()  # (BH, F, D)
        del M_kv_padded, M_z_padded

        # ──── 2. Recompute Q_post_relu, Q_for_num, etc. from qkv_normed ────
        # Mirror lines 1117-1141 of the bidi reference. We need:
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

        # ──── 3. Phase C̄: (Q_for_num, M_kv_active, dnum_bhfsd) → (dQ_kv, dM_C) ────
        # Mirror lines 1159-1161 of the bidi reference. M_combined = M_kv_active
        # for CP single-direction (no reverse contribution). phase_c_bwd
        # returns dM_C already trimmed to (BH, F, D, D).
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

        # ──── 4. Manual Z-chain VJP (mirror lines 1184-1189 of bidi ref) ────
        # In bidi: z_combined = z_fwd_full[:, 1:] + z_rev_full[:, :F].
        # In CP single-direction: z_combined = z_active (no reverse term).
        dden_bhfs = dden.reshape(BH, F, S).contiguous()  # bf16 / fp32 same as den dtype
        # dQ_z is bf16 (or matches qkv_normed dtype). Cast at the end of unrope.
        # Use unsqueeze convention from the reference: dden (BH, F, S) → (BH, F, S, 1);
        # z_active (BH, F, D) → (BH, F, 1, D); broadcast to (BH, F, S, D).
        dQ_z_bhfsd = (dden_bhfs.float().unsqueeze(-1) * z_active.unsqueeze(2)).to(dtype)
        # dz_C: contribution to dM_z from Q_for_den.
        # Q_for_den_bhfsd is (BH, F, S, D), dden_bhfs is (BH, F, S).
        dz_C = (Q_for_den_bhfsd.float() * dden_bhfs.float().unsqueeze(-1)).sum(dim=2)  # (BH, F, D)

        # ──── 5. Inverse-state-adapter VJP: (dM_C, dz_C) → (dS_kv, dS_z) ────
        # Forward inverse adapter (cp_scan_states_to_phase_c_states):
        #     M_kv_padded[..., :D, :D] = S_kv.transpose(-1, -2)
        #     M_z_padded[..., :D] = S_z
        # phase_c_bwd already returned dM_C trimmed to the active D×D slice,
        # so the slice step is free. Just transpose for KV; identity for Z.
        dS_kv = dM_C.transpose(-1, -2).contiguous()  # (BH, F, D, D) fp32
        dS_z = dz_C.contiguous()  # (BH, F, D) fp32
        del dM_C, dz_C, z_active

        # ──── 6. Q-channel rope/relu VJP — K side is structurally zero ────
        # Mirror lines 1241-1253 of bidi ref. K does NOT enter Phase C, so
        # dK_kv = dK_z = 0. fused_rope_unrope_bwd accepts the zero K inputs;
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

        # Reshape BHFSD → BNHD for the Q channel of dqkv_normed.
        dQ_normed_bnhd = bhfsd_to_bnhd(dQ_normed_bhfsd)
        del dQ_normed_bhfsd

        # ──── 7. Assemble dqkv_normed: Q channel = dQ_normed, K = 0, V = 0 ────
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


# ---------------------------------------------------------------------------
# End-to-end autograd composition wrapper.
# ---------------------------------------------------------------------------


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
) -> CpFusedGdnRawResult:
    """End-to-end differentiable CP fused GDN raw entry (num, den).

    Composes :class:`_CpFusedGdnPrep` → :func:`cp_frame_gdn_scan` →
    :class:`_CpFusedGdnOutput` with **full autograd** through
    ``qkv``, ``beta``, ``decay``, ``q_norm_weight``, ``k_norm_weight``.

    This is the production training-path entry. The forward-only
    :func:`cp_fused_gdn_raw_ag` calls :func:`phase_a` / :func:`phase_c`
    directly and cannot flow grads back through ``qkv`` / ``beta`` /
    norm-weights; use this function whenever ``requires_grad=True`` on
    any of those leaves.

    Pipeline:

      1. :class:`_CpFusedGdnPrep` — fuses RMSNorm + :func:`phase_a` +
         :func:`phase_a_to_cp_scan_transitions` with a hand-written VJP
         that composes :func:`phase_a_kv_bwd` + :func:`phase_a_z_bwd` +
         :func:`fused_rope_unrope_bwd` + RMSNorm VJP.
      2. :func:`cp_frame_gdn_scan` — already differentiable
         (:class:`FrameGDNScan` + :class:`_CPAllGatherMerge`).
      3. :class:`_CpFusedGdnOutput` — fuses
         :func:`cp_scan_states_to_phase_c_states` + :func:`phase_c` with
         a hand-written VJP that composes :func:`phase_c_bwd` + manual
         Z-chain VJP + inverse-state-adapter VJP + Q-channel
         :func:`fused_rope_unrope_bwd`.

    The Q channel of ``qkv`` accumulates grads ONLY from Output's
    backward (RMSNorm VJP for Q lives in Prep). The K channel
    accumulates grads from BOTH Output's backward (added to the
    ``qkv_normed`` K channel) AND Prep's backward (via Phase Ā KV/Z).
    The V channel accumulates ONLY from Prep's backward (Phase Ā KV
    returns ``dV``; Phase C does not consume V).

    Args:
        qkv: ``(B, N, 3, H, D)`` bf16/fp32 — local CP-rank slice. Channel
            0 = Q, 1 = K, 2 = V.
        beta: ``(B, H, F, S)`` bf16/fp32 — per-token update gate.
        decay: ``(B, H, F)`` bf16/fp32 — per-frame decay.
        q_norm_weight: ``(H*D,)`` fp32 or ``None`` (defaults to ones).
        k_norm_weight: ``(H*D,)`` fp32 or ``None`` (defaults to ones).
        rope_cos: ``(N, D)`` fp32 — CP-LOCAL RoPE cosines.
        rope_sin: ``(N, D)`` fp32 — CP-LOCAL RoPE sines.
        F: Local frame count (``N // S``).
        S: Spatial token count per frame.
        group: CP process group.
        k_scale: K scale factor used by Phase A (typically ``D ** -0.5``).
            Default ``1.0``.
        norm_eps: RMSNorm epsilon. Default ``1e-5``.
        eps: Currently unused at this layer; reserved for the final
            ``num / (den + eps)`` divide done by the caller.
        dot_precision: 0 = TF32 bf16 bridge (default); 1 = TF32 fp32
            bridge; 2 = IEEE fp32 + fp32 bridge.
        reverse_rank_order: If True, :func:`cp_frame_gdn_scan` traverses
            the rank order in reverse. Used by BidirectionalGDN's
            backward recurrence consumer.
        truncate_to_active: When provided, terminal state at global
            position ``truncate_to_active - 1`` is broadcast to all CP
            ranks and returned in the result. When ``None`` (default),
            the result's terminal-state fields are ``None``.
        use_checkpoint: When ``True``, wrap the Prep → CP scan → Output
            pipeline in ``torch.utils.checkpoint.checkpoint`` so the
            saved tensors are discarded after forward and recomputed
            during backward. Trades extra backward compute for lower
            forward peak memory.

            When ``None`` (default), auto-detect from autograd state:
            ``True`` iff :func:`torch.is_grad_enabled` and any input
            tensor has ``requires_grad=True``.

            Diagnostic override: setting env var
            ``CP_TRITON_BLOCK_FUSION_FORCE_NO_CHECKPOINT=1`` flips an
            otherwise-True resolved value to ``False``.

    Returns:
        :class:`CpFusedGdnRawResult` with:
            ``num`` ``(B, N_local, H, D)`` — raw numerator before output
            divide. Grad flows back to qkv/beta/decay/norm weights.
            ``den`` ``(B, H, N_local)`` — raw denominator. Same.
            ``terminal_state_kv`` / ``terminal_state_z`` — fp32, present
            only when ``truncate_to_active`` was set.

    Notes:
        * BLOCK_D is derived as ``triton.next_power_of_2(head_dim)``
          where ``head_dim = S_kv.shape[-1]``. For production D=112 this
          yields BLOCK_D=128. For test D=8 it yields BLOCK_D=8 (note:
          Triton MMA requires K>=16, so production tests use D>=16).
        * Numerator-only / camera-branch (``skip_z=True``) is not
          supported by this entry; use the dedicated camera-branch
          wrapper for that case.
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

    # Resolve the checkpoint flag. ``None`` -> auto-detect (True if grad
    # is enabled AND any input requires grad).
    use_checkpoint_resolved = _resolve_use_checkpoint(
        use_checkpoint,
        qkv,
        beta,
        decay,
        q_norm_weight,
        k_norm_weight,
    )

    def _inner_pipeline(
        qkv_in: Tensor,
        beta_in: Tensor,
        decay_in: Tensor,
        q_nw_in: Tensor | None,
        k_nw_in: Tensor | None,
        rope_cos_in: Tensor,
        rope_sin_in: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Composes Prep → cp_frame_gdn_scan → Output.

        Returns ``(num, den, terminal_state_kv, terminal_state_z)``;
        the terminal states are ``None`` when ``truncate_to_active`` is
        ``None`` and tensors otherwise.  ``torch.utils.checkpoint`` accepts
        ``None`` returns as long as the closure shape is consistent across
        forward and the recomputed forward in backward, which it is here
        (``truncate_to_active`` is captured from the outer scope).
        """
        # ── (1) Prep: RMSNorm + phase_a + adapter (autograd-aware) ──
        W_kv, U_kv, W_z, U_z, qkv_normed = _CpFusedGdnPrep.apply(
            qkv_in,
            beta_in,
            decay_in,
            q_nw_in,
            k_nw_in,
            rope_cos_in,
            rope_sin_in,
            F,
            S,
            float(k_scale),
            float(norm_eps),
            int(dot_precision),
        )

        # ── (2) CP scan (already differentiable via FrameGDNScan + _CPAllGatherMerge) ──
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
            # Defensive isinstance check so a future signature change
            # crashes loudly here instead of silently feeding garbage
            # downstream.
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

        # ── (3) BLOCK_D derivation: padded head dim for Phase C consumption ──
        # S_kv shape: (BH, F, head_dim, head_dim). For production D=112,
        # next_power_of_2 gives 128. For D=16 (test default), it stays 16.
        head_dim = S_kv.shape[-1]
        block_d = triton.next_power_of_2(head_dim)

        # ── (4) Output: inverse adapter + phase_c (autograd-aware) ──
        num_inner, den_inner = _CpFusedGdnOutput.apply(
            qkv_normed,
            rope_cos_in,
            rope_sin_in,
            S_kv,
            S_z,
            int(block_d),
            F,
            S,
            int(dot_precision),
        )
        return num_inner, den_inner, terminal_state_kv_inner, terminal_state_z_inner

    if use_checkpoint_resolved:
        # Discard the Prep/Output saved-tensor backing stores after the
        # forward pipeline returns; backward re-runs the inner pipeline
        # and recomputes the saves. ``use_reentrant=False`` is required
        # (and matches the eager path).
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
        )

    return CpFusedGdnRawResult(
        num=num,
        den=den,
        terminal_state_kv=terminal_state_kv,
        terminal_state_z=terminal_state_z,
    )


# ---------------------------------------------------------------------------
# Camera-branch CP fused numerator-only forward scan.
# ---------------------------------------------------------------------------


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

    The "fused" part of the camera-branch path lives outside this
    function: it is the upstream :func:`cam_prep_func_with_grad` Triton
    kernel which fuses RMSNorm + ReLU + K-scale + UCPE-projmat + RoPE on
    the raw QKV. This wrapper composes the fused-prep + CP scan + eager
    num-only output as one autograd graph.

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
        truncate_to_active: When set, ``cp_frame_gdn_scan`` masks padded
            positions and returns a terminal-state KV broadcast to all
            ranks. Surfaced as the second tuple element so callers can
            resume a local non-CP gen scan from that boundary state.
        eps_recurrence: Reserved for API symmetry; unused.
        use_checkpoint: When ``True``, wrap the transition build / scan /
            output projection pipeline in ``torch.utils.checkpoint`` so
            saved intermediates are discarded after forward and recomputed
            during backward. ``None`` auto-detects from autograd state;
            ``CP_TRITON_BLOCK_FUSION_FORCE_NO_CHECKPOINT=1`` forces
            ``False`` regardless.

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

    # Resolve the checkpoint flag. ``None`` -> auto-detect.
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
        """Composes transition build → cp_frame_gdn_scan → output projection.

        Returns ``(out, terminal_state_kv)``; the terminal state is
        ``None`` when ``truncate_to_active`` is ``None``.
        """

        # ── (1) Reshape (B, H, D, N) -> frame layout (B, H, F, D, S). ──
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

        # ── (2) Build transitions (single-path: KV only, Z zeroed). ──
        # Mirrors :func:`_build_transition_matrices` with ``k_rot`` used in
        # both spots and ``v`` carrying the input. Zero Z because the
        # downstream output projection ignores it and the scan's backward
        # returns zero gradients through the dummy Z slot.
        k_rot_beta = k_f * beta_f
        W_kv = decay_f * (I - torch.matmul(k_rot_beta, k_f.transpose(-1, -2)))
        U_kv = torch.matmul(v_f * beta_f, k_f.transpose(-1, -2))
        W_kv = W_kv.reshape(BH, F, D, D).contiguous()
        U_kv = U_kv.reshape(BH, F, D, D).contiguous()
        W_z = torch.zeros(BH, F, D, D, device=q_in.device, dtype=W_kv.dtype)
        U_z = torch.zeros(BH, F, D, device=q_in.device, dtype=W_kv.dtype)

        # ── (3) Distributed scan with autograd-aware all-gather merge. ──
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

        # ── (4) Output projection: out[b,h,f] = S_kv[b,h,f] @ q_rot[b,h,f] ──
        # Mirrors ``compiled_gdn_output_projection`` (frame_gdn/api.py:25) for
        # the num path. cp_frame_gdn_scan returns S_kv in right-multiply
        # convention (S_t = S_{t-1} @ W_t + U_t), so by the transpose
        # convention noted in the module docstring, M_t = S_t.T. The eager
        # camera output projection at
        # sana_gdn_camctrl_blocks.py:1958 uses ``torch.matmul(_Skv_fwd, _qrf)``
        # which IS the left-multiply M_t @ q_rot. We mirror that exactly.
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
