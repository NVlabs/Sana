"""Unidirectional stateful chunkwise GDN autograd Function with FULL state-grad
I/O (init AND final state gradients).

Why this exists
---------------

The existing ``FusedBiGDNChunkwiseFunction`` in
``fused_gdn_chunkwise_bwd.py``:

  1. Is bidirectional only (combines forward + reverse output via divide).
  2. Does not expose initial / final state as autograd-tracked tensors —
     callers cannot pass ``init_state_*`` or receive ``final_state_*``.
  3. Computes a "dM_init_kv" internally (see lines 1163, 261-267) but never
     surfaces it via ``backward``'s return tuple. This makes BPTT through
     an AR chain impossible: each step's ``final_state`` becomes the next
     step's ``init_state``, but without ``dM_init`` and ``dM_final`` flowing,
     the gradient signal cannot propagate across steps.

This new Function is the missing piece for AR-chain BPTT. It supports BOTH
directions:
  * ``direction=1`` (forward, default): matches the production
    ``fused_gdn_stateful_chunkwise(reverse=False, return_final_state=...)``
    semantics — accepts optional init state, optionally emits final state,
    and exposes both ``dM_init_*`` AND accepts ``dM_final_*`` as backward
    inputs.
  * ``direction=2`` (reverse): stateless per-chunk reset (matches the
    bidi-combine reverse path in ``_forward_main_branch_with_cache``).
    ``init_state_*`` and ``save_final_state`` are not allowed — there is
    no state I/O in the reverse direction. The Function returns ``out``
    only.

Math derivation (kv path; z is analogous)
----------------------------------------

Per-chunk forward recurrence (state path):
    M_0_in = init_state_kv  (zero if not passed)
    For c = 0..C-1:
        M_c = g_c * (I - P_kv_c) @ M_{c-1} + A_c
    final_state_kv = M_{C-1}

Phase C reads M_c (post-frame state) and computes per-frame output
    O_c = Q_c @ M_c
    out_c = O_c / (den_c + eps)
giving the autograd input ``dout``.

Going backward (reverse scan in time):

    dM_C[c] = dphase_C/dM_c                                  # (Phase C̄)
    Boundary at c = C-1:
        total_dM[C-1] = dM_C[C-1] + dM_final_kv             # ← NEW: accept upstream
    Propagate c → c-1 (for c = C-1..1):
        total_dM[c-1] = dM_C[c-1] + g_c * (I - P_kv_c)^T @ total_dM[c]
    At c = 0 boundary (special case for init-state grad):
        dM_init_kv = g_0 * (I - P_kv_0)^T @ total_dM[0]      # ← NEW: return upstream

The "dM_final"+"dM_init" boundary contract makes this Function the correct
building block for the chain
    state_t = f(state_{t-1}, x_t)
where gradients of ``x_t`` AND of ``state_{t-1}`` are both needed.

Important subtlety: when C == 1, the recurrence is
    M_0 = g_0 * (I - P_kv_0) @ init + A_0
    total_dM[0] = dM_C[0] + dM_final_kv
    dM_init_kv = g_0 * (I - P_kv_0)^T @ total_dM[0]
This is handled uniformly by initializing ``accum = dM_C[F-1] + dM_final_kv``
before the reverse scan loop runs (loop body executes F-1 times; for F=1 it
runs zero times and we go straight to the dM_init derivation).

Z-path parity
-------------

The existing FusedBiGDNChunkwiseFunction's z-bwd path (line 1193) sets
    total_dz_fwd[:, F-1] = dz_C[:, F-1]
which silently assumes dM_final_z = 0. This Function correctly includes
``dM_final_z`` at the F-1 seed and returns ``dM_init_z`` at the c=0 boundary.

Layout / shape contract
-----------------------

Mirrors ``fused_gdn_stateful_chunkwise``:
  qkv:            (B, N=F*S, 3, H, D)
  beta:           (B, H, F, S)
  decay:          (B, H, F)
  q/k_norm_weight:(H*D,)
  rope_cos/sin:   (N, D)
  init_state_kv:  (B, H, D, D)
  init_state_z:   (B, H, D, 1) or (B, H, D)
  save_final_state: bool — if True, the kernel writes the final state and
    those tensors become differentiable outputs (so they can accept
    dM_final_* from the upstream gradient).

Returns from .apply:
  if save_final_state:
    (out, final_state_kv, final_state_z) — all differentiable
  else:
    out                                  — single differentiable output
"""

from __future__ import annotations

import torch

from diffusion.model.ops.fused_gdn_chunkwise import (
    phase_a,
    phase_b_triton,
    phase_c,
)
from diffusion.model.ops.fused_gdn_chunkwise_bwd import (
    _resolve_bwd_block_s,
    fused_rope_relu_fwd,
    fused_rope_unrope_bwd,
    output_divide_bwd,
    phase_a_kv_bwd,
    phase_a_z_bwd,
    phase_c_bwd,
)

# ─────────────────────────────────────────────────────────────────────────────
# Forward-only Phase B̄ scan with state-grad seed + return.
# Implemented in PyTorch (fp32 small-D matmuls; the existing chunkwise bwd
# also uses a PyTorch fallback on every arch except Blackwell-DC, so this
# is consistent with the rest of the codebase).
# ─────────────────────────────────────────────────────────────────────────────


def _phase_b_fwd_only_bwd_pt(dM_C_fwd, P_all, g, dM_final_fwd):
    """Forward-only Phase B̄ reverse scan with state-grad I/O.

    Args:
      dM_C_fwd:     (BH, F, D, D) Phase C̄ injections per frame.
      P_all:        (BH, F, D, D) the P_kv (NOT I - P_kv) matrices.
      g:            (BH, F)       per-frame decay.
      dM_final_fwd: (BH, D, D)    upstream grad on final state (zero if absent).

    Returns:
      total_dM_fwd:  (BH, F, D, D)
      dM_init_fwd:   (BH, D, D)
    """
    BH, F, D, _ = dM_C_fwd.shape
    I_D = torch.eye(D, device=dM_C_fwd.device, dtype=dM_C_fwd.dtype)

    total_dM_fwd = torch.empty_like(dM_C_fwd)
    total_dM_fwd[:, F - 1] = dM_C_fwd[:, F - 1] + dM_final_fwd
    for f in range(F - 2, -1, -1):
        g_next = g[:, f + 1].view(BH, 1, 1)
        I_minus_P_next = I_D - P_all[:, f + 1]
        total_dM_fwd[:, f] = dM_C_fwd[:, f] + g_next * (I_minus_P_next.transpose(-2, -1) @ total_dM_fwd[:, f + 1])

    g0 = g[:, 0].view(BH, 1, 1)
    I_minus_P0 = I_D - P_all[:, 0]
    dM_init_fwd = g0 * (I_minus_P0.transpose(-2, -1) @ total_dM_fwd[:, 0])

    return total_dM_fwd, dM_init_fwd


def _phase_b_rev_only_bwd_pt(dM_C_rev, P_all, g):
    """Reverse-direction Phase B̄ forward-in-time scan (stateless).

    Reverse forward recurrence:
        M_rev[F-1] = 0
        For f = F-2..0:
            M_rev[f] = g_{f+1} * (I - P_{f+1}) @ M_rev[f+1] + A_{f+1}

    Backward (forward scan in time, no init/final state grads):
        total_dM_rev[0]   = dM_C_rev[0]
        For f = 0..F-2:
            total_dM_rev[f+1] = dM_C_rev[f+1]
                              + g_{f+1} * (I - P_{f+1})^T @ total_dM_rev[f]

    Args:
      dM_C_rev: (BH, F, D, D)
      P_all:    (BH, F, D, D)
      g:        (BH, F)

    Returns:
      total_dM_rev: (BH, F, D, D)
    """
    BH, F, D, _ = dM_C_rev.shape
    I_D = torch.eye(D, device=dM_C_rev.device, dtype=dM_C_rev.dtype)

    total_dM_rev = torch.empty_like(dM_C_rev)
    total_dM_rev[:, 0] = dM_C_rev[:, 0]
    for f in range(F - 1):
        g_next = g[:, f + 1].view(BH, 1, 1)
        I_minus_P_next = I_D - P_all[:, f + 1]
        total_dM_rev[:, f + 1] = dM_C_rev[:, f + 1] + g_next * (I_minus_P_next.transpose(-2, -1) @ total_dM_rev[:, f])
    return total_dM_rev


def _phase_b_z_rev_only_bwd_pt(dz_C_rev, P_z_all, g):
    """Reverse-direction Phase B̄ for the Z denominator stream (stateless).

    Args:
      dz_C_rev: (BH, F, D)
      P_z_all:  (BH, F, D, D)
      g:        (BH, F)

    Returns:
      total_dz_rev: (BH, F, D)
    """
    BH, F, D = dz_C_rev.shape
    I_D = torch.eye(D, device=dz_C_rev.device, dtype=dz_C_rev.dtype)

    total_dz_rev = torch.empty_like(dz_C_rev)
    total_dz_rev[:, 0] = dz_C_rev[:, 0]
    for f in range(F - 1):
        g_next = g[:, f + 1].view(BH, 1)
        I_minus_P_next = I_D - P_z_all[:, f + 1]
        total_dz_rev[:, f + 1] = dz_C_rev[:, f + 1] + g_next * (
            I_minus_P_next.transpose(-2, -1) @ total_dz_rev[:, f].unsqueeze(-1)
        ).squeeze(-1)
    return total_dz_rev


def _combine_rev_only_dA_dP_dg(total_dM_rev, M_rev_post, P_all, g):
    """Per-frame (dA_kv, dP_kv, dg_kv) from total_dM_rev, reverse-direction.

    Reverse rev-step at frame f+1 uses params at frame f+1 with state
    M_rev[f+1] as input, producing M_rev[f] as output. So:
      total_dM_shifted[f] = total_dM_rev[f-1]   for f >= 1
      total_dM_shifted[0] = 0
      M_prev_for_combine[f] = M_rev[f]          (M_rev[0] is irrelevant
                                                 because shift[0] is zero)

    Then standard per-frame combine on the shifted tensors:
      dA_f = total_dM_shifted[f]
      dP_f = -g_f * total_dM_shifted[f] @ M_prev[f]^T
      dg_f = sum( total_dM_shifted[f] * (I - P_f) @ M_prev[f] )
    """
    BH, F, D, _ = total_dM_rev.shape
    zero_DD = torch.zeros(BH, 1, D, D, device=total_dM_rev.device, dtype=total_dM_rev.dtype)
    total_dM_shifted = torch.cat([zero_DD, total_dM_rev[:, : F - 1]], dim=1).contiguous()
    return _combine_fwd_only_dA_dP_dg(total_dM_shifted, M_rev_post.contiguous(), P_all, g)


def _combine_rev_only_dB_dPz_dg_z(total_dz_rev, z_rev_post, P_z_all, g):
    """Reverse-direction analog of ``_combine_fwd_only_dB_dPz_dg_z``.

    Same shift-by-one logic as the kv combine: rev-step at frame f+1 uses
    parameters at frame f+1 and produces z_rev[f]. So:
      total_dz_shifted[f] = total_dz_rev[f-1]   for f >= 1
      total_dz_shifted[0] = 0
      z_prev_for_combine[f] = z_rev[f]          (z_rev[0] is irrelevant
                                                 because shift[0] is zero)
    """
    BH, F, D = total_dz_rev.shape
    zero_D = torch.zeros(BH, 1, D, device=total_dz_rev.device, dtype=total_dz_rev.dtype)
    total_dz_shifted = torch.cat([zero_D, total_dz_rev[:, : F - 1]], dim=1).contiguous()
    return _combine_fwd_only_dB_dPz_dg_z(total_dz_shifted, z_rev_post.contiguous(), P_z_all, g)


def _phase_b_z_fwd_only_bwd_pt(dz_C_fwd, P_z_all, g, dz_final_fwd):
    """Forward-only Phase B̄ reverse scan for the Z denominator stream.

    Args:
      dz_C_fwd:     (BH, F, D)
      P_z_all:      (BH, F, D, D)
      g:            (BH, F)
      dz_final_fwd: (BH, D)

    Returns:
      total_dz_fwd: (BH, F, D)
      dz_init_fwd:  (BH, D)
    """
    BH, F, D = dz_C_fwd.shape
    I_D = torch.eye(D, device=dz_C_fwd.device, dtype=dz_C_fwd.dtype)

    total_dz_fwd = torch.empty_like(dz_C_fwd)
    total_dz_fwd[:, F - 1] = dz_C_fwd[:, F - 1] + dz_final_fwd
    for f in range(F - 2, -1, -1):
        g_next = g[:, f + 1].view(BH, 1)
        I_minus_P_next = I_D - P_z_all[:, f + 1]
        total_dz_fwd[:, f] = dz_C_fwd[:, f] + g_next * (
            I_minus_P_next.transpose(-2, -1) @ total_dz_fwd[:, f + 1].unsqueeze(-1)
        ).squeeze(-1)

    g0 = g[:, 0].view(BH, 1)
    I_minus_P0 = I_D - P_z_all[:, 0]
    dz_init_fwd = g0 * (I_minus_P0.transpose(-2, -1) @ total_dz_fwd[:, 0].unsqueeze(-1)).squeeze(-1)

    return total_dz_fwd, dz_init_fwd


def _combine_fwd_only_dA_dP_dg(total_dM_fwd, M_fwd_prev, P_all, g):
    """Per-frame (dA_kv, dP_kv, dg_kv) from total_dM_fwd, forward only.

    Recurrence: M_f = g_f * (I - P_f) @ M_prev[f] + A_f
      dA_f = total_dM[f]
      dP_f = -g_f * total_dM[f] @ M_prev[f]^T
      dg_f = sum( total_dM[f] * (I - P_f) @ M_prev[f] )
    """
    BH, F, D, _ = total_dM_fwd.shape
    I_D = torch.eye(D, device=total_dM_fwd.device, dtype=total_dM_fwd.dtype)
    g_per = g.view(BH, F, 1, 1)
    I_minus_P = I_D - P_all

    dA_total = total_dM_fwd.clone()
    dP_total = -g_per * (total_dM_fwd @ M_fwd_prev.transpose(-2, -1))
    I_minus_P_M_fwd = I_minus_P @ M_fwd_prev
    dg_total = (total_dM_fwd * I_minus_P_M_fwd).sum(dim=(-2, -1))

    return dA_total, dP_total, dg_total


def _combine_fwd_only_dB_dPz_dg_z(total_dz_fwd, z_fwd_prev, P_z_all, g):
    """Per-frame (dB_z, dP_z, dg_z) from total_dz_fwd, forward only.

    Recurrence: z_f = g_f * (I - P_z_f) @ z_prev[f] + B_z_f
      dB_z_f = total_dz[f]
      dP_z_f = -g_f * outer(total_dz[f], z_prev[f])
      dg_z_f = sum( total_dz[f] * (I - P_z_f) @ z_prev[f] )
    """
    BH, F, D = total_dz_fwd.shape
    I_D = torch.eye(D, device=total_dz_fwd.device, dtype=total_dz_fwd.dtype)
    g_per = g.view(BH, F, 1, 1)
    I_minus_P = I_D - P_z_all

    dB_z_total = total_dz_fwd.clone()
    dP_z_total = -g_per * (total_dz_fwd.unsqueeze(-1) @ z_fwd_prev.unsqueeze(-2))
    Imp_z = (I_minus_P @ z_fwd_prev.unsqueeze(-1)).squeeze(-1)
    dg_z_total = (total_dz_fwd * Imp_z).sum(dim=-1)

    return dB_z_total, dP_z_total, dg_z_total


# ─────────────────────────────────────────────────────────────────────────────
# State-layout helpers (mirror fused_gdn_stateful_chunkwise padding).
# ─────────────────────────────────────────────────────────────────────────────


def _pad_state_kv_to_block(state_kv, BLOCK_D):
    """Pad caller-facing ``(B, H, D, D)`` state to kernel layout
    ``(B*H, BLOCK_D, BLOCK_D)``."""
    B, H, D_in, D_out = state_kv.shape
    if D_in != BLOCK_D or D_out != BLOCK_D:
        pad_in = BLOCK_D - D_in
        pad_out = BLOCK_D - D_out
        return torch.nn.functional.pad(
            state_kv.transpose(-1, -2).reshape(B * H, D_out, D_in), (0, pad_in, 0, pad_out)
        ).contiguous()
    return state_kv.transpose(-1, -2).reshape(B * H, BLOCK_D, BLOCK_D).contiguous()


def _pad_state_z_to_block(state_z, BLOCK_D):
    """Pad caller-facing z-state ``(B, H, D)`` or ``(B, H, D, 1)`` to
    kernel layout ``(B*H, BLOCK_D)``."""
    z_ = state_z.squeeze(-1) if state_z.dim() == 4 else state_z
    Bz, Hz, Dz = z_.shape
    if Dz != BLOCK_D:
        return torch.nn.functional.pad(z_.reshape(Bz * Hz, Dz), (0, BLOCK_D - Dz)).contiguous()
    return z_.reshape(Bz * Hz, BLOCK_D).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────────


class FusedGDNChunkwiseStatefulFunction(torch.autograd.Function):
    """Directional chunkwise stateful GDN with full state-grad I/O.

    ``direction=1`` (forward): supports init/final state I/O.
    ``direction=2`` (reverse): stateless per-chunk reset; no state I/O.
    """

    @staticmethod
    def forward(
        ctx,
        qkv,
        beta,
        decay,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        init_state_kv,
        init_state_z,
        F,
        S,
        k_scale,
        norm_eps,
        eps,
        dot_precision,
        BLOCK_S,
        save_final_state,
        direction,
    ):
        if BLOCK_S is None:
            BLOCK_S = _resolve_bwd_block_s()
        if direction not in (1, 2):
            raise ValueError(
                f"FusedGDNChunkwiseStatefulFunction: direction must be 1 (forward) or 2 (reverse); got {direction}."
            )
        if direction == 2 and (init_state_kv is not None or init_state_z is not None or save_final_state):
            raise ValueError(
                "FusedGDNChunkwiseStatefulFunction: reverse direction (direction=2) is stateless; "
                "init_state_kv / init_state_z must be None and save_final_state must be False."
            )
        B, N, three, H, D = qkv.shape
        C = H * D
        assert three == 3 and N == F * S
        if (init_state_kv is None) != (init_state_z is None):
            raise ValueError(
                "FusedGDNChunkwiseStatefulFunction: init_state_kv and init_state_z must be "
                "provided together (both None or both non-None)."
            )
        device = qkv.device
        fp32 = torch.float32
        # Track whether caller passed None so backward returns None grad
        # (autograd's contract: non-Tensor inputs must get None grads).
        q_nw_was_none = q_norm_weight is None
        k_nw_was_none = k_norm_weight is None
        if q_nw_was_none:
            q_norm_weight = torch.ones(C, device=device, dtype=fp32)
        if k_nw_was_none:
            k_norm_weight = torch.ones(C, device=device, dtype=fp32)

        # ── RMSNorm (channel-wise) — done outside the kernel so backward
        #    can analytically VJP it. Matches FusedBiGDNChunkwiseFunction.
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        q_inv_rms = torch.rsqrt(q_raw_v.float().pow(2).sum(dim=(-2, -1)) / C + norm_eps)
        k_inv_rms = torch.rsqrt(k_raw_v.float().pow(2).sum(dim=(-2, -1)) / C + norm_eps)
        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(qkv.dtype)
        qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(qkv.dtype)

        # ── Chunkwise forward (forward direction only).
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

        BLOCK_D = I_P_kv.shape[-1]
        init_kv_padded = None
        init_z_padded = None
        if init_state_kv is not None:
            init_kv_padded = _pad_state_kv_to_block(init_state_kv, BLOCK_D)
            init_z_padded = _pad_state_z_to_block(init_state_z, BLOCK_D)

        if save_final_state:
            # save_final_state is only valid for direction=1 (guarded above).
            M_fwd, z_fwd, _, _, final_kv_pad, final_z_pad = phase_b_triton(
                I_P_kv,
                A,
                I_P_z,
                B_z,
                decay,
                F=F,
                dot_precision=dot_precision,
                direction=direction,
                init_state_kv=init_kv_padded,
                init_state_z=init_z_padded,
                return_final_state=True,
            )
            M_use, z_use = M_fwd, z_fwd
        else:
            M_fwd, z_fwd, M_rev, z_rev = phase_b_triton(
                I_P_kv,
                A,
                I_P_z,
                B_z,
                decay,
                F=F,
                dot_precision=dot_precision,
                direction=direction,
                init_state_kv=init_kv_padded,
                init_state_z=init_z_padded,
            )
            final_kv_pad = None
            final_z_pad = None
            # For direction=2 the kernel writes its state into M_rev/z_rev
            # while M_fwd/z_fwd are placeholder dummies (see phase_b_triton).
            if direction == 2:
                M_use, z_use = M_rev, z_rev
            else:
                M_use, z_use = M_fwd, z_fwd

        num_out, den_out = phase_c(
            qkv_normed,
            dummy_inv,
            dummy_nw,
            rope_cos,
            rope_sin,
            M_use,
            z_use,
            F=F,
            S=S,
            dot_precision=dot_precision,
            accumulate=False,
        )
        total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)
        out = (num_out.float() / (total_den + eps)).to(qkv.dtype)

        if save_final_state:
            final_state_kv = final_kv_pad.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
            final_state_z = final_z_pad.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        else:
            final_state_kv = torch.empty(0, device=device, dtype=fp32)
            final_state_z = torch.empty(0, device=device, dtype=fp32)

        # NOTE: Phase A's A and B_z tensors are NOT saved because the bwd
        # recomputes per-frame (dA, dP, dg) and (dB_z, dP_z, dg_z) from
        # total_dM/total_dz and the M_prev/z_prev state series directly.
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
            num_out,
            den_out,
            I_P_kv,
            I_P_z,
            M_use,
            z_use,
            init_kv_padded if init_kv_padded is not None else torch.empty(0, device=device, dtype=fp32),
            init_z_padded if init_z_padded is not None else torch.empty(0, device=device, dtype=fp32),
        )
        ctx.shape = (B, N, H, D, F, S, C)
        ctx.k_scale = k_scale
        ctx.norm_eps = norm_eps
        ctx.eps = eps
        ctx.dot_precision = dot_precision
        ctx.BLOCK_S = BLOCK_S
        ctx.has_init_state = init_state_kv is not None
        ctx.save_final_state = save_final_state
        ctx.direction = direction
        ctx.q_nw_was_none = q_nw_was_none
        ctx.k_nw_was_none = k_nw_was_none
        # Track the original init_state_z dim() so backward returns a gradient
        # of the matching rank (autograd validates shape match).
        ctx.init_state_z_dim = init_state_z.dim() if init_state_z is not None else 0

        if save_final_state:
            return out, final_state_kv, final_state_z
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.save_final_state:
            dout, dfinal_kv_user, dfinal_z_user = grad_outputs
        else:
            (dout,) = grad_outputs
            dfinal_kv_user = None
            dfinal_z_user = None

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
            num_out,
            den_out,
            I_P_kv,
            I_P_z,
            M_fwd,
            z_fwd,
            init_kv_padded,
            init_z_padded,
        ) = ctx.saved_tensors
        B, N, H, D, F, S, C = ctx.shape
        k_scale, eps = ctx.k_scale, ctx.eps
        norm_eps = ctx.norm_eps  # noqa: F841
        dot_precision, BLOCK_S = ctx.dot_precision, ctx.BLOCK_S
        device = qkv.device
        fp32 = torch.float32
        dtype = qkv.dtype
        BH = B * H

        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)

        # ── 1. Output-divide VJP. ───────────────────────────────────────────
        dnum, dden = output_divide_bwd(dout, num_out, den_out, eps=eps, out_dtype=dtype)
        del num_out, den_out

        # ── 2. Reconstruct qkv_normed (matches forward exactly). ────────────
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(dtype)
        qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(dtype)

        # ── 3. Phase A intermediates (unpadded D×D, fp32). ──────────────────
        I_P_kv.shape[-1]
        I_D = torch.eye(D, device=device, dtype=fp32)
        P_kv_all = I_D[None, None] - I_P_kv[:, :, :D, :D].float()
        P_z_all = I_D[None, None] - I_P_z[:, :, :D, :D].float()
        del I_P_kv, I_P_z

        direction = ctx.direction
        # M_fwd/z_fwd here are the *saved* post-state series for the
        # active direction (M_fwd for direction=1, M_rev for direction=2,
        # both stored under the same slot — see ctx.save_for_backward).
        M_use_d = M_fwd[:, :, :D, :D].float()
        del M_fwd
        z_use_d = z_fwd[:, :, :D].float()
        del z_fwd

        if direction == 1:
            # Build the *pre-frame* state series M_fwd_full[f] = M_{f-1}.
            # M_fwd_full[0] = init_M0 (zero or user-provided).
            if init_kv_padded.numel() > 0:
                # Padded init layout: row-major M[V_feat, K_feat] (matches kernel).
                # The forward pad routine took user (B,H,D,D) -> transpose(-1,-2) ->
                # (B*H, BLOCK_D, BLOCK_D). To extract the unpadded D×D we just slice
                # the top-left; this gives us back kernel-layout M (V_feat, K_feat).
                init_M0 = init_kv_padded[:, :D, :D].float().reshape(B, H, D, D)
                init_z0 = init_z_padded[:, :D].float().reshape(B, H, D)
            else:
                init_M0 = torch.zeros(B, H, D, D, device=device, dtype=fp32)
                init_z0 = torch.zeros(B, H, D, device=device, dtype=fp32)

            init_M0_bh = init_M0.reshape(BH, 1, D, D)
            init_z0_bh = init_z0.reshape(BH, 1, D)
            M_fwd_full = torch.cat([init_M0_bh, M_use_d], dim=1)  # (BH, F+1, D, D)
            z_fwd_full = torch.cat([init_z0_bh, z_use_d], dim=1)
            del M_use_d, z_use_d
        else:
            # direction == 2 (reverse): M_use_d / z_use_d holds M_rev[f]
            # (post-rev-step state at frame f). No init/final state.
            M_rev_post = M_use_d.contiguous()
            z_rev_post = z_use_d.contiguous()
            del M_use_d, z_use_d

        # ── 4. Rope+relu recomputation (saves bwd memory). ──────────────────
        def bnhd_to_bhfsd(x):
            return x.permute(0, 2, 1, 3).reshape(B, H, F, S, D).reshape(BH, F, S, D).contiguous()

        def bhfsd_to_bnhd(x):
            return x.reshape(BH, F * S, D).reshape(B, H, N, D).permute(0, 2, 1, 3).contiguous()

        Q_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 0])
        K_normed_bhfsd = bnhd_to_bhfsd(qkv_normed[:, :, 1])
        V_bhfsd = bnhd_to_bhfsd(qkv[:, :, 2])
        del qkv_normed

        Q_post_relu_bhfsd, K_post_relu_bhfsd, Q_for_num_bhfsd, K_kv_bhfsd = fused_rope_relu_fwd(
            Q_normed_bhfsd,
            K_normed_bhfsd,
            rope_cos,
            rope_sin,
            k_scale,
            F,
            S,
        )
        del Q_normed_bhfsd, K_normed_bhfsd

        Q_for_den_bhfsd = Q_post_relu_bhfsd
        K_z_bhfsd = K_post_relu_bhfsd

        beta_bhfs = beta.reshape(BH, F, S).float()
        decay_bhf = decay.reshape(BH, F).float()
        dO_bhfsd = bnhd_to_bhfsd(dnum)
        dden_bhfs = dden.reshape(BH, F, S).contiguous()
        del dnum

        # ── 5. KV chain. ────────────────────────────────────────────────────
        if direction == 1:
            M_post = M_fwd_full[:, 1:].contiguous()
        else:
            M_post = M_rev_post
        dQ_kv, dM_C = phase_c_bwd(
            Q_for_num_bhfsd.contiguous(),
            M_post,
            dO_bhfsd,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        if direction == 1:
            # dM_final_fwd from upstream (or zero).
            if dfinal_kv_user is not None and dfinal_kv_user.numel() > 0:
                # User grad shape: (B, H, D, D) — caller-facing (K_feat, V_feat).
                # Kernel layout for state is (V_feat, K_feat) (it's transposed at the
                # boundary). Inverse: transpose(-1,-2) to convert user-grad back.
                dfinal_kv_kernel = dfinal_kv_user.float().transpose(-1, -2).contiguous().reshape(BH, D, D)
            else:
                dfinal_kv_kernel = torch.zeros(BH, D, D, device=device, dtype=fp32)

            total_dM_use, dM_init_kv_bh = _phase_b_fwd_only_bwd_pt(
                dM_C,
                P_kv_all,
                decay_bhf,
                dfinal_kv_kernel,
            )
            dA_total, dP_kv_total, dg_kv_total = _combine_fwd_only_dA_dP_dg(
                total_dM_use,
                M_fwd_full[:, :-1].contiguous(),
                P_kv_all,
                decay_bhf,
            )
        else:
            total_dM_use = _phase_b_rev_only_bwd_pt(dM_C, P_kv_all, decay_bhf)
            dA_total, dP_kv_total, dg_kv_total = _combine_rev_only_dA_dP_dg(
                total_dM_use,
                M_rev_post,
                P_kv_all,
                decay_bhf,
            )
            dM_init_kv_bh = None
        dK_kv, dV, dbeta_kv = phase_a_kv_bwd(
            K_kv_bhfsd.contiguous(),
            V_bhfsd.contiguous(),
            beta_bhfs,
            dA_total,
            dP_kv_total,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # ── 6. Z chain. ─────────────────────────────────────────────────────
        if direction == 1:
            z_post = z_fwd_full[:, 1:]
        else:
            z_post = z_rev_post
        dQ_z = (dden_bhfs.unsqueeze(-1) * z_post.unsqueeze(2)).to(dtype)
        dz_C = (Q_for_den_bhfsd.float() * dden_bhfs.unsqueeze(-1).float()).sum(dim=2)

        if direction == 1:
            if dfinal_z_user is not None and dfinal_z_user.numel() > 0:
                dfinal_z_kernel = (
                    dfinal_z_user.float().squeeze(-1).reshape(BH, D)
                    if dfinal_z_user.dim() == 4
                    else dfinal_z_user.float().reshape(BH, D)
                )
            else:
                dfinal_z_kernel = torch.zeros(BH, D, device=device, dtype=fp32)

            total_dz_use, dz_init_z_bh = _phase_b_z_fwd_only_bwd_pt(
                dz_C,
                P_z_all,
                decay_bhf,
                dfinal_z_kernel,
            )
            dB_z_total, dP_z_total, dg_z_total = _combine_fwd_only_dB_dPz_dg_z(
                total_dz_use,
                z_fwd_full[:, :-1].contiguous(),
                P_z_all,
                decay_bhf,
            )
        else:
            total_dz_use = _phase_b_z_rev_only_bwd_pt(dz_C, P_z_all, decay_bhf)
            dB_z_total, dP_z_total, dg_z_total = _combine_rev_only_dB_dPz_dg_z(
                total_dz_use,
                z_rev_post,
                P_z_all,
                decay_bhf,
            )
            dz_init_z_bh = None
        dK_z, dbeta_z = phase_a_z_bwd(
            K_z_bhfsd.contiguous(),
            beta_bhfs,
            dB_z_total,
            dP_z_total,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # ── 7. RoPE + ReLU + RMSNorm VJPs. ──────────────────────────────────
        dQ_normed_bhfsd, dK_normed_bhfsd = fused_rope_unrope_bwd(
            dQ_kv,
            dK_kv,
            dQ_z,
            dK_z,
            Q_post_relu_bhfsd,
            K_post_relu_bhfsd,
            rope_cos,
            rope_sin,
            k_scale,
            F,
            S,
        )
        del dQ_kv, dK_kv, dQ_z, dK_z, Q_post_relu_bhfsd, K_post_relu_bhfsd

        dQ_normed_bnhd = bhfsd_to_bnhd(dQ_normed_bhfsd)
        del dQ_normed_bhfsd
        dK_normed_bnhd = bhfsd_to_bnhd(dK_normed_bhfsd)
        del dK_normed_bhfsd
        dV_bnhd = bhfsd_to_bnhd(dV)
        del dV
        dbeta_total = (dbeta_kv + dbeta_z).reshape(B, H, F, S)
        del dbeta_kv, dbeta_z
        ddecay_total = (dg_kv_total + dg_z_total).reshape(B, H, F)

        # RMSNorm VJP.
        q_raw_f = q_raw_v.float()
        q_irms = q_inv_rms[:, :, None, None]
        gw_q = dQ_normed_bnhd * q_nw_hd[None, None]
        dq_nw = (dQ_normed_bnhd * q_raw_f * q_irms).sum(dim=(0, 1)).reshape(-1)
        corr_q = (gw_q * q_raw_f).sum(dim=(-2, -1), keepdim=True)
        dQ_raw = q_irms * gw_q - (q_irms**3) / C * q_raw_f * corr_q
        del dQ_normed_bnhd, gw_q, corr_q, q_raw_f

        k_raw_f = k_raw_v.float()
        k_irms = k_inv_rms[:, :, None, None]
        gw_k = dK_normed_bnhd * k_nw_hd[None, None]
        dk_nw = (dK_normed_bnhd * k_raw_f * k_irms).sum(dim=(0, 1)).reshape(-1)
        corr_k = (gw_k * k_raw_f).sum(dim=(-2, -1), keepdim=True)
        dK_raw = k_irms * gw_k - (k_irms**3) / C * k_raw_f * corr_k
        del dK_normed_bnhd, gw_k, corr_k, k_raw_f

        dqkv = torch.stack([dQ_raw.to(dtype), dK_raw.to(dtype), dV_bnhd.to(dtype)], dim=2)

        # ── 8. Init-state grads (reshape back to caller-facing layouts). ────
        if ctx.has_init_state:
            # direction=1 only — direction=2 has has_init_state=False (guarded).
            # dM_init_kv (kernel layout (V_feat, K_feat)) → user-facing (K_feat, V_feat).
            dinit_state_kv = dM_init_kv_bh.reshape(B, H, D, D).transpose(-1, -2).contiguous().to(fp32)
            # Match the rank the user passed in: (B,H,D,1) or (B,H,D).
            if ctx.init_state_z_dim == 4:
                dinit_state_z = dz_init_z_bh.reshape(B, H, D, 1).contiguous().to(fp32)
            else:
                dinit_state_z = dz_init_z_bh.reshape(B, H, D).contiguous().to(fp32)
        else:
            dinit_state_kv = None
            dinit_state_z = None

        # Return None for q_nw/k_nw grads if caller passed None (autograd
        # contract: non-Tensor inputs must get None grads).
        dq_nw_out = None if ctx.q_nw_was_none else dq_nw.to(q_norm_weight.dtype)
        dk_nw_out = None if ctx.k_nw_was_none else dk_nw.to(k_norm_weight.dtype)
        return (
            dqkv,
            dbeta_total.to(beta.dtype),
            ddecay_total.to(decay.dtype),
            dq_nw_out,
            dk_nw_out,
            None,  # rope_cos
            None,  # rope_sin
            dinit_state_kv,  # init_state_kv
            dinit_state_z,  # init_state_z
            None,  # F
            None,  # S
            None,  # k_scale
            None,  # norm_eps
            None,  # eps
            None,  # dot_precision
            None,  # BLOCK_S
            None,  # save_final_state
            None,  # direction
        )


def fused_gdn_chunkwise_stateful_autograd(
    qkv,
    beta,
    decay,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    F,
    S,
    *,
    init_state_kv=None,
    init_state_z=None,
    k_scale=1.0,
    norm_eps=1e-5,
    eps=1e-6,
    dot_precision=0,
    BLOCK_S=None,
    save_final_state=False,
    direction=1,
):
    """Directional chunkwise stateful GDN with full state-grad I/O.

    Args mirror ``fused_gdn_stateful_chunkwise`` with the addition of
    ``save_final_state`` and ``direction``.

    ``direction=1`` (default, forward): supports init/final state I/O.
    ``direction=2`` (reverse): stateless per-chunk reset; ``init_state_*``
        must be None and ``save_final_state`` must be False.

    Returns:
      out                                       if ``save_final_state=False``
      (out, final_state_kv, final_state_z)      if ``save_final_state=True``
    """
    return FusedGDNChunkwiseStatefulFunction.apply(
        qkv,
        beta,
        decay,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        init_state_kv,
        init_state_z,
        F,
        S,
        k_scale,
        norm_eps,
        eps,
        dot_precision,
        BLOCK_S,
        save_final_state,
        direction,
    )


__all__ = [
    "FusedGDNChunkwiseStatefulFunction",
    "fused_gdn_chunkwise_stateful_autograd",
]
