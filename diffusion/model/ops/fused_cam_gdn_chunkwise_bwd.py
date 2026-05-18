"""Cam-branch chunkwise stateful autograd Function with full state-grad I/O.

Option C: derive cam bwd as a num-only chunkwise specialization of the main
GDN chunkwise bwd. Cam-specific simplifications:

  * **Num-only**: no Z denominator scan.
  * **Skip-relu**: cam_prep applies ReLU BEFORE UCPE+RoPE; the post-UCPE
    values can have legitimate negatives that re-applying ReLU would clobber.
  * **Identity rope/rms**: q/k/v come in pre-prepped externally. We hand the
    kernel ones_inv_rms, ones_norm_weight, ones_cos, zeros_sin so the
    in-kernel prep is a no-op. k_scale=1.0 (cam_prep already applied it).
  * **State passing only on reverse=False**: matches ``cam_scan_chunkwise``.

Surface layout (cam's):
  q, k, v : (B, H, D, N) fp32
  beta    : (B, H, F, S) fp32
  decay   : (B, H, F)   fp32
  init_state / final_state : (B, H, D, D) fp32 (forward direction only)

Output: out (B, H, D, N) fp32; optional final_state when save_final_state=True.

For the reverse direction we run no-state forward through chunkwise with
reverse=True; cam never AR-caches the rev direction. Reverse bwd uses
phase_b_bidi_bwd-style forward-in-time scan with dM_init=0 and dM_final=0.
"""

from __future__ import annotations

import torch
import triton

from diffusion.model.ops.fused_gdn_chunkwise import (
    _cam_identity_tables,
    _default_dot_prec,
    phase_a,
    phase_b_triton,
    phase_c,
)
from diffusion.model.ops.fused_gdn_chunkwise_bwd import (
    _resolve_bwd_block_s,
    phase_a_kv_bwd,
    phase_c_bwd,
)
from diffusion.model.ops.fused_gdn_chunkwise_stateful_bwd import (
    _combine_fwd_only_dA_dP_dg,
    _phase_b_fwd_only_bwd_pt,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _pack_qkv_bnhd(q, k, v):
    """Repack ``(B, H, D, N)`` → ``(B, N, 3, H, D)`` (cam → chunkwise layout)."""
    B, H, D, N = q.shape
    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))
    return qkv


def _pad_state_kv_to_block_cam(state_kv, BLOCK_D):
    """Cam state pad: (B, H, D, D) → (B*H, BLOCK_D, BLOCK_D) WITHOUT transpose.

    Cam state convention matches the kernel's storage exactly (no transpose
    needed; see ``cam_scan_chunkwise`` line 1975).
    """
    B, H, D_in, D_out = state_kv.shape
    state_kv_f = state_kv.float().contiguous()
    if D_in == BLOCK_D and D_out == BLOCK_D:
        return state_kv_f.reshape(B * H, BLOCK_D, BLOCK_D).contiguous()
    pad_in = BLOCK_D - D_in
    pad_out = BLOCK_D - D_out
    return torch.nn.functional.pad(state_kv_f.reshape(B * H, D_in, D_out), (0, pad_out, 0, pad_in)).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ─────────────────────────────────────────────────────────────────────────────


class CamGDNChunkwiseStatefulFunction(torch.autograd.Function):
    """Cam-branch chunkwise scan with autograd + full state-grad I/O."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        decay,
        init_state,
        reverse,
        save_final_state,
        dot_precision,
        BLOCK_S,
    ):
        assert q.shape == k.shape == v.shape, "q/k/v shape mismatch"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert beta.is_contiguous() and decay.is_contiguous()
        assert q.dtype == torch.float32, f"cam scan requires fp32 (got {q.dtype})"

        if reverse and (init_state is not None or save_final_state):
            raise NotImplementedError("CamGDNChunkwiseStatefulFunction: state I/O is forward-only.")

        B, H, D, N = q.shape
        F_frames = beta.shape[2]
        assert N % F_frames == 0
        S = N // F_frames
        assert beta.shape == (B, H, F_frames, S)
        assert decay.shape == (B, H, F_frames)

        if dot_precision is None:
            dot_precision = _default_dot_prec()
        if BLOCK_S is None:
            BLOCK_S = _resolve_bwd_block_s()

        BLOCK_D = triton.next_power_of_2(D)
        F = F_frames
        device = q.device
        fp32 = torch.float32

        qkv_packed = _pack_qkv_bnhd(q, k, v)
        ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(B=B, N=N, H=H, D=D, device=device)

        I_P_kv, A, I_P_z, B_z = phase_a(
            qkv_packed,
            beta,
            ones_inv_rms,
            ones_inv_rms,
            ones_nw,
            ones_nw,
            ones_cos,
            zeros_sin,
            F=F,
            S=S,
            k_scale=1.0,
            norm_eps=1e-5,
            dot_precision=dot_precision,
            skip_relu=True,
            skip_z=True,
        )

        init_kv_padded = None
        init_z_padded = None
        if init_state is not None:
            init_kv_padded = _pad_state_kv_to_block_cam(init_state, BLOCK_D)
            init_z_padded = torch.zeros(B * H, BLOCK_D, device=device, dtype=fp32)

        direction = 2 if reverse else 1
        if save_final_state:
            M_fwd, _, M_rev, _, final_kv_pad, _ = phase_b_triton(
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
                skip_z=True,
            )
        else:
            M_fwd, _, M_rev, _ = phase_b_triton(
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
                skip_z=True,
            )
            final_kv_pad = None

        M_use = M_rev if reverse else M_fwd
        num_out, _ = phase_c(
            qkv_packed,
            ones_inv_rms,
            ones_nw,
            ones_cos,
            zeros_sin,
            M_use,
            M_use,
            F=F,
            S=S,
            dot_precision=dot_precision,
            skip_relu=True,
            num_only=True,
        )

        out = num_out.permute(0, 2, 3, 1).contiguous().to(fp32)

        if save_final_state and not reverse:
            final_state = final_kv_pad.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].contiguous()
        else:
            final_state = torch.empty(0, device=device, dtype=fp32)

        # NOTE: A (Phase A's K^T β V intermediate) is NOT saved because the bwd
        # recomputes per-frame (dA, dP, dg) from total_dM and M_prev directly.
        ctx.save_for_backward(
            q,
            k,
            v,
            beta,
            decay,
            I_P_kv,
            M_use,
            init_kv_padded if init_kv_padded is not None else torch.empty(0, device=device, dtype=fp32),
        )
        ctx.shape = (B, H, D, N, F, S)
        ctx.reverse = reverse
        ctx.has_init_state = init_state is not None
        ctx.save_final_state = save_final_state
        ctx.dot_precision = dot_precision
        ctx.BLOCK_S = BLOCK_S

        if save_final_state:
            return out, final_state
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.save_final_state:
            dout, dfinal_state_user = grad_outputs
        else:
            (dout,) = grad_outputs
            dfinal_state_user = None

        (
            q,
            k,
            v,
            beta,
            decay,
            I_P_kv,
            M_use_pad,
            init_kv_padded,
        ) = ctx.saved_tensors
        B, H, D, N, F, S = ctx.shape
        reverse = ctx.reverse
        dot_precision = ctx.dot_precision
        BLOCK_S = ctx.BLOCK_S
        device = q.device
        fp32 = torch.float32
        BH = B * H

        # ── 1. Reshape inputs to BHFSD for Phase Ā / C̄. ─────────────────────
        def bhdn_to_bhfsd(x):
            return x.reshape(B, H, D, F, S).permute(0, 1, 3, 4, 2).reshape(BH, F, S, D).contiguous()

        dO_bhfsd = bhdn_to_bhfsd(dout.float().contiguous())
        q_bhfsd = bhdn_to_bhfsd(q)
        k_bhfsd = bhdn_to_bhfsd(k)
        v_bhfsd = bhdn_to_bhfsd(v)

        M_post = M_use_pad[:, :, :D, :D].float().contiguous()

        I_D = torch.eye(D, device=device, dtype=fp32)
        P_kv_all = I_D[None, None] - I_P_kv[:, :, :D, :D].float()

        # ── 2. Phase C̄: dQ, dM_C. ──────────────────────────────────────────
        dQ_kv, dM_C = phase_c_bwd(
            q_bhfsd,
            M_post,
            dO_bhfsd,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # ── 3. Phase B̄: reverse scan. ──────────────────────────────────────
        decay_bhf = decay.reshape(BH, F).float()
        beta_bhfs = beta.reshape(BH, F, S).float()

        if reverse:
            # Reverse-direction forward-in-time scan.
            #   total_dM_rev[0]   = dM_C[0]
            #   total_dM_rev[f+1] = dM_C[f+1] + g[f+1] * (I - P[f+1])^T @ total_dM_rev[f]
            total_dM = torch.empty_like(dM_C)
            total_dM[:, 0] = dM_C[:, 0]
            for f in range(F - 1):
                g_next = decay_bhf[:, f + 1].view(BH, 1, 1)
                I_minus_P_next = I_D - P_kv_all[:, f + 1]
                total_dM[:, f + 1] = dM_C[:, f + 1] + g_next * (I_minus_P_next.transpose(-2, -1) @ total_dM[:, f])
            dinit_state = None

            # In the reverse direction's per-frame attribution, the rev step at
            # frame f uses parameters at frame f to step M_rev[f] → M_rev[f-1].
            # So the post-state grad `total_dM_rev[f-1]` contributes to params
            # at frame f, and frame 0 has no rev-contribution. This matches the
            # bidi bwd `combine_bidi_dA_dP_dg` (dM-shifted by -1, M_prev = M_rev[f]).
            #
            # Build:
            #   total_dM_shifted[f] = total_dM_rev[f-1]   for f >= 1
            #   total_dM_shifted[0] = 0
            #   M_prev_rev[f]       = M_rev[f]            for f >= 1
            #   M_prev_rev[0]       = 0                   (cancelled by shift's zero)
            zero_DD = torch.zeros(BH, 1, D, D, device=device, dtype=fp32)
            total_dM_shifted = torch.cat([zero_DD, total_dM[:, : F - 1]], dim=1).contiguous()
            # M_rev[f] is the post-rev-step state at frame f. M_post (= M_rev_d)
            # already has these. For frame 0 we use 0 (cancelled by the zero in
            # total_dM_shifted, so the value here is irrelevant — just needs to
            # be the same shape for broadcast).
            M_prev_for_combine = M_post.contiguous()
            # Replace total_dM with the shifted version so the existing combine
            # formulae produce the correct per-frame attribution.
            total_dM = total_dM_shifted
        else:
            if dfinal_state_user is not None and dfinal_state_user.numel() > 0:
                # Cam state: caller-facing layout IS kernel layout. Direct reshape.
                dfinal_kernel = dfinal_state_user.float().reshape(BH, D, D).contiguous()
            else:
                dfinal_kernel = torch.zeros(BH, D, D, device=device, dtype=fp32)

            total_dM, dM_init_bh = _phase_b_fwd_only_bwd_pt(
                dM_C,
                P_kv_all,
                decay_bhf,
                dfinal_kernel,
            )

            if init_kv_padded.numel() > 0:
                init_M0 = init_kv_padded[:, :D, :D].float().reshape(B, H, D, D)
            else:
                init_M0 = torch.zeros(B, H, D, D, device=device, dtype=fp32)
            init_M0_bh = init_M0.reshape(BH, 1, D, D)
            M_fwd_full = torch.cat([init_M0_bh, M_post], dim=1)
            M_prev_for_combine = M_fwd_full[:, :-1].contiguous()

            if ctx.has_init_state:
                dinit_state = dM_init_bh.reshape(B, H, D, D).contiguous().to(fp32)
            else:
                dinit_state = None

        # ── 4. Combine into (dA_kv, dP_kv, dg_kv). ──────────────────────────
        dA_total, dP_kv_total, dg_kv_total = _combine_fwd_only_dA_dP_dg(
            total_dM,
            M_prev_for_combine,
            P_kv_all,
            decay_bhf,
        )

        # ── 5. Phase Ā KV. ──────────────────────────────────────────────────
        dK_kv, dV_bhfsd, dbeta_kv = phase_a_kv_bwd(
            k_bhfsd,
            v_bhfsd,
            beta_bhfs,
            dA_total,
            dP_kv_total,
            D,
            BLOCK_S=BLOCK_S,
            dot_precision=dot_precision,
        )

        # ── 6. Reshape grads back to cam layout. ────────────────────────────
        def bhfsd_to_bhdn(x_bhfsd):
            return x_bhfsd.reshape(B, H, F, S, D).permute(0, 1, 4, 2, 3).reshape(B, H, D, N).contiguous()

        dq = bhfsd_to_bhdn(dQ_kv.float())
        dk = bhfsd_to_bhdn(dK_kv.float())
        dv = bhfsd_to_bhdn(dV_bhfsd.float())
        dbeta = dbeta_kv.reshape(B, H, F, S).to(beta.dtype)
        ddecay = dg_kv_total.reshape(B, H, F).to(decay.dtype)

        return (
            dq,
            dk,
            dv,
            dbeta,
            ddecay,
            dinit_state,
            None,  # reverse
            None,  # save_final_state
            None,  # dot_precision
            None,  # BLOCK_S
        )


def cam_gdn_chunkwise_stateful_autograd(
    q,
    k,
    v,
    beta,
    decay,
    *,
    init_state=None,
    reverse=False,
    save_final_state=False,
    dot_precision=None,
    BLOCK_S=None,
):
    """Cam-branch chunkwise scan with autograd + full state-grad I/O.

    Args mirror ``cam_scan_chunkwise`` / ``cam_scan_func``:
      q, k, v: (B, H, D, N) fp32 contiguous
      beta:    (B, H, F, S) fp32
      decay:   (B, H, F)   fp32
      init_state: optional (B, H, D, D) fp32 — forward-only state cache.
      reverse: scan direction (state I/O rejected when True).
      save_final_state: forward-only; surfaces a differentiable final_state.

    Returns:
      out (B, H, D, N) fp32, or (out, final_state (B, H, D, D)).
    """
    return CamGDNChunkwiseStatefulFunction.apply(
        q,
        k,
        v,
        beta,
        decay,
        init_state,
        reverse,
        save_final_state,
        dot_precision,
        BLOCK_S,
    )


__all__ = [
    "CamGDNChunkwiseStatefulFunction",
    "cam_gdn_chunkwise_stateful_autograd",
]
