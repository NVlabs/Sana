"""Raw ``(num, den)`` chunkwise stateful GDN wrapper.

The CP integration tests use this as a local non-CP reference for raw
numerator/denominator semantics. The caller owns the final divide and any
combination across scan directions.

Forward returns:
    direction=1, save_final_state=False : (num, den)
    direction=1, save_final_state=True  : (num, den, final_state_kv, final_state_z)
    direction=2 (reverse, stateless)    : (num, den)

Shapes (matching the legacy ``fused_gdn_func``):
    num:           (B, N, H, D)    — native ``phase_c`` dtype
                                    (fp32 when dot_precision >= 1, else bf16)
    den:           (B, H, N)       — native ``phase_c`` dtype
    final_state_kv:(B, H, D, D)    — fp32 (forward-only)
    final_state_z: (B, H, D, 1)    — fp32 (forward-only)

Returning ``(num, den)`` in their native dtype preserves precision through an
outside sum + divide. In mixed precision, bf16 qkv with fp32 dot accumulation
keeps the caller's final divide in fp32 before the final cast.

Backward accepts ``(dnum, dden[, dfinal_kv, dfinal_z])`` directly — no
implicit divide-VJP.
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
    phase_a_kv_bwd,
    phase_a_z_bwd,
    phase_c_bwd,
)
from diffusion.model.ops.fused_gdn_chunkwise_stateful_bwd import (
    _combine_fwd_only_dA_dP_dg,
    _combine_fwd_only_dB_dPz_dg_z,
    _combine_rev_only_dA_dP_dg,
    _combine_rev_only_dB_dPz_dg_z,
    _pad_state_kv_to_block,
    _pad_state_z_to_block,
    _phase_b_fwd_only_bwd_pt,
    _phase_b_rev_only_bwd_pt,
    _phase_b_z_fwd_only_bwd_pt,
    _phase_b_z_rev_only_bwd_pt,
)


class FusedGDNChunkwiseStatefulRawFunction(torch.autograd.Function):
    """Chunkwise stateful GDN that returns raw ``(num, den)`` (no divide).

    See module docstring for shape contract. Backward receives raw ``dnum`` /
    ``dden`` directly from ``grad_outputs`` and therefore does not run the
    final divide VJP.
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
        dot_precision,
        BLOCK_S,
        save_final_state,
        direction,
    ):
        if BLOCK_S is None:
            BLOCK_S = _resolve_bwd_block_s()
        if direction not in (1, 2):
            raise ValueError(f"FusedGDNChunkwiseStatefulRawFunction: direction must be 1 or 2; got {direction}.")
        if direction == 2 and (init_state_kv is not None or init_state_z is not None or save_final_state):
            raise ValueError(
                "FusedGDNChunkwiseStatefulRawFunction: reverse direction (direction=2) is "
                "stateless; init_state_* must be None and save_final_state must be False."
            )
        B, N, three, H, D = qkv.shape
        C = H * D
        assert three == 3 and N == F * S
        if (init_state_kv is None) != (init_state_z is None):
            raise ValueError(
                "FusedGDNChunkwiseStatefulRawFunction: init_state_kv and init_state_z must be "
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

        # RMSNorm (channel-wise) — done outside the kernel.
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        q_inv_rms = torch.rsqrt(q_raw_v.float().pow(2).sum(dim=(-2, -1)) / C + norm_eps)
        k_inv_rms = torch.rsqrt(k_raw_v.float().pow(2).sum(dim=(-2, -1)) / C + norm_eps)
        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(qkv.dtype)
        qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(qkv.dtype)

        # Phase A.
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

        # Return (num, den) in phase_c's native dtype (fp32 when
        # dot_precision >= 1, bf16 otherwise) — matches the reference
        # raw stateful path in fused_gdn_chunkwise.fused_gdn_stateful_chunkwise.
        # The caller (bidi combine in _forward_main_branch_with_cache) does
        # the summation and final divide outside, so they want maximum
        # precision in the intermediate (num, den).
        num_user = num_out
        den_user = den_out

        if save_final_state:
            final_state_kv = final_kv_pad.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D].transpose(-1, -2).contiguous()
            final_state_z = final_z_pad.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        else:
            final_state_kv = torch.empty(0, device=device, dtype=fp32)
            final_state_z = torch.empty(0, device=device, dtype=fp32)

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
            I_P_z,
            M_use,
            z_use,
            init_kv_padded if init_kv_padded is not None else torch.empty(0, device=device, dtype=fp32),
            init_z_padded if init_z_padded is not None else torch.empty(0, device=device, dtype=fp32),
        )
        ctx.shape = (B, N, H, D, F, S, C)
        ctx.k_scale = k_scale
        ctx.norm_eps = norm_eps
        ctx.dot_precision = dot_precision
        ctx.BLOCK_S = BLOCK_S
        ctx.has_init_state = init_state_kv is not None
        ctx.save_final_state = save_final_state
        ctx.direction = direction
        ctx.q_nw_was_none = q_nw_was_none
        ctx.k_nw_was_none = k_nw_was_none
        ctx.init_state_z_dim = init_state_z.dim() if init_state_z is not None else 0

        if save_final_state:
            return num_user, den_user, final_state_kv, final_state_z
        return num_user, den_user

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.save_final_state:
            dnum_user, dden_user, dfinal_kv_user, dfinal_z_user = grad_outputs
        else:
            dnum_user, dden_user = grad_outputs
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
            I_P_kv,
            I_P_z,
            M_fwd,
            z_fwd,
            init_kv_padded,
            init_z_padded,
        ) = ctx.saved_tensors
        B, N, H, D, F, S, C = ctx.shape
        k_scale = ctx.k_scale
        dot_precision, BLOCK_S = ctx.dot_precision, ctx.BLOCK_S
        device = qkv.device
        fp32 = torch.float32
        dtype = qkv.dtype
        BH = B * H

        q_nw_hd = q_norm_weight.reshape(H, D)
        k_nw_hd = k_norm_weight.reshape(H, D)

        # ── 1. Reshape raw (dnum, dden) into the kernel layouts the rest of
        #      the bwd expects. Matches the post-output_divide_bwd shapes in
        #      the divided wrapper exactly.
        # dnum: (B, N, H, D); dden: (B, H, N) — both in qkv.dtype.
        dnum = dnum_user.to(dtype).contiguous()
        dden = dden_user.to(dtype).contiguous()

        # ── 2. Reconstruct qkv_normed (matches forward exactly). ────────────
        q_raw_v = qkv[:, :, 0]
        k_raw_v = qkv[:, :, 1]
        qkv_normed = qkv.clone()
        qkv_normed[:, :, 0] = (q_raw_v.float() * q_inv_rms[:, :, None, None] * q_nw_hd[None, None]).to(dtype)
        qkv_normed[:, :, 1] = (k_raw_v.float() * k_inv_rms[:, :, None, None] * k_nw_hd[None, None]).to(dtype)

        # ── 3. Phase A intermediates (unpadded D×D, fp32). ──────────────────
        I_D = torch.eye(D, device=device, dtype=fp32)
        P_kv_all = I_D[None, None] - I_P_kv[:, :, :D, :D].float()
        P_z_all = I_D[None, None] - I_P_z[:, :, :D, :D].float()
        del I_P_kv, I_P_z

        direction = ctx.direction
        M_use_d = M_fwd[:, :, :D, :D].float()
        del M_fwd
        z_use_d = z_fwd[:, :, :D].float()
        del z_fwd

        if direction == 1:
            if init_kv_padded.numel() > 0:
                init_M0 = init_kv_padded[:, :D, :D].float().reshape(B, H, D, D)
                init_z0 = init_z_padded[:, :D].float().reshape(B, H, D)
            else:
                init_M0 = torch.zeros(B, H, D, D, device=device, dtype=fp32)
                init_z0 = torch.zeros(B, H, D, device=device, dtype=fp32)
            init_M0_bh = init_M0.reshape(BH, 1, D, D)
            init_z0_bh = init_z0.reshape(BH, 1, D)
            M_fwd_full = torch.cat([init_M0_bh, M_use_d], dim=1)
            z_fwd_full = torch.cat([init_z0_bh, z_use_d], dim=1)
            del M_use_d, z_use_d
        else:
            M_rev_post = M_use_d.contiguous()
            z_rev_post = z_use_d.contiguous()
            del M_use_d, z_use_d

        # ── 4. Rope+relu recomputation. ─────────────────────────────────────
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
            if dfinal_kv_user is not None and dfinal_kv_user.numel() > 0:
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

        # ── 8. Init-state grads. ────────────────────────────────────────────
        if ctx.has_init_state:
            dinit_state_kv = dM_init_kv_bh.reshape(B, H, D, D).transpose(-1, -2).contiguous().to(fp32)
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
            dinit_state_kv,
            dinit_state_z,
            None,  # F
            None,  # S
            None,  # k_scale
            None,  # norm_eps
            None,  # dot_precision
            None,  # BLOCK_S
            None,  # save_final_state
            None,  # direction
        )


def fused_gdn_chunkwise_stateful_raw_autograd(
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
    dot_precision=0,
    BLOCK_S=None,
    save_final_state=False,
    direction=1,
):
    """Directional chunkwise stateful GDN that returns raw ``(num, den)``.

    Directional chunkwise stateful GDN with the final divide omitted. The
    caller is responsible for combining ``num`` / ``den`` across directions.

    Returns:
      (num, den)                                            if save_final_state=False
      (num, den, final_state_kv, final_state_z)             if save_final_state=True
    """
    return FusedGDNChunkwiseStatefulRawFunction.apply(
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
        dot_precision,
        BLOCK_S,
        save_final_state,
        direction,
    )


__all__ = [
    "FusedGDNChunkwiseStatefulRawFunction",
    "fused_gdn_chunkwise_stateful_raw_autograd",
]
