"""Stateful chunkwise GDN backward helper scans.

The raw stateful wrapper reuses these small fp32 PyTorch scans to backprop
through forward and reverse chunkwise state recurrences.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _phase_b_fwd_only_bwd_pt(dM_C_fwd, P_all, g, dM_final_fwd):
    """Reverse-time backward scan for the forward state recurrence."""
    BH, num_frames, dim, _ = dM_C_fwd.shape
    I_D = torch.eye(dim, device=dM_C_fwd.device, dtype=dM_C_fwd.dtype)

    total_dM_fwd = torch.empty_like(dM_C_fwd)
    total_dM_fwd[:, num_frames - 1] = dM_C_fwd[:, num_frames - 1] + dM_final_fwd
    for frame in range(num_frames - 2, -1, -1):
        g_next = g[:, frame + 1].view(BH, 1, 1)
        I_minus_P_next = I_D - P_all[:, frame + 1]
        total_dM_fwd[:, frame] = dM_C_fwd[:, frame] + g_next * (
            I_minus_P_next.transpose(-2, -1) @ total_dM_fwd[:, frame + 1]
        )

    g0 = g[:, 0].view(BH, 1, 1)
    I_minus_P0 = I_D - P_all[:, 0]
    dM_init_fwd = g0 * (I_minus_P0.transpose(-2, -1) @ total_dM_fwd[:, 0])
    return total_dM_fwd, dM_init_fwd


def _phase_b_rev_only_bwd_pt(dM_C_rev, P_all, g):
    """Forward-time backward scan for the reverse state recurrence."""
    BH, num_frames, dim, _ = dM_C_rev.shape
    I_D = torch.eye(dim, device=dM_C_rev.device, dtype=dM_C_rev.dtype)

    total_dM_rev = torch.empty_like(dM_C_rev)
    total_dM_rev[:, 0] = dM_C_rev[:, 0]
    for frame in range(num_frames - 1):
        g_next = g[:, frame + 1].view(BH, 1, 1)
        I_minus_P_next = I_D - P_all[:, frame + 1]
        total_dM_rev[:, frame + 1] = dM_C_rev[:, frame + 1] + g_next * (
            I_minus_P_next.transpose(-2, -1) @ total_dM_rev[:, frame]
        )
    return total_dM_rev


def _phase_b_z_rev_only_bwd_pt(dz_C_rev, P_z_all, g):
    """Forward-time backward scan for the reverse denominator state."""
    BH, num_frames, dim = dz_C_rev.shape
    I_D = torch.eye(dim, device=dz_C_rev.device, dtype=dz_C_rev.dtype)

    total_dz_rev = torch.empty_like(dz_C_rev)
    total_dz_rev[:, 0] = dz_C_rev[:, 0]
    for frame in range(num_frames - 1):
        g_next = g[:, frame + 1].view(BH, 1)
        I_minus_P_next = I_D - P_z_all[:, frame + 1]
        total_dz_rev[:, frame + 1] = dz_C_rev[:, frame + 1] + g_next * (
            I_minus_P_next.transpose(-2, -1) @ total_dz_rev[:, frame].unsqueeze(-1)
        ).squeeze(-1)
    return total_dz_rev


def _phase_b_z_fwd_only_bwd_pt(dz_C_fwd, P_z_all, g, dz_final_fwd):
    """Reverse-time backward scan for the forward denominator state."""
    BH, num_frames, dim = dz_C_fwd.shape
    I_D = torch.eye(dim, device=dz_C_fwd.device, dtype=dz_C_fwd.dtype)

    total_dz_fwd = torch.empty_like(dz_C_fwd)
    total_dz_fwd[:, num_frames - 1] = dz_C_fwd[:, num_frames - 1] + dz_final_fwd
    for frame in range(num_frames - 2, -1, -1):
        g_next = g[:, frame + 1].view(BH, 1)
        I_minus_P_next = I_D - P_z_all[:, frame + 1]
        total_dz_fwd[:, frame] = dz_C_fwd[:, frame] + g_next * (
            I_minus_P_next.transpose(-2, -1) @ total_dz_fwd[:, frame + 1].unsqueeze(-1)
        ).squeeze(-1)

    g0 = g[:, 0].view(BH, 1)
    I_minus_P0 = I_D - P_z_all[:, 0]
    dz_init_fwd = g0 * (I_minus_P0.transpose(-2, -1) @ total_dz_fwd[:, 0].unsqueeze(-1)).squeeze(-1)
    return total_dz_fwd, dz_init_fwd


def _combine_fwd_only_dA_dP_dg(total_dM_fwd, M_fwd_prev, P_all, g):
    """Per-frame ``dA``, ``dP``, and ``dg`` for the forward state recurrence."""
    BH, num_frames, dim, _ = total_dM_fwd.shape
    I_D = torch.eye(dim, device=total_dM_fwd.device, dtype=total_dM_fwd.dtype)
    g_per = g.view(BH, num_frames, 1, 1)
    I_minus_P = I_D - P_all

    dA_total = total_dM_fwd.clone()
    dP_total = -g_per * (total_dM_fwd @ M_fwd_prev.transpose(-2, -1))
    dg_total = (total_dM_fwd * (I_minus_P @ M_fwd_prev)).sum(dim=(-2, -1))
    return dA_total, dP_total, dg_total


def _combine_fwd_only_dB_dPz_dg_z(total_dz_fwd, z_fwd_prev, P_z_all, g):
    """Per-frame ``dB_z``, ``dP_z``, and ``dg_z`` for the forward denominator recurrence."""
    BH, num_frames, dim = total_dz_fwd.shape
    I_D = torch.eye(dim, device=total_dz_fwd.device, dtype=total_dz_fwd.dtype)
    g_per = g.view(BH, num_frames, 1, 1)
    I_minus_P = I_D - P_z_all

    dB_z_total = total_dz_fwd.clone()
    dP_z_total = -g_per * (total_dz_fwd.unsqueeze(-1) @ z_fwd_prev.unsqueeze(-2))
    dg_z_total = (total_dz_fwd * (I_minus_P @ z_fwd_prev.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
    return dB_z_total, dP_z_total, dg_z_total


def _combine_rev_only_dA_dP_dg(total_dM_rev, M_rev_post, P_all, g):
    """Per-frame ``dA``, ``dP``, and ``dg`` for the reverse state recurrence."""
    BH, num_frames, dim, _ = total_dM_rev.shape
    zero = torch.zeros(BH, 1, dim, dim, device=total_dM_rev.device, dtype=total_dM_rev.dtype)
    shifted = torch.cat([zero, total_dM_rev[:, : num_frames - 1]], dim=1).contiguous()
    return _combine_fwd_only_dA_dP_dg(shifted, M_rev_post.contiguous(), P_all, g)


def _combine_rev_only_dB_dPz_dg_z(total_dz_rev, z_rev_post, P_z_all, g):
    """Per-frame ``dB_z``, ``dP_z``, and ``dg_z`` for the reverse denominator recurrence."""
    BH, num_frames, dim = total_dz_rev.shape
    zero = torch.zeros(BH, 1, dim, device=total_dz_rev.device, dtype=total_dz_rev.dtype)
    shifted = torch.cat([zero, total_dz_rev[:, : num_frames - 1]], dim=1).contiguous()
    return _combine_fwd_only_dB_dPz_dg_z(shifted, z_rev_post.contiguous(), P_z_all, g)


def _pad_state_kv_to_block(state_kv, BLOCK_D):
    """Pad caller-facing ``(B, H, D, D)`` state to ``(B*H, BLOCK_D, BLOCK_D)``."""
    B, H, D_in, D_out = state_kv.shape
    state = state_kv.transpose(-1, -2).reshape(B * H, D_out, D_in)
    if D_in != BLOCK_D or D_out != BLOCK_D:
        return F.pad(state, (0, BLOCK_D - D_in, 0, BLOCK_D - D_out)).contiguous()
    return state.contiguous()


def _pad_state_z_to_block(state_z, BLOCK_D):
    """Pad caller-facing z-state ``(B, H, D)`` or ``(B, H, D, 1)`` to ``(B*H, BLOCK_D)``."""
    z = state_z.squeeze(-1) if state_z.dim() == 4 else state_z
    B, H, dim = z.shape
    z = z.reshape(B * H, dim)
    if dim != BLOCK_D:
        return F.pad(z, (0, BLOCK_D - dim)).contiguous()
    return z.contiguous()


__all__ = [
    "_combine_fwd_only_dA_dP_dg",
    "_combine_fwd_only_dB_dPz_dg_z",
    "_combine_rev_only_dA_dP_dg",
    "_combine_rev_only_dB_dPz_dg_z",
    "_pad_state_kv_to_block",
    "_pad_state_z_to_block",
    "_phase_b_fwd_only_bwd_pt",
    "_phase_b_rev_only_bwd_pt",
    "_phase_b_z_fwd_only_bwd_pt",
    "_phase_b_z_rev_only_bwd_pt",
]
