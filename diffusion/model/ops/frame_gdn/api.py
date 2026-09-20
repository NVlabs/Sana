"""Frame-wise GDN transition helpers used by context parallel paths."""

from __future__ import annotations

import torch


def _build_transition_matrices(
    k_f: torch.Tensor,
    v_f: torch.Tensor,
    k_rot_f: torch.Tensor,
    beta_f: torch.Tensor,
    decay_f: torch.Tensor,
    I: torch.Tensor,
    BH: int,
    T: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build transition and input matrices for the frame-wise GDN scan.

    Args:
        k_f: Frame-reshaped keys, ``(B, H, T, D, S)``.
        v_f: Frame-reshaped values, ``(B, H, T, D, S)``.
        k_rot_f: Frame-reshaped rotated keys, ``(B, H, T, D, S)``.
        beta_f: Update gate, ``(B, H, T, 1, 1)`` or ``(B, H, T, 1, S)``.
        decay_f: Decay gate, ``(B, H, T, 1, 1)``.
        I: Identity matrix, ``(1, 1, 1, D, D)``.
        BH: ``B * H``.
        T: Number of frames.
        D: Head dimension.

    Returns:
        ``W_kv, U_kv, W_z, U_z`` in flattened ``(B*H, T, ...)`` layout.
    """
    k_rot_beta = k_rot_f * beta_f
    W_kv = decay_f * (I - torch.matmul(k_rot_beta, k_rot_f.transpose(-1, -2)))
    U_kv = torch.matmul(v_f * beta_f, k_rot_f.transpose(-1, -2))

    k_beta = k_f * beta_f
    W_z = decay_f * (I - torch.matmul(k_beta, k_f.transpose(-1, -2)))
    U_z = k_beta.sum(dim=-1)

    return (
        W_kv.reshape(BH, T, D, D).contiguous(),
        U_kv.reshape(BH, T, D, D).contiguous(),
        W_z.reshape(BH, T, D, D).contiguous(),
        U_z.reshape(BH, T, D).contiguous(),
    )


__all__ = ["_build_transition_matrices"]
