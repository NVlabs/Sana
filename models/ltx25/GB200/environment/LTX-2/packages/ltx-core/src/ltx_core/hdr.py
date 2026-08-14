"""HDR utilities: LogC3 compression for HDR IC-LoRA training and inference.
Provides compress/decompress and postprocess helpers for HDR video generation.
Used by ltx-pipelines for HDR IC-LoRA and by ltx-trainer for HDR validation.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


class LogC3:
    """ARRI LogC3 (EI 800) HDR compression.
    Maps linear [0, ∞) <-> LogC3 [0, 1] via the camera log curve. The log
    curve allocates more precision to shadows/midtones and compresses
    highlights smoothly. Callers are responsible for mapping the [0, 1]
    output to the VAE's [-1, 1] input range.
    """

    name = "LogC3"
    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591

    def compress(self, hdr: Tensor) -> Tensor:
        """Compress linear HDR [0, ∞) → LogC3 [0, 1]."""
        x = torch.clamp(hdr, min=0.0)
        log_part = self.C * torch.log10(self.A * x + self.B) + self.D
        lin_part = self.E * x + self.F
        logc = torch.where(x >= self.CUT, log_part, lin_part)
        return torch.clamp(logc, 0.0, 1.0)

    def compress_ldr(self, ldr: Tensor) -> Tensor:
        """Compress LDR [0, 1] → [0, 1] (no log curve, just clamp)."""
        return torch.clamp(ldr, 0.0, 1.0)

    def decompress(self, logc: Tensor) -> Tensor:
        """Decompress LogC3 [0, 1] → linear HDR [0, ∞)."""
        logc = torch.clamp(logc, 0.0, 1.0)
        cut_log = self.E * self.CUT + self.F
        lin_from_log = (torch.pow(10.0, (logc - self.D) / self.C) - self.B) / self.A
        lin_from_lin = (logc - self.F) / self.E
        return torch.where(logc >= cut_log, lin_from_log, lin_from_lin)

    def decompress_ldr(self, logc: Tensor) -> Tensor:
        """Decompress [0, 1] → LDR [0, 1] (identity clamp)."""
        return torch.clamp(logc, 0.0, 1.0)


def apply_hdr_decode_postprocess(
    decoded_video: Tensor,
    transform: Literal["logc3"] = "logc3",
) -> Tensor:
    """Apply HDR decompress to VAE decode output for HDR recovery.
    Args:
        decoded_video: Tensor from VAE decode in [0, 1], shape [B, C, F, H, W].
            Must be float32 for sufficient color resolution.
        transform: "logc3".
    Returns:
        HDR video tensor float32.
    """
    decoded_video = decoded_video.float()
    if transform == "logc3":
        return LogC3().decompress(decoded_video)
    raise ValueError(f"Unsupported HDR transform: {transform}")
