"""HDR working-space transfers: LogC3 and ACEScct.
Two transforms are supported:
* **LogC3**   — ARRI LogC3 (EI 800) camera log curve (HDR IC-LoRA default).
* **ACEScct** — AMPAS S-2016-001 ACEScct log working space (EXR condition round-trip).
Both map linear HDR [0, ∞) <-> a compressed [0, 1] signal. Callers are responsible
for mapping the [0, 1] signal to/from the VAE's [-1, 1] range (``to_vae_range`` on
encode, the VAE decoder's ``to_rgb`` on decode). Used by ltx-pipelines for EXR
conditioning / HDR IC-LoRA and by ltx-trainer for HDR validation.
"""

from __future__ import annotations

import enum

import torch
from torch import Tensor

from ltx_core.color.primaries import Primaries

__all__ = [
    "HDRTransfer",
    "TransferEncoding",
    "to_acescct_working_space",
    "to_hdr_linear",
]


class TransferEncoding(enum.Enum):
    """Transfer encoding of an HDR tensor (scene-linear vs already-compressed log)."""

    LINEAR = "linear"
    LOG = "log"


# --- curve helpers ----------------------------------------------------------------

_LOGC3_A = 5.555556
_LOGC3_B = 0.052272
_LOGC3_C = 0.247190
_LOGC3_D = 0.385537
_LOGC3_E = 5.367655
_LOGC3_F = 0.092809
_LOGC3_CUT = 0.010591

_ACESCCT_A_LIN = 10.5402377416545
_ACESCCT_B_LIN = 0.0729055341958355
_ACESCCT_X_BRK = 0.0078125
_ACESCCT_Y_BRK = 0.155251141552511
_ACESCCT_LOG_M = 17.52
_ACESCCT_LOG_B = 9.72


def _compress_logc3(hdr: Tensor) -> Tensor:
    x = torch.clamp(hdr, min=0.0)
    log_part = _LOGC3_C * torch.log10(_LOGC3_A * x + _LOGC3_B) + _LOGC3_D
    lin_part = _LOGC3_E * x + _LOGC3_F
    return torch.clamp(torch.where(x >= _LOGC3_CUT, log_part, lin_part), 0.0, 1.0)


def _decompress_logc3(logc: Tensor) -> Tensor:
    logc = torch.clamp(logc, 0.0, 1.0)
    cut_log = _LOGC3_E * _LOGC3_CUT + _LOGC3_F
    lin_from_log = (torch.pow(10.0, (logc - _LOGC3_D) / _LOGC3_C) - _LOGC3_B) / _LOGC3_A
    lin_from_lin = (logc - _LOGC3_F) / _LOGC3_E
    return torch.where(logc >= cut_log, lin_from_log, lin_from_lin)


def _compress_acescct(hdr: Tensor) -> Tensor:
    x = torch.clamp(hdr, min=0.0)
    log_part = (torch.log2(torch.clamp(x, min=1e-12)) + _ACESCCT_LOG_B) / _ACESCCT_LOG_M
    lin_part = _ACESCCT_A_LIN * x + _ACESCCT_B_LIN
    return torch.clamp(torch.where(x > _ACESCCT_X_BRK, log_part, lin_part), 0.0, 1.0)


def _decompress_acescct(ct: Tensor) -> Tensor:
    ct = torch.clamp(ct, 0.0, 1.0)
    lin_from_log = torch.pow(2.0, ct * _ACESCCT_LOG_M - _ACESCCT_LOG_B)
    lin_from_lin = (ct - _ACESCCT_B_LIN) / _ACESCCT_A_LIN
    return torch.where(ct > _ACESCCT_Y_BRK, lin_from_log, lin_from_lin)


class HDRTransfer(enum.Enum):
    """VAE HDR working-space transfer curve."""

    LOGC3 = "logc3"
    ACESCCT = "acescct"

    @property
    def display_name(self) -> str:
        """Human / EXR ``colorSpace`` tag name (e.g. ``ACEScct``, ``LogC3``)."""
        match self:
            case HDRTransfer.LOGC3:
                return "LogC3"
            case HDRTransfer.ACESCCT:
                return "ACEScct"

    @property
    def native_primaries(self) -> Primaries:
        """Linear primaries the curve is defined on (decompress lands here)."""
        match self:
            case HDRTransfer.LOGC3:
                return Primaries.REC709
            case HDRTransfer.ACESCCT:
                return Primaries.ACESCG

    def compress(self, hdr: Tensor) -> Tensor:
        """Compress linear HDR [0, ∞) → working-space [0, 1]."""
        match self:
            case HDRTransfer.LOGC3:
                return _compress_logc3(hdr)
            case HDRTransfer.ACESCCT:
                return _compress_acescct(hdr)

    def decompress(self, compressed: Tensor) -> Tensor:
        """Decompress working-space [0, 1] → linear HDR [0, ∞)."""
        match self:
            case HDRTransfer.LOGC3:
                return _decompress_logc3(compressed)
            case HDRTransfer.ACESCCT:
                return _decompress_acescct(compressed)

    def compress_ldr(self, ldr: Tensor) -> Tensor:
        """LDR [0, 1] → [0, 1] (identity clamp; transfer-agnostic)."""
        return torch.clamp(ldr, 0.0, 1.0)

    def to_working_space(
        self,
        video: Tensor,
        *,
        source_primaries: Primaries = Primaries.REC709,
    ) -> Tensor:
        """Map scene-linear HDR into this transfer's compressed [0, 1] working space.
        Rotates ``source_primaries`` into :attr:`native_primaries`, then compresses.
        Callers that already have working-space codes should skip this and pass them
        straight to ``to_vae_range``.
        """
        linear_native = source_primaries.to(self.native_primaries, video.float())
        return self.compress(linear_native.clamp(min=0.0))

    def to_linear(
        self,
        working: Tensor,
        *,
        out_primaries: Primaries = Primaries.REC709,
    ) -> Tensor:
        """Decompress this transfer's [0, 1] working space to scene-linear HDR.
        Decompresses into :attr:`native_primaries`, then converts into ``out_primaries``.
        Callers that need to keep log codes (e.g. log-tagged EXR round-trip) should
        skip this and write the working signal directly.
        """
        linear_native = self.decompress(working.float())
        return self.native_primaries.to(out_primaries, linear_native).clamp(min=0.0)


# --- thin wrappers for transitional call sites ------------------------------------


def to_acescct_working_space(
    video: Tensor,
    source_primaries: Primaries = Primaries.REC709,
) -> Tensor:
    """Compress scene-linear HDR into the ACEScct [0, 1] working space."""
    return HDRTransfer.ACESCCT.to_working_space(video, source_primaries=source_primaries)


def to_hdr_linear(
    working: Tensor,
    transfer: HDRTransfer = HDRTransfer.LOGC3,
    out_primaries: Primaries = Primaries.REC709,
) -> Tensor:
    """Decompress a [0, 1] working-space signal to scene-linear HDR."""
    return transfer.to_linear(working, out_primaries=out_primaries)
