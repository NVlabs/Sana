"""HDR colour-space policy for pipelines (``HDRColorSpace`` + resolve helpers)."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from pathlib import Path

import torch

from ltx_core.color.primaries import Primaries
from ltx_core.hdr import HDRTransfer


class HDRColorSpace(enum.Enum):
    """Explicit HDR source / working colour space (CLI ``--hdr``).
    ``None`` at call sites means SDR. Members:
    * ``SRGB_LINEAR`` / ``ACESCG`` — scene-linear; compress via ACEScct on load.
    * ``ACESCCT`` — already ACEScct log working codes; no load-time transfer.
    """

    SRGB_LINEAR = "srgb_linear"
    ACESCG = "acescg"
    ACESCCT = "acescct"

    @property
    def is_log_working(self) -> bool:
        """True when the VAE signal is already ACEScct log (no load-time transfer)."""
        return self is HDRColorSpace.ACESCCT

    @property
    def source_primaries(self) -> Primaries:
        match self:
            case HDRColorSpace.ACESCG | HDRColorSpace.ACESCCT:
                return Primaries.ACESCG
            case _:
                return Primaries.REC709

    @property
    def transfer(self) -> HDRTransfer:
        """Curve for the VAE working space (always defined; today always ACEScct).
        Whether to *apply* it depends on the call site:
        * load — compress only when not :attr:`is_log_working`
        * EXR write — decompress only when writing linear EXR
        * HLG — always decompress to Rec.709 linear
        """
        match self:
            case HDRColorSpace.SRGB_LINEAR | HDRColorSpace.ACESCG | HDRColorSpace.ACESCCT:
                return HDRTransfer.ACESCCT
            case _:
                raise ValueError(f"No transfer defined for {self!r}.")

    def exr_output_tags(self) -> tuple[Primaries, str]:
        """``(primaries, colorSpace_tag)`` for :func:`save_exr_tensor`."""
        match self:
            case HDRColorSpace.ACESCCT:
                return Primaries.ACESCG, "ACEScct"
            case HDRColorSpace.ACESCG:
                return Primaries.ACESCG, "ACEScg"
            case HDRColorSpace.SRGB_LINEAR:
                return Primaries.REC709, "sRGB"


def vae_dtype_for_hdr(hdr: HDRColorSpace | None, default: torch.dtype) -> torch.dtype:
    """VAE decode dtype: float32 for HDR, else ``default`` (pipelines use bf16 today)."""
    return torch.float32 if hdr is not None else default


def _has_exr_input(
    images: Iterable[object] = (),
    video_paths: Iterable[str | Path] = (),
) -> bool:
    from ltx_pipelines.utils.media_io.exr import is_exr_dir  # noqa: PLC0415

    for img in images:
        path = getattr(img, "path", img)
        if str(path).lower().endswith(".exr"):
            return True
    return any(is_exr_dir(p) for p in video_paths)


def resolve_hdr_color_space(
    images: Iterable[object] = (),
    video_paths: Iterable[str | Path] = (),
    hdr: HDRColorSpace | None = None,
) -> HDRColorSpace | None:
    """Return the explicit ``--hdr`` value, or raise if EXR inputs lack it.
    ``None`` means SDR.
    """
    if _has_exr_input(images, video_paths) and hdr is None:
        raise ValueError("EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT} to declare the source colour space.")
    return hdr


def decode_hdr_video(
    decoded_video: torch.Tensor,
    transfer: HDRTransfer,
    out_primaries: Primaries = Primaries.REC709,
) -> torch.Tensor:
    """VAE decode ``[F,H,W,C]`` in ``[0,1]`` → scene-linear HDR float32."""
    video = decoded_video.float().permute(3, 0, 1, 2).unsqueeze(0)
    hdr = transfer.to_linear(video, out_primaries=out_primaries)
    return hdr[0].permute(1, 2, 3, 0).contiguous()
