"""User-facing quantization-policy dispatch.
``ltx-core`` exposes one ``build_policy`` factory per backend. This module
provides the user-facing string-keyed dispatch used by CLI args and pipeline
defaults — keeping the enum out of ``ltx-core`` so adding/removing backends is
a single-file change here.
"""

from enum import Enum

from typing_extensions import assert_never

from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_cast import build_policy as _build_fp8_cast_policy
from ltx_core.quantization.fp8_scaled_mm import build_policy as _build_fp8_scaled_mm_policy
from ltx_core.quantization.nvfp4 import (
    ActScale,
    build_nvfp4_cast_policy,
    build_nvfp4_prequant_policy,
)


class QuantizationKind(str, Enum):
    FP8_CAST = "fp8-cast"
    FP8_SCALED_MM = "fp8-scaled-mm"
    NVFP4_CAST = "nvfp4-cast"
    NVFP4_PREQUANT = "nvfp4-prequant"

    def to_policy(self, checkpoint_path: str | None = None) -> QuantizationPolicy:
        """Build the :class:`QuantizationPolicy` for this kind.
        ``checkpoint_path`` is required for ``FP8_*`` and ``NVFP4_PREQUANT``.
        ``NVFP4_CAST`` ignores it (online BF16→NVFP4); the CLI still asks for
        a path so the missing-flag error stays uniform across kinds.
        """
        match self:
            case QuantizationKind.FP8_CAST:
                if checkpoint_path is None:
                    raise ValueError(f"{self.value} quantization requires checkpoint_path.")
                return _build_fp8_cast_policy(checkpoint_path)
            case QuantizationKind.FP8_SCALED_MM:
                if checkpoint_path is None:
                    raise ValueError(f"{self.value} quantization requires checkpoint_path.")
                return _build_fp8_scaled_mm_policy(checkpoint_path)
            case QuantizationKind.NVFP4_CAST:
                return build_nvfp4_cast_policy(act_scale=ActScale.FIXED_1)
            case QuantizationKind.NVFP4_PREQUANT:
                if checkpoint_path is None:
                    raise ValueError(f"{self.value} quantization requires checkpoint_path.")
                return build_nvfp4_prequant_policy(checkpoint_path, act_scale=ActScale.STATIC)
            case _:
                assert_never(self)
