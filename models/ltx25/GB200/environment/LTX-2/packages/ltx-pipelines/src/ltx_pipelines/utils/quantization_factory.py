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


class QuantizationKind(str, Enum):
    FP8_CAST = "fp8-cast"
    FP8_SCALED_MM = "fp8-scaled-mm"

    def to_policy(self, checkpoint_path: str | None = None) -> QuantizationPolicy:
        """Build the :class:`QuantizationPolicy` for this kind.
        ``checkpoint_path`` is required for both backends: ``FP8_SCALED_MM``
        uses it to discover the layer set from ``.weight_scale`` tensors,
        and ``FP8_CAST`` uses it to fold any prequant scales into the fp8
        weight at load time.
        """
        if checkpoint_path is None:
            raise ValueError(f"{self.value} quantization requires checkpoint_path.")
        match self:
            case QuantizationKind.FP8_CAST:
                return _build_fp8_cast_policy(checkpoint_path)
            case QuantizationKind.FP8_SCALED_MM:
                return _build_fp8_scaled_mm_policy(checkpoint_path)
            case _:
                assert_never(self)
