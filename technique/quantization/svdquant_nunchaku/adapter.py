"""Adapter for SVDQuant / Nunchaku."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "quantization/svdquant_nunchaku"
SOURCE_URLS = [
    "https://github.com/nunchaku-ai/nunchaku",
    "https://github.com/nunchaku-ai/deepcompressor",
]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.quantization.nunchaku_linear",
    "sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "nunchaku",
        "sglang.multimodal_gen.runtime.layers.quantization.nunchaku_linear",
    )
