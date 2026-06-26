"""Adapter for ModelOpt / FP8 diffusion quantization."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "quantization/modelopt_fp8"
SOURCE_URLS = ["https://github.com/NVIDIA/Model-Optimizer"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant.ModelOptFp8Config",
    "sglang.multimodal_gen.tools.build_modelopt_fp8_transformer",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "modelopt",
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
    )
