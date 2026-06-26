"""Adapter for NVFP4 quantization."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "quantization/nvfp4"
SOURCE_URLS = [
    "https://github.com/NVIDIA/TransformerEngine",
    "https://github.com/NVIDIA/Model-Optimizer",
]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant.ModelOptFp4Config",
    "sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "transformer_engine",
        "modelopt",
        "sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant",
    )
