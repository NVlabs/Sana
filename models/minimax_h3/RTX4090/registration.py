"""Register the RTX 4090 runtime without modifying the SGLang installation."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path


_PINNED_SOURCES = {
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3": (
        "5f87319969c446685ee93d422fc34a7c040defb238eff2274d664f2f8310e997"
    ),
    "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising": (
        "2325b039c055d2db8c320a00f65e2880816888fd5fa107b72cfd6a85be104399"
    ),
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.minimax_h3.stages.decoding": (
        "5e2cf87da11e0c744d6c7703d8151abd7da7937e1ad67d576fdbf6c678380954"
    ),
}


def _verify_module_source(module_name: str) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Pinned SGLang module is unavailable: {module_name}")
    path = Path(spec.origin)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = _PINNED_SOURCES[module_name]
    if actual != expected:
        raise RuntimeError(
            f"Unexpected SGLang source for {module_name}: {actual}; "
            f"expected {expected}"
        )


def register_runtime() -> str:
    """Install process-local model and pipeline registry entries."""

    profile = os.environ.get("H3_RTX4090_PROFILE", "dense").strip().lower()
    if profile not in {"dense", "sol", "fullopt"}:
        raise ValueError("H3_RTX4090_PROFILE must be dense, sol, or fullopt")

    _verify_module_source(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3"
    )
    if profile == "dense":
        return profile

    from model import MiniMaxH3DiTModel
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    ModelRegistry.register_model("MiniMaxH3DiTModel", MiniMaxH3DiTModel)
    ModelRegistry.register_model(
        "MiniMaxH3Transformer3DModel", MiniMaxH3DiTModel
    )

    if profile == "fullopt":
        _verify_module_source(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising"
        )
        _verify_module_source(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.minimax_h3.stages.decoding"
        )

        from pipeline import MiniMaxH3RTX4090Pipeline
        from sglang.multimodal_gen import registry

        registry._discover_and_register_pipelines()
        registry._PIPELINE_REGISTRY[
            MiniMaxH3RTX4090Pipeline.pipeline_name
        ] = MiniMaxH3RTX4090Pipeline

    return profile


__all__ = ["register_runtime"]
