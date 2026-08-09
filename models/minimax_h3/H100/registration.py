"""Register the H100 MiniMax-H3 model without patching SGLang."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from .profiles import (
    PINNED_SGLANG_MODEL_SHA256,
    HardwareProfile,
    RuntimeProfile,
    configure_runtime,
)


UPSTREAM_MODEL_MODULE = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"


def _verify_upstream_model() -> Path:
    spec = importlib.util.find_spec(UPSTREAM_MODEL_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Pinned SGLang module is unavailable: {UPSTREAM_MODEL_MODULE}")
    path = Path(spec.origin)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != PINNED_SGLANG_MODEL_SHA256:
        raise RuntimeError(
            f"Unexpected SGLang MiniMax-H3 source: {actual}; "
            f"expected {PINNED_SGLANG_MODEL_SHA256}"
        )
    return path


def register_runtime() -> tuple[HardwareProfile, RuntimeProfile]:
    """Install the process-local model entry used by sparse/cache profiles."""

    hardware, profile = configure_runtime()
    _verify_upstream_model()
    if not profile.sol_attention and profile.cache == "none":
        return hardware, profile

    from .model import MiniMaxH3DiTModel
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    ModelRegistry.register_model("MiniMaxH3DiTModel", MiniMaxH3DiTModel)
    ModelRegistry.register_model("MiniMaxH3Transformer3DModel", MiniMaxH3DiTModel)
    return hardware, profile


__all__ = ["register_runtime"]
