"""Adapter for Sol-Engine feature-norm token pruning."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "token_pruning/feature_norm_pruning"
SOURCE_URLS = ["https://github.com/NVlabs/Sana/tree/sol-engine"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.efficiency.techniques.token_prune.TokenPrune",
    "sglang.multimodal_gen.runtime.efficiency.techniques.token_prune.keep_indices",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "sglang.multimodal_gen.runtime.efficiency.techniques.token_prune"
    )


def get_technique_class():
    from sglang.multimodal_gen.runtime.efficiency.techniques.token_prune import (
        TokenPrune,
    )

    return TokenPrune
