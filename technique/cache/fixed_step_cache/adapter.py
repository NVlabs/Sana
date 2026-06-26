"""Adapter for Sol-Engine fixed-step cache."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "cache/fixed_step_cache"
SOURCE_URLS = ["https://github.com/NVlabs/Sana/tree/sol-engine"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.efficiency.techniques.step_cache.StepCache",
    "sglang.multimodal_gen.runtime.cache.ltx2_stage1_cache_core",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "sglang.multimodal_gen.runtime.efficiency.techniques.step_cache"
    )


def get_technique_class():
    from sglang.multimodal_gen.runtime.efficiency.techniques.step_cache import StepCache

    return StepCache
