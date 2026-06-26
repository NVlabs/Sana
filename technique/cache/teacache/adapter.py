"""Adapter for TeaCache."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "cache/teacache"
SOURCE_URLS = ["https://github.com/ali-vilab/TeaCache"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.efficiency.techniques.teacache.TeaCache",
    "sglang.multimodal_gen.runtime.cache.teacache.TeaCacheMixin",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "sglang.multimodal_gen.runtime.efficiency.techniques.teacache"
    )


def get_technique_class():
    from sglang.multimodal_gen.runtime.efficiency.techniques.teacache import TeaCache

    return TeaCache
