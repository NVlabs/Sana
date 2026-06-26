"""Adapter for EasyCache."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "cache/easycache"
SOURCE_URLS = ["https://github.com/H-EmbodVis/EasyCache"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.models.dits.sana_video",
    "sglang.multimodal_gen.runtime.cache.ltx2_stage1_cache_core",
    "scripts/sana/sana_video_sglang_run.py --easycache",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "sglang.multimodal_gen.runtime.models.dits.sana_video",
        "sglang.multimodal_gen.runtime.cache.ltx2_stage1_cache_core",
    )
