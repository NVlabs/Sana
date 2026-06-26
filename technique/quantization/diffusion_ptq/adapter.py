"""Reference adapter for diffusion PTQ families."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "quantization/diffusion_ptq"
SOURCE_URLS = [
    "https://github.com/Xiuyu-Li/q-diffusion",
    "https://github.com/42Shawn/PTQ4DM",
]
LOCAL_ENTRYPOINTS = ["reference family; no unified local runtime adapter yet"]


def dependency_status() -> dict[str, bool]:
    return _dependency_status("sglang.multimodal_gen.runtime.layers.quantization")
