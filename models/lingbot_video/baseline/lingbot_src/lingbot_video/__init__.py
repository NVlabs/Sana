from __future__ import annotations

import importlib
from typing import Any


_EXPORTS = {
    "FlowUniPCMultistepScheduler": (
        "lingbot_video.scheduling_flow_unipc",
        "FlowUniPCMultistepScheduler",
    ),
    "LingBotVideoImageToVideoPipeline": (
        "lingbot_video.pipeline_lingbot_video_i2v",
        "LingBotVideoImageToVideoPipeline",
    ),
    "LingBotVideoPipeline": (
        "lingbot_video.pipeline_lingbot_video",
        "LingBotVideoPipeline",
    ),
    "LingBotVideoTransformer3DModel": (
        "lingbot_video.transformer_lingbot_video",
        "LingBotVideoTransformer3DModel",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
