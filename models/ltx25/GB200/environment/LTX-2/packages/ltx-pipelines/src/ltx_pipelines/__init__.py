"""
LTX-2 Pipelines: High-level video generation pipelines and utilities.
This package provides ready-to-use pipelines for video generation:
- TI2VidOneStagePipeline: Text/image-to-video in a single stage
- T2AOneStagePipeline: Text-to-audio in a single stage (audio-only output)
- TI2VidTwoStagesPipeline: Two-stage generation with upsampling
- DistilledPipeline: Fast distilled two-stage generation
- ICLoraPipeline: Image/video conditioning with distilled LoRA
- LipDubPipeline: Lip dubbing with IC-LoRA and audio conditioning
- KeyframeInterpolationPipeline: Keyframe-based video interpolation
- RetakePipeline: Regenerate a time region (retake) of an existing video
For more detailed components and utilities, import from specific submodules
like `ltx_pipelines.utils.media_io` or `ltx_pipelines.utils.constants`.
Pipeline classes are imported lazily (PEP 562). Importing this package therefore
does not eagerly pull in every pipeline module, which keeps `import ltx_pipelines`
light and avoids the runpy double-import warning when a pipeline is run as a module
(e.g. `python -m ltx_pipelines.distilled`).
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
    from ltx_pipelines.lipdub import LipDubPipeline
    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

# Public name -> module that defines it. Used for lazy resolution in __getattr__.
_EXPORTS = {
    "A2VidPipelineTwoStage": "ltx_pipelines.a2vid_two_stage",
    "DistilledPipeline": "ltx_pipelines.distilled",
    "ICLoraPipeline": "ltx_pipelines.ic_lora",
    "KeyframeInterpolationPipeline": "ltx_pipelines.keyframe_interpolation",
    "LipDubPipeline": "ltx_pipelines.lipdub",
    "RetakePipeline": "ltx_pipelines.retake",
    "T2AOneStagePipeline": "ltx_pipelines.t2a_one_stage",
    "TI2VidOneStagePipeline": "ltx_pipelines.ti2vid_one_stage",
    "TI2VidTwoStagesPipeline": "ltx_pipelines.ti2vid_two_stages",
}

__all__ = [
    "A2VidPipelineTwoStage",
    "DistilledPipeline",
    "ICLoraPipeline",
    "KeyframeInterpolationPipeline",
    "LipDubPipeline",
    "RetakePipeline",
    "T2AOneStagePipeline",
    "TI2VidOneStagePipeline",
    "TI2VidTwoStagesPipeline",
]


def __getattr__(name: str) -> object:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
