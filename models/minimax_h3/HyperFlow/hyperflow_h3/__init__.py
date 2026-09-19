# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""HyperFlow: Video Rebirth's 8-step LoRA for MiniMax-H3, on top of the official diffusers Modular Pipeline.

Three calls on top of the official loading recipe::

    blocks = hyperflow_blocks("fl2va")                       # official workflow, two blocks swapped
    pipe = blocks.init_pipeline("MiniMaxAI/MiniMax-H3", components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    load_hyperflow_lora(pipe, "videorebirth/hyperflow")  # LoRA + two-time embedder + sigma grid
    if sol_attn_available():
        enable_sol_attention(pipe)                           # optional, NVIDIA Sol-Attn (SM80+)
"""

from .blocks import (
    WORKFLOWS,
    HyperFlowLoopDenoiser,
    HyperFlowSetTimestepsStep,
    configure_hyperflow_blocks,
    current_step,
    hyperflow_blocks,
    swap_hyperflow_blocks,
)
from .embedder import TwoTimeEmbedder
from .lora import (
    MANIFEST_NAME,
    HyperFlowMetadata,
    disable_hyperflow,
    enable_hyperflow,
    find_transformers,
    load_hyperflow_lora,
    read_metadata,
    resolve_weights,
)
from .schedule import (
    AUDIO_SHIFT,
    DEFAULT_SIGMAS_8STEP,
    VIDEO_SHIFT,
    build_row_time_pairs,
    endpoints_from_sigmas,
    shift_sigmas,
)
from .sol_attn import (
    HyperFlowSolAttnProcessor,
    SolAttnRecipe,
    disable_sol_attention,
    enable_sol_attention,
    sol_attn_available,
)

__version__ = "1.0.0"

__all__ = [
    "AUDIO_SHIFT",
    "DEFAULT_SIGMAS_8STEP",
    "MANIFEST_NAME",
    "VIDEO_SHIFT",
    "WORKFLOWS",
    "HyperFlowLoopDenoiser",
    "HyperFlowMetadata",
    "HyperFlowSetTimestepsStep",
    "HyperFlowSolAttnProcessor",
    "SolAttnRecipe",
    "TwoTimeEmbedder",
    "__version__",
    "build_row_time_pairs",
    "configure_hyperflow_blocks",
    "current_step",
    "disable_hyperflow",
    "disable_sol_attention",
    "enable_hyperflow",
    "enable_sol_attention",
    "endpoints_from_sigmas",
    "find_transformers",
    "hyperflow_blocks",
    "load_hyperflow_lora",
    "read_metadata",
    "resolve_weights",
    "shift_sigmas",
    "sol_attn_available",
    "swap_hyperflow_blocks",
]
