"""Multi-GPU utilities for the Gemma text encoder."""

from ltx_core.multigpu.gemma.accelerate_wrapper import AccelerateGemmaWrapper
from ltx_core.multigpu.gemma.loader import load_gemma_with_device_map

__all__ = ["AccelerateGemmaWrapper", "load_gemma_with_device_map"]
