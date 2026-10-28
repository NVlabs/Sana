"""Loads the unchanged author latent-upscaler node (upscaler_node.py, MIT) outside ComfyUI and applies it.

The node is the 3D-convolution upscaler from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler. Only ComfyUI's
registration / model lookup are shimmed; architecture, precision, normalization and target-size logic are the node's.
It maps a normalized native H3 latent z to D(U(N(z))), i.e. a normalized latent at the target size.
"""
import importlib.util
from pathlib import Path
import sys
import types

import torch

NODE = Path(__file__).resolve().parent / "upscaler_node.py"


def load(checkpoint):
    folder = types.ModuleType("folder_paths")
    folder.folder_names_and_paths = {"latent_upscale_models": ([], set())}
    folder.get_full_path_or_raise = lambda category, name: str(checkpoint)
    sys.modules["folder_paths"] = folder
    sys.modules["comfy.model_management"] = None
    sys.modules["comfy_api.latest"] = None
    spec = importlib.util.spec_from_file_location("author_upscaler", NODE)
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    node.io.NodeOutput = lambda value: (value,)
    return node


def upscale(node, native, target_shape):
    """native: normalized (24, T, h, w) latent -> normalized (24, T, H, W) at target_shape, FP32."""
    mode = {"mode": node.UpscaleMode.TARGET_DIMENSIONS, "width": target_shape[3] * 16, "height": target_shape[2] * 16}
    with torch.inference_mode():
        result, = node.MinimaxH3LatentUpscaler3D.execute(
            {"samples": native[None].float()}, "upscaler", mode, align=32, enable_temporal_chunking=False,
            force_unload=False, device="cuda", precision="bf16")
    return result["samples"].float()[0]
