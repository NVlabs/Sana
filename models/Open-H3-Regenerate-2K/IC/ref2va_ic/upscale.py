"""The author 3D-conv latent upscaler (minimax_h3_latent_upscaler_3d_conv_v1, LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
outside ComfyUI, as Open-H3-Regenerate-2K applies it: upscaler_node.py is the unchanged author node (MIT,
LICENSE-upscaler-node.txt); only ComfyUI's registration / model lookup are shimmed.  It maps a normalized H3 latent
[24, T, h, w] (the draft) to a normalized latent [24, T, H, W] at the target size (FP32).  Used by the up3 / up4
samplers as their starting point.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch

NODE = Path(__file__).resolve().parent / 'upscaler_node.py'


def load(checkpoint):
    folder = types.ModuleType('folder_paths')
    folder.folder_names_and_paths = {'latent_upscale_models': ([], set())}
    folder.get_full_path_or_raise = lambda category, name: str(checkpoint)
    sys.modules['folder_paths'] = folder
    sys.modules['comfy.model_management'] = None
    sys.modules['comfy_api.latest'] = None
    spec = importlib.util.spec_from_file_location('author_upscaler', NODE)
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    node.io.NodeOutput = lambda value: (value,)
    return node


STAGE = (90, 160)   # 2560x1440: larger targets take two passes (to 2K, then to the target), as Open-H3-Regenerate-2K


def upscale(node, native, shape, device='cuda'):
    if shape[2] > STAGE[0]:
        native = _pass(node, native, [*shape[:2], *STAGE], device)
    return _pass(node, native, shape, device)


def _pass(node, native, shape, device):
    mode = {'mode': node.UpscaleMode.TARGET_DIMENSIONS, 'width': shape[3] * 16, 'height': shape[2] * 16}
    with torch.inference_mode():
        result, = node.MinimaxH3LatentUpscaler3D.execute(
            {'samples': native[None].float()}, 'upscaler', mode, align=32, enable_temporal_chunking=False,
            force_unload=False, device=device, precision='bf16')
    out = result['samples'].float()[0]
    assert tuple(out.shape) == tuple(shape), (tuple(out.shape), shape)
    return out
