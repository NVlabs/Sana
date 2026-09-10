"""LBH normalized-H3 x2 before the existing H3-to-LTX adapter.

Loads the pinned author's model definitions unchanged, without ComfyUI folder
registration. The session owns conditioning, denoising and mux.
"""
from __future__ import annotations

import ast
from collections import Counter
import gc
import hashlib
from pathlib import Path
import re
import time


SOURCE_COMMIT = "d7c01b9011f2e8439493f6c02c29995a27df276f"
SOURCE_SHA256 = "744063b43e0f3eec23e2485cb7c65503069946ca9690906ecb548d7515cb89e2"
CHECKPOINT_REVISION = "13ccf95d85d120bdbc92c05b1247a6e147bf54bf"
CHECKPOINT_SHA256 = "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6"
CHECKPOINT_BYTES = 690592992
MODEL_SYMBOLS = ("normalization", "zero_module", "AttnBlock3D", "ResBlockEmb3D",
                 "TemporalConv", "LatentResizer3D", "_extract_upscaler_sd", "_detect_arch")
H3_INPUT = (1, 24, 37, 24, 42)
H3_UPSCALED = (1, 24, 37, 48, 84)
ADAPTER_OUTPUT = (1, 128, 17, 24, 42)
REFINER_INPUT = (1, 128, 16, 24, 42)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_definitions(source):
    """Select original AST nodes only; no Comfy imports or model math edits."""
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in MODEL_SYMBOLS]
    if tuple(node.name for node in nodes) != MODEL_SYMBOLS:
        raise ValueError("pinned LBH model definitions are missing or reordered")
    return ast.Module(body=nodes, type_ignores=[])


def normalization_statistics(source):
    """Read the pinned author's literal 24-channel node statistics only."""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"LATENTS_MEAN", "LATENTS_STD"}:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"LATENTS_MEAN", "LATENTS_STD"} or any(len(x) != 24 for x in values.values()):
        raise ValueError("pinned LBH node normalization statistics are missing")
    return values["LATENTS_MEAN"], values["LATENTS_STD"]


def load_upscaler(source_path, checkpoint_path, *, device, torch_module):
    """One preload using the author's architecture detector and strict loader."""
    from einops import rearrange
    from safetensors.torch import load_file

    torch = torch_module
    source_path, checkpoint_path = Path(source_path).resolve(strict=True), Path(checkpoint_path).resolve(strict=True)
    if file_sha256(source_path) != SOURCE_SHA256:
        raise ValueError("LBH source does not match the pinned d7c01 companion")
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES or file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("LBH checkpoint does not match the pinned BF16 release")
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
                 "gc": gc, "re": re, "rearrange": rearrange}
    source = source_path.read_text()
    exec(compile(model_definitions(source), str(source_path), "exec"), namespace)
    started = time.perf_counter()
    state = namespace["_extract_upscaler_sd"](load_file(str(checkpoint_path), device="cpu"))
    config = namespace["_detect_arch"](state)
    model = namespace["LatentResizer3D"](**config)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False).to(torch.bfloat16)
    mean, std = normalization_statistics(source)
    for name, values in (("comfy_latents_mean", mean), ("comfy_latents_std", std)):
        model.register_buffer(name, torch.tensor(values, device=device, dtype=torch.bfloat16).view(
            1, -1, 1, 1, 1), persistent=False)
    torch.cuda.synchronize()
    del state
    inventory = dict(Counter(str(parameter.dtype) for parameter in model.parameters()))
    if set(inventory) != {"torch.bfloat16"}:
        raise RuntimeError(f"unexpected LBH model dtype inventory: {inventory}")
    route = {
        "name": "h3_upscale_before_adapter", "scale": 2.0,
        "source": str(source_path), "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
        "checkpoint": str(checkpoint_path), "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES,
        "model_config": config, "compute_dtype": "torch.bfloat16",
        "actual_parameter_dtypes": inventory, "load_s_excluded": time.perf_counter() - started,
        "input_normalization": "released_h3_per_channel_mean_std",
        "output_normalization": "released_h3_per_channel_mean_std",
        "raw_normalization_roundtrip": True,
        "normalization_boundary": "Comfy_H3_VAEEncode_samples_already_normalized",
        "author_node_transform": "D(model(N(h3_normalized)))",
        "normalization_roundtrip_meaning": "author_node_N_and_D; incoming_and_outgoing_LATENT_are_normalized_H3",
        "enable_chunking": True, "temporal_chunk_size": 32,
        "temporal_overlap": config["temporal_kernel"] if config["temporal_every"] else 0,
        "h3_input_shape": list(H3_INPUT), "h3_upscaled_shape": list(H3_UPSCALED),
        "adapter_native_output_shape": list(ADAPTER_OUTPUT), "refiner_input_shape": list(REFINER_INPUT),
        "ltx_learned_x2_enabled": False,
        "ltx_learned_x2_loaded": False,
    }
    return model, route


def upscale_comfy_normalized_h3(model, normalized):
    """Match the author node around a normalized Comfy H3 LATENT.

    Comfy H3 VAEEncode already returns normalized H3. The author's node still
    applies its own N/model/D, so do not cancel those transforms at this seam.
    Calling model(x) directly remains available for reproducing old diagnostics.
    """
    mean, std = model.comfy_latents_mean, model.comfy_latents_std
    samples = normalized.to(device=mean.device, dtype=mean.dtype, copy=True)
    network_input = (samples - mean) / std
    output = model(network_input, scale=2.0, target_size=H3_UPSCALED[2:], enable_chunking=True)
    output = output * std + mean
    return output.to(dtype=normalized.dtype)
