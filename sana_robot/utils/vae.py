# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch._dynamo
from safetensors.torch import load_file

from sana_robot.utils.causal_vae import LTX2VideoEncoder3d

SINGLE_PASS_SHAPES = 64


def grouping_slots(encoder: LTX2VideoEncoder3d) -> list[int]:
    """Return the slots of the encoder's cache that hold the frame grouping state of a temporal downsampler instead of cached frames."""
    (slots, slot) = ([], 1)
    for block in encoder.down_blocks:
        slot += 2 * len(block.resnets)
        for downsampler in block.downsamplers:
            if downsampler.stride[0] > 1:
                slots.append(slot)
                slot += 1
            slot += 1
    return slots


def encode_in_batches(encode, clips: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return the latents of clips [N, 3, T, H, W] encoded ``batch_size`` at a time, the last batch filled with copies of its first clip so that the compiled encoder meets one batch shape whatever N is."""
    parts = []
    for start in range(0, clips.shape[0], batch_size):
        part = clips[start : start + batch_size]
        real = part.shape[0]
        if real < batch_size:
            part = torch.cat((part, part[:1].expand(batch_size - real, *part.shape[1:])))
        parts.append(encode(part)[:real])
    return torch.cat(parts)


def load_video_encoder(model_path, device, dtype=torch.bfloat16, compiled: bool = False):
    """Load the encoder of the LTX-2 causal video VAE.

    A whole video on a GPU is encoded in one pass over all its frames, through the encoder compiled for the shape
    of the video and with channels-last weights and frames: the encoding the released RoboDojo checkpoint is
    trained on, in bfloat16. A stream, and a video on the CPU, is encoded block by block on the cache of the
    encoder's layers: the first frame alone, then 8 frames at a time, every block after the cache the one before
    left. The encoder is causal in time, so the two give the same latents up to the rounding of the precision.

    Args:
        model_path: The directory of the VAE weights in the diffusers layout, with the ``vae`` subfolder.
        device: The device to encode on.
        dtype: The precision of the weights and of the latent normalization.
        compiled: Run the block-wise encoder of a stream in channels-last and compiled with ``torch.compile``,
            as a deployment does for its latency. The latents then differ from the eager ones by the rounding of
            the precision, and in bfloat16 by more for a video that starts with copies of its first frame; the
            first calls of every clip size take longer.

    Returns:
        ``encode(frames, stream=None)``: frames [B, 3, T, H, W] in [-1, 1] to normalized float32 latents
        [B, 128, T', H / 32, W / 32]. Without ``stream`` the frames are a whole video, T = 1 + 8 * n and
        T' = 1 + n; on a GPU every new shape of ``frames`` compiles the encoder once, and the encoder keeps
        channels-last weights from its first whole video on. ``stream`` is the state of one video, the frames
        the encoder's layers cache, as a list of tensors that starts empty and is updated in place: its first
        call takes the first frame and each later call the next 8 * n frames.
    """
    folder = Path(model_path, "vae")
    config = json.loads((folder / "config.json").read_text())
    weights = load_file(folder / "diffusion_pytorch_model.safetensors")
    encoder = LTX2VideoEncoder3d(
        in_channels=config["in_channels"],
        out_channels=config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        down_block_types=tuple(config["down_block_types"]),
        spatio_temporal_scaling=tuple(config["spatio_temporal_scaling"]),
        layers_per_block=tuple(config["layers_per_block"]),
        downsample_type=tuple(config["downsample_type"]),
        patch_size=config["patch_size"],
        patch_size_t=config["patch_size_t"],
        resnet_norm_eps=config["resnet_norm_eps"],
        is_causal=config["encoder_causal"],
        spatial_padding_mode=config["encoder_spatial_padding_mode"],
    )
    encoder.load_state_dict(
        {name[len("encoder.") :]: value for (name, value) in weights.items() if name.startswith("encoder.")},
        strict=True,
    )
    encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if compiled:
        encoder.to(memory_format=torch.channels_last_3d)
        encoder.compile(mode="default")
    mean = weights["latents_mean"].view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    std = weights["latents_std"].view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    scaling_factor = config["scaling_factor"]
    block = config["temporal_compression_ratio"]
    slots = grouping_slots(encoder)
    single_pass = None

    def encode_single_pass(frames: torch.Tensor) -> torch.Tensor:
        nonlocal single_pass
        if single_pass is None:
            limits = torch._dynamo.config
            limits.recompile_limit = max(limits.recompile_limit, SINGLE_PASS_SHAPES)
            encoder.to(memory_format=torch.channels_last_3d)
            single_pass = torch.compile(encoder.forward, dynamic=False)
        layout = torch.empty(
            frames.shape, dtype=frames.dtype, device=frames.device, memory_format=torch.channels_last_3d
        )
        return single_pass(layout.copy_(frames), causal=True, feat_cache=None, feat_idx=None)

    @torch.no_grad()
    def encode(frames: torch.Tensor, stream: list | None = None) -> torch.Tensor:
        frames = frames.to(device=device, dtype=dtype)
        if stream is None and frames.is_cuda and frames.shape[2] % block == 1:
            hidden = encode_single_pass(frames)
        else:
            cache = list(stream or ())
            if cache:
                for slot in slots:
                    cache.insert(slot, {"prefixed": True, "pending": None})
            (start, step, hidden) = (0, frames.shape[2] % block or block, [])
            while start < frames.shape[2]:
                hidden.append(
                    encoder(frames[:, :, start : start + step, :, :], causal=True, feat_cache=cache, feat_idx=[0])
                )
                (start, step) = (start + step, block)
            hidden = hidden[0] if len(hidden) == 1 else torch.cat(hidden, dim=2)
        latents = torch.chunk(hidden, 2, dim=1)[0]
        if stream is not None:
            stream[:] = [entry for entry in cache if torch.is_tensor(entry)]
        return ((latents - mean) * scaling_factor / std).float()

    return encode
