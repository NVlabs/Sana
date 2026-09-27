#!/usr/bin/env python3
"""Maintainer utility: package existing Diffusers components with fused weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    LTX2ConditionPipeline,
    LTX2VideoDiffusionDecoderModel,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

from sol_refiner_h3 import SoLRefinerH3Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True, help="Compatible LTX-2.5 Diffusers package"
    )
    parser.add_argument(
        "--transformer",
        required=True,
        help="Already-fused Diffusers transformer directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new directory")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        args.transformer, torch_dtype=torch.bfloat16
    )
    base = LTX2ConditionPipeline.from_pretrained(
        args.base,
        transformer=transformer,
        audio_vae=None,
        vocoder=None,
        torch_dtype=torch.bfloat16,
    )
    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        args.base, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
    )
    decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
        args.base, subfolder="diffusion_decoder", torch_dtype=torch.bfloat16
    )
    pipeline = SoLRefinerH3Pipeline(
        vae=base.vae,
        transformer=transformer,
        latent_upsampler=upsampler,
        diffusion_decoder=decoder,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=1, use_dynamic_shifting=False),
        text_encoder=base.text_encoder,
        tokenizer=base.tokenizer,
        connectors=base.connectors,
    )
    pipeline.save_pretrained(args.output, safe_serialization=True)


if __name__ == "__main__":
    main()
