#!/usr/bin/env python3
"""Refine an existing H3 video using a merged Diffusers model directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import torch
from diffusers.utils import export_to_video, load_video

from sol_refiner_h3 import SoLRefinerH3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Merged pipeline directory or published model repository",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder-seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new path")
    if not args.input.is_file():
        raise SystemExit("Input video does not exist")
    if not torch.cuda.is_available():
        raise SystemExit("An NVIDIA CUDA GPU is required")
    with av.open(str(args.input)) as source:
        fps = source.streams.video[0].average_rate
        if fps is None:
            raise SystemExit("Input video frame rate is unavailable")
        fps = float(fps)
    frames = load_video(str(args.input))
    pipe = SoLRefinerH3Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    result = pipe(
        frames,
        args.prompt,
        width=args.width,
        height=args.height,
        frame_rate=fps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        decoder_generator=torch.Generator("cuda").manual_seed(args.decoder_seed),
    ).frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result, str(args.output), fps=fps)


if __name__ == "__main__":
    main()
