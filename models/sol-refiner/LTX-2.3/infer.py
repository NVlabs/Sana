#!/usr/bin/env python3
"""Refine a video with the original LTX-2.3 SoL-Refiner."""

import argparse
import json
from pathlib import Path

import av
import torch
from diffusers.utils import export_to_video, load_video

from sol_refiner import SoLRefinerPipeline
from sol_refiner.sampling import DEFAULT_NEGATIVE_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="One-step or multi-step Diffusers model package"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["baseline", "sol"], default="baseline")
    parser.add_argument(
        "--teacache",
        action="store_true",
        help="Reuse predictions for the multi-step model",
    )
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--sol-tau", type=float, default=None)
    routing.add_argument("--sol-density", type=float, default=None)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("An NVIDIA CUDA GPU is required")
    if args.engine == "baseline" and (
        args.sol_tau is not None or args.sol_density is not None
    ):
        parser.error("Routing parameters require --engine sol")
    if args.teacache and args.engine != "sol":
        parser.error("Use --engine sol with --teacache")
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.width, args.height) <= 0 or args.width % 2 or args.height % 2:
        parser.error("Output dimensions must be positive and even")
    with av.open(str(args.input)) as source:
        rate = source.streams.video[0].average_rate
        if rate is None:
            parser.error("Input frame rate is unavailable")
        fps = float(rate)
    pipe = SoLRefinerPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    if args.teacache and pipe.config.variant == "one-step":
        parser.error("TeaCache is only supported for the multi-step model")
    if args.engine == "sol":
        pipe.enable_sol_engine(tau=args.sol_tau, density=args.sol_density)
    pipe.enable_model_cpu_offload()
    result = pipe(
        load_video(str(args.input)),
        args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frame_rate=fps,
        teacache=args.teacache,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result, str(args.output), fps=fps, macro_block_size=1)
    print(json.dumps(pipe.last_run))


if __name__ == "__main__":
    main()
