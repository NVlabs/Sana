#!/usr/bin/env python3
"""Sol-Engine HyperFlow inference; no CUDA import is needed for --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local MiniMax-H3 snapshot with both DiT partitions")
    parser.add_argument("--adapter", type=Path, required=True, help="Local HyperFlow 8-step safetensors file")
    parser.add_argument("--task", choices=("t2v", "i2v", "ref2va"), required=True)
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--prompt")
    text.add_argument("--prompt-file", type=Path)
    parser.add_argument("--image", type=Path, help="First frame, required for I2V")
    parser.add_argument("--reference-image", type=Path, action="append", default=[], help="Ordered Ref2VA image; repeat for up to nine images")
    parser.add_argument("--duration", type=int, choices=(5, 10, 15), default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-backend", choices=("sol_bsa", "dense"), default="sol_bsa")
    parser.add_argument("--lora-mode", choices=("fused", "separate"), default="fused")
    parser.add_argument("--keep-full-conditioner", action="store_true", help="Retain the unused conditioner tail and vocabulary head")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.task == "i2v" and args.image is None:
        parser.error("I2V requires --image")
    if args.task != "i2v" and args.image is not None:
        parser.error("--image is only valid for I2V")
    if args.task == "ref2va" and not 1 <= len(args.reference_image) <= 9:
        parser.error("Ref2VA requires 1–9 --reference-image inputs")
    if args.task != "ref2va" and args.reference_image:
        parser.error("--reference-image is only valid for Ref2VA")
    return args


def load_image(path):
    if path is None:
        return None
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def main(argv=None):
    args = parse_args(argv)
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8").strip()
    image = load_image(args.image)
    references = [load_image(path) for path in args.reference_image]
    from sol_hyperflow import HyperFlowInference
    from sol_hyperflow.config import Request

    Request(prompt, args.task, args.duration, args.seed, args.attention_backend,
            image, tuple(references)).payload()
    with HyperFlowInference(args.model, args.adapter, lora_mode=args.lora_mode,
                            trim_conditioner=not args.keep_full_conditioner) as engine:
        options = dict(task=args.task, duration=args.duration, image=image,
                       references=references, attention_backend=args.attention_backend)
        if args.warmup:
            engine.warmup(prompt=prompt, **options)
        media = engine.generate(prompt, seed=args.seed, **options)
        if media is not None:
            media.save(args.output)
            print(json.dumps(dict(engine.last_request, output=str(args.output),
                                  lora_mode=args.lora_mode, compute_quant="none",
                                  warmup_performed=args.warmup,
                                  timing_excludes=["loading", "warmup", "mp4_encoding"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
