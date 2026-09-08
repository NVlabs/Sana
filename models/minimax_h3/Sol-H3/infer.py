#!/usr/bin/env python3
"""Command-line entry point for 768p MiniMax-H3 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h3_runtime import MiniMaxH3Inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MiniMax-H3 checkpoint directory or repo ID")
    parser.add_argument("--adapter", required=True, help="FastH3 dense-datafree adapter file")
    parser.add_argument(
        "--attention-backend",
        choices=("dense", "sol", "sol_bsa"),
        default="sol_bsa",
        help="Attention backend (default: sol_bsa)",
    )
    parser.add_argument(
        "--compute-quant",
        choices=("none", "mxfp8"),
        default="none",
        help="DiT linear compute mode (default: none/BF16)",
    )
    parser.add_argument("--task", choices=("t2v", "i2v", "ref2va"), required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--image", type=Path, help="Required for i2v; PNG, JPEG, or WebP")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="TYPE:PATH",
        help="Ordered Ref2VA input; repeat with image:, video:, or audio:",
    )
    parser.add_argument(
        "--reference-image-resize-mode",
        choices=("match", "diffusers"),
        default="match",
        help="Ref2VA image sizing: fast match mode (default) or official Diffusers 2048 mode",
    )
    parser.add_argument("--duration", type=int, choices=(5, 10, 15), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def load_references(specs: list[str]):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference

    references = []
    kinds = []
    for spec in specs:
        kind, separator, path = spec.partition(":")
        kind = kind.lower().strip()
        path = path.strip()
        if not separator or kind not in {"image", "video", "audio"} or not path:
            raise SystemExit(
                f"invalid --reference {spec!r}; use image:PATH, video:PATH, or audio:PATH"
            )
        references.append(MiniMaxH3Reference(**{kind: path}))
        kinds.append(kind)
    if references and not ({"image", "video"} & set(kinds)):
        raise SystemExit("Ref2VA audio must be paired with an image or video reference")
    return references


def main() -> int:
    args = parse_args()
    if bool(args.prompt) == bool(args.prompt_file):
        raise SystemExit("pass exactly one of --prompt or --prompt-file")
    if args.task == "i2v" and args.image is None:
        raise SystemExit("--image is required for i2v")
    if args.task == "t2v" and args.image is not None:
        raise SystemExit("--image is only valid for i2v")
    if args.task == "ref2va" and args.image is not None:
        raise SystemExit("use --reference image:PATH for Ref2VA, not --image")
    if args.task == "ref2va" and not args.reference:
        raise SystemExit("pass at least one --reference for Ref2VA")
    if args.task != "ref2va" and args.reference:
        raise SystemExit("--reference is only valid for Ref2VA")
    if args.task == "ref2va" and args.attention_backend == "sol":
        raise SystemExit("Ref2VA supports --attention-backend dense or sol_bsa")

    prompt = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8").strip()
    )
    references = load_references(args.reference)

    with MiniMaxH3Inference(
        args.model,
        args.adapter,
        attention_backend=args.attention_backend,
        task=args.task,
        reference_image_resize_mode=args.reference_image_resize_mode,
        compute_quant=args.compute_quant,
    ) as engine:
        if args.warmup:
            engine.warmup(
                duration=args.duration,
                prompt=prompt,
                image=args.image,
                references=references or None,
            )
        result = engine.generate(
            prompt,
            duration=args.duration,
            seed=args.seed,
            image=args.image,
            references=references or None,
        )
        if result is not None:
            result.save(args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output.resolve()),
                        "task": args.task,
                        "duration": args.duration,
                        "seed": args.seed,
                        "attention_backend": args.attention_backend,
                        "compute_quant": args.compute_quant,
                        "reference_image_resize_mode": (
                            args.reference_image_resize_mode if args.task == "ref2va" else None
                        ),
                        "inference_s": round(result.elapsed_s, 3),
                    },
                    ensure_ascii=False,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
