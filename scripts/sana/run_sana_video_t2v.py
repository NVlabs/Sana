#!/usr/bin/env python3
"""Text-to-video generation with the SANA-Video 2B diffusers pipeline.

Mirrors the official HF docs example:
  https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_video

Recommended dtypes (per the model card): transformer + text_encoder in
bfloat16, VAE in float32.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video

DEFAULT_PROMPT = (
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully "
    "measuring flour, while the dog is stirring the batter with a wooden spoon. "
    "The kitchen is cozy, with sunlight streaming through the window."
)
DEFAULT_NEG = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEG)
    p.add_argument("--motion-scale", type=int, default=30)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/sana_video/sana_video_t2v.mp4")
    p.add_argument(
        "--vae-tiling",
        action="store_true",
        help="enable VAE spatial tiling to bound decode memory at high resolution",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(
        f"[env] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(
            f"[env] gpu={torch.cuda.get_device_name(0)} "
            f"cap={torch.cuda.get_device_capability(0)}",
            flush=True,
        )

    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    pipe = SanaVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.to("cuda")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
        print("[vae] spatial tiling enabled", flush=True)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    prompt = args.prompt + f" motion score: {args.motion_scale}."
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(
        f"[infer] {args.width}x{args.height} frames={args.frames} "
        f"steps={args.steps} guidance={args.guidance_scale} seed={args.seed}",
        flush=True,
    )
    t1 = time.time()
    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        frames=args.frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
    ).frames[0]
    print(f"[infer] done in {time.time() - t1:.1f}s ({len(video)} frames)", flush=True)

    export_to_video(video, args.output, fps=args.fps)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[save] {args.output} ({size_mb:.2f} MB)", flush=True)
    print("RUN_COMPLETE_MARKER", flush=True)


if __name__ == "__main__":
    main()
