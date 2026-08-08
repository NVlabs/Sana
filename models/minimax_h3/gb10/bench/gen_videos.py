#!/usr/bin/env python3
"""Generate videos for arbitrary prompts, accelerated or as the reference.

Three configurations, all with every lossless fusion on:

* `accelerated` — Sol-Attn at tau=1 and FirstBlockCache at 0.08, the row the ablation ends on.
* `cache_only` — the cache, dense attention. The two approximations are not interchangeable:
  the cache reuses a whole step's residual and so blurs *time*, while Sol-Attn drops attention
  blocks and so changes what each step attends to. When an accelerated sample lands on a
  visibly different scene rather than a slightly different one, this separates which of the two
  moved it.
* `reference` — neither. Not the released BF16 model either: the fusions are bit-identical to
  the unfused path, so this is numerically the baseline while taking a third less wall time to
  produce. That makes it the right denominator — the only differences visible between the
  videos are the approximations under test, with nothing else moving.

The VAE is left exactly as it ships in both; the tile batching measured elsewhere is
deliberately not applied, so the decode path is the stock one.

Prompts come from a JSON file rather than the command line because these contain quotation
marks and apostrophes that a shell would eat, and because a prompt that reached the model
subtly mangled would look like a model problem.

Speed is not measured here. This is a full request on a machine whose other tenants are not
controlled, so its wall time says as much about them as about the model; the numbers to quote
come from `bench_dit.py --sweep`, which interleaves variants inside one process.
"""

from __future__ import annotations

import paths

paths.setup()

import argparse
import json
import os
import time

import torch

import cache_line
import h3_fp8
import gpu_infer

# What the ablation's last row is made of.
SOL = dict(tau=1.0, dense_blocks=2, dense_first_steps=10)
CACHE_THRESHOLD = 0.08


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True,
                        help="JSON: [{'name': ..., 'prompt': ...}, ...]")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(paths.OUTPUT_DIR / "videos"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--config", choices=("accelerated", "cache_only", "reference"),
                        default="accelerated")
    parser.add_argument("--suffix", default="",
                        help="appended to each output name, so a second config cannot "
                             "silently overwrite the first")
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as handle:
        jobs = json.load(handle)
    os.makedirs(args.output_dir, exist_ok=True)

    args.prompt_file = str(paths.PROMPT_FILE)
    args.prompt = jobs[0]["prompt"]
    gpu_infer.LOSSLESS = dict(fuse_qkv=True, quantizer="triton", fuse_adaln=True,
                           fuse_rope=True, fuse_swiglu=True)
    pipe, placement = gpu_infer.build_pipeline(args, **gpu_infer.LOSSLESS)
    use_sol = args.config == "accelerated"
    use_cache = args.config in ("accelerated", "cache_only")
    described = ", ".join(
        ([f"sol tau={SOL['tau']}"] if use_sol else [])
        + ([f"cache {CACHE_THRESHOLD}"] if use_cache else [])
    ) or "lossless kernels only"
    print(f"[gen] loaded in {placement['load_s']:.1f}s, "
          f"{args.width}x{args.height} {args.num_frames}f {args.steps} steps, "
          f"{args.config}: {described}", flush=True)

    from diffusers.models.transformers import transformer_minimax_h3 as h3
    from diffusers.utils.export_utils import encode_video

    dense_dispatch = h3.dispatch_attention_fn

    for job in jobs:
        # Rebuilt per prompt: the dispatch carries a call counter that the step-indexed policy
        # (`dense_first_steps`) reads, and the cache carries the previous sample's residual.
        # Either one leaking across samples would make the second video depend on the first.
        if use_sol:
            dispatch = h3_fp8.make_sol_attn_dispatch(num_steps=args.steps, **SOL)
            dispatch.reset()
            h3.dispatch_attention_fn = dispatch
        else:
            h3.dispatch_attention_fn = dense_dispatch

        cache_line.remove_cache(pipe.transformer)
        counter = None
        if use_cache:
            cache_line.apply_cache(pipe.transformer, threshold=CACHE_THRESHOLD)
            cache_line.reset_cache(pipe.transformer)
            counter = cache_line.SkipCounter(pipe.transformer)

        print(f"\n[gen] {job['name']}: {job['prompt'][:80]}...", flush=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        state = pipe(
            prompt=job["prompt"], height=args.height, width=args.width,
            num_frames=args.num_frames, num_inference_steps=args.steps,
            generator=torch.Generator().manual_seed(args.seed),
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        skipped = str(counter) if counter else "cache off"
        if counter:
            counter.restore()

        path = os.path.join(args.output_dir, f"{job['name']}{args.suffix}.mp4")
        encode_video(state.get("videos")[0], fps=24, output_path=path,
                     audio=state.get("audio")[0], audio_sample_rate=state.get("sampling_rate"))
        print(f"[gen] {job['name'] + args.suffix:22s} -> {path}\n"
              f"      {skipped}, wall {wall:.0f}s (wall is not a speed measurement)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
