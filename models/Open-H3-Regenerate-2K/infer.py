"""Open-H3-Regenerate-2K inference on one GPU.

    # prompt -> HyperFlow 768P draft -> 2560x1440 refine
    python infer.py --weights ./weights --hyperflow --prompt "A red fox trots through fresh snow ..." --output fox.mp4

    # existing normalized H3 latent ([24, 37, 48, 84] at 768P or [24, 37, 36, 64] at 576P) -> 2560x1440 refine
    python infer.py --weights ./weights --input-latents native.pt --prompt "..." --output out.mp4

    # draft video -> H3 VAE encode (1344x768 by default, --input-size 576p for 1024x576) -> 2560x1440 refine
    python infer.py --weights ./weights --input-video draft.mp4 --prompt "..." --output out.mp4

    # unaccelerated reference forward
    python infer.py ... --fwd standard

    # two refine steps from a lower start noise (0.7) instead of three
    python infer.py ... --steps 2

    # another output size (768p / 1080p / 2k / 4k), e.g. 4K from a refined 768P latent, re-noised to 0.8
    python infer.py ... --input-latents refined768.pt --resolution 4k --start-sigma 0.8

    # released H3 VAE decoder instead of the fast one (the default with --fwd sol_engine)
    python infer.py ... --vae standard

--warmup / --runs repeat the request for timing; per-stage seconds go to <output>.json.
"""
import argparse
import json
from pathlib import Path

import torch

from open_h3.pipeline import INPUT_SIZES, NATIVE_SHAPES, RESOLUTIONS, SCHEDULES, Pipeline, encode_mp4, encode_prompt, encode_video

HYPERFLOW_FILE = "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="directory filled by download.py")
    parser.add_argument("--base-model", type=Path, help="MiniMax-H3 Diffusers folder (default: <weights>/base_model)")
    parser.add_argument("--fwd", choices=["sol_engine", "standard"], default="sol_engine")
    parser.add_argument("--steps", type=int, choices=sorted(SCHEDULES), default=3,
                        help="refine steps: 3 = sigmas 0.909375, 0.857, 0.667; 2 = sigmas 0.7, 0.494 (shift 6)")
    parser.add_argument("--start-sigma", type=float,
                        help="start noise of the refine; the steps follow the shift-6 rule (default: the --steps grid)")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="2k", help="output size (default 2560x1440)")
    parser.add_argument("--vae", choices=["standard", "fast"],
                        help="video decoder: released H3 VAE or the fast H3 VAE decoder (default: fast with sol_engine)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--input-latents", type=Path,
                        help="torch.save dict with a normalized 'video_latents' [24,37,48,84] (768P) or [24,37,36,64] (576P)")
    parser.add_argument("--input-video", type=Path, help="draft video; the first 121 frames at 24 fps are encoded")
    parser.add_argument("--input-size", choices=sorted(INPUT_SIZES), default="768p",
                        help="size the draft video is encoded at: 768p = 1344x768, 576p = 1024x576")
    parser.add_argument("--vae-seed", type=int, default=42, help="seed of the VAE posterior sample for --input-video")
    parser.add_argument("--hyperflow", action="store_true", help="generate the 768P latent with the HyperFlow adapter")
    parser.add_argument("--hf-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1766580932, help="refine noise seed")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()
    base = args.base_model or args.weights / "base_model"
    if [args.input_latents is not None, args.input_video is not None, args.hyperflow].count(True) != 1:
        parser.error("give exactly one of --input-latents, --input-video or --hyperflow")

    native = None
    if args.input_latents is not None:
        native = torch.load(args.input_latents, map_location="cpu", weights_only=True)["video_latents"]
        native = native[0] if native.ndim == 5 else native
        if tuple(native.shape) not in NATIVE_SHAPES.values() or not torch.isfinite(native).all():
            raise ValueError(f"expected a finite normalized latent of shape {sorted(NATIVE_SHAPES.values())}, got {tuple(native.shape)}")

    torch.cuda.set_device(0)
    if args.input_video is not None:
        native = encode_video(base, args.input_video, args.input_size, args.vae_seed)
    text = encode_prompt(base, args.prompt)
    vae = args.vae or ("fast" if args.fwd == "sol_engine" else "standard")
    pipe = Pipeline(args.weights, base, fwd=args.fwd, steps=args.steps, start_sigma=args.start_sigma,
                    resolution=args.resolution, vae=vae,
                    hyperflow=args.weights / HYPERFLOW_FILE if args.hyperflow else None)
    print(f"LOADED fwd={args.fwd} vae={vae} resolution={args.resolution} sigmas={[round(x, 4) for x in pipe.sigmas]} "
          f"gpu-mem={torch.cuda.memory_allocated() / 2**30:.1f}GB", flush=True)

    for i in range(args.warmup):
        _, _, times = pipe(text, native, args.hf_seed, args.seed)
        print(f"WARMUP {i} " + " ".join(f"{k}={v:.3f}" for k, v in times.items()), flush=True)
    runs = []
    for i in range(args.runs):
        frames, latents, times = pipe(text, native, args.hf_seed, args.seed)
        runs.append(times)
        print(f"RUN {i} " + " ".join(f"{k}={v:.3f}" for k, v in times.items()), flush=True)
    median = {k: sorted(r[k] for r in runs)[len(runs) // 2] for k in runs[0]}
    print("MEDIAN " + " ".join(f"{k}={v:.3f}" for k, v in median.items()), flush=True)

    encode_mp4(frames, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"fwd": args.fwd, "vae": vae, "resolution": args.resolution, "steps": args.steps, "input": str(args.input_video or args.input_latents or "hyperflow"), "sigmas": pipe.sigmas, "prompt": args.prompt, "seed": args.seed,
         "runs": runs, "median": median}, indent=1))
    if args.save_latents:   # "video_latents" = the refined latent, so a 768P result can be refined again at a larger size
        torch.save({**latents, "video_latents": latents["refined"]}, args.output.with_suffix(".latents.pt"))
    print(f"VIDEO_READY {args.output}", flush=True)


if __name__ == "__main__":
    main()
