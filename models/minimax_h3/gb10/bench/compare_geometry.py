"""How different is a retiled decode, in the space the viewer actually sees?

The sweep reports `max|d|` on the decoder's raw output, and on a real latent that reaches 5.23
for an untiled height. That number is close to useless on its own: it is one worst pixel out of
174 million, in ImageNet-normalized units, before the clamp to [0, 1]. Five rows of frames put
side by side look identical.

So this measures the same comparison in uint8 RGB after the pipeline's own postprocessing —
PSNR, mean error, and the fraction of pixels a viewer could distinguish — and writes a
difference map, because *where* the error sits decides what it is. Error along the tile seams
is blending; error spread over the frame is the decoder behaving differently at a tile size it
was not tuned for.

One framing note: the 3x4 default is not ground truth. It is what the shipped recipe produces,
so a difference is a deviation from the official output, not automatically a degradation —
the untiled height does no blending at all and could as easily be the better picture.
"""

from __future__ import annotations

import paths

paths.setup()

import argparse

import torch

import vae_shard
from bench_vae import decode_as_pipeline, grid_of, load_vae, set_geometry, to_uint8


def compare(reference: torch.Tensor, other: torch.Tensor) -> str:
    """Both `(T, H, W, 3)` uint8. Reported on the 0-255 scale the viewer sees."""
    a = reference.float()
    b = other.float()
    delta = (a - b).abs()
    mse = (delta**2).mean().item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0**2 / mse)).item()
    total = delta.numel()
    return (f"PSNR {psnr:6.2f} dB   mean|d| {delta.mean():6.3f}   max|d| {delta.max():5.0f}   "
            f">2: {(delta > 2).sum() / total:6.2%}   >8: {(delta > 8).sum() / total:6.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--tile-heights", type=int, nargs="+", default=[272, 288, 480])
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--capture-steps", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    latent_path = (paths.CAPTURE_DIR / f"{args.width}x{args.height}"
                   / f"vae_latent_s{args.capture_steps}.pt")
    if not latent_path.exists():
        raise SystemExit(f"no {args.capture_steps}-step latent at {latent_path}")
    z = torch.load(latent_path).to(f"cuda:{args.device}")
    vae = load_vae(args.device)

    vae_shard.unpatch_batched_tiles()
    set_geometry(vae, 256, args.tile_width, args.overlap)
    reference = to_uint8(decode_as_pipeline(vae, z)[0])
    print(f"reference: {grid_of(vae, args.height, args.width)[1]}\n", flush=True)

    maps = {}
    for tile_h in args.tile_heights:
        vae_shard.unpatch_batched_tiles()
        set_geometry(vae, tile_h, args.tile_width, args.overlap)
        vae_shard.patch_batched_tiles(vae, batch=args.batch)
        frames = to_uint8(decode_as_pipeline(vae, z)[0])
        print(f"h={tile_h:<4d} {compare(reference, frames)}", flush=True)
        # Averaged over time so a seam, which sits at the same rows in every frame, separates
        # from per-frame noise, which does not.
        maps[tile_h] = (reference.float() - frames.float()).abs().mean(dim=(0, 3))

    # Batching at the default geometry, for scale: the same table row with no blending change.
    vae_shard.unpatch_batched_tiles()
    set_geometry(vae, 256, args.tile_width, args.overlap)
    vae_shard.patch_batched_tiles(vae, batch=args.batch)
    print(f"\nbatch only {compare(reference, to_uint8(decode_as_pipeline(vae, z)[0]))}",
          flush=True)
    vae_shard.unpatch_batched_tiles()

    from PIL import Image

    out_dir = paths.OUTPUT_DIR / "vae_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Scaled to each map's own peak: the question is where the error is, not how big.
    strip = torch.cat([(m / m.max().clamp(min=1e-6) * 255).byte() for m in maps.values()], dim=1)
    Image.fromarray(strip.cpu().numpy()).save(out_dir / "difference.png")
    print(f"\nwrote {out_dir / 'difference.png'} "
          f"(left to right: {', '.join(f'h={h}' for h in maps)}; each scaled to its own peak)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
