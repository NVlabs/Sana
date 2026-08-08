"""Time the video VAE's tiled decode across batch sizes and tile geometries.

The decode is 92.6% of what a request costs outside the denoising loop. Counting what it
actually does says where the room is: 36 transformer layers at dim 2048, run 84 times (12
spatial tiles x 7 temporal clips) on 1796 tokens each. That is 627 TFLOPs, and 29.41 s of it
is 21.3 TFLOPS against a measured fp16 ceiling of 96 — 22% of the part. 87% of those FLOPs are
plain GEMM, so the shortfall is not arithmetic, it is that each GEMM only sees M=1796.

Two levers, close to orthogonal:

* **Batch the tiles.** All twelve are independent and identically shaped, so they can go
  through as one GEMM of M=21552 instead of twelve of M=1796.
* **Choose the tile size.** `_split_tiles` takes the fewest tiles that cover the frame and
  pushes all the slack into the overlaps. At 832x480 the height needs 3 tiles of 256 to cover
  480, leaving 288 pixels of slack that become overlap — 144 a seam, 56% of a tile. Twelve
  tiles decode 786,432 pixels of a 399,360-pixel frame. A height of 272 needs only 2 tiles at
  the minimum 64 overlap, covering 544 instead of 768.

Only the first can be bit-exact. Changing the geometry changes which pixels are blended and
with what weights, so it is checked by eye as well as by norm.

Both are measured against one recorded latent, interleaved in one process, because absolute
timings drift between processes and ratios within one do not.

Stage one records the latent and exits; stage two starts clean with the VAE alone. That split
is not organisational: the denoiser and conditioner hold 56 GB, and on a part that shares one
memory pool with the host, carrying them into a batched decode does not make it fail — it
makes it spill and crawl, which cost one run forty minutes with nothing to show.
"""

from __future__ import annotations

import paths

paths.setup()

import argparse
import json
from pathlib import Path

import torch

import vae_shard


def bench(fn, warmup=1, iters=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return min(samples), out


class _Captured(Exception):
    """Raised to unwind out of the pipeline once the latent is in hand."""


def capture_latent(args, path: Path) -> None:
    """Run the pipeline far enough to record what reaches the VAE, then stop."""
    import gpu_infer

    args.prompt = json.load(open(args.prompt_file))["prompt"]
    pipe, _ = gpu_infer.build_pipeline(args, fuse_qkv=True, quantizer="triton", fuse_adaln=True,
                                    fuse_rope=True, fuse_swiglu=True)
    grabbed = {}
    original = pipe.vae.decode

    def capture(z, *rest, **kw):
        grabbed.setdefault("z", z.detach().clone())
        raise _Captured

    pipe.vae.decode = capture
    try:
        with torch.no_grad():
            pipe(prompt=args.prompt, height=args.height, width=args.width,
                 num_frames=args.num_frames, num_inference_steps=args.steps,
                 generator=torch.Generator().manual_seed(0))
    except _Captured:
        pass
    pipe.vae.decode = original

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grabbed["z"].cpu(), path)
    print(f"recorded {tuple(grabbed['z'].shape)} -> {path}", flush=True)


def load_vae(device: int):
    """The video VAE alone, built by the pipeline's own loader so it is the same model.

    The class lists encoder, decoder and both convs in `_keep_in_fp32_modules`, and those four
    are the entire model, so `load_components(dtype=torch.bfloat16)` — what the pipeline
    calls — returns a VAE that is float32 throughout. That is deliberate and documented: the
    released checkpoint is float32 and the verified recipe is float16 autocast *over* float32
    weights, which `decoders.py` applies at the call site.

    Reproducing that recipe is the whole reason this does not simply cast. fp32 weights run
    at 119.3 s; casting them to bf16 runs at 29.35 s but is a different model; the pipeline's
    own instrumented decode is 29.41 s. Only the autocast reading makes all three agree, and
    it is the one with fp32 master weights.
    """
    from diffusers import ComponentsManager, ModularPipeline

    pipe = ModularPipeline.from_pretrained(paths.h3_snapshot(),
                                           components_manager=ComponentsManager())
    pipe.load_components(names=["vae"], dtype=torch.bfloat16)
    return pipe.vae.to(f"cuda:{device}").eval()


def grid_of(vae, height: int, width: int) -> tuple[int, str]:
    """The tile layout a geometry produces, and how much of it is redundant."""
    _, y_lengths, y_overlaps = vae._split_tiles(
        height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height
    )
    _, x_lengths, x_overlaps = vae._split_tiles(
        width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width
    )
    tiles = len(y_lengths) * len(x_lengths)
    covered = sum(y_lengths) * sum(x_lengths)
    return tiles, (f"{len(y_lengths)}x{len(x_lengths)}={tiles} tiles, "
                   f"overlap h={y_overlaps or [0]} w={x_overlaps or [0]}, "
                   f"{covered / (height * width):.2f}x redundant")


def set_geometry(vae, tile_h: int, tile_w: int, overlap: int) -> None:
    vae.enable_tiling(tile_sample_min_height=tile_h, tile_sample_min_width=tile_w,
                      tile_sample_min_overlap_height=overlap,
                      tile_sample_min_overlap_width=overlap)


def decode_as_pipeline(vae, z: torch.Tensor) -> torch.Tensor:
    """`vae.decode` under the recipe `decoders.py` uses: float16 autocast over float32 weights.

    Not an optimisation and not incidental — it is what makes the standalone timing mean the
    same thing as the 29.4 s in the request breakdown. Without it the same weights decode in
    119.3 s.
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        return vae.decode(z, return_dict=False)[0]


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """`(3, T, H, W)` -> `(T, H, W, 3)` uint8, undoing the VAE's output normalization.

    The decoder emits ImageNet-normalized RGB, not [-1, 1]; `decoders.py` reverses it with the
    ImageNet constants before handing the video to the processor. Treating it as [-1, 1]
    produces a plausible-looking picture with the wrong contrast and colour, which is worse
    than an obviously broken one.
    """
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MINIMAX_H3_PIXEL_MEAN,
        MINIMAX_H3_PIXEL_STD,
    )

    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=video.device).view(-1, 1, 1, 1)
    std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=video.device).view(-1, 1, 1, 1)
    rgb = (video.float() * std + mean).clamp(0, 1)
    return (rgb * 255).round().byte().permute(1, 2, 3, 0).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 12])
    # 256 is what ships. 272 is the smallest height that covers 480 in two tiles at the
    # minimum overlap; 480 stops tiling the height altogether. Counting the transformer's
    # FLOPs over the resulting grids gives 627 / 447 / 477 / 435 TFLOPs — the last is fewest
    # because four large tiles waste less on overlap than twelve small ones, and the attention
    # that grows with tile size is only 13% of the work to begin with. Width stays tiled: one
    # 832-wide tile pushes attention to 47% and the total back up to 523.
    parser.add_argument("--tile-heights", type=int, nargs="+", default=[256, 272, 288, 480],
                        help="256 is the shipped default; 480 leaves the height untiled")
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--recapture", action="store_true")
    # Two steps is enough to get a correctly shaped latent, and the decode's cost is entirely
    # shape-driven, so the timings do not care. The pictures do: a two-step latent decodes to
    # noise, which says nothing about whether a geometry change is visible. Capture at 50 for
    # anything that will be looked at.
    parser.add_argument("--capture-steps", type=int, default=2)
    args = parser.parse_args()
    args.num_frames, args.seed = 124, 0
    args.steps = args.capture_steps
    args.prompt_file = str(paths.PROMPT_FILE)

    latent_path = (paths.CAPTURE_DIR / f"{args.width}x{args.height}"
                   / f"vae_latent_s{args.capture_steps}.pt")
    if args.recapture or not latent_path.exists():
        capture_latent(args, latent_path)

    vae = load_vae(args.device)
    z = torch.load(latent_path).to(f"cuda:{args.device}")
    print(f"latent {tuple(z.shape)}  {z.dtype}; "
          f"{torch.cuda.memory_allocated(args.device) / 2**30:.1f} GiB resident", flush=True)

    # The reference is the shipped geometry decoded one tile at a time, because that is what
    # the 29.4 s in the request breakdown was spent on.
    default_h = args.tile_heights[0]
    vae_shard.unpatch_batched_tiles()
    set_geometry(vae, default_h, args.tile_width, args.overlap)
    print(f"reference geometry: {grid_of(vae, args.height, args.width)[1]}", flush=True)
    base_ms, reference = bench(lambda: decode_as_pipeline(vae, z))

    frames = {"reference": reference[0]}
    print(f"\n{'tile h':>7s} {'batch':>7s} {'ms':>8s} {'speedup':>8s}  pixels")
    print(f"{default_h:>7d} {'seq':>7s} {base_ms:8.0f} {'1.00x':>8s}  reference", flush=True)

    for tile_h in args.tile_heights:
        set_geometry(vae, tile_h, args.tile_width, args.overlap)
        num_tiles, description = grid_of(vae, args.height, args.width)
        print(f"\n  h={tile_h}: {description}", flush=True)

        for batch in ["seq"] + [b for b in args.batches if b <= num_tiles]:
            vae_shard.unpatch_batched_tiles()
            if batch != "seq":
                vae_shard.patch_batched_tiles(vae, batch=batch)
            try:
                ms, out = bench(lambda: decode_as_pipeline(vae, z))
            except torch.OutOfMemoryError:
                print(f"{tile_h:>7d} {str(batch):>7s} {'OOM':>8s}", flush=True)
                torch.cuda.empty_cache()
                continue

            if out.shape == reference.shape:
                delta = (out.float() - reference.float()).abs().max().item()
                verdict = "EXACT" if delta == 0 else f"max|d|={delta:.2e}"
            else:
                verdict = f"shape {tuple(out.shape)}"
            print(f"{tile_h:>7d} {str(batch):>7s} {ms:8.0f} {base_ms / ms:7.2f}x  {verdict}",
                  flush=True)
            frames.setdefault(f"h{tile_h}", out[0])

    vae_shard.unpatch_batched_tiles()

    # Geometry is not bit-exact by construction, so it gets looked at, not only normed.
    from PIL import Image

    out_dir = paths.OUTPUT_DIR / "vae_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = [0.0, 0.33, 0.66, 1.0]
    rows = [torch.cat(list(to_uint8(v)[[int(p * (v.shape[1] - 1)) for p in picks]]), dim=1)
            for v in frames.values()]
    Image.fromarray(torch.cat(rows, dim=0).numpy()).save(out_dir / "geometry.png")
    print(f"\nwrote {out_dir / 'geometry.png'} (rows: {', '.join(frames)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
