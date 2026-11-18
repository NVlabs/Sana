"""Open-H3-Regenerate-2K-IC: regenerate a low-resolution video (e.g. a 768P MiniMax-H3 draft) at 1080P / 2K / 4K.

    torchrun --nproc_per_node 8 infer.py --weights ./weights --input draft.mp4 --output draft_2k.mp4 \
        --caption "the prompt the draft was generated from (optional)"
    torchrun --nproc_per_node 8 infer.py --weights ./weights --jobs jobs.jsonl      # {"input", "output", "caption"}

Stages (all clips of a run go through each stage before the next one starts, so every model is loaded once):
  1. rank 0: read each draft (24 fps, 124 frames) onto the LQ canvas, H3 VAE latent, Qwen3-VL prompt embeddings;
  2. every rank: Ref2VA transformer (transformer_ref) + HyperFlow 8-step adapter (merged) + LMS Sharp at --sharp
     (merged; 0 = off) + the IC LoRA, FSDP-sharded
     over the ranks with Ulysses context parallelism (world size 1, 2, 4, 7 or 8);
  3. sampling (CFG 1, same noise on every rank) with each --sampler:
       up3 (default): the draft latent through the author latent upscaler to the target size, noised to sigma 0.835,
                      then the last 3 steps of the HyperFlow 8-step grid (0.835, 0.697, 0.469, 0);
       hf8:           pure noise and the full HyperFlow 8-step grid;
       up4:           as up3 from sigma 0.923 (4 steps);
  4. decode (the outputs spread over the ranks): the fast non-causal 24-layer H3 VAE decoder of
     Open-H3-Regenerate-2K / StreamRender-H3 (vae_fast/; engine "fast" = FlashAttention-4 + Triton + CUDA graphs on
     Blackwell, else "ref" = plain PyTorch); --vae standard, or a clip too large for engine "ref" on the GPU, uses the
     released H3 VAE (FP32 tiled); then H.264 with the draft's audio, cut to the draft's length.
`--weights` is the folder written by download.py.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ref2va_ic import cp as cp_lib
from ref2va_ic import encode, hyperflow, layout as lay, lora, media, sharp as sharp_lib

# output resolution -> (LQ canvas h, w, target h, w, rows kept).  1080P and 3840P are computed at 1088 / 2176 rows
# (whole latent patches) and the centre 1080 / 2160 rows are kept.  '3840p' (alias '4k') is 3840x2160,
# '4k-dci' 4096x2304.  768P targets take 384/512/576-row
# drafts (--lq).
RESOLUTIONS = {'1080p': (768, 1344, 1088, 1920, 1080), '2k': (768, 1344, 1440, 2560, 1440),
               '3840p': (768, 1344, 2176, 3840, 2160), '4k-dci': (768, 1344, 2304, 4096, 2304)}
RESOLUTIONS['4k'] = RESOLUTIONS['3840p']
LQ_768P = {'384': (384, 672), '512': (512, 896), '576': (576, 1024)}


def log(*parts):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(time.strftime('%H:%M:%S'), *parts, flush=True)


def sigmas_for(sampler, meta, shift=12.):
    """(sigma grid incl. terminal 0, init) for --sampler."""
    grid = hyperflow.sigmas(meta, shift)                               # 9 points, 8 steps
    if sampler == 'hf8':
        return grid, 'noise'
    if sampler in ('up4', 'up3'):        # the last 4 / 3 HyperFlow steps from the upscaled draft (0.923 / 0.835)
        return grid[8 - int(sampler[2:]):], 'upscaled'
    raise ValueError(sampler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--caption', default='')
    ap.add_argument('--jobs', type=Path, help='jsonl of {"input", "output", "caption"}')
    ap.add_argument('--resolution', choices=[*RESOLUTIONS, '768p'], default='2k')
    ap.add_argument('--lq', choices=list(LQ_768P), help='draft height for --resolution 768p')
    ap.add_argument('--sampler', default='up3', help='comma-separated, from up3 (default), hf8, up4; with several, '
                    'each output gets the sampler name as a suffix')
    ap.add_argument('--vae', choices=['fast', 'standard'], default='fast', help='decoder (see above)')
    ap.add_argument('--vae-engine', choices=['auto', 'fast', 'ref'], default='auto', help='--vae fast engine')
    ap.add_argument('--sharp', type=float, default=None, help='LMS Sharp strength (default: config.json; 0 = off)')
    ap.add_argument('--seed', type=int, default=0, help='VAE posterior sample and sampling noise (seed mod 2^62)')
    ap.add_argument('--save-latent', action='store_true', help='also write <output>.latent.pt (normalized latent)')
    ap.add_argument('--attention', default='_flash_3_hub', choices=['_flash_3_hub', 'native'])
    args = ap.parse_args()
    jobs = ([json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()] if args.jobs else
            [dict(input=str(args.input), output=str(args.output), caption=args.caption)])
    if args.resolution == '768p':
        assert args.lq, '--resolution 768p needs --lq 384|512|576'
        lq_h, lq_w, hq_h, hq_w, keep = *LQ_768P[args.lq], 768, 1344, 768
    else:
        lq_h, lq_w, hq_h, hq_w, keep = RESOLUTIONS[args.resolution]
    W = args.weights
    base = W / 'base_model'
    with open(W / 'config.json') as f:
        config = json.load(f)

    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    rank = dist.get_rank() if world > 1 else 0
    assert cp_lib.HEADS % world == 0, 'world size must divide the 56 heads (1, 2, 4, 7, 8)'

    # 1. drafts -> latents + prompt embeddings (rank 0), shared with every rank
    encoded = None
    if rank == 0:
        t0 = time.time()
        drafts = []
        for job in jobs:
            pixels, real, fps = media.read_draft(job['input'], lq_h, lq_w)
            drafts.append((pixels, real))
            log('draft', job['input'], f'{real} frames at 24 fps (source {fps:.2f} fps)')
        vae = encode.load_vae(base, device)
        latents = [encode.vae_encode(vae, p, args.seed) for p, _ in drafts]
        del vae
        torch.cuda.empty_cache()
        text = encode.load_text_encoder(base, device)
        prompts = [encode.encode_prompt(text, p, job.get('caption', '')) for (p, _), job in zip(drafts, jobs)]
        del text
        torch.cuda.empty_cache()
        encoded = [dict(latent=z, prompt=pr, frames=real) for z, pr, (_, real) in zip(latents, prompts, drafts)]
        log(f'encoded {len(jobs)} draft(s) in {time.time() - t0:.0f} s')
    if world > 1:
        box = [encoded]
        dist.broadcast_object_list(box, 0)
        encoded = box[0]

    # 2. Ref2VA transformer + HyperFlow + IC LoRA
    from diffusers import MiniMaxH3Transformer3DModel
    t0 = time.time()
    model = MiniMaxH3Transformer3DModel.from_pretrained(str(base), subfolder='transformer_ref', torch_dtype=torch.bfloat16,
                                                        low_cpu_mem_usage=True, local_files_only=True)
    model.requires_grad_(False)
    meta = hyperflow.install(model, W / config['components']['hyperflow'], device)
    strength = config['sampler']['sharp'] if args.sharp is None else args.sharp
    if strength:
        sharp_lib.install(model, W / config['components']['sharp_lora'], W / config['components']['sharp_curve'],
                          strength, device)
    count = lora.install(model, W / config['components']['lora'])
    if args.attention == '_flash_3_hub':
        from diffusers.models.attention_dispatch import (AttentionBackendName, _HUB_KERNELS_REGISTRY,
                                                         _maybe_download_kernel_for_backend)
        kernel = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
        kernel.revision, kernel.version = config['attention']['fa3_hub_revision'], None
        _maybe_download_kernel_for_backend(AttentionBackendName._FLASH_3_HUB)
    model.set_attention_backend(args.attention)
    if world > 1:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        mesh = init_device_mesh('cuda', (1, world), mesh_dim_names=('replicate', 'shard'))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for block in model.transformer_blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32,
                                                                     cast_forward_inputs=False))
        model.eval()
        cp_lib.enable_cp(model, world, args.attention)
    else:
        model.to(device).eval()
    plans = [(name, *sigmas_for(name, meta)) for name in args.sampler.split(',')]
    upscaler = None
    if any(init == 'upscaled' for *_, init in plans):
        from ref2va_ic import upscale
        upscaler = upscale.load(W / config['components']['latent_upsampler'])
    audio = load_file(str(W / config['components']['silence_audio']))['audio_latents']
    log(f'model ready in {time.time() - t0:.0f} s: HyperFlow ({meta["merged_modules"]} linears), IC LoRA ({count} '
        f'linears), Sharp {strength}, cp {world}, samplers', {name: [round(s, 4) for s in sig] for name, sig, _ in plans})

    # 3. sampling
    target_shape = [24, encoded[0]['latent'].shape[1], hq_h // 16, hq_w // 16]
    rows = target_shape[1] * target_shape[2] // 2 * target_shape[3] // 2
    results = []
    for job, enc in zip(jobs, encoded):
        ref, aud, prompt, layout, n = lay.pack(enc['latent'], target_shape, enc['prompt'], audio, device)
        up = None
        if upscaler is not None:
            up = torch.empty((1, rows, 96), device=device, dtype=torch.float32)
            if rank == 0:   # computed once, so every rank starts from the same latent
                up.copy_(lay.patch_rows(upscale.upscale(upscaler, enc['latent'].to(device), target_shape, 'cuda')))
            if world > 1:
                dist.broadcast(up, 0)
        for name, sigmas, init in plans:
            rng = torch.Generator(device=device).manual_seed(args.seed % (1 << 62))   # same noise for every sampler
            anchor = .999 * ref.float() + .001 * torch.randn(ref.shape, generator=rng, device=device)
            x = torch.randn((1, rows, 96), generator=rng, device=device, dtype=torch.float32)
            if init == 'upscaled':
                x = (1. - sigmas[0]) * up + sigmas[0] * x
            t0 = time.time()
            with torch.inference_mode():
                for sigma, nxt in zip(sigmas[:-1], sigmas[1:]):
                    times, _ = lay.row_timesteps(layout, sigma)
                    model.time_embedder.endpoint_timesteps = hyperflow.endpoints(times, lay.canonical_sigma(sigma), nxt)
                    with torch.autocast('cuda', enabled=False):
                        v = cp_lib.forward(model, torch.cat([anchor, x], 1), aud, prompt, layout, sigma, world)
                    x = x + (sigma - nxt) * v[:, n:].float()
                    del v
            assert torch.isfinite(x).all()
            output = Path(job['output'])
            if len(plans) > 1:
                output = output.with_name(f'{output.stem}_{name}{output.suffix}')
            results.append((job, enc, name, output, lay.unpatch_rows(x.cpu(), target_shape).to(torch.bfloat16)))
            log(job['input'], f'{name}: {len(sigmas) - 1} steps in {time.time() - t0:.1f} s')
            del anchor, x
        del ref, aud, prompt, layout, up
        torch.cuda.empty_cache()

    # 4. decode: every rank holds every result (the CP forward returns all rows), so rank r decodes results r, r + world, ...
    if world > 1:
        dist.barrier()
    del model
    torch.cuda.empty_cache()
    mine = results[rank::world]
    if mine:
        fast = vae = None
        if args.vae == 'fast':
            from ref2va_ic.fast_vae import FastVAE
            fast = FastVAE(W / config['components']['vae_fast'], base, engine=args.vae_engine)
        for job, enc, name, output, z in mine:
            t0 = time.time()
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.save_latent:
                torch.save(dict(video_latents=z, sampler=name, seed=args.seed), output.with_suffix('.latent.pt'))
            if fast is not None and fast.fits(z):
                decoder = f'fast VAE ({fast.engine})'
                frames = fast.decode_frames(z).cpu().numpy()[:enc['frames']]
            else:
                decoder = 'H3 VAE'
                if vae is None:
                    vae = encode.load_vae(base, device)
                frames = encode.vae_decode(vae, z)[:enc['frames']]
            top = (frames.shape[1] - keep) // 2
            frames = frames[:, top:top + keep]
            media.write_video(frames, output, audio_from=job['input'])
            print(time.strftime('%H:%M:%S'), f'rank {rank} wrote', output, f'{frames.shape[2]}x{frames.shape[1]}, '
                  f'{len(frames)} frames, {decoder} decode {time.time() - t0:.0f} s', flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
