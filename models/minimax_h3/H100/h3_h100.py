#!/usr/bin/env python3
"""MiniMax-H3 T2VA on N x 80 GB GPUs (H100): Ulysses over the packed sequence + per-layer weight sharding.

    torchrun --nproc_per_node 4 h3_h100.py --model-path /models/MiniMax-H3-diffusers --attention sol

Same request, kernels and Sol-Attn policy as the GB200 single-GPU runtime; what differs is placement:
the pipeline is loaded on the host, the small components go to every GPU, the DiT blocks and the
Qwen3-VL decoder layers are sharded 1/N per rank and rebuilt per layer by an all-gather that hides
behind compute (`shard.LayerShards`), and the block stack runs Ulysses-sharded (`forward_mp`).
On one GPU the same layers stay in pinned host memory and are copied in per layer instead (offload mode);
below four GPUs the text encoder is offloaded that way too, which is what lets two GPUs hold the rest.
`build_engine` is shared with the HTTP server (`serve_h100.py`).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
GB200 = os.path.join(HERE, "..", "GB200")
sys.path[:0] = [HERE, GB200]


def log(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[h3-h100] {msg}", flush=True)


def gib(n):
    return round(n / 2**30, 1)


def init_dist() -> SimpleNamespace:
    """torchrun env -> device, default NCCL group, and the weight-gather group (None on one GPU)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    # weight gathers on their own communicator; one GPU: no group = per-layer offload from pinned host memory
    wgroup = dist.new_group(backend="nccl") if world > 1 else None
    return SimpleNamespace(world=world, rank=rank, device=device, wgroup=wgroup)


def build_engine(model_path: str, attention: str) -> SimpleNamespace:
    """Load, place and arm the pipeline on this rank (torchrun env); every rank must call it."""
    d = init_dist()
    world, rank, device, wgroup = d.world, d.rank, d.device, d.wgroup

    import shard
    import vae_shard
    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(model_path, components_manager=ComponentsManager())
    pipe.load_components(dtype=torch.bfloat16)          # host memory
    tr, te = pipe.transformer, pipe.text_encoder
    for name, comp in pipe.components.items():
        if isinstance(comp, torch.nn.Module) and comp is not tr and comp is not te:
            comp.to(device)
    log(f"loaded on host in {time.perf_counter() - t0:.1f}s; small components on {device}")

    t1 = time.perf_counter()
    dit, sparse = place_dit(tr, attention, device, wgroup)
    te_name, te_layers = shard.largest_layer_list(te)
    shard.move_except(te, te_layers, device)
    tes = shard.LayerShards(te_layers, wgroup if world >= 4 else None, device)
    tes.install_pre_hooks()
    vae_shard.install(pipe.vae, batched=True, compile_mode="default")
    install_fast_postprocess(pipe)
    torch.cuda.synchronize()

    def placed(s):
        res = f", {len(s.resident)} whole" if getattr(s, "resident", None) else ""
        return f"{'offloaded' if s.offload else 'sharded'}{res}, {gib(s.resident_bytes)} GiB/rank"
    log(f"placed in {time.perf_counter() - t1:.1f}s: DiT {len(dit.layers)} blocks ({placed(dit)}), "
        f"text encoder {te_name} {len(tes.layers)} layers ({placed(tes)}); "
        f"allocated {gib(torch.cuda.memory_allocated())} GiB")
    if os.environ.get("H3_CAPTURE_DIR"):
        install_capture(os.environ["H3_CAPTURE_DIR"], rank)
    if os.environ.get("H3_PHASE_TIMING") == "1":
        install_phase_timing(rank)
        install_post_timing(pipe)
    return SimpleNamespace(pipe=pipe, sparse=sparse, world=world, rank=rank, device=device)


def place_dit(tr, attention: str, device, wgroup):
    """Attention core, AdaLN precompute (armed), weight shards and the Ulysses block stack for the DiT.

    Shared by `build_engine` and the DiT-only replay tool, so both run exactly the same block stack.
    Returns (LayerShards of the blocks, Sol-Attn object or None)."""
    import adaln
    import balance
    import forward_mp
    import shard
    import sol_attn_h3

    # Hopper: dense attention (dense arm, Sol's dense steps / layers and its sink rows) on FlashAttention-4's sm90
    # kernel instead of SDPA, which lands on FA2 there; same result to bf16 rounding. H3_DENSE_ATTN=sdpa keeps SDPA.
    #   fa4t  default: the all-to-all's strided q/k/v views straight in, tile 192x128, mma_pv_is_rs=False
    #   fa4s  the same, but letting FA4 pick the tile
    #   fa4   what shipped before: three `.contiguous()` copies of those views, default tile
    # The copies are not worth paying for at the per-rank head counts we run (H = 56/world), and the
    # 192x128 tile is the fastest config that is bit-identical to FA4's default (kbench2, job 21779620,
    # 1 GPU, S=38247, D=128, bf16, ms): H=28 31.23 -> 29.70, H=14 16.15 -> 15.48, H=7 7.82 -> 7.66.
    # Measured on the real stack (replay, job 21790107, median step): g8 787.1 -> 773.5 ms (-1.7%),
    # g4 1498.9 -> 1449.3 ms (-3.3%). The Kineto profile says why: 13.7 ms/step of `direct_copy` at g8.
    mode = os.environ.get("H3_DENSE_ATTN", "fa4t")
    if torch.cuda.get_device_capability(device)[0] == 9 and mode.startswith("fa4"):
        from flash_attn.cute.interface import _flash_attn_fwd

        kw = {"tile_mn": (192, 128), "mma_pv_is_rs": False, "intra_wg_overlap": True} if mode == "fa4t" else {}
        contig = (lambda t: t.unsqueeze(0).contiguous()) if mode == "fa4" else (lambda t: t.unsqueeze(0))

        def fa4_dense(q, k, v):                         # (rows, heads, dim) in and out
            return _flash_attn_fwd(contig(q), contig(k), contig(v), **kw)[0].squeeze(0)
        sol_attn_h3._dense = fa4_dense
        log(f"dense attention: FA4 sm90 ({mode})")

    # Sol-Attn and AdaLN precompute are installed exactly as on GB200; the block stack is ours
    sparse = None
    if attention == "sol":
        sparse = sol_attn_h3.install(tr, tau=1.0, thresh_type="diag", dense_steps=10, dense_layers=2,
                                     sink_mode="prefix")
        attn_core = sparse
    else:
        attn_core = sol_attn_h3._dense
    adaln.enable_adaln_precompute(tr)

    shard.move_except(tr, tr.transformer_blocks, device)
    # H3_DIT_RESIDENT=K keeps K of the 50 blocks whole on the GPU (no per-step weight all-gather for them).
    # Costs (1 - 1/world) * 0.77 GiB per block per rank, and only pays where a layer's compute is too
    # short to hide its successor's gather — which is only at 8 GPUs. Replay sweep (job 21794717):
    # g8 K20 -2.1%, K30 -2.8%, K40 -3.7% of the step; g4 K20 -0.7% and g2 K40 -0.6%, i.e. noise.
    # Board harness, 20/20 valid, same node (job 21806489): g8 K20 40.66 -> 39.67, K30 -> 39.30 s, peak
    # 50.2 / 56.7 GB. K40 does not fit: it OOMs at 79.0 GiB during warmup (job 21818143). So 30 at g8.
    world = dist.get_world_size(wgroup) if wgroup is not None else 1
    dit = shard.LayerShards(tr.transformer_blocks, wgroup, device,
                            exclude=lambda n: n.startswith("adaln_proj."),   # freed by the AdaLN precompute
                            resident=int(os.environ.get("H3_DIT_RESIDENT", 30 if world == 8 else 0)))
    # H3_BALANCE=1 sizes each rank's row shard by its measured speed instead of splitting the packed
    # sequence evenly: the eight H100s of a node sustain clocks 6-7% apart and the step costs the
    # slowest of them (job 21819864). Mathematically the same model — a row's output depends only on
    # that row and the weights, and attention is per head, so only the GEMMs' M changes.
    # H3_BALANCE=1 sizes each rank's row shard by the speed it is measured to run at, instead of
    # splitting the packed sequence evenly: the H100s of a node compute a step up to 6% apart and the
    # step costs the slowest of them. Mathematically the same model — a row's output depends only on
    # that row and the weights, and the head split is untouched, so only the GEMMs' M changes.
    # H3_BALANCE_PROBE=1 measures and reports without changing the split.
    bal = balance.make(world, dist.get_rank() if world > 1 else 0)
    if bal is not None:
        log(f"row balancing {'on' if not bal.measure_only else 'probe only'} "
            f"(beta {balance.BETA}, alpha {balance.ALPHA}, at {balance.UPDATE_AT})")
    forward_mp.install(tr, attn_core, group=None, block_pre=dit.before, balancer=bal)
    return dit, sparse


def install_fast_postprocess(pipe) -> None:
    """Make the decoded frames uint8 on the GPU instead of float32-in-numpy-then-PIL on the host.

    `post_s` at g4 is 3.03 s of which the VAE decode is only 1.05 s and the audio VAE 0.06 s (job
    21807778); the remaining ~1.9 s is this output path. Shipped, `MiniMaxH3VideoDecodeStep` hands
    `VideoProcessor.postprocess_video(..., output_type="pil")` a float32 (1, 3, 124, 768, 1344) tensor
    and that does, on one host thread: `.cpu().permute(0,2,3,1).float().numpy()` (a 1.53 GB pageable
    device->host copy), `(x * 255).round().astype("uint8")` over 384 M floats, and 124
    `PIL.Image.fromarray` calls — after which `serve._write_mp4` immediately turns the 124 PIL images
    back into one uint8 array (`np.stack([np.asarray(f) ...])`, charged to `encode_s`).

    So do the scale/round/cast on the GPU and copy 384 MB of uint8 across instead of 1.53 GB of float32,
    in the exact layout `_write_mp4` wants. Bit-identical: the input tensor is the same, `torch.round`
    and `np.round` are both round-half-to-even, the product is float32 either way, and the values are in
    [0, 255] before the cast so the truncation to uint8 is exact (`clamp` is only a guard).
    `do_normalize=False` in this block's processor config, so the denormalize step it replaces is a no-op.

    The return value keeps the shape of the shipped one — a list with one entry per batch item — but the
    entry is a `(T, H, W, 3)` uint8 ndarray rather than a list of PIL images. `_write_mp4` accepts both
    (it calls `np.asarray` and skips its own conversion when the dtype is already uint8).
    `H3_FAST_POST=0` restores the shipped path.
    """
    if os.environ.get("H3_FAST_POST", "1") != "1":
        return
    vp = getattr(pipe, "video_processor", None)
    if vp is None or getattr(vp.config, "do_normalize", False):
        log(f"fast postprocess NOT installed (video_processor={vp!r})")
        return
    original = vp.postprocess_video

    def postprocess_video(video, output_type: str = "np", **kw):
        if output_type != "pil" or kw or not video.is_cuda:
            return original(video, output_type=output_type, **kw)
        u8 = (video.float() * 255).round().clamp_(0, 255).to(torch.uint8)      # (B, C, T, H, W)
        return [f.permute(1, 2, 3, 0).contiguous().cpu().numpy() for f in u8]  # (T, H, W, C) per item

    vp.postprocess_video = postprocess_video
    log("fast postprocess installed (uint8 frames built on the GPU)")


CAPTURE_STEPS = (0, 24, 48)


def install_capture(out_dir: str, rank: int, steps=CAPTURE_STEPS) -> None:
    """Rank 0 saves the first request's denoiser inputs (+ the velocities it produced at `steps`) to
    `out_dir/capture.pt`, for the DiT-only replay tool. Off unless H3_CAPTURE_DIR is set."""
    from diffusers.modular_pipelines.minimax_h3 import denoise as h3_denoise

    inner = h3_denoise.MiniMaxH3LoopDenoiser.__call__
    state = {"done": rank != 0, "cap": None}

    def cpu(t):
        return t.detach().to("cpu", copy=True) if torch.is_tensor(t) else t

    @torch.no_grad()
    def call(self, components, bs, i, t):
        if state["done"]:
            return inner(self, components, bs, i, t)
        if i == 0:
            state["cap"] = {
                "prompt_embeds": cpu(bs.prompt_embeds), "token_tags": cpu(bs.token_tags),
                "position_ids": cpu(bs.position_ids), "video_indices": cpu(bs.video_indices),
                "audio_indices": cpu(bs.audio_indices), "text_indices": cpu(bs.text_indices),
                "row_timestep_plan": [(cpu(a), cpu(b)) for a, b in bs.row_timestep_plan],
                "steps": {}}
        if i in steps:
            state["cap"]["steps"][i] = {"latents": cpu(bs.latents), "audio_latents": cpu(bs.audio_latents)}
        out = inner(self, components, bs, i, t)
        if i in steps:
            state["cap"]["steps"][i].update(noise_pred=cpu(bs.noise_pred), audio_noise_pred=cpu(bs.audio_noise_pred))
        if i == len(bs.row_timestep_plan) - 1:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(state["cap"], os.path.join(out_dir, "capture.pt"))
            log(f"captured denoiser inputs (steps {sorted(state['cap']['steps'])}) to {out_dir}/capture.pt")
            state["done"], state["cap"] = True, None
        return out

    h3_denoise.MiniMaxH3LoopDenoiser.__call__ = call


def install_phase_timing(rank: int) -> None:
    """Log `[h3-phase] {...}` per request on rank 0: pre-denoise (text encoder, packing), denoise loop,
    per-step median, post (VAE decode). Synchronizes at the loop boundaries only. H3_PHASE_TIMING=1."""
    from diffusers.modular_pipelines.minimax_h3 import denoise as h3_denoise

    wrapper = h3_denoise.MiniMaxH3DenoiseLoopWrapper
    inner_loop, inner_step = wrapper.__call__, h3_denoise.MiniMaxH3LoopDenoiser.__call__
    marks = {}

    def loop(self, components, state):
        torch.cuda.synchronize()
        marks["loop0"] = time.perf_counter()
        marks["events"] = []
        out = inner_loop(self, components, state)
        torch.cuda.synchronize()
        marks["loop1"] = time.perf_counter()
        return out

    def step(self, components, bs, i, t):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        marks.setdefault("events", []).append(ev)
        return inner_step(self, components, bs, i, t)

    wrapper.__call__ = loop
    h3_denoise.MiniMaxH3LoopDenoiser.__call__ = step
    install_phase_timing.marks = marks


def install_post_timing(pipe) -> None:
    """Break `post_s` down by wrapping the decode entry points, for whichever of them this pipeline has.

    post_s is 3.0 s of the 77.2 s request at g4 and is the bulk of what we give away to SGLang, so it
    needs to be attributed before it can be attacked. Timed with CUDA events (recorded around each call
    and read once at report time), so the tile loop is not serialised by a synchronize per call.
    Called only from `install_phase_timing`, i.e. only under H3_PHASE_TIMING=1.
    """
    spans: dict = {}

    def wrap(owner, attr, label):
        fn = getattr(owner, attr, None)
        if fn is None:
            return
        def timed(*args, **kw):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            out = fn(*args, **kw)
            b.record()
            spans.setdefault(label, []).append((a, b))
            return out
        setattr(owner, attr, timed)

    vae, avae = getattr(pipe, "vae", None), getattr(pipe, "audio_vae", None)
    if vae is not None:
        wrap(vae, "decode", "vae_decode")
        wrap(vae, "_decode_clip", "vae_clip")       # after vae_shard.install, this is the sharded one
    if avae is not None:
        wrap(avae, "decode", "audio_decode")
    install_post_timing.spans = spans


def phase_report(t_start: float, t_end: float) -> dict | None:
    marks = getattr(install_phase_timing, "marks", None)
    if not marks or "loop0" not in marks:
        return None
    evs = marks["events"]
    steps = sorted(evs[i].elapsed_time(evs[i + 1]) for i in range(len(evs) - 1))
    rep = {"pre_s": round(marks["loop0"] - t_start, 3), "denoise_s": round(marks["loop1"] - marks["loop0"], 3),
           "step_med_ms": round(steps[len(steps) // 2], 1) if steps else None,
           "step_first_ms": round(evs[0].elapsed_time(evs[1]), 1) if len(evs) > 1 else None,
           "post_s": round(t_end - marks["loop1"], 3), "total_s": round(t_end - t_start, 3)}
    spans = getattr(install_post_timing, "spans", None)
    if spans:
        rep["post"] = {k: [len(v), round(sum(a.elapsed_time(b) for a, b in v), 1)] for k, v in spans.items()}
        spans.clear()
    return rep


def run_request(engine, prompt, height, width, num_frames, steps, seed):
    """One request on every rank in lockstep; returns (pipeline state, seconds)."""
    g = torch.Generator().manual_seed(seed)
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    state = engine.pipe(prompt=prompt, height=height, width=width, num_frames=num_frames,
                        num_inference_steps=steps, generator=g)
    torch.cuda.synchronize()
    dist.barrier()
    t_end = time.perf_counter()
    rep = phase_report(t, t_end)
    if rep is not None:
        log_phase(rep)
    return state, t_end - t


def log_phase(rep: dict) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[h3-phase] {json.dumps(rep)}", flush=True)


def write_state_mp4(state, path, preset="medium", crf=18):
    import serve                                         # ../GB200/serve.py
    videos = getattr(state, "videos", None)
    if videos is None:
        videos = getattr(state, "frames")
    serve._write_mp4(videos[0], getattr(state, "audio", None), int(getattr(state, "sampling_rate", 32000) or 32000),
                     path, preset, crf)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--attention", choices=("sol", "dense"), default="sol")
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--num-frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--measure", type=int, default=2)
    p.add_argument("--output-dir", default="outputs")
    a = p.parse_args()

    prompt = json.load(open(a.prompt_file))["prompt"]
    engine = build_engine(a.model_path, a.attention)

    def request():
        return run_request(engine, prompt, a.height, a.width, a.num_frames, a.steps, a.seed)

    for i in range(a.warmup):
        _, s = request()
        log(f"warmup {i}: {s:.2f}s peak {gib(torch.cuda.max_memory_allocated())} GiB")
    torch.cuda.reset_peak_memory_stats()
    times, state = [], None
    for i in range(a.measure):
        state, s = request()
        times.append(s)
        log(f"request {i}: {s:.3f}s")
    res = {"world": engine.world, "attention": a.attention, "request_s": [round(x, 3) for x in times],
           "request_median": round(sorted(times)[len(times) // 2], 3),
           "peak_gib": gib(torch.cuda.max_memory_allocated()),
           "sparse_calls": getattr(engine.sparse, "last_request_sparse_calls", None)}
    log("RESULT " + json.dumps(res))
    if engine.rank == 0:
        os.makedirs(a.output_dir, exist_ok=True)
        write_state_mp4(state, os.path.join(a.output_dir, f"h3_{a.attention}_n{engine.world}.mp4"))
        json.dump(res, open(os.path.join(a.output_dir, f"result_{a.attention}_n{engine.world}.json"), "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
