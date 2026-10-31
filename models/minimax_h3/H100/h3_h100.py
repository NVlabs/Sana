#!/usr/bin/env python3
"""MiniMax-H3 T2VA on N x 80 GB GPUs (H100): Ulysses over the packed sequence + per-layer weight sharding.

    torchrun --nproc_per_node 4 h3_h100.py --model-path /models/MiniMax-H3-diffusers --attention sol

Same request, kernels and Sol-Attn policy as the GB200 single-GPU runtime; what differs is placement:
the pipeline is loaded on the host, the small components go to every GPU, the DiT blocks and the
Qwen3-VL decoder layers are sharded 1/N per rank and rebuilt per layer by an all-gather that hides
behind compute (`shard.LayerShards`), and the block stack runs Ulysses-sharded (`forward_mp`).
On one GPU the same layers stay in pinned host memory and are copied in per layer instead (offload mode).
"""
import argparse
import json
import os
import sys
import time

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

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    # weight gathers on their own communicator; one GPU: no group = per-layer offload from pinned host memory
    wgroup = dist.new_group(backend="nccl") if world > 1 else None
    prompt = json.load(open(a.prompt_file))["prompt"]

    import adaln
    import forward_mp
    import shard
    import sol_attn_h3
    import vae_shard
    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(a.model_path, components_manager=ComponentsManager())
    pipe.load_components(dtype=torch.bfloat16)          # host memory
    tr, te = pipe.transformer, pipe.text_encoder
    for name, comp in pipe.components.items():
        if isinstance(comp, torch.nn.Module) and comp is not tr and comp is not te:
            comp.to(device)
    log(f"loaded on host in {time.perf_counter() - t0:.1f}s; small components on {device}")

    # Hopper: dense attention (dense arm, Sol's dense steps / layers and its sink rows) on FlashAttention-4's sm90
    # kernel instead of SDPA, which lands on FA2 there; same result to bf16 rounding. H3_DENSE_ATTN=sdpa keeps SDPA.
    if torch.cuda.get_device_capability(device)[0] == 9 and os.environ.get("H3_DENSE_ATTN", "fa4") == "fa4":
        from flash_attn.cute.interface import _flash_attn_fwd

        def fa4_dense(q, k, v):                         # (rows, heads, dim) in and out
            return _flash_attn_fwd(q.unsqueeze(0).contiguous(), k.unsqueeze(0).contiguous(),
                                   v.unsqueeze(0).contiguous())[0].squeeze(0)
        sol_attn_h3._dense = fa4_dense
        log("dense attention: FA4 sm90")

    # Sol-Attn and AdaLN precompute are installed exactly as on GB200; the block stack is ours
    sparse = None
    if a.attention == "sol":
        sparse = sol_attn_h3.install(tr, tau=1.0, thresh_type="diag", dense_steps=10, dense_layers=2,
                                     sink_mode="prefix")
        attn_core = sparse
    else:
        attn_core = sol_attn_h3._dense
    adaln.enable_adaln_precompute(tr)

    t1 = time.perf_counter()
    shard.move_except(tr, tr.transformer_blocks, device)
    dit = shard.LayerShards(tr.transformer_blocks, wgroup, device,
                            exclude=lambda n: n.startswith("adaln_proj."))   # freed by the AdaLN precompute
    te_name, te_layers = shard.largest_layer_list(te)
    shard.move_except(te, te_layers, device)
    tes = shard.LayerShards(te_layers, wgroup, device)
    tes.install_pre_hooks()
    forward_mp.install(tr, attn_core, group=None, block_pre=dit.before)
    undo_vae = vae_shard.install(pipe.vae, batched=True, compile_mode="default")
    torch.cuda.synchronize()
    log(f"{'offloaded' if dit.offload else 'sharded'} in {time.perf_counter() - t1:.1f}s: "
        f"DiT {len(dit.layers)} blocks ({gib(dit.resident_bytes)} GiB/rank), "
        f"text encoder {te_name} {len(tes.layers)} layers ({gib(tes.resident_bytes)} GiB/rank); "
        f"allocated {gib(torch.cuda.memory_allocated())} GiB")

    def request():
        g = torch.Generator().manual_seed(a.seed)
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        state = pipe(prompt=prompt, height=a.height, width=a.width, num_frames=a.num_frames,
                     num_inference_steps=a.steps, generator=g)
        torch.cuda.synchronize()
        dist.barrier()
        return state, time.perf_counter() - t

    for i in range(a.warmup):
        _, s = request()
        log(f"warmup {i}: {s:.2f}s peak {gib(torch.cuda.max_memory_allocated())} GiB")
    torch.cuda.reset_peak_memory_stats()
    times, state = [], None
    for i in range(a.measure):
        state, s = request()
        times.append(s)
        log(f"request {i}: {s:.3f}s")
    res = {"world": world, "attention": a.attention, "request_s": [round(x, 3) for x in times],
           "request_median": round(sorted(times)[len(times) // 2], 3),
           "peak_gib": gib(torch.cuda.max_memory_allocated()),
           "sparse_calls": getattr(sparse, "last_request_sparse_calls", None)}
    log("RESULT " + json.dumps(res))
    if rank == 0:
        os.makedirs(a.output_dir, exist_ok=True)
        import serve
        videos = getattr(state, "videos", None)
        if videos is None:
            videos = getattr(state, "frames")
        serve._write_mp4(videos[0], getattr(state, "audio", None), int(getattr(state, "sampling_rate", 32000) or 32000),
                         os.path.join(a.output_dir, f"h3_{a.attention}_n{world}.mp4"))
        json.dump(res, open(os.path.join(a.output_dir, f"result_{a.attention}_n{world}.json"), "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
