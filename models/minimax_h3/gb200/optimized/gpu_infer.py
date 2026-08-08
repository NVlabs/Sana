#!/usr/bin/env python3
"""MiniMax-H3 optimized driver. Env-driven, one technique per switch.

Called by `scripts/run_minimax_h3_gpu.sh`, which tees stdout to `run.log`. Runs a warmup request
and then a measured request over the prompt file, and writes the standard artifacts into
`$H3_OUTPUT_DIR`: `out.mp4`, `benchmark.json`, `run_config.json`.

The acceleration line, in the order the switches compose:

    H3_ULYSSES_DEGREE   context parallelism over the packed sequence. H3 is a single-stream
                        Omni-Transformer with one distilled branch, so there is no CFG branch to
                        split; the only axis is the sequence, sharded by `tensor_split` (38247 rows
                        do not divide by 8) and exchanged head-wise by Ulysses.
    H3_KERNEL           lossless kernel work: the fused QKV kept packed through one all-to-all
                        instead of three, two Triton relayout kernels replacing a 5-D
                        `permute().contiguous()`, and the AdaLN modulation table precomputed once
                        (bit-identical, and it frees 23.9 GB of projection weights).
    H3_SOL_ATTN         Sol-Attn on the packed self-attention, inside the Ulysses exchange — the
                        one point where a rank holds the whole sequence, so a sequence-level
                        operator can be correct under context parallelism.
    H3_CACHE            FirstBlockCache over the 50-block stack.
    H3_SHARD_VAE        the video VAE decode split across the ranks, batched, optionally compiled.

Every switch is independent, so `benchmark.json` from two runs that differ in one variable is a
clean A/B. The candidate manifests in `candidates/minimax_h3_*.toml` pin the combinations that
were measured.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist


_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.join(os.path.dirname(_HERE), "baseline")
for _vendored in (os.path.join(_BASELINE, "vendor_site"),
                  os.path.join(_BASELINE, "diffusers_src", "src")):
    if os.path.isdir(_vendored):
        sys.path.insert(0, _vendored)
sys.path.insert(0, _HERE)

from gpu_infer_cp import (  # noqa: E402
    GpuTimer, _round, build_pipeline, enable_context_parallel, instrument, run_request,
)


def log(message: str) -> None:
    print(f"[h3] {message}", flush=True)


def _s(key, default=None):
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _i(key, default):
    value = os.environ.get(key)
    return int(value) if value not in (None, "") else default


def _f(key, default):
    value = os.environ.get(key)
    return float(value) if value not in (None, "") else default


def install_stack(pipe, config: dict):
    """Arm the requested techniques. Returns (uninstall, sparse_handle).

    Order is load-bearing and matches what was measured:

      fusions before Ulysses  — Ulysses replaces `attn.forward`, and must be the outer patch.
      AdaLN once              — it frees the projection weights, so it is irreversible in one
                                direction and is armed for the whole process rather than per run.
      Sol-Attn before Ulysses — it is handed in as the exchange's `attention_fn`, not installed
                                into the model's attention.
      cache last              — `fusion_install` assigns `block.forward` directly, so installing
                                fusions after the cache would displace the registered cache hook
                                while its registry still believed it was live.
    """
    import adaln
    import cache_line
    import fusion_install
    import sol_attn_h3
    import ulysses_custom
    import vae_shard

    undo, sparse = [], None
    world = dist.get_world_size() if dist.is_initialized() else 1

    if config["kernel"]:
        undo.append(fusion_install.install(pipe.transformer))
        adaln.enable_adaln_precompute(pipe.transformer)

    if config["sol_attn"]:
        sparse = sol_attn_h3.install(
            pipe.transformer, tau=config["sol_tau"], thresh_type=config["sol_thresh_type"],
            dense_steps=config["sol_dense_steps"], dense_layers=config["sol_dense_layers"],
            sink_mode=config["sol_sink_mode"])
        undo.append(lambda: sol_attn_h3.uninstall(pipe.transformer))

    if config["kernel"] and world > 1:
        undo.append(ulysses_custom.install(pipe.transformer, packed=True, attention_fn=sparse))

    if config["cache"] is not None:
        undo.append(lambda: cache_line.disable_cache(pipe.transformer))
        cache_line.enable_cache(pipe.transformer, method="first_block_cache",
                                threshold=config["cache"])

    if config["shard_vae"]:
        undo.append(vae_shard.install(pipe.vae, batched=config["vae_batched"],
                                      compile_mode=config["vae_compile"]))
    return undo, sparse


def main() -> int:
    rank = _i("RANK", 0)
    local_rank = _i("LOCAL_RANK", 0)
    world_size = _i("WORLD_SIZE", 1)
    # Bind the device before NCCL initialises and hand it over explicitly: otherwise NCCL infers it
    # from the global rank, which is only right when the rank->GPU mapping is homogeneous.
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                                device_id=torch.device(device))

    config = {
        "model_path": _s("H3_MODEL_PATH"),
        "prompt_file": _s("H3_PROMPT_FILE"),
        "height": _i("H3_HEIGHT", 768),
        "width": _i("H3_WIDTH", 1344),
        "num_frames": _i("H3_NUM_FRAMES", 124),
        "steps": _i("H3_STEPS", 50),
        "seed": _i("H3_SEED", 0),
        "ulysses_degree": _i("H3_ULYSSES_DEGREE", world_size),
        "kernel": _s("H3_KERNEL", "1") == "1",
        "sol_attn": _s("H3_SOL_ATTN", "0") == "1",
        "sol_tau": _f("H3_SOL_TAU", 1.0),
        "sol_thresh_type": _s("H3_SOL_THRESH_TYPE", "diag"),
        "sol_dense_steps": _i("H3_SOL_DENSE_STEPS", 10),
        "sol_dense_layers": _i("H3_SOL_DENSE_LAYERS", 2),
        "sol_sink_mode": _s("H3_SOL_SINK_MODE", "prefix"),
        "cache": _f("H3_CACHE_THRESHOLD", 0.0) or None,
        "shard_vae": _s("H3_SHARD_VAE", "1") == "1",
        "vae_batched": _s("H3_VAE_BATCHED", "1") == "1",
        "vae_compile": _s("H3_VAE_COMPILE") or None,
        "warmup_requests": _i("H3_WARMUP", 1),
        "output_dir": _s("H3_OUTPUT_DIR", "outputs"),
    }
    if not config["model_path"]:
        raise SystemExit("H3_MODEL_PATH is required")
    prompt = _s("H3_PROMPT")
    if prompt is None:
        if not config["prompt_file"]:
            raise SystemExit("H3_PROMPT or H3_PROMPT_FILE is required")
        with open(config["prompt_file"]) as handle:
            prompt = json.load(handle)["prompt"]

    # `build_pipeline` and `run_request` are shared with the baseline driver and take an
    # argparse-shaped object, so the measured code path is literally the same one.
    args = SimpleNamespace(
        model_path=config["model_path"], attention_backend=_s("H3_ATTENTION_BACKEND") or None,
        prompt=prompt, height=config["height"], width=config["width"],
        num_frames=config["num_frames"], steps=config["steps"], seed=config["seed"])

    pipe, meta = build_pipeline(args, device)
    if world_size > 1 and config["ulysses_degree"] > 1:
        enable_context_parallel(pipe, config["ulysses_degree"],
                                _s("H3_ULYSSES_ANYTHING", "1") == "1", rank)

    undo, sparse = install_stack(pipe, config)

    timers = {name: GpuTimer() for name in ("denoise", "decode")}
    originals = [
        (pipe.transformer, "forward", instrument(pipe.transformer, timers["denoise"])),
        (pipe.vae, "decode", instrument(pipe.vae, timers["decode"], method="decode")),
    ]
    try:
        # An untimed warmup absorbs kernel compilation, the Sol-Attn correctness gate, and the
        # first-call route materialisation. None of it repeats in the measured request.
        for _ in range(config["warmup_requests"]):
            run_request(pipe, args)
        for timer in timers.values():
            timer.reset()
        torch.cuda.reset_peak_memory_stats()

        state, request_s = run_request(pipe, args)
    finally:
        for module, method, original in originals:
            setattr(module, method, original)

    denoise_s = _round(timers["denoise"].total())
    decode_s = _round(timers["decode"].total())
    benchmark = {
        "schema": 2,
        "total_s": round(request_s, 3),
        "denoise_s": denoise_s,
        "decode_s": decode_s,
        # The fixed remainder — text/audio encoding, packing, scheduling, output assembly — is
        # ~2.1 s and does not move with the denoise work, so the hot path is reported separately.
        "hot_path_s": round((denoise_s or 0) + (decode_s or 0), 3),
        "peak_memory_mib": round(
            torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2),
        "world_size": world_size,
        "config": {k: v for k, v in config.items() if k != "prompt"},
    }
    if sparse is not None:
        sparse._close_request()
        benchmark["sol_attn"] = sparse.stats()

    if rank == 0:
        os.makedirs(config["output_dir"], exist_ok=True)
        with open(os.path.join(config["output_dir"], "benchmark.json"), "w") as handle:
            json.dump(benchmark, handle, indent=2)
        with open(os.path.join(config["output_dir"], "run_config.json"), "w") as handle:
            json.dump(benchmark["config"], handle, indent=2)
        log(f"total {benchmark['total_s']}s  denoise {denoise_s}s  decode {decode_s}s  "
            f"hot {benchmark['hot_path_s']}s  peak {benchmark['peak_memory_mib']} MiB")
        export = getattr(state, "videos", None) or getattr(state, "frames", None)
        if export is not None:
            from diffusers.utils import export_to_video
            export_to_video(export[0], os.path.join(config["output_dir"], "out.mp4"), fps=24)

    for u in reversed(undo):
        u()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
