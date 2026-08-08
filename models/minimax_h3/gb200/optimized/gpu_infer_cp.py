#!/usr/bin/env python3
"""MiniMax-H3 optimized runtime — acceleration line #1: Ulysses context parallelism.

Same pipeline, same precision, same attention backend and same eager DiT as the registered
baseline. The only change is that the packed sequence is sharded across ranks inside the block
stack, which makes this a topology change rather than an approximation: no cache, no sparsity, no
quantization, no compile. `H3_ULYSSES_DEGREE=1` skips `enable_parallelism` entirely and reproduces
the baseline placement, which is the off-identity control the eval profile requires.

Each rank holds a full copy of the weights (134 GB of 186 GB) — CP shards activations and attention
work, not parameters. Text encoding and VAE decode run replicated on every rank; only the block
stack is sharded, and only rank 0 writes artifacts.

Determinism across ranks: the request draws from a CPU `torch.Generator`, so every rank draws the
same conditioning, video and audio noise, and every rank encodes the same prompt. Ranks therefore
agree on everything outside the sharded region without any broadcast.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist


# The vendored diffusers and its huggingface_hub live with the baseline: they are the pristine
# upstream dependency, shared rather than duplicated, so both runtimes are provably the same code.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.join(os.path.dirname(_HERE), "baseline")
for _vendored in (os.path.join(_BASELINE, "vendor_site"), os.path.join(_BASELINE, "diffusers_src", "src")):
    if os.path.isdir(_vendored):
        sys.path.insert(0, _vendored)
sys.path.insert(0, _HERE)


class GpuTimer:
    """Accumulates device time for one component with CUDA events."""

    def __init__(self) -> None:
        self.spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def span(self, device: torch.device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            start.record()
        try:
            yield
        finally:
            with torch.cuda.device(device):
                end.record()
            self.spans.append((start, end))

    def total(self) -> float | None:
        # None, not 0.0: a timer that never fired means the wrapper missed its call site, and that
        # must not be readable as "this component costs nothing".
        if not self.spans:
            return None
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.spans) / 1000.0

    def reset(self) -> None:
        """Drop the warmup's spans so `total()` describes the measured request alone."""
        self.spans.clear()

    def __len__(self) -> int:
        return len(self.spans)


def _module_device(module: torch.nn.Module) -> torch.device:
    """Device a module's work lands on, for modules that may hold no parameters at all.

    `MiniMaxH3RotaryPosEmbed` is the case that matters: it carries only a non-persistent `inv_freq`
    buffer, so `next(module.parameters())` raises StopIteration rather than returning a device.
    """
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cuda", torch.cuda.current_device())


def instrument(module: torch.nn.Module, timer: GpuTimer, method: str = "forward"):
    original = getattr(module, method)
    device = _module_device(module)

    def wrapped(*args, **kwargs):
        with timer.span(device):
            return original(*args, **kwargs)

    setattr(module, method, wrapped)
    return original


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def probe_sequence_length(transformer, out: dict) -> None:
    """Record the packed sequence length, read off the rope module's full-sequence input.

    `rope` is called once per forward with the whole `position_ids` — the plan splits its *output*,
    not its input — so this sees the global length before any sharding. The number decides whether
    plain equipartition sharding is legal (`seq_len % ulysses_degree == 0`) or whether the run needs
    `ulysses_anything`, and diffusers builds this sequence without padding so it is not a round
    number by construction.
    """
    original = transformer.rope.forward

    def wrapped(position_ids, *args, **kwargs):
        out.setdefault("packed_sequence_length", int(position_ids.shape[0]))
        return original(position_ids, *args, **kwargs)

    transformer.rope.forward = wrapped


def build_pipeline(args, device: str) -> tuple[object, dict]:
    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.perf_counter()
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(args.model_path, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to(device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)

    return pipe, {"load_s": load_s}


def enable_context_parallel(pipe, degree: int, ulysses_anything: bool, rank: int) -> dict:
    from cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask
    from diffusers.models._modeling_parallel import ContextParallelConfig

    assert_no_attention_mask(pipe.transformer)
    pipe.transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=degree, ulysses_anything=ulysses_anything),
        cp_plan=MINIMAX_H3_CP_PLAN,
    )
    if rank == 0:
        print(f"[h3] context parallel: ulysses_degree={degree} ulysses_anything={ulysses_anything}", flush=True)
    return {"ulysses_degree": degree, "ulysses_anything": ulysses_anything}


def run_request(pipe, args) -> tuple[object, float]:
    generator = torch.Generator().manual_seed(args.seed)
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t0 = time.perf_counter()
    state = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
    )
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return state, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("H3_MODEL_PATH"))
    parser.add_argument("--prompt-file", default=os.environ.get("H3_PROMPT_FILE"))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--height", type=int, default=int(os.environ.get("H3_HEIGHT", 768)))
    parser.add_argument("--width", type=int, default=int(os.environ.get("H3_WIDTH", 1344)))
    parser.add_argument("--num-frames", type=int, default=int(os.environ.get("H3_NUM_FRAMES", 124)))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("H3_STEPS", 50)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("H3_SEED", 0)))
    parser.add_argument("--warmup-requests", type=int, default=int(os.environ.get("H3_WARMUP", 1)))
    parser.add_argument("--measure-requests", type=int, default=int(os.environ.get("H3_MEASURE", 1)))
    parser.add_argument("--ulysses-degree", type=int, default=int(os.environ.get("H3_ULYSSES_DEGREE", 4)))
    parser.add_argument(
        "--ulysses-anything",
        type=int,
        default=int(os.environ.get("H3_ULYSSES_ANYTHING", 1)),
        help="Uneven sequence sharding. Needed unless the packed length divides the degree exactly.",
    )
    parser.add_argument("--attention-backend", default=os.environ.get("H3_ATTENTION_BACKEND") or None)
    parser.add_argument("--output-dir", default=os.environ.get("H3_OUTPUT_DIR", "outputs"))
    parser.add_argument("--tag", default=os.environ.get("H3_TAG", "cp"))
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if not args.model_path:
        parser.error("--model-path (or H3_MODEL_PATH) is required")
    if args.prompt is None:
        if not args.prompt_file:
            parser.error("--prompt or --prompt-file (or H3_PROMPT_FILE) is required")
        with open(args.prompt_file) as handle:
            args.prompt = json.load(handle)["prompt"]

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Bind the device *before* NCCL initializes and hand it the device explicitly: otherwise NCCL
    # infers it from the global rank, which is only right when the rank->GPU mapping is homogeneous.
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(device)
        )

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    pipe, meta = build_pipeline(args, device)
    if rank == 0:
        print(f"[h3] loaded in {meta['load_s']:.2f}s  device={device} world_size={world_size}", flush=True)

    probe = {}
    probe_sequence_length(pipe.transformer, probe)

    cp_meta = {"ulysses_degree": 1, "ulysses_anything": False}
    if args.ulysses_degree > 1:
        cp_meta = enable_context_parallel(pipe, args.ulysses_degree, bool(args.ulysses_anything), rank)

    for i in range(args.warmup_requests):
        _, warm_s = run_request(pipe, args)
        if rank == 0:
            print(f"[h3] warmup {i + 1}/{args.warmup_requests}: {warm_s:.2f}s  "
                  f"packed_seq={probe.get('packed_sequence_length')}", flush=True)

    timers = {"denoise": GpuTimer(), "decode": GpuTimer(), "audio_decode": GpuTimer(), "encode": GpuTimer()}
    instrument(pipe.transformer, timers["denoise"])
    instrument(pipe.vae, timers["decode"], method="decode")
    instrument(pipe.audio_vae, timers["audio_decode"], method="decode")
    # The encoder block calls `text_encoder.model(...)`, not the top-level forward, so that the
    # language-model head never runs; wrapping `text_encoder.forward` would never fire.
    instrument(pipe.text_encoder.model, timers["encode"])

    torch.cuda.reset_peak_memory_stats()
    requests = []
    state = None
    for i in range(args.measure_requests):
        state, request_s = run_request(pipe, args)
        requests.append(request_s)
        if rank == 0:
            print(f"[h3] request {i + 1}/{args.measure_requests}: {request_s:.3f}s", flush=True)

    if rank != 0:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return 0

    seq_len = probe.get("packed_sequence_length")
    result = {
        "tag": args.tag,
        "variant": f"diffusers_modular_resident_bf16_eager_ulysses{cp_meta['ulysses_degree']}",
        "technique": "topology/context_parallel" if cp_meta["ulysses_degree"] > 1 else "none",
        "model_path": args.model_path,
        "task": "t2va",
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": 24,
        "steps": args.steps,
        "seed": args.seed,
        "world_size": world_size,
        **cp_meta,
        "packed_sequence_length": seq_len,
        "sequence_divides_degree": (None if seq_len is None else seq_len % max(cp_meta["ulysses_degree"], 1) == 0),
        "load_s": round(meta["load_s"], 3),
        "request_s": [round(v, 3) for v in requests],
        "request_s_median": round(sorted(requests)[len(requests) // 2], 3),
        "denoise_gpu_s": _round(timers["denoise"].total()),
        "dit_evals": len(timers["denoise"]),
        "encode_gpu_s": _round(timers["encode"].total()),
        "video_decode_gpu_s": _round(timers["decode"].total()),
        "audio_decode_gpu_s": _round(timers["audio_decode"].total()),
        "peak_memory_mib": {
            f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 1024**2)
            for i in range(torch.cuda.device_count())
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "attention_backend": args.attention_backend or "default",
        "warmup_requests": args.warmup_requests,
    }
    result["per_step_gpu_ms"] = (
        round(result["denoise_gpu_s"] * 1000 / result["dit_evals"], 2)
        if result["denoise_gpu_s"] and result["dit_evals"]
        else None
    )
    accounted = sum(
        v for v in (result["denoise_gpu_s"], result["encode_gpu_s"],
                    result["video_decode_gpu_s"], result["audio_decode_gpu_s"]) if v
    )
    result["unaccounted_s"] = round(result["request_s_median"] - accounted, 3)

    result_path = os.path.join(args.output_dir, f"{args.tag}_result.json")
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)

    if not args.no_export and state is not None:
        from diffusers.utils.export_utils import encode_video

        video_path = os.path.join(args.output_dir, f"{args.tag}.mp4")
        encode_video(
            state.get("videos")[0],
            fps=24,
            output_path=video_path,
            audio=state.get("audio")[0],
            audio_sample_rate=state.get("sampling_rate"),
        )
        print(f"[h3] wrote {video_path}", flush=True)

    print(f"[h3] wrote {result_path}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
