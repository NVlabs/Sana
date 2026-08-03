#!/usr/bin/env python3
"""MiniMax-H3 baseline inference on GB200 — pristine diffusers, no acceleration technique.

This is the Sol-Engine *baseline* contract for `minimax_h3`: the lightest correct way to run the
model, so every later speedup has an honest denominator. It is the upstream Modular Diffusers
integration (PR #14355) run as-is — eager DiT, released BF16/FP32 precision policy, default
attention backend, no cache, no sparsity, no compile, no quantization.

Placement is single-card resident: the 33B DiT (61.7 GB bf16), the Qwen3-VL conditioner (62.1 GB)
and both VAEs all sit on one GB200. 134 GB of weights fit in 186 GB with room for the packed
sequence, so nothing is offloaded and nothing streams and the measured latency is pure compute.
This is also the plainest upstream usage — a bare `pipe.to(device)`.

The upstream two-card recipe (conditioner on a second card through a `device_map`) does *not*
work here, and the reason is structural rather than a tuning miss:
`ModularPipeline._execution_device` walks `self.components` in insertion order and returns the
device of the first module carrying an accelerate hook. `text_encoder` is the first entry of
`MINIMAX_H3_COMPONENTS`, so a `device_map` on it makes the *whole* pipeline resolve to that card —
latents, timesteps and layout all get built there while the DiT sits on the other one, and the
denoise step dies on mixed devices. The modular blocks assume one execution device throughout
(every block reads `components._execution_device`), so splitting components needs a patch, which a
baseline must not carry. `--encoder-device` is kept for smaller cards where the split is required.

There is no context parallelism here on purpose: `MiniMaxH3Transformer3DModel` declares no
`_cp_plan`, so diffusers cannot shard the packed sequence across cards even though
`MiniMaxH3AttnProcessor` already carries `_parallel_config` and forwards it to
`dispatch_attention_fn`. Wiring Ulysses over those 4 cards is optimization line #1, measured
against this file.

Timing. The authoritative number is `request_s`: the wall interval of one `pipe(...)` call with
both models already resident, CUDA synchronized on both sides. The GPU sub-intervals come from
CUDA events around the components themselves, so they are real device time, not host intervals:
`denoise_gpu_s` sums the DiT evaluations, `decode_gpu_s` the two VAE decodes, `encode_gpu_s` the
conditioner. They do not add up to `request_s` — packing, layout and scheduler work sit between
them on the host.
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


# The vendored diffusers (PR #14355 / branch `minimax-h3`) has to win over whatever the env has
# installed, so it goes on the front of the path before diffusers is first imported. `vendor_site`
# carries huggingface_hub >= 1.23, which that diffusers requires and the shared env does not have —
# the env is a registered read-only asset, so it is shadowed rather than upgraded.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _vendored in (os.path.join(_HERE, "vendor_site"), os.path.join(_HERE, "diffusers_src", "src")):
    if os.path.isdir(_vendored):
        sys.path.insert(0, _vendored)


class GpuTimer:
    """Accumulates device time for one component with CUDA events.

    Events are recorded on the component's own stream and only synchronized once, at `total()`, so
    instrumenting the 50 DiT evaluations does not serialize the pipeline the way a per-call
    `torch.cuda.synchronize()` would.
    """

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

    def __len__(self) -> int:
        return len(self.spans)


def instrument(module: torch.nn.Module, timer: GpuTimer, method: str = "forward"):
    """Wrap one method of a component so every call lands in `timer`."""
    original = getattr(module, method)

    def wrapped(*args, **kwargs):
        device = next(module.parameters()).device
        with timer.span(device):
            return original(*args, **kwargs)

    setattr(module, method, wrapped)
    return original


def _round(value: float | None, digits: int = 3) -> float | None:
    """Keep a missed timer as None instead of collapsing it to a number."""
    return None if value is None else round(value, digits)


def build_pipeline(args) -> tuple[object, dict]:
    from diffusers import ComponentsManager, ModularPipeline
    from transformers import Qwen3VLForConditionalGeneration

    dit_device = f"cuda:{args.dit_device}"
    encoder_device = f"cuda:{args.encoder_device}"

    t0 = time.perf_counter()
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(args.model_path, components_manager=manager)

    if encoder_device != dit_device:
        # Loading the conditioner straight onto its own card avoids ever materializing 62 GB on the
        # DiT's card, which a load-then-move would do.
        pipe.update_components(
            text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path,
                subfolder="text_encoder",
                dtype=torch.bfloat16,
                device_map={"": encoder_device},
            ),
        )

    pipe.load_components(dtype=torch.bfloat16)
    if encoder_device == dit_device:
        pipe.to(dit_device)
    else:
        pipe.transformer.to(dit_device)
        pipe.vae.to(dit_device)
        pipe.audio_vae.to(dit_device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)

    return pipe, {"load_s": load_s, "dit_device": dit_device, "encoder_device": encoder_device}


def run_request(pipe, args, seed: int, timers: dict | None) -> tuple[object, float]:
    generator = torch.Generator().manual_seed(seed)
    torch.cuda.synchronize()
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
    return state, time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("H3_MODEL_PATH"), required=False)
    parser.add_argument("--prompt-file", default=os.environ.get("H3_PROMPT_FILE"))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--height", type=int, default=int(os.environ.get("H3_HEIGHT", 768)))
    parser.add_argument("--width", type=int, default=int(os.environ.get("H3_WIDTH", 1344)))
    parser.add_argument("--num-frames", type=int, default=int(os.environ.get("H3_NUM_FRAMES", 124)))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("H3_STEPS", 50)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("H3_SEED", 0)))
    parser.add_argument("--warmup-requests", type=int, default=int(os.environ.get("H3_WARMUP", 1)))
    parser.add_argument("--measure-requests", type=int, default=int(os.environ.get("H3_MEASURE", 1)))
    parser.add_argument("--dit-device", type=int, default=int(os.environ.get("H3_DIT_DEVICE", 0)))
    parser.add_argument("--encoder-device", type=int, default=int(os.environ.get("H3_ENCODER_DEVICE", 0)))
    parser.add_argument("--attention-backend", default=os.environ.get("H3_ATTENTION_BACKEND") or None)
    parser.add_argument("--output-dir", default=os.environ.get("H3_OUTPUT_DIR", "outputs"))
    parser.add_argument("--tag", default=os.environ.get("H3_TAG", "baseline"))
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if not args.model_path:
        parser.error("--model-path (or H3_MODEL_PATH) is required")
    if args.prompt is None:
        if not args.prompt_file:
            parser.error("--prompt or --prompt-file (or H3_PROMPT_FILE) is required")
        with open(args.prompt_file) as handle:
            args.prompt = json.load(handle)["prompt"]

    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    pipe, placement = build_pipeline(args)
    print(f"[h3] loaded in {placement['load_s']:.2f}s  "
          f"dit={placement['dit_device']} encoder={placement['encoder_device']}", flush=True)

    for i in range(args.warmup_requests):
        _, warm_s = run_request(pipe, args, seed=args.seed, timers=None)
        print(f"[h3] warmup {i + 1}/{args.warmup_requests}: {warm_s:.2f}s", flush=True)

    # Instrumented only for the measured requests, so warmup autotuning never lands in the numbers.
    timers = {"denoise": GpuTimer(), "decode": GpuTimer(), "audio_decode": GpuTimer(), "encode": GpuTimer()}
    instrument(pipe.transformer, timers["denoise"])
    instrument(pipe.vae, timers["decode"], method="decode")
    instrument(pipe.audio_vae, timers["audio_decode"], method="decode")
    # The conditioner is invoked as `text_encoder.model(...)`, not through the top-level forward:
    # the encoder block fires the accelerate hook by hand and calls the inner module so the
    # vocabulary-wide language-model head never runs. Wrapping `text_encoder.forward` would never
    # fire (it silently reported 0.0 in job 5813128).
    instrument(pipe.text_encoder.model, timers["encode"])

    torch.cuda.reset_peak_memory_stats()
    requests = []
    state = None
    for i in range(args.measure_requests):
        state, request_s = run_request(pipe, args, seed=args.seed, timers=timers)
        requests.append(request_s)
        print(f"[h3] request {i + 1}/{args.measure_requests}: {request_s:.3f}s", flush=True)

    result = {
        "tag": args.tag,
        "variant": "diffusers_modular_resident_bf16_eager",
        "model_path": args.model_path,
        "task": "t2va",
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": 24,
        "steps": args.steps,
        "seed": args.seed,
        "load_s": round(placement["load_s"], 3),
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
    # What the component timers do not account for: text encoding when its timer missed, plus
    # packing, layout and scheduler work on the host.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
