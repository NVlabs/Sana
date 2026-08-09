#!/usr/bin/env python3
"""Registered MiniMax-H3 *gb10* runtime entrypoint — one GB10, FP8, no context parallelism.

Same contract as the baseline and optimized entrypoints: the candidate manifest sets OUT_DIR
and the H3_* switches, this builds the pipeline and writes `benchmark.json` and `out.mp4` into
it. Every acceleration is env-gated, so `run_minimax_h3_gpu.sh` is identical for every
candidate in `candidates/minimax_h3_gb10_*.toml`.

What this variant is
--------------------
MiniMax-H3 t2va on one GB10, from the pruned FP8 checkpoint.

Sol-Engine's own baseline (`models/minimax_h3/GB200/gpu_infer.py`) puts the 33B DiT, the
Qwen3-VL-32B conditioner and both VAEs resident on one 186 GB GB200 and calls that the
denominator. None of that fits here: GB10 has 128 GB of *unified* memory shared with the
host, and the released BF16 weights alone are 134 GB. Two substitutions close the gap and
neither changes the pipeline's structure:

* the DiT is the ComfyUI `pruned_fp8_scaled` checkpoint, loaded through `build.py` — 23.3 GB,
  with 250 of its 300 quantised linears running on the FP8 tensor cores;
* the conditioner is `Qwen/Qwen3-VL-32B-Instruct-FP8` — 33 GB against 62 GB, and structurally
  the same model (H3's `text_encoder/config.json` is stock Qwen3-VL-32B-Instruct: 64 layers,
  hidden 5120, vocab 151936, `rope_theta` 5e6).

Context parallelism, the first stage of Sol-Engine's acceleration line, has nothing to do on
one card. Sol-Attn runs through the released entry point with nothing bypassed and nothing
forked: `_backend_for_arch` has no CuTe kernel for GB10's SM121 and returns "triton", and the
951-row exact KV sink the policy calls for is `sink_tokens`. Both were true only after the
`sol-engine` head moved; `fusion_install.load_sol_attn` records what the port carried before
and the measurement that let it be deleted.

The timing block is kept as Sol-Engine writes it — CUDA events around the components, one
warmup request excluded — so the numbers stay comparable in shape, though not in value.
"""

from __future__ import annotations

import paths

paths.setup()

import argparse
import glob
import json
import os
import platform
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("HF_HOME", str(paths.HF_CACHE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import torch

QWEN_FP8 = "Qwen/Qwen3-VL-32B-Instruct-FP8"


H3_SNAPSHOT = paths.h3_snapshot()
QWEN_FP8_PATH = paths.qwen_fp8()
_QWEN_QUANT = json.load(open(f"{QWEN_FP8_PATH}/config.json"))["quantization_config"]
DIT_CHECKPOINT = paths.dit_checkpoint()


class GpuTimer:
    """Accumulates device time for one component with CUDA events.

    Events are recorded on the component's own stream and synchronized once, at `total()`, so
    instrumenting every DiT evaluation does not serialize the pipeline.
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
        # None, not 0.0: a timer that never fired means the wrapper missed its call site, and
        # that must not read as "this component costs nothing".
        if not self.spans:
            return None
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.spans) / 1000.0

    def __len__(self) -> int:
        return len(self.spans)


def instrument(module: torch.nn.Module, timer: GpuTimer, method: str = "forward"):
    original = getattr(module, method)

    def wrapped(*args, **kwargs):
        device = next(module.parameters()).device
        with timer.span(device):
            return original(*args, **kwargs)

    setattr(module, method, wrapped)
    return original


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _localise_component_index(snapshot: str) -> None:
    """Point `modular_model_index.json` at the directory it is sitting in.

    Each component in that file records a *hub repo id* plus a subfolder, so `load_components`
    resolves them by asking huggingface.co and only finds them locally if the download happens
    to sit in the HuggingFace cache layout. Anything fetched another way — a ModelScope mirror,
    a manual copy — sends the pipeline to the network instead, and on a machine that cannot
    reach it that is a retry loop rather than an error.

    Rewriting the ids to the local path removes the question. Idempotent, and the original is
    kept beside it.
    """
    index = os.path.join(snapshot, "modular_model_index.json")
    backup = index + ".hub"
    with open(index) as handle:
        spec = json.load(handle)

    changed = False
    for entry in spec.values():
        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict):
            path = entry[2].get("pretrained_model_name_or_path")
            if path and not os.path.isabs(path):
                entry[2]["pretrained_model_name_or_path"] = snapshot
                changed = True
    if not changed:
        return

    if not os.path.exists(backup):
        os.replace(index, backup)
        with open(backup) as handle:
            pass
    with open(index, "w") as handle:
        json.dump(spec, handle, indent=2)
    print(f"[h3] component index localised to {snapshot}", flush=True)


def build_pipeline(args, **transformer_options):
    from diffusers import ComponentsManager, ModularPipeline
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    from build import build_pruned_fp8_transformer
    from transformers_compat import patch_processor, patch_text_encoder

    device = f"cuda:{args.device}"
    t0 = time.perf_counter()

    config = json.load(open(f"{H3_SNAPSHOT}/transformer/config.json"))
    transformer, info = build_pruned_fp8_transformer(
        DIT_CHECKPOINT, config, device=device, **transformer_options
    )
    print(f"[h3] DiT: {torch.cuda.memory_allocated(args.device) / 2**30:.2f} GiB, "
          f"{info['quantized_layers']} quantised linears", flush=True)

    # The FP8 conditioner's exclusion list has to be spelled the way transformers reads it, and
    # it has to be edited *into the config*: Qwen ships the vLLM/compressed-tensors key
    # `ignored_layers`, transformers' `FineGrainedFP8Config` reads `modules_to_not_convert`, and
    # passing a corrected `quantization_config=` to `from_pretrained` does not help — a checkpoint
    # that carries its own config wins, with only a `UserWarning` to say so.
    #
    # Left uncorrected, transformers converts the whole model, including the 27-block vision
    # tower and `lm_head` that the checkpoint deliberately left in BF16. Those layers then find
    # no `weight_scale_inv` to load, silently stay on meta — `missing_keys` is empty, because
    # nothing was missing from the file — and `dispatch_model` dies moving them to the device.
    # Only the 64 language-model layers are quantised: 448 scales = 64 x 7 projections.
    config = AutoConfig.from_pretrained(QWEN_FP8_PATH)
    config.quantization_config["modules_to_not_convert"] = _QWEN_QUANT["ignored_layers"]

    # `device_map` loads the conditioner straight onto the card instead of materialising it on
    # the host first, which on a unified-memory part would double-count against the same pool.
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_FP8_PATH, config=config, dtype=torch.bfloat16, device_map={"": device}
    )
    print(f"[h3] + conditioner: {torch.cuda.memory_allocated(args.device) / 2**30:.2f} GiB",
          flush=True)

    _localise_component_index(H3_SNAPSHOT)
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(H3_SNAPSHOT, components_manager=manager)
    pipe.update_components(transformer=transformer, text_encoder=text_encoder)
    pipe.load_components(dtype=torch.bfloat16)

    # The branch targets transformers 5.x; this environment is 4.57. See `transformers_compat`.
    patch_processor(pipe.processor)
    patch_text_encoder(pipe.text_encoder)

    # Only the two VAEs and the tokenizers are left to place; the two big models are already
    # on the card and `pipe.to(...)` would walk them again.
    pipe.vae.to(device)
    pipe.audio_vae.to(device)
    torch.cuda.synchronize()

    load_s = time.perf_counter() - t0
    print(f"[h3] all components: {torch.cuda.memory_allocated(args.device) / 2**30:.2f} GiB",
          flush=True)
    return pipe, {"load_s": load_s, "device": device}


# Every acceleration is opt-in through the environment, so one entrypoint serves every
# candidate in `candidates/minimax_h3_gb10_*.toml` and the run script is identical for all of
# them. This mirrors `optimized/gpu_infer.py`, which gates its Ulysses and cache line the same
# way; what differs is only which techniques exist on a single card.
def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False")


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "")
    return float(raw) if raw not in ("", "none", "None") else None


def lossless_kernels() -> dict:
    """The bit-identical fusions. Verified against `dit_reference/`, not asserted."""
    if not _flag("H3_KERNEL_FUSIONS", "1"):
        return {}
    return dict(fuse_qkv=True, quantizer="triton", fuse_adaln=True,
                fuse_rope=True, fuse_swiglu=True)


def apply_approximations(pipe, steps: int) -> dict:
    """Sol-Attn and the cache: the two techniques that change the output.

    Rebuilt per request rather than installed once. The Sol-Attn dispatch carries a call
    counter that the step-indexed policy reads, and the cache carries the previous sample's
    residual; either leaking across requests would make the second sample depend on the first.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    import cache_line
    import fusion_install

    applied = {}
    tau = _optional_float("H3_SOL_ATTN_TAU")
    if tau is not None:
        dispatch = fusion_install.make_sol_attn_dispatch(
            tau=tau,
            dense_blocks=int(os.environ.get("H3_SOL_ATTN_DENSE_BLOCKS", 2)),
            dense_first_steps=int(os.environ.get("H3_SOL_ATTN_DENSE_FIRST_STEPS", 10)),
            num_steps=steps,
        )
        dispatch.reset()
        h3.dispatch_attention_fn = dispatch
        applied["sol_attn_tau"] = tau

    threshold = _optional_float("H3_CACHE_THRESHOLD")
    cache_line.remove_cache(pipe.transformer)
    if threshold is not None:
        cache_line.apply_cache(pipe.transformer, threshold=threshold)
        cache_line.reset_cache(pipe.transformer)
        applied["cache_threshold"] = threshold

    batch = int(os.environ.get("H3_VAE_TILE_BATCH", 0))
    if batch:
        import vae_shard

        vae_shard.patch_batched_tiles(pipe.vae, batch=batch)
        applied["vae_tile_batch"] = batch
    return applied


def run_request(pipe, args, seed: int):
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
    # Defaults are a bring-up cell, not the published one: GB10 has ~1/30th of a GB200's
    # memory bandwidth, so the official 1344x768/124f/50-step cell is a long run to discover
    # a wiring bug in. Pass --official for it.
    #
    # Only the canvas and the step count shrink. The frame count cannot: the pipeline accepts
    # `17 * n + 5` frames between 5 and 15 seconds at 24 fps, so 124 (n=7) is both the floor
    # and what the published cell uses. `height` / `width` must be multiples of 32.
    parser.add_argument("--height", type=int, default=int(os.environ.get("H3_HEIGHT", 256)))
    parser.add_argument("--width", type=int, default=int(os.environ.get("H3_WIDTH", 448)))
    parser.add_argument("--num-frames", type=int, default=int(os.environ.get("H3_NUM_FRAMES", 124)))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("H3_STEPS", 8)))
    parser.add_argument("--official", action="store_true",
                        help="the published cell: 1344x768, 124 frames, 50 steps")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("H3_SEED", 0)))
    parser.add_argument("--warmup-requests", type=int, default=int(os.environ.get("H3_WARMUP", 0)))
    parser.add_argument("--measure-requests", type=int, default=int(os.environ.get("H3_MEASURE", 1)))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=os.environ.get("H3_PROMPT_FILE", str(paths.PROMPT_FILE)))
    parser.add_argument("--output-dir", default=os.environ.get("H3_OUTPUT_DIR", "outputs"))
    parser.add_argument("--tag", default=os.environ.get("H3_TAG", "gb10"))
    parser.add_argument("--attention-backend", default=os.environ.get("H3_ATTENTION_BACKEND") or None)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if args.official:
        args.height, args.width, args.num_frames, args.steps = 768, 1344, 124, 50

    if args.prompt is None:
        with open(args.prompt_file) as handle:
            args.prompt = json.load(handle)["prompt"]

    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    pipe, placement = build_pipeline(args, **lossless_kernels())
    approximations = apply_approximations(pipe, args.steps)
    if approximations:
        print(f"[h3] approximations: {approximations}", flush=True)
    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)
    print(f"[h3] loaded in {placement['load_s']:.2f}s on {placement['device']}  "
          f"{args.width}x{args.height} {args.num_frames}f {args.steps} steps", flush=True)

    for i in range(args.warmup_requests):
        _, warm_s = run_request(pipe, args, seed=args.seed)
        print(f"[h3] warmup {i + 1}/{args.warmup_requests}: {warm_s:.2f}s", flush=True)

    timers = {k: GpuTimer() for k in ("denoise", "decode", "audio_decode", "encode")}
    instrument(pipe.transformer, timers["denoise"])
    instrument(pipe.vae, timers["decode"], method="decode")
    instrument(pipe.audio_vae, timers["audio_decode"], method="decode")
    # The encoder block calls `text_encoder.model(...)` directly to skip the LM head, so
    # wrapping `text_encoder.forward` would never fire.
    instrument(pipe.text_encoder.model, timers["encode"])

    torch.cuda.reset_peak_memory_stats()
    requests, state = [], None
    for i in range(args.measure_requests):
        state, request_s = run_request(pipe, args, seed=args.seed)
        requests.append(request_s)
        print(f"[h3] request {i + 1}/{args.measure_requests}: {request_s:.3f}s", flush=True)

    result = {
        "tag": args.tag,
        "variant": "diffusers_modular_pruned_fp8_scaled_single_gb10",
        "dit_checkpoint": DIT_CHECKPOINT,
        "text_encoder": QWEN_FP8,
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
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(args.device) / 1024**2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "attention_backend": args.attention_backend or "default",
        "kernel_fusions": bool(lossless_kernels()),
        **approximations,
        "warmup_requests": args.warmup_requests,
    }
    result["per_step_gpu_ms"] = (
        round(result["denoise_gpu_s"] * 1000 / result["dit_evals"], 2)
        if result["denoise_gpu_s"] and result["dit_evals"] else None
    )
    accounted = sum(v for v in (result["denoise_gpu_s"], result["encode_gpu_s"],
                                result["video_decode_gpu_s"], result["audio_decode_gpu_s"]) if v)
    result["unaccounted_s"] = round(result["request_s_median"] - accounted, 3)

    result_path = os.path.join(args.output_dir, "benchmark.json")
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)

    if not args.no_export and state is not None:
        from diffusers.utils.export_utils import encode_video

        video_path = os.path.join(args.output_dir, "out.mp4")
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
