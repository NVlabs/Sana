#!/usr/bin/env python3
"""Wan2.2 A14B fixed-contract driver with experiment-local cache adapters."""

import json
import os
import shutil
import statistics
import time
from pathlib import Path

import torch


DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def log(message):
    print(f"[wan22] {message}", flush=True)


def _s(key, default=None):
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _i(key, default):
    value = os.environ.get(key)
    return int(value) if value not in (None, "") else default


def _f(key, default):
    value = os.environ.get(key)
    return float(value) if value not in (None, "") else default


def _enabled(key):
    return _s(key, "0").strip().lower() in {"1", "true", "yes", "on"}


def _save_frames(frames, destination):
    import numpy as np
    from PIL import Image

    destination.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(array).save(destination / f"f_{index:05d}.png")


def main():
    context_parallel = _enabled("WAN22_CONTEXT_PARALLEL")
    rank = 0
    local_rank = 0
    world_size = 1
    if context_parallel:
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    is_main = rank == 0
    device = torch.device("cuda", local_rank)

    out = Path(os.environ["OUT_DIR"])
    if is_main:
        (out / "videos").mkdir(parents=True, exist_ok=True)
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / "frames_by_prompt").mkdir(parents=True, exist_ok=True)

    weights = os.environ["WAN22_WEIGHTS"]
    label = _s("WAN22_MODEL_LABEL", "wan22")
    height = _i("WAN22_HEIGHT", 704)
    width = _i("WAN22_WIDTH", 1280)
    num_frames = _i("WAN22_FRAMES", 121)
    steps = _i("WAN22_STEPS", 50)
    guidance = _f("WAN22_GUIDANCE", 5.0)
    guidance_2 = _f("WAN22_GUIDANCE2", None)
    flow_shift = _f("WAN22_FLOW_SHIFT", None)
    fps = _i("WAN22_FPS", 24)
    seed = _i("WAN22_SEED", 1024)
    num_prompts = _i("WAN22_NUM_PROMPTS", 5)
    warmup_passes = _i("WAN22_WARMUP_PASSES", 1)
    cache_method = _s("WAN22_CACHE_METHOD", "").strip().lower()
    if cache_method in {"off", "none", "0", "false"}:
        cache_method = ""
    # Cache runs on the frozen 4-rank Ulysses substrate OR single-GPU (world_size==1);
    # the controller's distributed reductions already no-op when dist is uninitialized.
    if cache_method and context_parallel and world_size != 4:
        raise RuntimeError(
            "cache config must execute on the frozen four-rank Ulysses substrate "
            f"(context_parallel={context_parallel}, world_size={world_size})"
        )

    prompts = []
    prompt_file = _s("WAN22_VAL_PROMPTS")
    if prompt_file:
        prompt_file = (
            prompt_file
            if os.path.isabs(prompt_file)
            else str(Path(os.environ.get("AUTOVIDEO_REPO_ROOT", ".")) / prompt_file)
        )
        if Path(prompt_file).exists():
            prompts = [item["prompt"] for item in json.loads(Path(prompt_file).read_text())]
    if not prompts:
        prompts = [
            "A majestic lion strides across the golden savanna, its powerful frame "
            "glistening under the warm afternoon sun. Low angle, cinematic."
        ]
    prompts = prompts[: max(1, num_prompts)]

    log(
        f"rank={rank}/{world_size} local_rank={local_rank} | torch {torch.__version__} "
        f"| cuda={torch.cuda.is_available()} "
        f"| device={torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'}"
    )
    log(f"model={label} weights={weights}")
    log(
        f"cfg H{height} W{width} F{num_frames} steps{steps} g{guidance} g2{guidance_2} "
        f"shift{flow_shift} fps{fps} seed{seed} prompts={len(prompts)} warmup={warmup_passes}"
    )

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    load_start = time.time()
    vae = AutoencoderKLWan.from_pretrained(weights, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(weights, vae=vae, torch_dtype=torch.bfloat16)
    if flow_shift is not None:
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        log(f"scheduler flow_shift set to {flow_shift}")
    moe = getattr(pipe, "transformer_2", None) is not None
    pipe.to(device)
    pipe.set_progress_bar_config(disable=not is_main)

    # Install the verified lossless kernel stack while the original forty-block
    # expert modules are still directly reachable.  Context-parallel hooks are
    # attached next, then the cache proxy wraps those already-patched blocks.
    from wan_kernel_runtime import WanKernelRuntime

    kernel_runtime = WanKernelRuntime(pipe, out)

    parallelism = {"kind": "none", "world_size": 1}
    if context_parallel:
        from diffusers import ContextParallelConfig

        degree = _i("WAN22_CP_DEGREE", world_size)
        ring_degree = _i("WAN22_CP_RING_DEGREE", 1)
        ulysses_degree = _i("WAN22_CP_ULYSSES_DEGREE", degree)
        if degree != world_size or ring_degree * ulysses_degree != world_size:
            raise ValueError(
                "context-parallel degrees must cover the torchrun world: "
                f"ring={ring_degree} ulysses={ulysses_degree} world={world_size}"
            )
        if (ring_degree, ulysses_degree) not in ((1, 4), (2, 2)):
            raise ValueError(
                "certified Wan topology config are ring1/ulysses4 or "
                f"ring2/ulysses2, got ring={ring_degree} ulysses={ulysses_degree}"
            )
        for model_name in ("transformer", "transformer_2"):
            model = getattr(pipe, model_name, None)
            if model is not None:
                model.enable_parallelism(
                    config=ContextParallelConfig(
                        ring_degree=ring_degree, ulysses_degree=ulysses_degree
                    )
                )
        parallelism = {
            "kind": "context_parallel_ring_ulysses",
            "world_size": world_size,
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "dense_attention_preserved": True,
        }
        log(f"enabled frozen exact dense context parallelism: {parallelism}")

    pisa_step_tracking = None
    reset_pisa_step_tracking = None
    pisa_step_tracking_summary = None
    if "pisa_attention" in kernel_runtime.stack:
        from wan_kernel_optimizations import (
            install_pisa_step_tracking,
            reset_pisa_step_tracking,
            pisa_step_tracking_summary,
        )

        pisa_step_tracking = install_pisa_step_tracking(pipe)
        kernel_runtime.optimization_activations["pisa_step_tracking"] = pisa_step_tracking
        log(f"PISA step clock active: {pisa_step_tracking}")

    if kernel_runtime.active:
        log(
            "kernel runtime active "
            f"profile={kernel_runtime.profile_enabled} stack={list(kernel_runtime.stack)}"
        )

    cache_runtime = None
    if cache_method:
        # Install after enable_parallelism so the original first block retains
        # Diffusers' registered sequence-split hook inside the proxy.
        from cache_controller import install_cache

        cache_runtime = install_cache(pipe, steps)
        if cache_runtime is None:
            raise RuntimeError("WAN22_CACHE_METHOD was set but no cache runtime was installed")
        log(
            f"cache enabled: family={cache_method} payload=blocks1-39 residual "
            "with fresh CP-sharding block0"
        )
    else:
        log("cache disabled: all original transformer blocks execute")

    log(
        f"pipeline ready in {time.time() - load_start:.1f}s | "
        f"scheduler={type(pipe.scheduler).__name__} | MoE={moe}"
    )

    def generate(prompt, tag, prompt_index=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        step_times = []
        controller = cache_runtime.new_controller() if cache_runtime is not None else None
        if reset_pisa_step_tracking is not None:
            reset_pisa_step_tracking(prompt_index)

        def callback(pipeline, index, timestep, callback_kwargs):
            step_times.append(time.time())
            return callback_kwargs

        kwargs = {
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEG,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": callback,
        }
        if guidance_2 is not None:
            kwargs["guidance_scale_2"] = guidance_2

        torch.cuda.synchronize(device)
        start = time.time()
        kernel_runtime.start_pass(tag)
        try:
            result = pipe(**kwargs)
        except TypeError as error:
            removed = []
            for unsupported in ("guidance_scale_2", "callback_on_step_end"):
                if unsupported in kwargs and unsupported in str(error):
                    removed.append(unsupported)
                    kwargs.pop(unsupported)
            if not removed:
                raise
            log(f"unsupported pipeline kwargs {removed} ({error!r}); retrying")
            if cache_runtime is not None:
                cache_runtime.clear_controller()
                controller = cache_runtime.new_controller()
            step_times.clear()
            torch.cuda.synchronize(device)
            start = time.time()
            result = pipe(**kwargs)
        torch.cuda.synchronize(device)
        total = time.time() - start
        kernel_runtime.finish_pass(tag)
        denoise = (step_times[-1] - step_times[0]) if len(step_times) >= 2 else None
        decode = (time.time() - step_times[-1]) if step_times else None
        cache_stats = controller.finalize() if controller is not None else None
        if cache_runtime is not None:
            cache_runtime.clear_controller()
        cache_note = ""
        if cache_stats is not None:
            cache_note = (
                f" cache={cache_stats['method']} reused={cache_stats['reuse_steps']}/"
                f"{cache_stats['total_steps']} pattern={cache_stats['hit_pattern']}"
            )
        log(
            f"[{tag}] total={total:.2f}s denoise~{denoise and round(denoise, 2)} "
            f"decode~{decode and round(decode, 2)}{cache_note}"
        )
        return result.frames[0], total, denoise, decode, cache_stats

    log(f"=== warmup ({warmup_passes} pass) ===")
    for _ in range(max(0, warmup_passes)):
        generate(prompts[0], "warmup", prompt_index=0)

    log("=== measured pass ===")
    samples = []
    cache_records = []
    for prompt_index, prompt in enumerate(prompts):
        frames, total, denoise, decode, cache_stats = generate(
            prompt, f"p{prompt_index}", prompt_index=prompt_index
        )
        sample = {
            "prompt_id": f"p{prompt_index}",
            "total_s": total,
            "denoise_s": denoise,
            "decode_s": decode,
        }
        if cache_stats is not None:
            sample["cache"] = {
                key: cache_stats[key]
                for key in ("compute_steps", "reuse_steps", "hit_rate", "hit_pattern")
            }
            cache_records.append({"prompt_id": f"p{prompt_index}", "stats": cache_stats})
        samples.append(sample)

        if is_main:
            video_path = out / "videos" / f"prompt_{prompt_index:02d}.mp4"
            export_to_video(frames, str(video_path), fps=fps)
            grouped_dir = out / "frames_by_prompt" / f"p{prompt_index:02d}"
            _save_frames(frames, grouped_dir)
            if prompt_index == 0:
                shutil.copy(video_path, out / "out.mp4")
                for frame_path in sorted(grouped_dir.glob("*.png")):
                    shutil.copy(frame_path, out / "frames" / frame_path.name)

    def median(key):
        values = [sample[key] for sample in samples if sample[key] is not None]
        return statistics.median(values) if values else None

    total_median = median("total_s")
    denoise_median = median("denoise_s")
    decode_median = median("decode_s")
    kernel_summary = kernel_runtime.summary()
    cache_summary = None
    if cache_records:
        first = cache_records[0]["stats"]
        cache_summary = {
            "method": cache_method,
            "measured_prompt_count": len(cache_records),
            "signal_source": first["signal_source"],
            "reuse_payload": first["reuse_payload"],
            "refresh_rule": first["refresh_rule"],
            "off_path": first["off_path"],
            "placement": first["placement"],
            "parameters": first["parameters"],
            "compute_steps_median": statistics.median(
                record["stats"]["compute_steps"] for record in cache_records
            ),
            "reuse_steps_median": statistics.median(
                record["stats"]["reuse_steps"] for record in cache_records
            ),
            "hit_rate_median": statistics.median(
                record["stats"]["hit_rate"] for record in cache_records
            ),
            "hit_patterns": [record["stats"]["hit_pattern"] for record in cache_records],
        }

    run_config = {
        "model": label,
        "weights": weights,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance,
        "guidance_scale_2": guidance_2,
        "flow_shift": flow_shift,
        "fps": fps,
        "seed": seed,
        "num_gpus": world_size,
        "moe": moe,
        "prompt_count": len(prompts),
        "warmup_passes": warmup_passes,
        "dit_calls_per_step": 2 if guidance > 1.0 else 1,
        "expected_dit_calls_per_prompt": steps * (2 if guidance > 1.0 else 1),
        "parallelism": parallelism,
        "cache_method": cache_method or "off",
        "kernel_stack": list(kernel_runtime.stack),
    }
    if pisa_step_tracking_summary is not None:
        run_config["pisa_step_tracking"] = pisa_step_tracking_summary()
    benchmark = {
        "schema_version": 2,
        "model_id": label,
        "pipeline": "WanPipeline",
        "total_s": total_median,
        "denoise_s": denoise_median,
        "decode_s": decode_median,
        "timing_scope": "text_to_video_hot_after_warmup_pass",
        "warm_steady_state": True,
        "baseline_class": "cache_config" if cache_method else "pristine_cp4_control",
        "config": run_config,
        "aggregate": {
            "total_s": total_median,
            "denoise_s": denoise_median,
            "decode_s": decode_median,
            "prompt_count": len(prompts),
            "reduction": "median",
        },
        "samples": samples,
        "cache": cache_summary,
        "kernel_runtime": kernel_summary,
        "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }

    if is_main:
        (out / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
        (out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        if cache_records:
            (out / "cache_stats.json").write_text(
                json.dumps({"schema_version": 1, "summary": cache_summary, "prompts": cache_records}, indent=2)
                + "\n"
            )
        log(
            f"=== DONE === median total_s={total_median:.2f}s "
            f"denoise_s={denoise_median and round(denoise_median, 2)} over {len(samples)} prompt(s)"
        )
        log(f"artifacts in {out}")

    if context_parallel:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
