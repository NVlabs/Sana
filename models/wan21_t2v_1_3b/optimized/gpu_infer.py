#!/usr/bin/env python3
"""Wan2.2 T2V baseline driver (vanilla diffusers WanPipeline). Env-driven.

Called by scripts/run_wan22_*_gpu.sh (which tees stdout to run.log). Runs a
warmup pass + a measured pass over the validation prompts, then writes the
standard artifacts into $OUT_DIR:
  out.mp4, videos/prompt_NN.mp4, frames/f_%05d.png,
  benchmark.json (schema 2: total_s + denoise_s + decode_s + config + samples),
  run_config.json

Handles BOTH Wan2.2-TI2V-5B (single transformer) and Wan2.2-T2V-A14B (MoE:
transformer + transformer_2, dual guidance via WAN22_GUIDANCE2). Single GPU.
"""
import json
import os
import statistics
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

# Official Wan2.2 default negative prompt (Wan-Video/Wan2.2).
DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def log(m):
    print(f"[wan22] {m}", flush=True)


def _s(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


def _i(k, d):
    v = os.environ.get(k)
    return int(v) if v not in (None, "") else d


def _f(k, d):
    v = os.environ.get(k)
    return float(v) if v not in (None, "") else d


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _profile_summary(prof, structural):
    """Reduce a Kineto trace to durable, bounded hot-path evidence."""

    def us(event, *names):
        for name in names:
            value = getattr(event, name, None)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    operators = []
    for event in prof.key_averages(group_by_input_shape=True):
        self_device_us = us(event, "self_device_time_total", "self_cuda_time_total")
        device_total_us = us(event, "device_time_total", "cuda_time_total")
        cpu_total_us = us(event, "cpu_time_total")
        operators.append(
            {
                "name": str(getattr(event, "key", getattr(event, "name", "unknown"))),
                "count": int(getattr(event, "count", 0) or 0),
                "self_device_ms": self_device_us / 1000.0,
                "device_total_ms": device_total_us / 1000.0,
                "cpu_total_ms": cpu_total_us / 1000.0,
                "input_shapes": _jsonable(getattr(event, "input_shapes", None)),
            }
        )
    operators.sort(key=lambda row: (row["self_device_ms"], row["device_total_ms"]), reverse=True)

    kernel_totals = defaultdict(lambda: {"count": 0, "device_ms": 0.0})
    cuda_event_count = 0
    for event in prof.events():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" not in device_type:
            continue
        cuda_event_count += 1
        name = str(getattr(event, "name", "unknown"))
        row = kernel_totals[name]
        row["count"] += 1
        row["device_ms"] += us(event, "device_time_total", "self_device_time_total", "cuda_time_total") / 1000.0
    kernels = [
        {"name": name, "count": row["count"], "device_ms": row["device_ms"]}
        for name, row in kernel_totals.items()
    ]
    kernels.sort(key=lambda row: row["device_ms"], reverse=True)

    layout_names = (
        "copy_",
        "clone",
        "contiguous",
        "transpose",
        "permute",
        "reshape",
        "view",
        "cat",
        "to_copy",
    )
    layout = [row for row in operators if any(name in row["name"] for name in layout_names)]
    pointwise_names = ("aten::add", "aten::mul", "aten::sub", "aten::div", "aten::silu", "aten::gelu")
    pointwise = [row for row in operators if row["name"].startswith(pointwise_names)]

    return {
        "schema_version": 1,
        "timing_scope": "warm_single_prompt_generation_full_hot_path",
        "warmup_policy": "two complete generations finished before the profiled measured generation",
        "profiler": {
            "activities": ["cpu", "cuda"],
            "record_shapes": True,
            "profile_memory": True,
            "with_flops": True,
            "cuda_event_count": cuda_event_count,
        },
        "structural": structural,
        "top_operators_by_self_device_time": operators[:100],
        "top_cuda_kernels_by_device_time": kernels[:100],
        "layout_and_copy_operators": layout[:100],
        "pointwise_operators": pointwise[:100],
    }


def main():
    out = Path(os.environ["OUT_DIR"])
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    weights = os.environ["WAN22_WEIGHTS"]
    label = _s("WAN22_MODEL_LABEL", "wan22")
    H, W, F = _i("WAN22_HEIGHT", 704), _i("WAN22_WIDTH", 1280), _i("WAN22_FRAMES", 121)
    STEPS = _i("WAN22_STEPS", 50)
    G = _f("WAN22_GUIDANCE", 5.0)
    G2 = _f("WAN22_GUIDANCE2", None)
    SHIFT = _f("WAN22_FLOW_SHIFT", None)
    FPS = _i("WAN22_FPS", 24)
    SEED = _i("WAN22_SEED", 1024)
    NUMP = _i("WAN22_NUM_PROMPTS", 5)
    WARM = _i("WAN22_WARMUP_PASSES", 1)
    PROFILE = bool(_i("WAN22_KERNEL_PROFILE", 0))
    INTEGRATED_PAIR = bool(_i("WAN22_INTEGRATED_GENERATION_PAIR", 0))

    prompts = []
    pf = _s("WAN22_VAL_PROMPTS")
    if pf:
        pf = pf if os.path.isabs(pf) else str(Path(os.environ.get("AUTOVIDEO_REPO_ROOT", ".")) / pf)
        if Path(pf).exists():
            prompts = [d["prompt"] for d in json.loads(Path(pf).read_text())]
    if not prompts:
        prompts = ["A majestic lion strides across the golden savanna, its powerful "
                   "frame glistening under the warm afternoon sun. Low angle, cinematic."]
    prompts = prompts[: max(1, NUMP)]

    log(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
        f"| device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"model={label} weights={weights}")
    log(f"cfg H{H} W{W} F{F} steps{STEPS} g{G} g2{G2} shift{SHIFT} fps{FPS} seed{SEED} "
        f"prompts={len(prompts)} warmup={WARM}")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(weights, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(weights, vae=vae, torch_dtype=torch.bfloat16)
    if SHIFT is not None:
        from diffusers import UniPCMultistepScheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=SHIFT)
        log(f"scheduler flow_shift set to {SHIFT}")
    moe = getattr(pipe, "transformer_2", None) is not None
    pipe.to("cuda")
    log(f"pipeline ready in {time.time() - t0:.1f}s | scheduler={type(pipe.scheduler).__name__} | MoE={moe}")

    # --- Optional PISA sparse self-attention (gated on WAN22_PISA_DENSITY) ---
    # Ported from the verified 14B Wan PISA adapter. The processor hook is
    # installed BEFORE the kernel runtime's regional_compile so the compiled
    # block captures the PISA-hooked attn1; the triton kernel graph-breaks out of
    # the compiled region, so WAN22_COMPILE_FULLGRAPH must be 0 for this variant.
    _reset_pisa_step_clock = None
    _pisa_step_summary = None
    _pisa_mod = None
    if os.environ.get("WAN22_PISA_DENSITY"):
        # Load the 14B PISA adapter by explicit file path (importlib) rather than
        # via sys.path, so this runtime's own `wan_kernel_runtime` module is never
        # shadowed by the 14B one (both are named identically but differ).
        import importlib.util as _ilu
        _repo = Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
        _pisa_dir = os.environ.get(
            "WAN22_PISA_ADAPTER_DIR",
            str(_repo / "models/wan22_t2v_a14b/optimized"),
        )
        _pisa_file = str(Path(_pisa_dir) / "wan_kernel_optimizations.py")
        _spec = _ilu.spec_from_file_location("wan14b_pisa_adapter", _pisa_file)
        _pisa_mod = _ilu.module_from_spec(_spec)
        # Register under its unique name so torch.compile/dynamo can resolve the
        # module when it traces the PISA processor. Unique name => no shadowing of
        # this runtime's own `wan_kernel_runtime`.
        import sys as _sys
        _sys.modules["wan14b_pisa_adapter"] = _pisa_mod
        _spec.loader.exec_module(_pisa_mod)
        _pisa_mod.install_pisa_attention(pipe, out)
        log(
            f"PISA attention installed density={os.environ.get('WAN22_PISA_DENSITY')} "
            f"block={os.environ.get('WAN22_PISA_BLOCK_SIZE')}"
        )

    # --- Optional SOL Attention (PISA2 CuTe DSL, SM100), gated on WAN22_SOL_ATTN ---
    # Installs the sol_attn dispatch hook over diffusers' dispatch_attention_fn,
    # BEFORE regional_compile so the compiled block captures it. The CuteDSL kernel
    # graph-breaks out of the compiled region, so WAN22_COMPILE_FULLGRAPH must be 0.
    # Dense fallback for non-128 / causal / cross-attn / non-SM100.
    if os.environ.get("WAN22_SOL_ATTN") == "1":
        import sys as _sys_sol
        _repo_sol = os.environ.get(
            "AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3])
        )
        if _repo_sol not in _sys_sol.path:
            _sys_sol.path.insert(0, _repo_sol)
        from techniques.sparse_backends.sol_attn_backend import (
            make_sol_attn_dispatch, sol_attn_begin_forward, install_wan_morton_forward,
        )
        from diffusers.models.transformers import transformer_wan as _twan_sol
        _sol_kw = dict(
            target_density=float(os.environ.get("WAN22_SOL_DENSITY", "0.05")),
            dense_steps=int(os.environ.get("WAN22_SOL_DENSE_STEPS", "0")),
            dense_layers=os.environ.get("WAN22_SOL_DENSE_LAYERS", ""),
        )
        _sol_tau = os.environ.get("WAN22_SOL_TAU")
        if _sol_tau is not None:
            _sol_kw["tau"] = float(_sol_tau)
        # GLOBAL Morton3D reorder: permute tokens + RoPE once at the block stack, not
        # per attention call (FFN/norm/residual are per-token, order-invariant).
        _grid = None
        if os.environ.get("WAN22_SOL_REORDER", "1") == "1":
            _pt = list(pipe.transformer.config.patch_size)
            _vt = getattr(pipe, "vae_scale_factor_temporal", 4)
            _vs = getattr(pipe, "vae_scale_factor_spatial", 8)
            _gF = 1 + int(os.environ.get("WAN22_FRAMES", "81")) // (_vt * _pt[0])
            _gH = int(os.environ.get("WAN22_HEIGHT", "480")) // (_vs * _pt[1])
            _gW = int(os.environ.get("WAN22_WIDTH", "832")) // (_vs * _pt[2])
            _grid = (_gF, _gH, _gW)
            _tok = install_wan_morton_forward(pipe.transformer, _grid)
            log(f"Morton3D GLOBAL reorder installed grid={_grid} tokens={_tok}")
        # Per-call hook does NOT reorder (grid omitted); global reorder handles it.
        _twan_sol.dispatch_attention_fn = make_sol_attn_dispatch(
            _twan_sol.dispatch_attention_fn, **_sol_kw
        )
        pipe.transformer.register_forward_pre_hook(lambda _m, _a: sol_attn_begin_forward())
        log(f"SOL Attention installed tau={_sol_tau} density_target={_sol_kw['target_density']} "
            f"dense_steps={_sol_kw['dense_steps']} dense_layers='{_sol_kw['dense_layers']}' "
            f"global_reorder={_grid is not None}")

    from wan_kernel_runtime import WanKernelRuntime

    kernel_runtime = WanKernelRuntime(pipe, out)
    if kernel_runtime.active:
        log(
            "kernel runtime active "
            f"stack={list(kernel_runtime.stack)} active_candidate={kernel_runtime.active_candidate or None} "
            f"initialization={kernel_runtime.activation['initialization_s']:.3f}s"
        )

    # Compose the independently verified cache controller outside the exact
    # kernel stack. EasyCache captures the already-transformed DiT forward, so
    # every refresh executes the complete kernel-optimized model while reuse
    # calls return only the recorded transform vector.
    from cache_runtime import maybe_enable_cache

    cache_runtime = maybe_enable_cache(pipe, STEPS)
    if cache_runtime is not None:
        log(f"cache enabled: {cache_runtime.describe()}")

    # PISA step clock must wrap model.forward OUTSIDE the cache wrapper so it
    # counts true denoising steps (EasyCache short-circuits inside the forward).
    if os.environ.get("WAN22_PISA_DENSITY") and _pisa_mod is not None:
        _pisa_mod.install_pisa_step_tracking(pipe)
        _reset_pisa_step_clock = _pisa_mod.reset_pisa_step_tracking
        _pisa_step_summary = _pisa_mod.pisa_step_tracking_summary
        log("PISA step clock installed (outer of cache)")

    model_calls = {"count": 0, "first_input": None}
    hook_handles = []
    if PROFILE:
        transformer = pipe.transformer

        def count_model_calls(module, args, kwargs):
            model_calls["count"] += 1
            if model_calls["first_input"] is None:
                hidden = kwargs.get("hidden_states") if kwargs else None
                timestep = kwargs.get("timestep") if kwargs else None
                encoder = kwargs.get("encoder_hidden_states") if kwargs else None
                model_calls["first_input"] = {
                    "hidden_states": {
                        "shape": list(hidden.shape),
                        "dtype": str(hidden.dtype),
                        "stride": list(hidden.stride()),
                    },
                    "timestep": {
                        "shape": list(timestep.shape),
                        "dtype": str(timestep.dtype),
                    },
                    "encoder_hidden_states": {
                        "shape": list(encoder.shape),
                        "dtype": str(encoder.dtype),
                        "stride": list(encoder.stride()),
                    },
                }

        hook_handles.append(transformer.register_forward_pre_hook(count_model_calls, with_kwargs=True))

    profile_result = None

    def gen(prompt, tag):
        nonlocal profile_result
        if cache_runtime is not None:
            cache_runtime.begin_generation(tag)
        if _reset_pisa_step_clock is not None:
            _reset_pisa_step_clock(0)
        g = torch.Generator(device="cuda").manual_seed(SEED)
        step_t = []
        calls_before = model_calls["count"]

        def cb(p, i, t, kw):
            step_t.append(time.time())
            return kw

        kw = dict(prompt=prompt, negative_prompt=DEFAULT_NEG, height=H, width=W,
                  num_frames=F, guidance_scale=G, num_inference_steps=STEPS,
                  generator=g, callback_on_step_end=cb)
        if G2 is not None:
            kw["guidance_scale_2"] = G2
        torch.cuda.synchronize()
        s = time.time()
        kernel_runtime.start_pass(tag)
        profile_this = PROFILE and tag == "p0"
        if profile_this:
            profile_ctx = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
            )
        else:
            profile_ctx = nullcontext()
        try:
            with profile_ctx as prof:
                r = pipe(**kw)
        except TypeError as e:
            for bad in ("guidance_scale_2", "callback_on_step_end"):
                if bad in kw and bad in str(e):
                    log(f"{bad} not accepted ({e!r}); retrying without it")
                    kw.pop(bad)
            step_t.clear()
            torch.cuda.synchronize()
            s = time.time()
            r = pipe(**kw)
        torch.cuda.synchronize()
        total = time.time() - s
        dit_ledger = kernel_runtime.finish_pass(tag)
        # denoise ~= span across step callbacks; decode ~= time after last step.
        denoise = (step_t[-1] - step_t[0]) if len(step_t) >= 2 else None
        decode = (time.time() - step_t[-1]) if step_t else None
        calls = model_calls["count"] - calls_before
        if profile_this:
            cfg = pipe.transformer.config
            structural = {
                "prompt_count": 1,
                "steps_per_prompt": STEPS,
                "model_calls_per_step": calls // STEPS if STEPS else None,
                "dit_calls_profiled": calls,
                "blocks_per_dit": len(pipe.transformer.blocks),
                "block_calls_profiled": calls * len(pipe.transformer.blocks),
                "self_attention_calls_profiled": calls * len(pipe.transformer.blocks),
                "cross_attention_calls_profiled": calls * len(pipe.transformer.blocks),
                "ffn_calls_profiled": calls * len(pipe.transformer.blocks),
                "model_config": {
                    "num_layers": cfg.num_layers,
                    "num_attention_heads": cfg.num_attention_heads,
                    "attention_head_dim": cfg.attention_head_dim,
                    "inner_dim": cfg.num_attention_heads * cfg.attention_head_dim,
                    "ffn_dim": cfg.ffn_dim,
                    "patch_size": list(cfg.patch_size),
                    "in_channels": cfg.in_channels,
                    "out_channels": cfg.out_channels,
                },
                "first_dit_input": model_calls["first_input"],
                "transformer_dtype": str(pipe.transformer.dtype),
                "fused_projections_enabled": bool(
                    getattr(pipe.transformer.blocks[0].attn1, "fused_projections", False)
                ),
            }
            profile_result = _profile_summary(prof, structural)
            profile_result["measured"] = {
                "profiled_generation_total_s_including_profiler_overhead": total,
                "profiled_denoise_callback_span_s_including_profiler_overhead": denoise,
                "profiled_decode_tail_s_including_profiler_overhead": decode,
            }
            (out / "kernel_profile.json").write_text(json.dumps(profile_result, indent=2))
            log(
                f"[{tag}] wrote kernel_profile.json with {calls} DiT calls and "
                f"{profile_result['profiler']['cuda_event_count']} CUDA events"
            )
        log(
            f"[{tag}] total={total:.2f}s denoise~{denoise and round(denoise,2)} "
            f"decode~{decode and round(decode,2)} "
            f"dit_calls={dit_ledger.get('call_count', calls if PROFILE else 'not_counted')}"
        )
        return r.frames[0], total, denoise, decode, calls

    log(f"=== warmup ({WARM} pass) ===")
    warmup_samples = []
    for _ in range(max(0, WARM)):
        _, warm_total, warm_den, warm_dec, _ = gen(prompts[0], "warmup")
        warmup_samples.append(
            {"total_s": warm_total, "denoise_s": warm_den, "decode_s": warm_dec}
        )

    integrated_generation_pair = None
    if INTEGRATED_PAIR:
        pair_samples = {"off_warmups": [], "on_warmups": []}

        def pair_generation(enabled, tag):
            kernel_runtime.set_composed_stack(enabled)
            frames, total, denoise, decode, _ = gen(prompts[0], tag)
            del frames
            return {"total_s": total, "denoise_s": denoise, "decode_s": decode}

        log("=== integrated full-generation pair: OFF warmups ===")
        for index in range(2):
            pair_samples["off_warmups"].append(
                pair_generation(False, f"pair_off_warmup_{index}")
            )
        off_measured = pair_generation(False, "pair_off_measured")

        log("=== integrated full-generation pair: ON warmups ===")
        for index in range(2):
            pair_samples["on_warmups"].append(
                pair_generation(True, f"pair_on_warmup_{index}")
            )
        on_measured = pair_generation(True, "pair_on_measured")
        kernel_runtime.set_composed_stack(True)
        integrated_generation_pair = {
            "schema_version": 1,
            "comparison_scope": "full_composed_stack_OFF_eager_unpacked_uncached_vs_ON_compiled_packed_exact_cache",
            "timing_scope": "warm_single_prompt_generation_after_two_excluded_warmups_per_state",
            "prompt_count_per_state": 1,
            "steps_per_prompt": STEPS,
            "model_calls_per_step": 2 if G > 1.0 else 1,
            "dit_calls_per_prompt": STEPS * (2 if G > 1.0 else 1),
            "blocks_per_dit": len(pipe.transformer.blocks),
            "off": off_measured,
            "on": on_measured,
            "warmups": pair_samples,
            "speedup": {
                "total": off_measured["total_s"] / on_measured["total_s"],
                "denoise": off_measured["denoise_s"] / on_measured["denoise_s"],
            },
            "output_comparison_used": False,
            "generation_output_dependency": False,
        }
        (out / "integrated_generation_pair.json").write_text(
            json.dumps(integrated_generation_pair, indent=2)
        )
        log(
            "integrated generation pair "
            f"OFF={off_measured['total_s']:.2f}s ON={on_measured['total_s']:.2f}s "
            f"speedup={integrated_generation_pair['speedup']['total']:.4f}x"
        )

    paired_dit = kernel_runtime.run_paired_dit_benchmark()
    if paired_dit is not None:
        log(
            f"paired DiT {paired_dit['candidate']} OFF={paired_dit['off']['median_ms']:.3f}ms "
            f"ON={paired_dit['on']['median_ms']:.3f}ms "
            f"speedup={paired_dit['median_speedup']:.4f}x"
        )
    torch.cuda.reset_peak_memory_stats()

    log("=== measured pass ===")
    samples = []
    for i, p in enumerate(prompts):
        frames, total, den, dec, calls = gen(p, f"p{i}")
        samples.append(
            {
                "prompt_id": f"p{i}",
                "total_s": total,
                "denoise_s": den,
                "decode_s": dec,
                "dit_calls": calls if PROFILE else STEPS * 2,
            }
        )
        mp4 = out / "videos" / f"prompt_{i:02d}.mp4"
        export_to_video(frames, str(mp4), fps=FPS)
        if i == 0:
            import shutil
            shutil.copy(mp4, out / "out.mp4")
            try:
                import numpy as np
                from PIL import Image
                for j, fr in enumerate(frames):
                    a = np.asarray(fr)
                    if a.dtype != np.uint8:
                        a = (a * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(a).save(out / "frames" / f"f_{j:05d}.png")
            except Exception as e:
                log(f"frame dump skipped (non-fatal): {e!r}")

    def med(key):
        vals = [s[key] for s in samples if s[key] is not None]
        return statistics.median(vals) if vals else None

    tot, den, dec = med("total_s"), med("denoise_s"), med("decode_s")
    bench = {
        "schema_version": 2,
        "model_id": label,
        "pipeline": "WanPipeline",
        "total_s": tot,
        "denoise_s": den,
        "decode_s": dec,
        "timing_scope": "text_to_video_hot_after_warmup_pass",
        "warm_steady_state": True,
        "baseline_class": "pristine_unoptimized",
        "config": {
            "model": label, "weights": weights, "height": H, "width": W,
            "num_frames": F, "steps": STEPS, "guidance_scale": G, "guidance_scale_2": G2,
            "flow_shift": SHIFT, "fps": FPS, "seed": SEED, "num_gpus": 1,
            "moe": moe, "prompt_count": len(prompts), "warmup_passes": WARM,
            "model_calls_per_step": 2, "dit_calls_per_prompt": STEPS * 2,
            "kernel_profile_enabled": PROFILE,
        },
        "aggregate": {"total_s": tot, "denoise_s": den, "decode_s": dec,
                      "prompt_count": len(prompts), "reduction": "median"},
        "samples": samples,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }
    if profile_result is not None:
        bench["kernel_profile"] = "kernel_profile.json"
        bench["profile_timing_warning"] = "Profiled timing includes Kineto overhead and is not a speed candidate."
    kernel_summary = kernel_runtime.finalize()
    cache_summary = cache_runtime.summary() if cache_runtime is not None else None
    bench["kernel_runtime"] = kernel_summary
    bench["cache_method"] = cache_summary
    bench["warmup_samples"] = warmup_samples
    bench["paired_dit_benchmark"] = paired_dit
    bench["integrated_generation_pair"] = integrated_generation_pair
    bench["memory"] = {
        "measured_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "measured_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    (out / "benchmark.json").write_text(json.dumps(bench, indent=2))
    (out / "timing_accounting.json").write_text(
        json.dumps(
            {
                "timing_scope": "warm_generation_after_two_excluded_warmups",
                "warmup_samples": warmup_samples,
                "paired_dit_benchmark": paired_dit,
                "integrated_generation_pair": integrated_generation_pair,
                "kernel_activation": kernel_runtime.activation,
                "cache_method": cache_summary,
                "prompt_count": len(prompts),
                "steps_per_prompt": STEPS,
                "model_calls_per_step": 2 if G > 1.0 else 1,
                "dit_calls_per_prompt": STEPS * (2 if G > 1.0 else 1),
                "measured_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "measured_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            indent=2,
        )
    )
    (out / "run_config.json").write_text(json.dumps(bench["config"], indent=2))
    log(f"=== DONE === median total_s={tot:.2f}s denoise_s={den and round(den,2)} over {len(samples)} prompt(s)")
    if cache_summary is not None:
        log(f"cache totals: {cache_summary['totals']}")
    log(f"artifacts in {out}")

    for handle in hook_handles:
        handle.remove()


if __name__ == "__main__":
    main()
