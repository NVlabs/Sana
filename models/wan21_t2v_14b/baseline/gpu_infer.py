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

    def gen(prompt, tag):
        g = torch.Generator(device="cuda").manual_seed(SEED)
        step_t = []

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
        try:
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
        # denoise ~= span across step callbacks; decode ~= time after last step.
        denoise = (step_t[-1] - step_t[0]) if len(step_t) >= 2 else None
        decode = (time.time() - step_t[-1]) if step_t else None
        log(f"[{tag}] total={total:.2f}s denoise~{denoise and round(denoise,2)} decode~{decode and round(decode,2)}")
        return r.frames[0], total, denoise, decode

    log(f"=== warmup ({WARM} pass) ===")
    for _ in range(max(0, WARM)):
        gen(prompts[0], "warmup")

    log("=== measured pass ===")
    samples = []
    for i, p in enumerate(prompts):
        frames, total, den, dec = gen(p, f"p{i}")
        samples.append({"prompt_id": f"p{i}", "total_s": total, "denoise_s": den, "decode_s": dec})
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
        },
        "aggregate": {"total_s": tot, "denoise_s": den, "decode_s": dec,
                      "prompt_count": len(prompts), "reduction": "median"},
        "samples": samples,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }
    (out / "benchmark.json").write_text(json.dumps(bench, indent=2))
    (out / "run_config.json").write_text(json.dumps(bench["config"], indent=2))
    log(f"=== DONE === median total_s={tot:.2f}s denoise_s={den and round(den,2)} over {len(samples)} prompt(s)")
    log(f"artifacts in {out}")


if __name__ == "__main__":
    main()
