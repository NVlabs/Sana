#!/usr/bin/env python3
"""Bernini t2v baseline driver.

Drives the vendored, self-contained Bernini code under
`runtime/bernini_baseline/bernini_src/` (a clean copy of the upstream pristine
tree). It runs the official full-Bernini t2v recipe over the 5-prompt
validation set with a warmup pass + a measured pass, then normalizes the
measured (hot) calls into the standard run bundle: out.mp4 (+ per-prompt
videos), frames/, benchmark.json (schema 2), run_config.json.

Self-containment: the CODE is in-repo (bernini_src). Only two large external
assets are referenced by absolute path (they cannot live in git): the model
WEIGHTS ($BERNINI_WEIGHTS) passed to `--config`, and the vendored third-party
python libs ($BERNINI_DEPS) added to PYTHONPATH. Nothing is symlinked into the
code tree.

Invoked by scripts/run_bernini_gpu.sh (launch.sh exports OUT_DIR + [env]).
Stdlib-only; shells out to $PYTHON_BIN for the torch work.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median


RUNTIME_DIR = Path(__file__).resolve().parent          # runtime/bernini_baseline
REPO_ROOT = RUNTIME_DIR.parent.parent                  # Sol-LTX-Infer (or experiment worktree)
BERNINI_SRC = RUNTIME_DIR / "bernini_src"              # vendored pristine code (in-repo)
HOT_DRIVER = RUNTIME_DIR / "bernini_hot_infer.py"


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and (value is None or value == ""):
        raise SystemExit(f"[bernini_baseline] missing required env: {name}")
    return "" if value is None else str(value)


def find_ffmpeg() -> str | None:
    cand = env("BERNINI_FFMPEG")
    if cand and Path(cand).exists():
        return cand
    return shutil.which("ffmpeg")


def extract_frames(video: Path, frames_dir: Path) -> int:
    ff = find_ffmpeg()
    if ff is None:
        print("[bernini_baseline] WARN: no ffmpeg; skipping frame extraction", flush=True)
        return 0
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", str(video),
         "-start_number", "1", str(frames_dir / "f_%05d.png")],
        check=False,
    )
    return len(list(frames_dir.glob("f_*.png")))


def parse_hot_timing_stdout(text: str) -> list[dict]:
    pat = re.compile(
        r"\[HOT_TIMING\]\s+call=(?P<call>\d+)\s+vit_mllm=(?P<vit>[\d.]+)s\s+"
        r"t5=(?P<t5>[\d.]+)s\s+diffusion=(?P<diff>[\d.]+)s\s+vae_decode=(?P<vae>[\d.]+)s\s+"
        r"text_to_vae_decode=(?P<t2v>[\d.]+|n/a)s?\s+pipeline_total_with_save=(?P<total>[\d.]+)s"
    )
    out = []
    for m in pat.finditer(text):
        t2v = m.group("t2v")
        out.append({
            "call_id": int(m.group("call")),
            "vit_mllm": float(m.group("vit")),
            "t5": float(m.group("t5")),
            "diffusion": float(m.group("diff")),
            "vae_decode": float(m.group("vae")),
            "text_to_vae_decode": None if t2v == "n/a" else float(t2v),
            "pipeline_total_with_save": float(m.group("total")),
        })
    return out


def load_val_prompts() -> list[dict]:
    override = env("BERNINI_VAL_PROMPTS")
    path = Path(override) if override else (REPO_ROOT / "models/bernini/prompts/t2v_val5.json")
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise SystemExit(f"[bernini_baseline] validation prompts not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise SystemExit(f"[bernini_baseline] bad validation prompts file: {path}")
    return data


def med(vals: list[float]) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return median(vals) if vals else None


def main() -> int:
    out_dir = Path(env("OUT_DIR", required=True)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not BERNINI_SRC.is_dir():
        raise SystemExit(f"[bernini_baseline] vendored pristine code missing: {BERNINI_SRC}")
    if not HOT_DRIVER.exists():
        raise SystemExit(f"[bernini_baseline] timing driver missing: {HOT_DRIVER}")

    python_bin = env("PYTHON_BIN", "/usr/bin/python3.12")
    weights = env("BERNINI_WEIGHTS", required=True)     # abs path to ByteDance/Bernini-Diffusers
    if not Path(weights).is_dir():
        raise SystemExit(f"[bernini_baseline] weights dir not found: {weights}")
    deps = env("BERNINI_DEPS")                          # colon-sep abs paths (.gpu-site-pure:.gpu-site)
    nproc = env("BERNINI_NPROC_PER_NODE", "4")
    ulysses = env("BERNINI_ULYSSES", "4")
    warmup_passes = int(env("BERNINI_WARMUP_PASSES", "1"))

    sampling = [
        ("--num_frames", env("BERNINI_NUM_FRAMES", "81")),
        ("--max_image_size", env("BERNINI_MAX_IMAGE_SIZE", "842")),
        ("--height", env("BERNINI_HEIGHT", "480")),
        ("--width", env("BERNINI_WIDTH", "848")),
        ("--num_inference_steps", env("BERNINI_STEPS", "50")),
        ("--flow_shift", env("BERNINI_FLOW_SHIFT", "5.0")),
        ("--seed", env("BERNINI_SEED", "42")),
        ("--fps", env("BERNINI_FPS", "16")),
        ("--omega_txt", env("BERNINI_OMEGA_TXT", "4")),
        ("--omega_tgt", env("BERNINI_OMEGA_TGT", "0.5")),
        ("--omega_img", env("BERNINI_OMEGA_IMG", "1")),
        ("--omega_vid", env("BERNINI_OMEGA_VID", "1")),
        ("--omega_scale", env("BERNINI_OMEGA_SCALE", "1")),
        ("--vit_denoising_step", env("BERNINI_VIT_DENOISING_STEP", "5")),
        ("--vit_txt_cfg", env("BERNINI_VIT_TXT_CFG", "1.2")),
        ("--vit_img_cfg", env("BERNINI_VIT_IMG_CFG", "1.0")),
        ("--guidance_mode", env("BERNINI_GUIDANCE_MODE", "vae_txt_vit_wapg")),
    ]

    prompts = load_val_prompts()
    scratch = out_dir / "bernini_raw"
    videos_dir = out_dir / "videos"
    scratch.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # warmup pass(es) over all prompts, then one measured pass.
    inputs: list[dict] = []
    for w in range(max(warmup_passes, 0)):
        for p in prompts:
            inputs.append({"task_type": p.get("task_type", "t2v"), "prompt": p["prompt"],
                           "output": str(scratch / f"warmup{w}_{p['prompt_id']}.mp4")})
    measure_start = len(inputs)
    for p in prompts:
        inputs.append({"task_type": p.get("task_type", "t2v"), "prompt": p["prompt"],
                       "output": str(videos_dir / f"{p['prompt_id']}.mp4")})
    inputs_path = scratch / "t2v_inputs.json"
    inputs_path.write_text(json.dumps(inputs, indent=2, ensure_ascii=False))
    timing_json = scratch / "timing.json"

    cmd = [python_bin, "-m", "torch.distributed.run", "--standalone",
           "--nproc-per-node", str(nproc), str(HOT_DRIVER),
           "--config", weights, "--ulysses", str(ulysses)]
    for flag, val in sampling:
        cmd += [flag, str(val)]
    cmd += ["--inputs", str(inputs_path)]

    run_env = os.environ.copy()
    pypath = [str(BERNINI_SRC)]
    if deps:
        pypath = deps.split(":") + pypath
    run_env["PYTHONPATH"] = ":".join(pypath)
    run_env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    for key, sub in [("HF_HOME", ".cache/hf"), ("TORCH_HOME", ".cache/torch"),
                     ("TORCHINDUCTOR_CACHE_DIR", ".cache/inductor"),
                     ("TORCH_EXTENSIONS_DIR", ".cache/torch_ext"),
                     ("TRITON_CACHE_DIR", ".cache/triton"), ("XDG_CACHE_HOME", ".cache/xdg")]:
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        run_env[key] = str(d)
    (out_dir / ".tmp").mkdir(parents=True, exist_ok=True)
    run_env["TMPDIR"] = run_env["TMP"] = run_env["TEMP"] = str(out_dir / ".tmp")
    run_env["BERNINI_TIMING_JSON"] = str(timing_json)
    run_env.setdefault("FSDP", "1")
    run_env.setdefault("NCCL_NET_PLUGIN", "none")

    print(f"[bernini_baseline] PRISTINE unoptimized baseline", flush=True)
    print(f"[bernini_baseline] src={BERNINI_SRC}", flush=True)
    print(f"[bernini_baseline] weights={weights}", flush=True)
    print(f"[bernini_baseline] {len(prompts)} prompts x (warmup {warmup_passes} + measure 1) "
          f"= {len(inputs)} calls; measured = last {len(prompts)}", flush=True)
    print("[bernini_baseline] cmd=" + " ".join(shlex.quote(c) for c in cmd), flush=True)

    proc = subprocess.Popen(cmd, cwd=str(BERNINI_SRC), env=run_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"[bernini_baseline] torchrun failed rc={proc.returncode}")

    records = None
    if timing_json.exists():
        try:
            records = json.loads(timing_json.read_text())
        except (OSError, json.JSONDecodeError):
            records = None
    if not records:
        records = parse_hot_timing_stdout("".join(captured))
    if not records or len(records) < measure_start + 1:
        raise SystemExit("[bernini_baseline] insufficient hot-timing records captured")

    measured = records[measure_start:measure_start + len(prompts)]
    per_prompt = []
    for p, rec in zip(prompts, measured):
        per_prompt.append({"prompt_id": p["prompt_id"], **{k: rec.get(k) for k in
                          ("text_to_vae_decode", "diffusion", "vae_decode", "t5",
                           "vit_mllm", "pipeline_total_with_save")}})

    total_s = med([r["text_to_vae_decode"] for r in per_prompt])
    denoise_s = med([r["diffusion"] for r in per_prompt])
    decode_s = med([r["vae_decode"] for r in per_prompt])
    t5_s = med([r["t5"] for r in per_prompt])
    vit_s = med([r["vit_mllm"] for r in per_prompt])
    save_s = med([r["pipeline_total_with_save"] for r in per_prompt])

    # canonical single reference video = first prompt (polar bear)
    ref_pid = prompts[0]["prompt_id"]
    ref_video = videos_dir / f"{ref_pid}.mp4"
    frame_count = 0
    if ref_video.exists():
        shutil.copy2(ref_video, out_dir / "out.mp4")
        frame_count = extract_frames(out_dir / "out.mp4", out_dir / "frames")
    else:
        print(f"[bernini_baseline] WARN: reference video missing: {ref_video}", flush=True)

    benchmark = {
        "schema_version": 2,
        "total_s": total_s,
        "denoise_s": denoise_s,
        "decode_s": decode_s,
        "text_encoder_s": t5_s,
        "vit_mllm_s": vit_s,
        "pipeline_total_with_save_s": save_s,
        "timing_scope": "text_to_vae_decode_hot_after_warmup_pass",
        "warm_steady_state": True,
        "baseline_class": "pristine_unoptimized",
        "timings": {"generate_s": total_s, "wall_total_s": total_s,
                    "diffusion_s": denoise_s, "vae_decode_s": decode_s,
                    "t5_s": t5_s, "vit_mllm_s": vit_s},
        "stage_seconds": {"vit_mllm": vit_s, "t5": t5_s, "diffusion": denoise_s,
                          "vae_decode": decode_s, "text_to_vae_decode": total_s,
                          "pipeline_total_with_save": save_s},
        "aggregate": {"total_s": total_s, "sample_mean_s": total_s, "denoise_s": denoise_s,
                      "decode_s": decode_s, "text_encoder_s": t5_s,
                      "prompt_count": len(prompts), "warmup_passes": warmup_passes,
                      "reduction": "median_over_validation_set"},
        "samples": per_prompt,
        "config": {
            "model": "bernini_diffusers", "task_type": "t2v",
            "num_frames": int(env("BERNINI_NUM_FRAMES", "81")),
            "height": int(env("BERNINI_HEIGHT", "480")),
            "width": int(env("BERNINI_WIDTH", "848")),
            "fps": int(env("BERNINI_FPS", "16")),
            "steps": int(env("BERNINI_STEPS", "50")),
            "flow_shift": float(env("BERNINI_FLOW_SHIFT", "5.0")),
            "seed": int(env("BERNINI_SEED", "42")),
            "guidance_mode": env("BERNINI_GUIDANCE_MODE", "vae_txt_vit_wapg"),
            "num_gpus": int(nproc), "ulysses": int(ulysses),
            "validation_set": [p["prompt_id"] for p in prompts],
            "frame_count_extracted": frame_count,
        },
    }
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (out_dir / "run_config.json").write_text(json.dumps(
        {"weights": weights, "bernini_src": str(BERNINI_SRC), "cmd": cmd,
         "validation_set": [p["prompt_id"] for p in prompts], "all_call_records": records},
        indent=2) + "\n")

    print(f"[bernini_baseline] DONE (median over {len(prompts)} prompts): "
          f"text_to_vae_decode={total_s} denoise={denoise_s} decode={decode_s} "
          f"text_enc={t5_s} planner={vit_s} frames={frame_count}", flush=True)
    for r in per_prompt:
        print(f"  - {r['prompt_id']}: t2v_decode={r['text_to_vae_decode']} "
              f"diffusion={r['diffusion']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
