#!/usr/bin/env python3
"""Sana 5B video baseline wrapper.

This script runs the private Hugging Face `yitongl/sana_video` minimal inference
bundle and normalizes its output to the autovideo artifact contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DATASET_REPO = "yitongl/sana_video"
BUNDLE_FILE = "standalone/sana5b_720p193_minimal_infer.zip"
BUNDLE_DIRNAME = "sana5b_720p193_minimal_infer"
CHECKPOINT_FILE = (
    "Sana_5B_480px_QwenNext_ltxvae23_selfflow_pertoken_subattnres_v2_multires_multifps_sft/"
    "checkpoints/epoch_7_step_2107015/model_ema.pth"
)
DEFAULT_PROMPT = (
    "A coastal city at sunrise, warm light reflecting on glass towers and calm "
    "water, pedestrians crossing a waterfront promenade, cinematic camera motion."
)


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    for raw in (
        os.environ.get("HF_TOKEN_PATH", ""),
        str(Path.home() / ".cache/huggingface/token"),
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path.read_text().strip()
    return ""


def hf_dataset_url(filename: str) -> str:
    quoted = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{quoted}"


def download_dataset_file(filename: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(hf_dataset_url(filename), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(dst)


def asset_root() -> Path:
    raw = os.environ.get("SANA_VIDEO_ASSET_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache/huggingface")
    return (Path(hf_home).expanduser() / "sana_video").resolve()


def config_base() -> Path:
    for key in ("AUTOVIDEO_REPO_ROOT", "AUTOVIDEO_RUNTIME_ROOT"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def resolve_config_path(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def ensure_bundle(root: Path) -> Path:
    raw_bundle = os.environ.get("SANA_VIDEO_BUNDLE_ROOT")
    bundle_root = (
        resolve_config_path(raw_bundle, config_base())
        if raw_bundle
        else (root / BUNDLE_DIRNAME).resolve()
    )
    if (bundle_root / "scripts/run_infer.sh").exists():
        return bundle_root

    raw_reference = os.environ.get("SANA_VIDEO_REFERENCE_BUNDLE_ROOT")
    if raw_reference:
        reference_root = resolve_config_path(raw_reference, config_base())
        if (reference_root / "scripts/run_infer.sh").exists():
            return reference_root

    if not env_flag("SANA_VIDEO_AUTO_DOWNLOAD_BUNDLE", "1"):
        raise SystemExit(
            f"Sana bundle missing at {bundle_root}. Set SANA_VIDEO_BUNDLE_ROOT or "
            "SANA_VIDEO_AUTO_DOWNLOAD_BUNDLE=1."
        )

    zip_path = Path(os.environ.get("SANA_VIDEO_BUNDLE_ZIP") or root / BUNDLE_FILE).expanduser().resolve()
    if not zip_path.exists():
        print(f"[asset] downloading {DATASET_REPO}:{BUNDLE_FILE} -> {zip_path}", flush=True)
        download_dataset_file(BUNDLE_FILE, zip_path)

    print(f"[asset] extracting {zip_path} -> {root}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    if not (bundle_root / "scripts/run_infer.sh").exists():
        raise SystemExit(f"Extracted bundle did not create expected run script: {bundle_root}")
    return bundle_root


def materialize_symlink(src: Path, dst: Path, required_child: str | None = None) -> None:
    if dst.exists() or dst.is_symlink():
        if required_child is None or (dst / required_child).exists():
            return
        if dst.is_dir() and not any(dst.iterdir()):
            dst.rmdir()
        else:
            raise SystemExit(
                f"Asset destination exists but is incomplete: {dst}. "
                f"Expected child: {required_child}"
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def ensure_assets(bundle_root: Path) -> tuple[Path, Path]:
    ckpt = Path(os.environ.get("SANA_VIDEO_CKPT") or bundle_root / "models/checkpoints/model_ema.pth")
    ckpt = ckpt.expanduser().resolve()
    vae_root = Path(os.environ.get("SANA_VIDEO_VAE_ROOT") or bundle_root / "models/LTX-2.3-Diffusers")
    vae_root = vae_root.expanduser().resolve()

    if env_flag("SANA_VIDEO_PREPARE_ASSETS", "0"):
        script = bundle_root / "scripts/download_assets.py"
        link_mode = os.environ.get("SANA_VIDEO_ASSET_LINK_MODE", "symlink")
        cmd = [sys.executable, str(script), "--root", str(bundle_root), "--link-mode", link_mode]
        print("[asset] preparing checkpoint/VAE via bundled downloader", flush=True)
        subprocess.run(cmd, cwd=bundle_root, check=True)
        ckpt = Path(os.environ.get("SANA_VIDEO_CKPT") or bundle_root / "models/checkpoints/model_ema.pth").resolve()
        vae_root = Path(os.environ.get("SANA_VIDEO_VAE_ROOT") or bundle_root / "models/LTX-2.3-Diffusers").resolve()

    if os.environ.get("SANA_VIDEO_VAE_ROOT"):
        materialize_symlink(
            vae_root,
            bundle_root / "models/LTX-2.3-Diffusers",
            "vae/config.json",
        )

    missing = []
    if not ckpt.exists():
        missing.append(f"checkpoint {ckpt}")
    if not (bundle_root / "models/LTX-2.3-Diffusers/vae/config.json").exists():
        missing.append(f"VAE {bundle_root / 'models/LTX-2.3-Diffusers/vae'}")
    if missing:
        raise SystemExit(
            "Missing Sana assets: "
            + "; ".join(missing)
            + ". Set SANA_VIDEO_PREPARE_ASSETS=1, or set SANA_VIDEO_CKPT and "
            "SANA_VIDEO_VAE_ROOT to existing shared assets. Checkpoint source: "
            + f"{DATASET_REPO}:{CHECKPOINT_FILE}"
        )
    return ckpt, bundle_root / "models/LTX-2.3-Diffusers"


def write_prompt_file(bundle_root: Path) -> Path:
    raw = os.environ.get("SANA_VIDEO_PROMPTS")
    if raw:
        path = resolve_config_path(raw, config_base())
        if not path.exists():
            raise SystemExit(f"SANA_VIDEO_PROMPTS does not exist: {path}")
        return path
    prompt = os.environ.get("PROMPT") or os.environ.get("SANA_VIDEO_PROMPT") or DEFAULT_PROMPT
    path = bundle_root / "prompts/autovideo_prompt.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt.rstrip() + "\n")
    return path


def run_bundle(bundle_root: Path, out_dir: Path, ckpt: Path, prompts: Path) -> tuple[float, dict]:
    raw_out = out_dir / "sana_raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    hot_timing_path = raw_out / "hot-timing.json"
    hot_timing_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": os.environ.get("SANA_VIDEO_INFER_PYTHON") or sys.executable,
            "PATH": f"{Path(sys.executable).parent}:{env.get('PATH', '')}",
            "CKPT": str(ckpt),
            "PROMPTS": str(prompts),
            "OUT_DIR": str(raw_out),
            "NP": os.environ.get("SANA_VIDEO_NP", os.environ.get("NP", "1")),
            "SAMPLE_NUMS": os.environ.get("SANA_VIDEO_SAMPLE_NUMS", os.environ.get("SAMPLE_NUMS", "1")),
            "NUM_FRAMES": os.environ.get("SANA_VIDEO_NUM_FRAMES", "193"),
            "FPS": os.environ.get("SANA_VIDEO_FPS", "24"),
            "IMAGE_SIZE": os.environ.get("SANA_VIDEO_IMAGE_SIZE", "720"),
            "CFG_SCALE": os.environ.get("SANA_VIDEO_CFG_SCALE", "8"),
            "FLOW_SHIFT": os.environ.get("SANA_VIDEO_FLOW_SHIFT", "12"),
            "STEP": os.environ.get("SANA_VIDEO_STEPS", "50"),
            "MOTION_SCORE": os.environ.get("SANA_VIDEO_MOTION_SCORE", "20"),
            "DATASET": os.environ.get("SANA_VIDEO_DATASET", "sana5b_minimal"),
            "FORWARD_CACHE_METHOD": os.environ.get("FORWARD_CACHE_METHOD", "none"),
            "SANA_HOT_BENCHMARK": os.environ.get("SANA_HOT_BENCHMARK", "1"),
            "SANA_HOT_WARMUP_SAMPLES": os.environ.get("SANA_HOT_WARMUP_SAMPLES", "1"),
            "SANA_HOT_TIMING_OUTPUT": str(hot_timing_path),
        }
    )
    start = time.perf_counter()
    subprocess.run(["bash", "scripts/run_infer.sh"], cwd=bundle_root, env=env, check=True)
    process_wall_s = time.perf_counter() - start
    try:
        hot_timing = json.loads(hot_timing_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Sana inference did not produce a valid hot timing artifact: {hot_timing_path}") from exc
    timing = hot_timing.get("timing") if isinstance(hot_timing, dict) else {}
    aggregate = hot_timing.get("aggregate") if isinstance(hot_timing, dict) else {}
    if (
        timing.get("scope") != "warm_single_sample_text_encoder_through_vae_decode"
        or timing.get("warm_steady_state") is not True
        or not isinstance(aggregate.get("sample_total_s"), (int, float))
    ):
        raise SystemExit(f"Sana hot timing artifact has an invalid contract: {hot_timing_path}")
    return process_wall_s, hot_timing


def find_generated_video(out_dir: Path) -> Path:
    videos = sorted((out_dir / "sana_raw").glob("vis/**/*.mp4"))
    if not videos:
        raise SystemExit(f"Sana inference completed but no mp4 was found under {out_dir / 'sana_raw/vis'}")
    return videos[0]


def copy_video_and_frames(video: Path, out_dir: Path, fps: int) -> int:
    target = out_dir / "out.mp4"
    if video.resolve() != target.resolve():
        shutil.copy2(video, target)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        count = 0
        for idx, frame in enumerate(iio.imiter(target), start=1):
            iio.imwrite(frames_dir / f"f_{idx:05d}.png", frame)
            count = idx
        return count
    except Exception as exc:
        print(f"[warn] frame extraction deferred to collect_run.py: {type(exc).__name__}: {exc}", flush=True)
        return 0


def main() -> int:
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = asset_root()
    bundle_root = ensure_bundle(root)
    ckpt, vae_root = ensure_assets(bundle_root)
    prompts = write_prompt_file(bundle_root)

    process_wall_s, hot_timing = run_bundle(bundle_root, out_dir, ckpt, prompts)
    video = find_generated_video(out_dir)
    fps = int(float(os.environ.get("SANA_VIDEO_FPS", "24")))
    frame_count = copy_video_and_frames(video, out_dir, fps)
    sample_nums = int(float(os.environ.get("SANA_VIDEO_SAMPLE_NUMS", os.environ.get("SAMPLE_NUMS", "1"))))

    aggregate = hot_timing["aggregate"]
    timing_contract = dict(hot_timing["timing"])
    timing_contract.update({"prompt_count": sample_nums, "stage_isolated": True})
    benchmark = {
        "schema_version": 2,
        "total_s": aggregate["sample_total_s"],
        "mean_s": aggregate["sample_mean_s"],
        "text_encoder_s": aggregate["text_encoder_s"],
        "denoise_s": aggregate["denoise_s"],
        "decode_s": aggregate["vae_decode_s"],
        "timings": {
            "generate_s": aggregate["sample_total_s"],
            "wall_total_s": aggregate["sample_total_s"],
            "process_wall_s": process_wall_s,
        },
        "stage_seconds": {
            "text_encoder": aggregate["text_encoder_s"],
            "denoise": aggregate["denoise_s"],
            "vae_decode": aggregate["vae_decode_s"],
        },
        "aggregate": aggregate,
        "samples": hot_timing["samples"],
        "timing": timing_contract,
        "timing_scope": timing_contract["scope"],
        "timing_contract": timing_contract,
        "diagnostics": {"bundled_process_wall_s": process_wall_s},
        "config": {
            "model": "sana_video_5b",
            "dataset_repo": DATASET_REPO,
            "bundle_file": BUNDLE_FILE,
            "checkpoint_file": CHECKPOINT_FILE,
            "checkpoint_path": str(ckpt),
            "vae_root": str(vae_root),
            "num_frames": int(float(os.environ.get("SANA_VIDEO_NUM_FRAMES", "193"))),
            "fps": fps,
            "image_size": int(float(os.environ.get("SANA_VIDEO_IMAGE_SIZE", "720"))),
            "steps": int(float(os.environ.get("SANA_VIDEO_STEPS", "50"))),
            "sample_nums": sample_nums,
            "cfg_scale": float(os.environ.get("SANA_VIDEO_CFG_SCALE", "8")),
            "flow_shift": float(os.environ.get("SANA_VIDEO_FLOW_SHIFT", "12")),
            "motion_score": int(float(os.environ.get("SANA_VIDEO_MOTION_SCORE", "20"))),
            "forward_cache_method": os.environ.get("FORWARD_CACHE_METHOD", "none"),
            "prompts_path": str(prompts),
        },
    }
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    (out_dir / "run_config.json").write_text(json.dumps(benchmark["config"], indent=2, sort_keys=True) + "\n")
    print(
        f"[sana-video] DONE hot_mean_s={aggregate['sample_mean_s']:.3f} "
        f"hot_total_s={aggregate['sample_total_s']:.3f} process_wall_s={process_wall_s:.3f} "
        f"video={out_dir / 'out.mp4'} "
        f"frames={frame_count or 'deferred'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
