"""Unified evaluation script for world-model benchmark.

Evaluates generated videos across multiple metrics:
1. VBench visual quality (benchmark custom-input dimensions + total score)
2. Camera accuracy summary if Pi3 pose results already exist
3. Revisit frame consistency (PSNR/SSIM between evaluation pairs)
4. Temporal degradation (VBench on sliding windows)

## Expected Directory Structure

Each method produces results in this layout:
```
results/
├── <method_name>/
│   ├── simple_60s/
│   │   ├── game_style_001_generated.mp4
│   │   ├── indoor_001_generated.mp4
│   │   └── ...
│   ├── hard_60s/
│   │   ├── game_style_001_generated.mp4
│       └── ...
```

Evaluation results are saved to:
```
results/<method_name>/eval/
├── <split>/
│   ├── camera_accuracy.json         # Pi3 pose metrics per scene, if present
│   ├── vbench_scores.json           # VBench dimensions + total score
│   ├── revisit_consistency.json     # PSNR/SSIM per evaluation pair
│   ├── temporal_degradation.json    # VBench per 10s window
│   └── summary.json                 # Aggregated summary
```

## Usage

```bash
# Evaluate everything (requires GPU)
python tools/metrics/sana_wm/eval_unified.py \\
    --method_dir results/sana_wm_v1 \\
    --split simple_60s

# Specific metrics only
python tools/metrics/sana_wm/eval_unified.py \\
    --method_dir results/sana_wm_v1 \\
    --split simple_60s \\
    --metrics vbench revisit
```
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def _manifest_path_from_meta(meta_path: str) -> str | None:
    """Infer the matching run_manifest.jsonl path from scene_trajectories_v2.json."""
    candidate = Path(meta_path).parent / "sanawm_export_v2" / "run_manifest.jsonl"
    if candidate.exists():
        return str(candidate)
    return None


def _load_manifest_prompts(manifest_path: str | None) -> dict[str, str]:
    """Load scene prompts from a SANA-WM run manifest."""
    prompts: dict[str, str] = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return prompts
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = row.get("id")
            prompt = row.get("prompt")
            if scene_id and prompt:
                prompts[scene_id] = prompt
    return prompts


def _resolve_vbench_info_path(vbench_module) -> str:
    """Find VBench_full_info.json for either source-tree or pip-installed VBench."""
    candidates = [
        Path(PROJECT_ROOT) / "local_libs" / "VBench" / "vbench" / "VBench_full_info.json",
        Path(vbench_module.__file__).resolve().parent / "VBench_full_info.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not find VBench_full_info.json. Install VBench or provide local_libs/VBench.")


# ============================================================================
# VBench dimension definitions (all 16 from VBench 1.0)
# ============================================================================

VBENCH_QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
]

VBENCH_SEMANTIC_DIMS = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]

VBENCH_ALL_DIMS = VBENCH_QUALITY_DIMS + VBENCH_SEMANTIC_DIMS

# I2V-specific dimensions
VBENCH_I2V_DIMS = [
    "i2v_subject",
    "i2v_background",
]

# Official weights (from VBench constant.py)
VBENCH_DIM_WEIGHTS = {d: 1.0 for d in VBENCH_ALL_DIMS}
VBENCH_DIM_WEIGHTS["dynamic_degree"] = 0.5
QUALITY_WEIGHT = 4.0
SEMANTIC_WEIGHT = 1.0

# Normalization bounds (from VBench constant.py)
NORMALIZE_DIC = {
    "subject_consistency": {"Min": 0.1462, "Max": 1.0},
    "background_consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal_flickering": {"Min": 0.6293, "Max": 1.0},
    "motion_smoothness": {"Min": 0.7060, "Max": 0.9975},
    "dynamic_degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic_quality": {"Min": 0.0, "Max": 1.0},
    "imaging_quality": {"Min": 0.0, "Max": 1.0},
    "object_class": {"Min": 0.0, "Max": 1.0},
    "multiple_objects": {"Min": 0.0, "Max": 1.0},
    "human_action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial_relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 1.0},
    "appearance_style": {"Min": 0.0, "Max": 1.0},
    "temporal_style": {"Min": 0.0, "Max": 1.0},
    "overall_consistency": {"Min": 0.0, "Max": 1.0},
    "i2v_subject": {"Min": 0.0, "Max": 1.0},
    "i2v_background": {"Min": 0.0, "Max": 1.0},
}


def normalize_score(dim: str, raw: float) -> float:
    """Normalize a raw VBench score to [0, 1] using official bounds."""
    if dim not in NORMALIZE_DIC:
        return raw
    mn = NORMALIZE_DIC[dim]["Min"]
    mx = NORMALIZE_DIC[dim]["Max"]
    if mx - mn < 1e-8:
        return raw
    return max(0.0, min(1.0, (raw - mn) / (mx - mn)))


def compute_vbench_total(scores: dict[str, float]) -> dict[str, float]:
    """Compute VBench quality, semantic, and total scores.

    Follows the official VBench formula:
      Quality = weighted_avg(7 quality dims, dynamic_degree weight=0.5)
      Semantic = avg(9 semantic dims)
      Total = (4 * Quality + 1 * Semantic) / 5

    Args:
        scores: Dict mapping dimension name to raw score.

    Returns:
        Dict with quality_score, semantic_score, total_score (all normalized).
    """
    norm = {d: normalize_score(d, s) for d, s in scores.items()}

    q_dims = [d for d in VBENCH_QUALITY_DIMS if d in norm]
    if q_dims:
        q_total_w = sum(VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in q_dims)
        quality = sum(norm[d] * VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in q_dims) / q_total_w
    else:
        quality = 0.0

    s_dims = [d for d in VBENCH_SEMANTIC_DIMS if d in norm]
    if s_dims:
        s_total_w = sum(VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in s_dims)
        semantic = sum(norm[d] * VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in s_dims) / s_total_w
    else:
        semantic = 0.0

    total = (QUALITY_WEIGHT * quality + SEMANTIC_WEIGHT * semantic) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    return {
        "quality_score": round(quality, 4),
        "semantic_score": round(semantic, 4),
        "total_score": round(total, 4),
        "normalized_per_dim": {d: round(v, 4) for d, v in norm.items()},
    }


# ============================================================================
# Metric: Revisit frame consistency (PSNR / SSIM)
# ============================================================================


def _psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def _ssim_channel(c1: np.ndarray, c2: np.ndarray) -> float:
    """Compute SSIM for a single channel (uint8)."""
    from scipy.ndimage import uniform_filter

    c1 = c1.astype(np.float64)
    c2 = c2.astype(np.float64)
    k = 11

    mu1 = uniform_filter(c1, k)
    mu2 = uniform_filter(c2, k)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(c1**2, k) - mu1_sq
    sigma2_sq = uniform_filter(c2**2, k) - mu2_sq
    sigma12 = uniform_filter(c1 * c2, k) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two uint8 RGB images."""
    ssim_vals = []
    for c in range(min(img1.shape[2], 3)):
        ssim_vals.append(_ssim_channel(img1[:, :, c], img2[:, :, c]))
    return float(np.mean(ssim_vals))


def _remap_frame_index(frame_idx: int, ref_fps: float, video_fps: float, n_video_frames: int) -> int:
    """Convert a frame index from reference FPS to video FPS via time mapping.

    Evaluation pairs are stored at ref_fps (16fps). If the video was generated
    at a different FPS, we map through time: time = frame / ref_fps, then
    video_frame = round(time * video_fps).

    Args:
        frame_idx: Frame index at ref_fps.
        ref_fps: FPS used when computing evaluation pairs (typically 16).
        video_fps: Actual FPS of the generated video.
        n_video_frames: Total frames in the video (for clamping).

    Returns:
        Corresponding frame index in the video.
    """
    if abs(ref_fps - video_fps) < 0.1:
        return min(frame_idx, n_video_frames - 1)
    time_sec = frame_idx / ref_fps
    target = round(time_sec * video_fps)
    return min(max(target, 0), n_video_frames - 1)


def _detect_video_fps(video_path: str) -> float:
    """Detect FPS from a video file."""
    try:
        from decord import VideoReader

        vr = VideoReader(video_path)
        return float(vr.get_avg_fps())
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return float(fps)
    except Exception:
        pass
    return 16.0  # Default fallback


def _frame_to_numpy(frame) -> np.ndarray:
    """Convert a decord/imageio/torch frame object to a numpy array."""
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    if isinstance(frame, np.ndarray):
        return frame
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        return frame.detach().cpu().numpy()
    if hasattr(frame, "numpy"):
        return frame.numpy()
    return np.asarray(frame)


@contextlib.contextmanager
def _vbench_torch_load_compat():
    """Allow official VBench checkpoints to load under PyTorch 2.6+.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    Some official VBench checkpoints still serialize plain Python containers
    that are rejected by that safer default. During VBench's own model loading,
    opt back into the pre-2.6 behavior for those trusted public checkpoints.
    """
    import torch

    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compat_load
    try:
        yield
    finally:
        torch.load = original_load


_LPIPS_MODEL = None


def _get_lpips_model(device: str = "cuda"):
    """Lazily load the LPIPS AlexNet model (singleton)."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips as lpips_pkg

        _LPIPS_MODEL = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad_(False)
    return _LPIPS_MODEL


def compute_lpips(img1: np.ndarray, img2: np.ndarray, device: str = "cuda") -> float:
    """Compute LPIPS (AlexNet) between two uint8 RGB images."""
    import torch as _torch

    model = _get_lpips_model(device)
    # Convert to [-1, 1] tensors of shape (1, 3, H, W)
    t1 = _torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    t2 = _torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    t1 = t1.to(device)
    t2 = t2.to(device)
    with _torch.no_grad():
        d = model(t1, t2)
    return float(d.item())


def eval_revisit_consistency(
    video_dir: str,
    benchmark_meta: dict,
    max_pairs_per_scene: int = 5,
    ref_fps: float = 16.0,
    cache_dir: str | None = None,
    use_lpips: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate revisit frame consistency using PSNR, SSIM, and (optionally) LPIPS.

    For each scene, extracts the evaluation frame pairs (where camera revisits
    the same viewpoint) and computes pixel-level similarity metrics.

    FPS-aware: evaluation pairs are stored at ref_fps (16fps). If the video
    was generated at a different FPS, frame indices are remapped through time
    to ensure we compare the correct timestamps.

    Supports resume: per-scene results are cached in cache_dir. Cache entries
    that lack LPIPS are upgraded in-place when use_lpips=True (they are
    re-evaluated only for the LPIPS column, preserving prior PSNR/SSIM).

    Args:
        video_dir: Directory containing {scene_id}_generated.mp4 files.
        benchmark_meta: Parsed scene_trajectories_v2.json.
        max_pairs_per_scene: Max evaluation pairs to use per scene.
        ref_fps: FPS at which evaluation pairs were computed.
        cache_dir: Directory for per-scene intermediate results. None = no caching.
        use_lpips: If True, also compute LPIPS (AlexNet) per pair.
        device: CUDA device for LPIPS model.

    Returns:
        Dict with per-scene and aggregate PSNR/SSIM/LPIPS scores.
    """
    try:
        from decord import VideoReader
    except ImportError:
        import imageio.v3 as iio

        VideoReader = None

    scenes = benchmark_meta.get("scenes", [])
    results = {}
    all_psnr, all_ssim, all_lpips = [], [], []

    # Resume support: per-scene cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    n_cached = 0
    n_lpips_upgraded = 0

    for si, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        video_path = os.path.join(video_dir, f"{scene_id}_generated.mp4")
        if not os.path.exists(video_path):
            continue

        eval_pairs = scene.get("evaluation_pairs", [])
        if not eval_pairs:
            continue

        # Check per-scene cache
        cache_file = os.path.join(cache_dir, f"{scene_id}.json") if cache_dir else None
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                # If LPIPS requested but not cached, upgrade in place
                pairs_cached = cached.get("pairs", [])
                needs_lpips = use_lpips and pairs_cached and any("lpips" not in p for p in pairs_cached)
                if not needs_lpips:
                    results[scene_id] = cached
                    all_psnr.extend([p["psnr"] for p in pairs_cached])
                    all_ssim.extend([p["ssim"] for p in pairs_cached])
                    if use_lpips:
                        all_lpips.extend([p["lpips"] for p in pairs_cached if "lpips" in p])
                    n_cached += 1
                    continue
                # Need LPIPS upgrade: read video frames to compute LPIPS for cached pairs
                try:
                    if VideoReader is not None:
                        vr_local = VideoReader(video_path)
                        n_frames_local = len(vr_local)
                    else:
                        frames_local = iio.imread(video_path)
                        n_frames_local = frames_local.shape[0]
                except Exception:
                    print(f"  WARNING: corrupted video {scene_id}, skipping LPIPS upgrade")
                    results[scene_id] = cached
                    all_psnr.extend([p["psnr"] for p in pairs_cached])
                    all_ssim.extend([p["ssim"] for p in pairs_cached])
                    n_cached += 1
                    continue

                upgraded_pairs = []
                scene_lpips_local = []
                for p in pairs_cached:
                    fa = min(p["frame_a"], n_frames_local - 1)
                    fb = min(p["frame_b"], n_frames_local - 1)
                    if "lpips" in p:
                        upgraded_pairs.append(p)
                        scene_lpips_local.append(p["lpips"])
                        continue
                    if VideoReader is not None:
                        img_a = _frame_to_numpy(vr_local[fa])
                        img_b = _frame_to_numpy(vr_local[fb])
                    else:
                        img_a = frames_local[fa]
                        img_b = frames_local[fb]
                    lp = compute_lpips(img_a, img_b, device=device)
                    p2 = dict(p)
                    p2["lpips"] = round(lp, 4)
                    upgraded_pairs.append(p2)
                    scene_lpips_local.append(lp)

                cached["pairs"] = upgraded_pairs
                if scene_lpips_local:
                    cached["mean_lpips"] = round(float(np.mean(scene_lpips_local)), 4)
                results[scene_id] = cached
                all_psnr.extend([p["psnr"] for p in upgraded_pairs])
                all_ssim.extend([p["ssim"] for p in upgraded_pairs])
                all_lpips.extend([p["lpips"] for p in upgraded_pairs if "lpips" in p])
                n_lpips_upgraded += 1
                with open(cache_file, "w") as f:
                    json.dump(cached, f, indent=2)
                continue
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted, re-evaluate

        pairs = sorted(eval_pairs, key=lambda p: p.get("quality_score", 999))[:max_pairs_per_scene]

        try:
            if VideoReader is not None:
                vr = VideoReader(video_path)
                n_frames = len(vr)
            else:
                frames_all = iio.imread(video_path)
                n_frames = frames_all.shape[0]
        except Exception:
            print(f"  WARNING: corrupted video {scene_id}, skipping")
            continue

        # Detect video FPS for frame index remapping
        video_fps = _detect_video_fps(video_path)

        scene_psnr, scene_ssim, scene_lpips = [], [], []
        scene_pairs = []

        for pair in pairs:
            # Remap frame indices from ref_fps to video_fps
            fa = _remap_frame_index(pair["frame_a"], ref_fps, video_fps, n_frames)
            fb = _remap_frame_index(pair["frame_b"], ref_fps, video_fps, n_frames)
            if fa == fb:
                continue  # Degenerate after remap

            if VideoReader is not None:
                img_a = _frame_to_numpy(vr[fa])
                img_b = _frame_to_numpy(vr[fb])
            else:
                img_a = frames_all[fa]
                img_b = frames_all[fb]

            p = _psnr(img_a, img_b)
            s = compute_ssim(img_a, img_b)
            pair_record = {
                "frame_a": fa,
                "frame_b": fb,
                "psnr": round(p, 2),
                "ssim": round(s, 4),
                "distance_m": pair.get("distance_m"),
                "angle_diff_deg": pair.get("angle_diff_deg"),
            }
            scene_psnr.append(p)
            scene_ssim.append(s)
            if use_lpips:
                lp = compute_lpips(img_a, img_b, device=device)
                pair_record["lpips"] = round(lp, 4)
                scene_lpips.append(lp)
            scene_pairs.append(pair_record)

        if scene_psnr:
            scene_result = {
                "mean_psnr": round(float(np.mean(scene_psnr)), 2),
                "mean_ssim": round(float(np.mean(scene_ssim)), 4),
                "n_pairs": len(scene_psnr),
                "pairs": scene_pairs,
            }
            if use_lpips and scene_lpips:
                scene_result["mean_lpips"] = round(float(np.mean(scene_lpips)), 4)
            results[scene_id] = scene_result
            all_psnr.extend(scene_psnr)
            all_ssim.extend(scene_ssim)
            if use_lpips:
                all_lpips.extend(scene_lpips)

            # Save per-scene cache
            if cache_file:
                with open(cache_file, "w") as f:
                    json.dump(scene_result, f, indent=2)

        if (si + 1) % 20 == 0:
            print(f"  Revisit: {si+1}/{len(scenes)} scenes ({n_cached} cached, {n_lpips_upgraded} lpips-upgraded)")

    # Per-category breakdown
    categories: dict[str, list] = {}
    for sid, r in results.items():
        cat = "_".join(sid.split("_")[:-1])
        categories.setdefault(cat, []).append(r)

    per_category = {}
    for cat, rs in sorted(categories.items()):
        cat_psnr = [r["mean_psnr"] for r in rs]
        cat_ssim = [r["mean_ssim"] for r in rs]
        cat_lpips = [r["mean_lpips"] for r in rs if "mean_lpips" in r]
        entry = {
            "mean_psnr": round(float(np.mean(cat_psnr)), 2),
            "mean_ssim": round(float(np.mean(cat_ssim)), 4),
            "n_scenes": len(rs),
        }
        if cat_lpips:
            entry["mean_lpips"] = round(float(np.mean(cat_lpips)), 4)
        per_category[cat] = entry

    summary = {
        "overall_mean_psnr": round(float(np.mean(all_psnr)), 2) if all_psnr else 0.0,
        "overall_mean_ssim": round(float(np.mean(all_ssim)), 4) if all_ssim else 0.0,
        "n_scenes_evaluated": len(results),
        "n_total_pairs": len(all_psnr),
        "per_category": per_category,
    }
    if all_lpips:
        summary["overall_mean_lpips"] = round(float(np.mean(all_lpips)), 4)
        summary["n_lpips_pairs"] = len(all_lpips)

    return {"summary": summary, "per_scene": results}


# ============================================================================
# Metric: VBench evaluation
# ============================================================================


def _build_stripped_video_dir(video_dir: str, output_dir: str) -> str:
    """Re-encode every ``*_generated.mp4`` in ``video_dir`` with frame 0 dropped.

    Refiner methods (e.g. SANA-WM + LTX-2 sink-bidirectional refiner) only
    refine frames 1..T-1; frame 0 is a verbatim copy of the input image and
    looks visually mismatched ("flickering") relative to the rest of the
    clip. To compare apples-to-apples with non-refiner methods at evaluation
    time we drop that first frame *online*, never modifying the raw outputs.

    The stripped clips are cached at ``{output_dir}/_stripped_videos/`` and
    rebuilt only when the source mtime is newer than the cached copy. The
    returned path can be used as a drop-in replacement for ``video_dir`` by
    any downstream tool that scans ``*_generated.mp4`` files (notably
    VBench's ``custom_input`` mode).
    """
    import imageio.v3 as iio

    try:
        from decord import VideoReader

        _have_decord = True
    except ImportError:
        VideoReader = None  # type: ignore[assignment]
        _have_decord = False

    src_dir = Path(video_dir)
    dst_dir = Path(output_dir) / "_stripped_videos"
    dst_dir.mkdir(parents=True, exist_ok=True)

    n_built = 0
    n_cached = 0
    for src in sorted(src_dir.glob("*_generated.mp4")):
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            n_cached += 1
            continue
        try:
            if _have_decord:
                vr = VideoReader(str(src))
                frames = _frame_to_numpy(vr[1:])
            else:
                frames_all = iio.imread(str(src))
                frames = frames_all[1:]
            fps = _detect_video_fps(str(src))
            iio.imwrite(str(dst), frames, fps=int(round(fps)))
            n_built += 1
        except Exception as e:
            print(f"  WARNING: failed to strip first frame of {src.name}: {e}")
    if n_built or n_cached:
        print(f"  Stripped first frame: {n_built} re-encoded, {n_cached} cached -> {dst_dir}")
    return str(dst_dir)


def eval_vbench(
    video_dir: str,
    output_dir: str,
    dimensions: list[str] | None = None,
    device: str = "cuda",
    benchmark_meta: dict | None = None,
    benchmark_manifest: str | None = None,
    skip_first_frame: bool = False,
) -> dict[str, float]:
    """Run VBench evaluation on all videos in a directory.

    Args:
        video_dir: Directory containing *_generated.mp4 files.
        output_dir: Where to save VBench results.
        dimensions: List of dimensions to evaluate. None = default 9.
        device: CUDA device.
        benchmark_meta: Parsed scene_trajectories_v2.json for real text prompts.
            Used by CLIP-based dims (overall_consistency, temporal_style).
        benchmark_manifest: Matching run_manifest.jsonl for prompt text.
        skip_first_frame: If True, evaluate on copies with frame 0 removed
            (relevant for refiner methods that only refine frames 1..T-1).

    Returns:
        Dict mapping dimension name to overall score.
    """
    vbench_root = os.path.join(PROJECT_ROOT, "local_libs", "VBench")
    sys.path.insert(0, vbench_root)

    try:
        import vbench as vbench_module
        from vbench import VBench
    except ImportError:
        print("ERROR: VBench not importable. Ensure local_libs/VBench is available.")
        return {}
    try:
        vbench_info_path = _resolve_vbench_info_path(vbench_module)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return {}

    if dimensions is None:
        # Dims that work in custom_input mode (no detectron2/T2V prompts needed)
        # Not supported in custom_input: object_class, multiple_objects, color,
        # spatial_relationship, scene, appearance_style (need VBench built-in prompts)
        dimensions = [
            "subject_consistency",
            "background_consistency",
            "temporal_flickering",
            "motion_smoothness",
            "aesthetic_quality",
            "imaging_quality",
            "dynamic_degree",
            "overall_consistency",
            "temporal_style",
        ]

    # VBench discovers videos via ``os.listdir(videos_path)``, so any other mp4
    # in the directory (e.g. ``{scene}_combined.mp4`` written by the inference
    # script) doubles the video count and triggers a "Number of prompts should
    # match with number of videos" failure. Sidecars (combined.mp4, prompt.txt,
    # first_frame.png, trajectory.png, camera_motion.txt) get relocated into a
    # ``_sidecars/`` subdir so that ``video_dir`` only exposes
    # ``*_generated.mp4`` to VBench. Idempotent: noop if all sidecars are
    # already inside ``_sidecars/``.
    sidecar_dir = Path(video_dir) / "_sidecars"
    sidecar_suffixes = (
        "_combined.mp4",
        "_first_frame.png",
        "_trajectory.png",
        "_camera_motion.txt",
        "_prompt.txt",
    )
    moved = 0
    for entry in Path(video_dir).iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith("_generated.mp4"):
            continue
        if any(entry.name.endswith(s) for s in sidecar_suffixes):
            sidecar_dir.mkdir(exist_ok=True)
            target = sidecar_dir / entry.name
            if target.exists():
                target.unlink()
            entry.rename(target)
            moved += 1
    if moved:
        print(f"  Moved {moved} sidecar file(s) to {sidecar_dir} (VBench cleanup)")

    # If requested (refiner methods), point VBench at copies with frame 0 dropped.
    if skip_first_frame:
        video_dir = _build_stripped_video_dir(video_dir, output_dir)

    videos = sorted(Path(video_dir).glob("*_generated.mp4"))
    if not videos:
        print(f"No *_generated.mp4 files found in {video_dir}")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Build prompt_list for custom_input mode
    # Real text prompts are needed for CLIP-based dims (overall_consistency, temporal_style)
    scene_prompts = _load_manifest_prompts(benchmark_manifest)
    if benchmark_meta:
        for scene in benchmark_meta.get("scenes", []):
            sid = scene.get("scene_id", "")
            # Try to get prompt from scene metadata or linked prompts
            prompt = scene.get("prompt", sid)
            if sid and sid not in scene_prompts:
                scene_prompts[sid] = prompt if prompt else sid

    # Also try loading public manifests directly.
    if not scene_prompts:
        for candidate in [
            "data/SANA-WM-Bench/benchmark_v2_smooth_60s/sanawm_export_v2/run_manifest.jsonl",
            "data/SANA-WM-Bench/benchmark_v2_hard_60s/sanawm_export_v2/run_manifest.jsonl",
        ]:
            if os.path.exists(candidate):
                with open(candidate, encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line.strip())
                        scene_prompts[row["id"]] = row.get("prompt", row["id"])
                break

    prompt_list = {}
    for v in videos:
        sid = v.stem.replace("_generated", "")
        prompt_list[v.name] = scene_prompts.get(sid, sid)

    # Resume: load existing per-dimension results (skip NaN/invalid)
    # In custom_input mode, VBench saves to {name}_eval_results.json (not {name}_{dim}_...)
    scores = {}
    skipped = 0
    for dim in dimensions:
        # Check both naming patterns (custom_input vs standard)
        for pattern in [f"eval_{dim}_eval_results.json", f"eval_{dim}_{dim}_eval_results.json"]:
            result_file = os.path.join(output_dir, pattern)
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    if dim in data and len(data[dim]) > 0:
                        val = float(data[dim][0])
                        if not np.isnan(val) and len(data[dim]) > 1 and len(data[dim][1]) > 0:
                            scores[dim] = val
                            skipped += 1
                            break
                        else:
                            os.remove(result_file)
                except (json.JSONDecodeError, KeyError, ValueError):
                    os.remove(result_file)

    remaining = [d for d in dimensions if d not in scores]
    print(f"  {len(videos)} videos, {len(dimensions)} dimensions " f"({skipped} cached, {len(remaining)} to evaluate)")

    for i, dim in enumerate(remaining):
        print(f"  [{skipped+i+1}/{len(dimensions)}] {dim} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            bench = VBench(device, vbench_info_path, output_dir)
            with _vbench_torch_load_compat():
                bench.evaluate(
                    videos_path=str(video_dir),
                    name=f"eval_{dim}",
                    dimension_list=[dim],
                    mode="custom_input",
                    prompt_list=prompt_list,
                    local=(dim == "subject_consistency"),
                )

            # Check both naming patterns
            result_file = None
            for pat in [f"eval_{dim}_eval_results.json", f"eval_{dim}_{dim}_eval_results.json"]:
                candidate = os.path.join(output_dir, pat)
                if os.path.exists(candidate):
                    result_file = candidate
                    break

            if result_file:
                with open(result_file) as f:
                    data = json.load(f)
                if dim in data and len(data[dim]) > 0:
                    val = float(data[dim][0])
                    if not np.isnan(val):
                        scores[dim] = val
                        print(f"{val:.4f} ({time.time()-t0:.1f}s)")
                    else:
                        print(f"NaN ({time.time()-t0:.1f}s)")
                        os.remove(result_file)
                else:
                    print(f"no results ({time.time()-t0:.1f}s)")
            else:
                print(f"result file not found ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    missing = [dim for dim in dimensions if dim not in scores]
    if missing:
        raise RuntimeError(f"VBench failed for requested dimension(s): {', '.join(missing)}")

    return scores


# ============================================================================
# Metric: Camera accuracy (Pi3X)
# ============================================================================


def eval_camera_accuracy(
    video_dir: str,
    benchmark_meta: dict,
    output_path: str,
    device: str = "cuda",
    benchmark_manifest: str | None = None,
) -> dict[str, Any]:
    """Evaluate camera pose accuracy using Pi3X.

    Checks for existing results first. If not found, prints the command
    to run separately (requires accelerate launch for multi-GPU).

    Args:
        video_dir: Directory with generated videos.
        benchmark_meta: Benchmark metadata with GT poses.
        output_path: Where to save results.
        device: CUDA device.

    Returns:
        Dict with per-scene pose metrics, or empty if not yet evaluated.
    """
    # Check if results already exist
    existing = os.path.join(video_dir, "eval_poses.json")
    if os.path.exists(existing):
        print(f"  Loading existing results: {existing}")
        with open(existing) as f:
            return json.load(f)

    eval_script = os.path.join(PROJECT_ROOT, "tools", "metrics", "sana_wm", "eval_benchmark_poses.py")
    print(f"  Camera accuracy requires multi-GPU accelerate launch.")
    print(f"  Run separately:")
    print("    accelerate launch --num_processes=8 --mixed_precision=bf16 \\")
    cmd = f"      {eval_script} --result_folder {video_dir}"
    if benchmark_manifest:
        cmd += f" --manifest {benchmark_manifest}"
    print(cmd)
    return {}


def _ensure_roterr_degrees(camera_results: dict[str, Any]) -> dict[str, Any]:
    """Normalize per-scene RotErr values to degrees.

    Current eval_benchmark_poses.py writes degree-valued RotErr with
    ``RotErr_unit='deg'``. Older caches omitted the unit and stored radians.
    """
    rows = [v for v in camera_results.values() if isinstance(v, dict) and "RotErr" in v]
    if not rows:
        return camera_results
    for v in rows:
        unit = str(v.get("RotErr_unit", "")).lower()
        if unit == "rad" or (unit != "deg" and abs(float(v["RotErr"])) <= math.pi + 1e-6):
            v["RotErr"] = float(v["RotErr"]) * 180.0 / math.pi
        v["RotErr_unit"] = "deg"
    return camera_results


# ============================================================================
# Metric: Temporal degradation
# ============================================================================


def eval_temporal_degradation(
    video_dir: str,
    output_dir: str,
    window_sec: float = 10.0,
    fps: float = 16.0,
    dimensions: list[str] | None = None,
    skip_first_frame: bool = False,
) -> dict[str, Any]:
    """Evaluate VBench metrics on sliding time windows to measure degradation.

    Splits each video into N-second windows, evaluates VBench on each,
    and reports how quality changes over time.

    Args:
        video_dir: Directory with generated videos.
        output_dir: Where to save window clips and results.
        window_sec: Window size in seconds.
        fps: Video frame rate.
        dimensions: VBench dimensions to evaluate per window.
        skip_first_frame: If True, evaluate on copies with frame 0 removed
            (relevant for refiner methods that only refine frames 1..T-1).
            Equivalent to shifting all window boundaries one frame later.

    Returns:
        Dict with per-window quality scores and degradation trend.
    """
    if dimensions is None:
        dimensions = [
            "subject_consistency",
            "background_consistency",
            "temporal_flickering",
            "imaging_quality",
        ]

    try:
        from decord import VideoReader
    except ImportError:
        print("WARNING: decord not available for temporal degradation.")
        return {}

    import imageio.v3 as iio

    if skip_first_frame:
        video_dir = _build_stripped_video_dir(video_dir, output_dir)

    videos = sorted(Path(video_dir).glob("*_generated.mp4"))
    if not videos:
        return {}

    window_frames = int(window_sec * fps)
    print(f"  Windows: {window_sec}s ({window_frames} frames), {len(dimensions)} dims")

    # Split videos into windows
    window_dirs = {}
    for video_path in videos:
        scene_id = video_path.stem.replace("_generated", "")
        vr = VideoReader(str(video_path))
        n_frames = len(vr)
        n_windows = max(1, n_frames // window_frames)

        for w in range(n_windows):
            start = w * window_frames
            end = min(start + window_frames, n_frames)
            if end - start < 16:
                continue

            window_name = f"w{w}_{int(w*window_sec)}s-{int((w+1)*window_sec)}s"
            window_dir = os.path.join(output_dir, "temporal_windows", window_name)
            os.makedirs(window_dir, exist_ok=True)

            clip_path = os.path.join(window_dir, f"{scene_id}_generated.mp4")
            if not os.path.exists(clip_path):
                frames = _frame_to_numpy(vr[start:end])
                iio.imwrite(clip_path, frames, fps=int(fps))

            window_dirs.setdefault(window_name, window_dir)

    # Evaluate each window
    results = {}
    for window_name, window_dir in sorted(window_dirs.items()):
        print(f"  Window: {window_name}")
        result_dir = os.path.join(output_dir, "temporal_results", window_name)
        scores = eval_vbench(window_dir, result_dir, dimensions=dimensions)
        if scores:
            results[window_name] = scores

    # Compute degradation trend
    if results:
        windows = sorted(results.keys())
        trend = {}
        for dim in dimensions:
            vals = [results[w].get(dim, 0) for w in windows]
            if len(vals) >= 2:
                trend[dim] = {
                    "first_window": round(vals[0], 4),
                    "last_window": round(vals[-1], 4),
                    "degradation": round(vals[0] - vals[-1], 4),
                    "per_window": [round(v, 4) for v in vals],
                }

        return {"windows": results, "trend": trend}

    return {}


# ============================================================================
# Main orchestrator
# ============================================================================

AVAILABLE_METRICS = ["vbench", "revisit", "camera", "temporal"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluation for world-model benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method_dir",
        type=str,
        required=True,
        help="Method directory (e.g., results/sana_wm_v1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="simple",
        help="Benchmark split: simple, hard, simple_60s, hard_60s",
    )
    parser.add_argument(
        "--benchmark_meta",
        type=str,
        default=None,
        help="Path to scene_trajectories_v2.json (auto-detected from split).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=AVAILABLE_METRICS,
        choices=AVAILABLE_METRICS,
        help="Which metrics to evaluate. Default: all.",
    )
    parser.add_argument(
        "--vbench_dims",
        nargs="+",
        default=None,
        help=(
            "VBench dimensions. Default: benchmark custom-input dimensions "
            "(7 quality dims plus overall_consistency and temporal_style)."
        ),
    )
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--max_pairs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--ref_fps",
        type=float,
        default=16.0,
        help="FPS at which evaluation pairs were computed (default: 16). "
        "Video FPS is auto-detected; frame indices are remapped via time.",
    )
    parser.add_argument(
        "--revisit_lpips",
        action="store_true",
        help="Also compute LPIPS (AlexNet) for revisit pairs. Upgrades existing PSNR/SSIM cache in-place.",
    )
    parser.add_argument(
        "--skip_first_frame",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help=(
            "Drop frame 0 of every generated video before VBench / temporal "
            "degradation evaluation. Refiner methods only refine frames "
            "1..T-1 (frame 0 is a verbatim copy of the input image), so "
            "including frame 0 introduces a flicker/seam that biases visual "
            "metrics. ``auto`` enables this whenever ``method_info.json`` "
            "contains a ``refiner`` key. Revisit pairs never reference frame "
            "0 in our benchmark, so PSNR/SSIM/LPIPS are unaffected."
        ),
    )
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    video_dir = str(method_dir / args.split)
    eval_dir = str(method_dir / "eval" / args.split)
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.isdir(video_dir):
        print(f"ERROR: Video directory not found: {video_dir}")
        print(f"Expected: {method_dir}/<split>/*_generated.mp4")
        sys.exit(1)

    n_videos = len(list(Path(video_dir).glob("*_generated.mp4")))

    # Load method_info.json if present (describes the model variant)
    method_info_path = method_dir / "method_info.json"
    method_info = {}
    if method_info_path.exists():
        with open(method_info_path) as f:
            method_info = json.load(f)

    # Resolve --skip_first_frame: ``auto`` -> True iff this method_info
    # advertises a refiner (refiners only modify frames 1..T-1).
    if args.skip_first_frame == "yes":
        skip_first_frame = True
    elif args.skip_first_frame == "no":
        skip_first_frame = False
    else:
        skip_first_frame = bool(method_info.get("refiner"))

    print(f"Method:  {method_dir.name}")
    if method_info:
        print(f"  Model:   {method_info.get('model', 'unknown')}")
        print(f"  Variant: {method_info.get('variant', 'unknown')}")
        print(f"  FPS:     {method_info.get('fps', 'auto-detect')}")
    print(f"Split:   {args.split}")
    print(f"Videos:  {n_videos}")
    print(f"Metrics: {args.metrics}")
    print(f"Output:  {eval_dir}")
    print(f"Skip frame 0: {skip_first_frame} (--skip_first_frame={args.skip_first_frame})")

    # Auto-detect benchmark metadata
    benchmark_meta = None
    benchmark_manifest = None
    if args.benchmark_meta:
        with open(args.benchmark_meta) as f:
            benchmark_meta = json.load(f)
        benchmark_manifest = _manifest_path_from_meta(args.benchmark_meta)
    else:
        split_to_dir = {
            "simple": "benchmark_v2_smooth",
            "hard": "benchmark_v2_hard",
            "simple_60s": "benchmark_v2_smooth_60s",
            "hard_60s": "benchmark_v2_hard_60s",
        }
        bench_dir = split_to_dir.get(args.split, "benchmark_v2_smooth")
        meta_path = os.path.join("data", "SANA-WM-Bench", bench_dir, "scene_trajectories_v2.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                benchmark_meta = json.load(f)
            benchmark_manifest = _manifest_path_from_meta(meta_path)
            print(f"Metadata: {meta_path}")

    t_start = time.time()
    # Preserve any prior metric blocks (e.g. vbench from a previous run) so a
    # narrower invocation (--metrics revisit) does not erase them.
    summary_path = os.path.join(eval_dir, "summary.json")
    summary: dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            summary = {}
    summary.update(
        {
            "method": method_dir.name,
            "method_info": method_info if method_info else summary.get("method_info"),
            "split": args.split,
            "n_videos": n_videos,
            "skip_first_frame": skip_first_frame,
        }
    )

    # ---- 1. VBench ----
    if "vbench" in args.metrics:
        print(f"\n{'='*60}")
        print("VBench Visual Quality (benchmark custom-input dimensions + total score)")
        print(f"{'='*60}")
        vbench_results = eval_vbench(
            video_dir,
            eval_dir,
            dimensions=args.vbench_dims,
            device=args.device,
            benchmark_meta=benchmark_meta,
            benchmark_manifest=benchmark_manifest,
            skip_first_frame=skip_first_frame,
        )

        if vbench_results:
            total = compute_vbench_total(vbench_results)
            vbench_output = {
                "raw_scores": {d: round(s, 4) for d, s in vbench_results.items()},
                **total,
            }
            with open(os.path.join(eval_dir, "vbench_scores.json"), "w") as f:
                json.dump(vbench_output, f, indent=2)
            summary["vbench"] = {
                "quality_score": total["quality_score"],
                "semantic_score": total["semantic_score"],
                "total_score": total["total_score"],
                "n_dimensions": len(vbench_results),
            }
            print(
                f"\nVBench Total: {total['total_score']:.4f} "
                f"(Quality={total['quality_score']:.4f}, Semantic={total['semantic_score']:.4f})"
            )

    # ---- 2. Revisit consistency ----
    if "revisit" in args.metrics and benchmark_meta:
        print(f"\n{'='*60}")
        print("Revisit Frame Consistency (PSNR / SSIM)")
        print(f"{'='*60}")
        revisit_results = eval_revisit_consistency(
            video_dir,
            benchmark_meta,
            max_pairs_per_scene=args.max_pairs,
            ref_fps=args.ref_fps,
            cache_dir=os.path.join(eval_dir, "revisit_cache"),
            use_lpips=args.revisit_lpips,
            device=args.device,
        )
        with open(os.path.join(eval_dir, "revisit_consistency.json"), "w") as f:
            json.dump(revisit_results, f, indent=2)
        s = revisit_results["summary"]
        summary["revisit"] = s
        lpips_str = f", LPIPS={s['overall_mean_lpips']:.4f}" if "overall_mean_lpips" in s else ""
        print(
            f"\nRevisit: PSNR={s['overall_mean_psnr']:.2f} dB, "
            f"SSIM={s['overall_mean_ssim']:.4f}{lpips_str} ({s['n_total_pairs']} pairs)"
        )
        if "per_category" in s:
            for cat, cs in s["per_category"].items():
                cat_lpips = f", LPIPS={cs['mean_lpips']:.4f}" if "mean_lpips" in cs else ""
                print(f"  {cat}: PSNR={cs['mean_psnr']:.2f}, SSIM={cs['mean_ssim']:.4f}{cat_lpips}")

    # ---- 3. Camera accuracy ----
    if "camera" in args.metrics and benchmark_meta:
        print(f"\n{'='*60}")
        print("Camera Pose Accuracy (Pi3X)")
        print(f"{'='*60}")
        camera_results = eval_camera_accuracy(
            video_dir,
            benchmark_meta,
            output_path=os.path.join(eval_dir, "camera_accuracy.json"),
            device=args.device,
            benchmark_manifest=benchmark_manifest,
        )
        if camera_results:
            camera_results = _ensure_roterr_degrees(camera_results)
            with open(os.path.join(eval_dir, "camera_accuracy.json"), "w") as f:
                json.dump(camera_results, f, indent=2)
            rot_errs = [v.get("RotErr", 0) for v in camera_results.values() if isinstance(v, dict)]
            trans_errs = [v.get("TransErr_rel", 0) for v in camera_results.values() if isinstance(v, dict)]
            if rot_errs:
                summary["camera"] = {
                    "mean_rot_err_deg": round(float(np.mean(rot_errs)), 3),
                    "mean_trans_err_rel": round(float(np.mean(trans_errs)), 3),
                    "n_scenes": len(rot_errs),
                }
                print(
                    f"\nCamera: RotErr={summary['camera']['mean_rot_err_deg']:.3f}°, "
                    f"TransErr={summary['camera']['mean_trans_err_rel']:.3f}"
                )

    # ---- 4. Temporal degradation ----
    if "temporal" in args.metrics:
        print(f"\n{'='*60}")
        print(f"Temporal Degradation ({args.window_sec}s windows)")
        print(f"{'='*60}")
        temporal_results = eval_temporal_degradation(
            video_dir,
            os.path.join(eval_dir, "temporal"),
            window_sec=args.window_sec,
            skip_first_frame=skip_first_frame,
        )
        if temporal_results:
            with open(os.path.join(eval_dir, "temporal_degradation.json"), "w") as f:
                json.dump(temporal_results, f, indent=2)
            if "trend" in temporal_results:
                summary["temporal_degradation"] = temporal_results["trend"]
                for dim, trend in temporal_results["trend"].items():
                    print(
                        f"  {dim}: {trend['first_window']:.4f} → {trend['last_window']:.4f} "
                        f"(degradation={trend['degradation']:+.4f})"
                    )

    # ---- Save summary ----
    # Re-read summary.json just before writing: another eval job (e.g. vbench)
    # may have concurrently added its metric block since we loaded the file
    # at the start of this run. Merge their updates under any keys we didn't
    # touch in this invocation (vbench/revisit/camera/temporal_degradation/
    # method_info). Without this, a long-running temporal job can clobber a
    # vbench block that completed meanwhile.
    summary["eval_time_sec"] = round(time.time() - t_start, 1)
    summary_path = os.path.join(eval_dir, "summary.json")
    metrics_evaluated = set(args.metrics)
    preserve_keys = set()
    if "vbench" not in metrics_evaluated:
        preserve_keys.add("vbench")
    if "revisit" not in metrics_evaluated:
        preserve_keys.add("revisit")
    if "camera" not in metrics_evaluated:
        preserve_keys.add("camera")
    if "temporal" not in metrics_evaluated:
        preserve_keys.add("temporal_degradation")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as _f:
                disk = json.load(_f) or {}
            for _k in preserve_keys:
                if _k in disk and _k not in summary:
                    summary[_k] = disk[_k]
            # Always prefer disk method_info if ours is empty (never overwrite a valid one).
            if not summary.get("method_info") and disk.get("method_info"):
                summary["method_info"] = disk["method_info"]
        except (json.JSONDecodeError, OSError):
            pass
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE ({summary['eval_time_sec']:.1f}s)")
    print(f"{'='*60}")
    print(f"Results: {eval_dir}/")

    # Print final summary table
    print(f"\n{'Metric':<30} {'Score':>10}")
    print("-" * 42)
    if "vbench" in summary:
        v = summary["vbench"]
        print(f"{'VBench Total':<30} {v['total_score']:>10.4f}")
        print(f"{'  Quality Score':<30} {v['quality_score']:>10.4f}")
        print(f"{'  Semantic Score':<30} {v['semantic_score']:>10.4f}")
    if "revisit" in summary:
        r = summary["revisit"]
        print(f"{'Revisit PSNR (dB)':<30} {r['overall_mean_psnr']:>10.2f}")
        print(f"{'Revisit SSIM':<30} {r['overall_mean_ssim']:>10.4f}")
        if "overall_mean_lpips" in r:
            print(f"{'Revisit LPIPS':<30} {r['overall_mean_lpips']:>10.4f}")
    if "camera" in summary:
        c = summary["camera"]
        print(f"{'Camera RotErr (deg)':<30} {c['mean_rot_err_deg']:>10.3f}")
        print(f"{'Camera TransErr (rel)':<30} {c['mean_trans_err_rel']:>10.3f}")


if __name__ == "__main__":
    main()
