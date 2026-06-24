"""Evaluate camera pose accuracy for the SANA-WM benchmark.

Runs Pi3 on generated videos and compares estimated poses with GT poses
from the benchmark NPZ files. This does not require GT videos -- only GT
pose trajectories from the benchmark manifest.

Usage:
    accelerate launch --num_processes=8 --mixed_precision=bf16 \
        tools/metrics/sana_wm/eval_benchmark_poses.py \
        --result_folder output/wm_benchmark/scale_30/ \
        --manifest data/SANA-WM-Bench/benchmark_v2_smooth_60s/sanawm_export_v2/run_manifest.jsonl
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

# Add Pi3 to path
pi3_path = os.path.join(os.path.dirname(__file__), "Pi3")
if os.path.exists(pi3_path):
    sys.path.append(pi3_path)

try:
    from .pose_utils import metric, relative_pose, run_pi3_inference_batch
except ImportError:
    from pose_utils import metric, relative_pose, run_pi3_inference_batch


def load_manifest(manifest_path: str) -> Dict[str, dict]:
    """Load benchmark manifest, keyed by scene_id."""
    scenes = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            scenes[entry["id"]] = entry
    return scenes


def _dataset_root_from_manifest(manifest_path: str) -> str:
    """Return the dataset root for a SANA-WM manifest path."""
    export_dir = os.path.dirname(os.path.abspath(manifest_path))
    split_dir = os.path.dirname(export_dir)
    return os.path.dirname(split_dir)


def _resolve_camera_path(camera_path: str, dataset_root: str) -> str:
    """Resolve dataset-relative camera paths from `run_manifest.jsonl`."""
    if os.path.isabs(camera_path):
        return camera_path
    return os.path.join(dataset_root, camera_path)


def load_gt_poses(camera_path: str, num_frames: int = 961) -> torch.Tensor:
    """Load GT c2w poses from sanawm NPZ.

    Returns:
        (T, 4, 4) float32 tensor of camera-to-world poses, relativized to first frame.
    """
    data = np.load(camera_path)
    c2w = data["c2w"][:num_frames].astype(np.float32)  # (T, 4, 4)
    poses = torch.from_numpy(c2w).float()
    # Relativize to first frame
    poses = relative_pose(poses, "left")
    return poses


def _auto_detect_manifest(result_folder: str) -> str:
    """Infer the matching benchmark manifest from a result_folder path.

    Looks at the trailing path component (split name) and maps to the
    correct sanawm export directory.

    Args:
        result_folder: e.g. ".../results/sana_wm_bidirectional/simple_60s"

    Returns:
        Path to the run_manifest.jsonl for that split.
    """
    split_to_bench = {
        "simple": "benchmark_v2_smooth",
        "simple_60s": "benchmark_v2_smooth_60s",
        "hard": "benchmark_v2_hard",
        "hard_60s": "benchmark_v2_hard_60s",
    }
    split = os.path.basename(os.path.normpath(result_folder))
    bench_dir = split_to_bench.get(split)
    if bench_dir is None:
        # Fallback: default to smooth_60s
        bench_dir = "benchmark_v2_smooth_60s"
    return os.path.join("data", "SANA-WM-Bench", bench_dir, "sanawm_export_v2", "run_manifest.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark camera control evaluation")
    parser.add_argument("--result_folder", type=str, required=True, help="Folder with scene_id_generated.mp4 files")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to run_manifest.jsonl. If unset, auto-detected from result_folder split name.",
    )
    parser.add_argument("--pi3_ckpt", type=str, default="yyfz233/Pi3")
    parser.add_argument("--interval", type=int, default=4, help="Frame sampling interval for Pi3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Output JSON (default: result_folder/eval_poses.json)")
    parser.add_argument(
        "--dump_trajectories",
        action="store_true",
        help="Also dump raw aligned Pi3 / GT trajectories as NPZ (under <result_folder>/trajectories/).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated scene IDs to restrict the evaluation "
            "(e.g. 'game_style_001,indoor_003'). Forces re-evaluation of "
            "just those scenes even if already in the cache."
        ),
    )
    parser.add_argument(
        "--skip_first_frame",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help=(
            "Drop frame 0 of every video (and its corresponding GT pose) "
            "before computing pose-error metrics. Refiner methods only "
            "refine frames 1..T-1 -- frame 0 is a verbatim copy of the "
            "input image so its rotation/translation are trivially correct "
            "and would optimistically inflate the score. ``auto`` enables "
            "this whenever ``method_info.json`` (sibling of result_folder, "
            "i.e. ``<result_folder>/../method_info.json``) contains a "
            "``refiner`` key."
        ),
    )
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = _auto_detect_manifest(args.result_folder)
    return args


def _resolve_skip_first_frame(arg_val: str, result_folder: str) -> bool:
    """Resolve ``--skip_first_frame {auto,yes,no}`` against ``method_info.json``.

    Pose evaluation runs over a per-split subdir (e.g.
    ``.../sana_wm_bidirectional_refined/simple_60s``) but ``method_info.json``
    lives at the *parent* method directory. ``auto`` enables stripping iff
    that file is present and contains a ``refiner`` key.
    """
    if arg_val == "yes":
        return True
    if arg_val == "no":
        return False
    method_info_path = os.path.join(os.path.dirname(os.path.normpath(result_folder)), "method_info.json")
    if not os.path.exists(method_info_path):
        return False
    try:
        with open(method_info_path) as _f:
            method_info = json.load(_f)
    except (json.JSONDecodeError, OSError):
        return False
    return bool(method_info.get("refiner"))


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index

    skip_first_frame = _resolve_skip_first_frame(args.skip_first_frame, args.result_folder)
    if accelerator.is_main_process:
        print(f"Skip frame 0: {skip_first_frame} (--skip_first_frame={args.skip_first_frame})")

    # ---- Load manifest ----
    manifest = load_manifest(args.manifest)
    dataset_root = _dataset_root_from_manifest(args.manifest)

    # ---- Find generated videos ----
    gen_videos = sorted(glob(os.path.join(args.result_folder, "*_generated.mp4")))
    pairs = []
    for vpath in gen_videos:
        basename = os.path.basename(vpath).replace("_generated.mp4", "")
        if basename in manifest:
            pairs.append((basename, vpath, _resolve_camera_path(manifest[basename]["camera_path"], dataset_root)))

    if not pairs:
        print(f"No matching scenes found in {args.result_folder}")
        return

    if accelerator.is_main_process:
        print(f"Found {len(pairs)} scenes to evaluate")

    # ---- Load Pi3 model ----
    try:
        from pi3.models.pi3 import Pi3
    except ImportError:
        if accelerator.is_main_process:
            print("Error: Pi3 not found. Install Pi3 or place it on PYTHONPATH.")
        return
    pi3_model = Pi3.from_pretrained(args.pi3_ckpt).to(device)
    pi3_model.requires_grad_(False)

    # ---- Load existing cache ----
    output_path = args.output or os.path.join(args.result_folder, "eval_poses.json")
    cache = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            cache = json.load(f)
        # Cache entries record both the frame-0 mode and RotErr unit. Legacy
        # entries lack ``RotErr_unit`` and used radians, while current summaries
        # and paper tables expect degrees. Drop mismatched cache records so we
        # never silently mix radians with degree-labeled outputs.
        before = len(cache)
        cache = {
            sid: rec
            for sid, rec in cache.items()
            if (
                not isinstance(rec, dict)
                or (bool(rec.get("skip_first_frame", False)) == skip_first_frame and rec.get("RotErr_unit") == "deg")
            )
        }
        invalidated = before - len(cache)
        if invalidated and accelerator.is_main_process:
            print(
                f"Invalidated {invalidated}/{before} cached entries "
                f"(skip_first_frame mismatch or RotErr unit is not degrees)"
            )

    # Optional explicit scene filter (forces re-eval of just those)
    if args.scenes:
        keep = {s.strip() for s in args.scenes.split(",") if s.strip()}
        pairs = [(sid, vp, cp) for sid, vp, cp in pairs if sid in keep]
        # Don't skip via cache when forcing re-eval
        pairs_to_run = pairs
        if accelerator.is_main_process:
            print(f"Restricting to {len(pairs_to_run)} explicitly requested scenes: {sorted(keep)}")
    else:
        # Filter out already-cached scenes
        pairs_to_run = [(sid, vp, cp) for sid, vp, cp in pairs if sid not in cache]
        if accelerator.is_main_process:
            print(f"Skipping {len(pairs) - len(pairs_to_run)} cached, evaluating {len(pairs_to_run)}")

    if args.dump_trajectories and accelerator.is_main_process:
        os.makedirs(os.path.join(args.result_folder, "trajectories"), exist_ok=True)

    # ---- Distribute across GPUs ----
    local_results = {}
    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    with accelerator.split_between_processes(pairs_to_run) as my_pairs:
        for i in tqdm(range(0, len(my_pairs), args.batch_size), desc=f"GPU {rank}", disable=rank != 0):
            batch = my_pairs[i : i + args.batch_size]
            video_paths = [vp for _, vp, _ in batch]

            # Run Pi3 on generated videos
            pi3_results = run_pi3_inference_batch(pi3_model, video_paths, device, interval=args.interval)

            for (scene_id, vpath, cam_path), pi3_res in zip(batch, pi3_results):
                if pi3_res is None:
                    print(f"Pi3 failed for {scene_id}, skipping")
                    continue

                try:
                    # Estimated poses from Pi3 (c2w)
                    poses_est = pi3_res["pose"].float()  # (N_sampled, 4, 4)

                    # Refiners only modify frames 1..T-1; frame 0 is a verbatim
                    # copy of the input image so its pose error is trivially
                    # zero and would optimistically inflate the metric. Drop
                    # the first sampled pose (and the matching GT pose) before
                    # relativization so the reported error reflects only the
                    # refined frames.
                    if skip_first_frame:
                        poses_est = poses_est[1:]
                    poses_est = relative_pose(poses_est, "left")

                    # GT poses from benchmark NPZ
                    poses_gt_full = load_gt_poses(cam_path)

                    # Subsample GT to match Pi3's interval sampling
                    gt_indices = list(range(0, len(poses_gt_full), args.interval))
                    if skip_first_frame:
                        gt_indices = gt_indices[1:]
                    # Ensure same length
                    n = min(len(poses_est), len(gt_indices))
                    poses_est = poses_est[:n]
                    poses_gt = poses_gt_full[gt_indices[:n]].to(device)
                    if skip_first_frame:
                        # Re-relativize GT to its (new) first kept frame to
                        # match the relativization above on the est side.
                        poses_gt = relative_pose(poses_gt, "left")

                    # Compute metrics
                    rot_err, trans_err, cammc = metric(poses_gt, poses_est)

                    local_results[scene_id] = {
                        "RotErr": rot_err,
                        "RotErr_unit": "deg",
                        "TransErr_rel": trans_err,
                        "CamMC_rel": cammc,
                        "scene_type": manifest[scene_id].get("scene_type", "unknown"),
                        "n_frames": n,
                    }

                    if args.dump_trajectories:
                        # Save raw (relativized) trajectories as NPZ for
                        # external 3D plotting; no further alignment applied
                        # here so that downstream figures can decide their
                        # own normalization.
                        traj_dir = os.path.join(args.result_folder, "trajectories")
                        os.makedirs(traj_dir, exist_ok=True)
                        np.savez(
                            os.path.join(traj_dir, f"{scene_id}_pi3.npz"),
                            est=poses_est.detach().cpu().numpy(),
                            gt=poses_gt.detach().cpu().numpy(),
                            interval=args.interval,
                            n_frames=n,
                        )
                except Exception as e:
                    print(f"Error evaluating {scene_id}: {e}")

    # ---- Gather results from all GPUs ----
    from accelerate.utils import gather_object

    all_results_list = gather_object([local_results])

    if accelerator.is_main_process:
        # Merge all results
        merged = dict(cache)
        for r in all_results_list:
            merged.update(r)
        # Tag every per-scene record with the eval mode so downstream code
        # (eval_unified.py) can tell whether frame 0 was excluded.
        for _v in merged.values():
            if isinstance(_v, dict):
                _v.setdefault("skip_first_frame", skip_first_frame)

        # Save full results
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Saved {len(merged)} results to {output_path}")

        # ---- Print summary per scene_type ----
        categories = {}
        for scene_id, m in merged.items():
            cat = m.get("scene_type", "unknown")
            if cat not in categories:
                categories[cat] = {"RotErr": [], "TransErr_rel": [], "CamMC_rel": []}
            categories[cat]["RotErr"].append(m["RotErr"])
            categories[cat]["TransErr_rel"].append(m["TransErr_rel"])
            categories[cat]["CamMC_rel"].append(m["CamMC_rel"])

        print("\n" + "=" * 70)
        print(f"{'Category':<20} {'N':>4} {'RotErr(deg)':>12} {'TransErr':>10} {'CamMC':>10}")
        print("-" * 70)

        all_rot, all_trans, all_cammc = [], [], []
        for cat in sorted(categories.keys()):
            n = len(categories[cat]["RotErr"])
            rot = np.mean(categories[cat]["RotErr"])
            trans = np.mean(categories[cat]["TransErr_rel"])
            cammc = np.mean(categories[cat]["CamMC_rel"])
            print(f"{cat:<20} {n:>4} {rot:>12.4f} {trans:>10.4f} {cammc:>10.4f}")
            all_rot.extend(categories[cat]["RotErr"])
            all_trans.extend(categories[cat]["TransErr_rel"])
            all_cammc.extend(categories[cat]["CamMC_rel"])

        print("-" * 70)
        total_n = len(all_rot)
        print(
            f"{'OVERALL':<20} {total_n:>4} {np.mean(all_rot):>12.4f} {np.mean(all_trans):>10.4f} {np.mean(all_cammc):>10.4f}"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
