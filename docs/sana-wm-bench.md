# SANA-WM 80-Scene Benchmark

This guide documents the SANA-WM benchmark release used for the 80-scene
world-model evaluation. The release scope is intentionally narrow:

- 80 fixed scenes from `scene_set/kept_scenes.txt`.
- 60 second trajectories only.
- Sana-WM trajectory exports only: `sanawm_export_v2/*.npz`.
- No 24 second settings, non-Sana-WM exports, or baseline model outputs.
- Public scene IDs are anonymized within each category as `<category>_001` to
  `<category>_020`.

The benchmark contains two splits:

| Split | Directory | Trajectory set | Scenes | Frames |
|---|---|---:|---:|---:|
| `simple_60s` | `benchmark_v2_smooth_60s` | simple | 80 | 961 @ 16 fps |
| `hard_60s` | `benchmark_v2_hard_60s` | hard | 80 | 961 @ 16 fps |

Each Sana-WM `.npz` stores:

- `c2w`: `(961, 4, 4)` camera-to-world matrices.
- `intrinsics`: `(961, 3, 3)` camera intrinsics in source-image pixels.
- `fps`: scalar, `16`.
- `num_frames`: scalar, `961`.

## Dataset Layout

The public Hugging Face dataset is published at `Efficient-Large-Model/SANA-WM-Bench` with this layout:

```text
SANA-WM-Bench/
|-- README.md
|-- images/
|   `-- <scene_id>.png
|-- scene_set/
|   `-- kept_scenes.txt
|-- benchmark_v2_smooth_60s/
|   |-- scene_trajectories_v2.json
|   `-- sanawm_export_v2/
|       |-- run_manifest.jsonl
|       `-- <scene_id>.npz
|-- benchmark_v2_hard_60s/
|   |-- scene_trajectories_v2.json
|   `-- sanawm_export_v2/
|       |-- run_manifest.jsonl
|       `-- <scene_id>.npz
```

The uploaded manifests should use dataset-relative paths:

- `image_path`: `images/<scene_id>.png`
- `camera_path`: `<split_dir>/sanawm_export_v2/<scene_id>.npz`

## Download The Public Release

Download the benchmark from Hugging Face:

```bash
huggingface-cli download Efficient-Large-Model/SANA-WM-Bench \
  --repo-type dataset \
  --local-dir data/SANA-WM-Bench
```

The rest of this guide assumes:

```bash
BENCH=data/SANA-WM-Bench
```

## Run One Scene

Download or point `BENCH` at the release directory:

```bash
BENCH=data/SANA-WM-Bench
SPLIT=benchmark_v2_smooth_60s
SCENE=game_style_001
WORK=results/sana_wm_bench_inputs/$SPLIT/$SCENE
mkdir -p "$WORK"
export BENCH SPLIT SCENE WORK

python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

bench = Path(os.environ["BENCH"])
split = os.environ["SPLIT"]
scene = os.environ["SCENE"]
work = Path(os.environ["WORK"])

manifest = bench / split / "sanawm_export_v2/run_manifest.jsonl"
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
row = next(r for r in rows if r["id"] == scene)
traj = np.load(bench / row["camera_path"])

np.save(work / f"{scene}_c2w.npy", traj["c2w"])
np.save(work / f"{scene}_intrinsics.npy", traj["intrinsics"])
(work / f"{scene}.txt").write_text(row["prompt"], encoding="utf-8")
PY

python inference_video_scripts/wm/inference_sana_wm.py \
  --image "$BENCH/images/$SCENE.png" \
  --prompt "$WORK/$SCENE.txt" \
  --camera "$WORK/${SCENE}_c2w.npy" \
  --intrinsics "$WORK/${SCENE}_intrinsics.npy" \
  --num_frames 961 \
  --fps 16 \
  --step 60 \
  --cfg_scale 5.0 \
  --flow_shift 8.0 \
  --sampling_algo flow_euler_ltx \
  --seed 42 \
  --refiner_seed 42 \
  --no_action_overlay \
  --output_dir "results/sana_wm_bidirectional_refined/simple_60s" \
  --name "$SCENE"
```

The output is:

```text
results/sana_wm_bidirectional_refined/simple_60s/<scene_id>_generated.mp4
```

The LTX-2 refiner is enabled by default. Do not pass `--no_refiner` for the
benchmark run.

## Run All 80 Scenes With A Scheduler

For Slurm-based clusters, this example runs one scene per task. Adjust
array concurrency for the available GPU capacity.

```bash
cat > run_sana_wm_bench.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=swm_bench
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --array=0-79%8
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

REPO=/path/to/Sana
PY=/path/to/python
BENCH=data/SANA-WM-Bench
METHOD=sana_wm_bidirectional_refined
SPLIT_DIR=${SPLIT_DIR:-benchmark_v2_smooth_60s}
SPLIT_NAME=${SPLIT_NAME:-simple_60s}
export BENCH METHOD SPLIT_DIR SPLIT_NAME

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

SCENE=$($PY - <<'PY'
import json
import os
from pathlib import Path
bench = Path(os.environ["BENCH"])
split = os.environ["SPLIT_DIR"]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
rows = [json.loads(line) for line in (bench / split / "sanawm_export_v2/run_manifest.jsonl").read_text().splitlines() if line.strip()]
print(rows[idx]["id"])
PY
)

WORK="$REPO/results/sana_wm_bench_inputs/$SPLIT_NAME/$SCENE"
OUT="$REPO/results/$METHOD/$SPLIT_NAME"
mkdir -p "$WORK" "$OUT"
export SCENE WORK OUT

$PY - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
bench = Path(os.environ["BENCH"])
split = os.environ["SPLIT_DIR"]
scene = os.environ["SCENE"]
work = Path(os.environ["WORK"])
rows = [json.loads(line) for line in (bench / split / "sanawm_export_v2/run_manifest.jsonl").read_text().splitlines() if line.strip()]
row = next(r for r in rows if r["id"] == scene)
traj = np.load(bench / row["camera_path"])
np.save(work / f"{scene}_c2w.npy", traj["c2w"])
np.save(work / f"{scene}_intrinsics.npy", traj["intrinsics"])
(work / f"{scene}.txt").write_text(row["prompt"], encoding="utf-8")
PY

$PY inference_video_scripts/wm/inference_sana_wm.py \
  --image "$BENCH/images/$SCENE.png" \
  --prompt "$WORK/$SCENE.txt" \
  --camera "$WORK/${SCENE}_c2w.npy" \
  --intrinsics "$WORK/${SCENE}_intrinsics.npy" \
  --num_frames 961 --fps 16 \
  --step 60 --cfg_scale 5.0 --flow_shift 8.0 \
  --sampling_algo flow_euler_ltx \
  --seed 42 --refiner_seed 42 \
  --no_action_overlay \
  --output_dir "$OUT" \
  --name "$SCENE"
EOF

mkdir -p logs
sbatch --export=ALL,SPLIT_DIR=benchmark_v2_smooth_60s,SPLIT_NAME=simple_60s run_sana_wm_bench.sbatch
sbatch --export=ALL,SPLIT_DIR=benchmark_v2_hard_60s,SPLIT_NAME=hard_60s run_sana_wm_bench.sbatch
```

Each split is complete when it has 80 files matching `*_generated.mp4`.

## Metrics

The benchmark reports four metric families:

| Metric family | Primary output | Notes |
|---|---|---|
| VBench | `eval/<split>/vbench_scores.json` and raw `eval_*_eval_results.json` | Benchmark table setting uses 9 VBench dimensions: the 7 quality dimensions plus `overall_consistency` and `temporal_style`. |
| Revisit consistency | `eval/<split>/revisit_consistency.json` | PSNR, SSIM, and LPIPS on trajectory revisit frame pairs. |
| Camera / pose accuracy | `<split>/eval_poses.json` | Pi3X-estimated pose error against the benchmark camera path; aggregate fields are `RotErr`, `TransErr_rel`, and `CamMC_rel`. |
| Temporal degradation | `eval/<split>/temporal_degradation.json` | VBench quality decay across 10s windows; aggregate `Delta IQ` is first-window minus last-window imaging quality. |

For the benchmark comparison table, report columns with these exact
conventions:

- `VBench Overall`: `vbench_scores.json["quality_score"] * 100`. This is the
  normalized visual-quality aggregate over the available quality dimensions and
  is the source of values such as `79.55` and `81.10`.
- `VBench total_score`: keep as a raw 0-1 diagnostic only. It includes the
  semantic dimensions available in this 9-dimension run and should not be used
  as the table's `VBench Overall`.
- `RotErr`: mean `RotErr` in degrees from `eval_poses.json`.
- `TransErr` and `CamMC`: means of `TransErr_rel` and `CamMC_rel`.
- `Delta IQ`: `temporal_degradation.json["trend"]["imaging_quality"]["degradation"]`.

Use the evaluator scripts from your evaluation environment together with
this result layout:

```text
results/sana_wm_bidirectional_refined/
|-- method_info.json
|-- simple_60s/
|   `-- <scene_id>_generated.mp4
|-- hard_60s/
|   `-- <scene_id>_generated.mp4
`-- eval/
    |-- simple_60s/
    `-- hard_60s/
```

Keep the generated videos clean for metrics: use `*_generated.mp4`, not
overlay/combined videos.

Run VBench, revisit consistency, and temporal degradation with the unified
evaluator. Run from the model repository root and pass the benchmark metadata
explicitly:

```bash
BENCH=data/SANA-WM-Bench
METHOD_DIR=results/sana_wm_bidirectional_refined
PYTHONPATH="$PWD:${PYTHONPATH:-}" python /path/to/eval_unified.py \
  --method_dir "$METHOD_DIR" \
  --split simple_60s \
  --benchmark_meta "$BENCH/benchmark_v2_smooth_60s/scene_trajectories_v2.json" \
  --metrics vbench revisit temporal \
  --vbench_dims \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    overall_consistency temporal_style \
  --revisit_lpips \
  --window_sec 10 \
  --skip_first_frame auto

PYTHONPATH="$PWD:${PYTHONPATH:-}" python /path/to/eval_unified.py \
  --method_dir "$METHOD_DIR" \
  --split hard_60s \
  --benchmark_meta "$BENCH/benchmark_v2_hard_60s/scene_trajectories_v2.json" \
  --metrics vbench revisit temporal \
  --vbench_dims \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    overall_consistency temporal_style \
  --revisit_lpips \
  --window_sec 10 \
  --skip_first_frame auto
```

Camera accuracy is GPU-backed and should be run as a separate 8-GPU Slurm job.
Because the release manifest stores dataset-relative `camera_path` entries, run
the pose script from the dataset root while using absolute result and output
paths:

```bash
cat > run_sana_wm_pose_eval.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=swm_pose80
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

PY=/path/to/python
ACCEL=/path/to/accelerate
BENCH=data/SANA-WM-Bench
METHOD_DIR=/path/to/results/sana_wm_bidirectional_refined

cd "$BENCH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_pose() {
  local split="$1"
  local split_dir="$2"
  "$ACCEL" launch --num_processes=8 --mixed_precision=bf16 \
    /path/to/eval_benchmark_poses.py \
    --result_folder "$METHOD_DIR/$split" \
    --manifest "$BENCH/$split_dir/sanawm_export_v2/run_manifest.jsonl" \
    --interval 4 \
    --batch_size 1 \
    --output "$METHOD_DIR/$split/eval_poses.json" \
    --skip_first_frame auto
}

run_pose simple_60s benchmark_v2_smooth_60s
run_pose hard_60s benchmark_v2_hard_60s

"$PY" /path/to/aggregate_results.py \
  --results_root "$(dirname "$METHOD_DIR")" \
  --json_out "$METHOD_DIR/metrics_80scene/aggregate_results.json" \
  --md_out "$METHOD_DIR/metrics_80scene/aggregate_results.md"
EOF

mkdir -p logs
sbatch run_sana_wm_pose_eval.sbatch
```

Aggregate outputs into `metrics_80scene/aggregate_results.json` and
`metrics_80scene/aggregate_results.md`. The SANA-WM rows should include
`simple_60s` and `hard_60s` only.
