#!/usr/bin/env bash
# Example launch script for Sana-WM camera-controlled video inference.
#
# Weights default to the public Hugging Face release and are downloaded
# on first use. Override with --config / --model_path / --refiner_* to
# point at local files instead.
#
# Camera trajectory: either pass --camera <(F,4,4).npy> or roll one out
# from the action DSL via --action (w/s = forward/back, a/d = yaw, i/k = pitch,
# j/l = strafe; held keys ease in/out with light inertia). Intrinsics are
# optional — if you don't pass --intrinsics, we estimate them with Pi3X (and
# abort if the estimated FOV is outside [25°, 120°]).
set -euo pipefail

python inference_video_scripts/inference_sana_wm.py \
  --image      asset/sana_wm/demo_0.png \
  --prompt     asset/sana_wm/demo_0.txt \
  --action     "w-100,dw-60,w-100,aw-60" \
  --output_dir results/sana_wm_demo \
  --name       demo_0 \
  --num_frames 321 \
  --fps        16 \
  --step       60 \
  --cfg_scale  5.0 \
  --flow_shift 8.0
