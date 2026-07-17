#!/usr/bin/env bash
#SBATCH -J wan720-sm100
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
# HSG's QOS requires a whole four-GPU node; CUDA_VISIBLE_DEVICES uses only GPU 0.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 04:00:00

set -euo pipefail

SOL_REPO=${SOL_REPO:?set SOL_REPO to this Sol-Attn checkout}
WAN_REPO=${WAN_REPO:?set WAN_REPO to the patched Sparse-VideoGen checkout}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to an output directory}
ENV_ACTIVATE=${ENV_ACTIVATE:?set ENV_ACTIVATE to the Wan environment activation script}
PROMPT="A sweeping drone shot follows a red vintage sports car racing along a coastal cliff road at golden hour. Waves crash below, sunlight flickers through sea mist, the camera banks smoothly around the car, cinematic realistic motion."

source "$ENV_ACTIVATE"
mkdir -p "$OUT_ROOT"
cd "$WAN_REPO"

export PYTHONPATH="$SOL_REPO:$WAN_REPO${WAN_FAST_KERNELS:+:$WAN_FAST_KERNELS}${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PISA2_AUTO_CALIBRATE=1
export PISA2_CALIBRATE_SAMPLE_HEADS=2
export PISA2_CALIBRATE_COARSE_START=0.0
export PISA2_CALIBRATE_COARSE_END=4.0
export PISA2_SM100_CORRECTNESS_GATE=1
export PISA2_SM100_EVENT_LOG="$OUT_ROOT/sm100_events.jsonl"
export PISA2_CALIBRATION_LOG="$OUT_ROOT/calibration.json"
export CUDA_VISIBLE_DEVICES=0

git status --short --branch >"$OUT_ROOT/wan_git_status.txt"
git rev-parse HEAD >"$OUT_ROOT/wan_commit.txt"
git -C "$SOL_REPO" status --short --branch >"$OUT_ROOT/sol_git_status.txt"
git -C "$SOL_REPO" rev-parse HEAD >"$OUT_ROOT/sol_commit.txt"
nvidia-smi -L >"$OUT_ROOT/nvidia_smi_l.txt"
python -c 'import sys, torch, triton, cutlass; print(sys.version); print(torch.__version__, torch.version.cuda); print(triton.__version__); print(cutlass.__file__)' \
  >"$OUT_ROOT/environment.txt"

/usr/bin/time -f 'real_sec=%e\nmax_rss_kb=%M' -o "$OUT_ROOT/time.txt" \
  python wan_t2v_inference.py \
    --model_id Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --pattern pisa2 \
    --pisa2_impl sm100-colmask \
    --prompt "$PROMPT" \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_inference_steps 50 \
    --seed 37 \
    --first_layers_fp 0.025 \
    --first_times_fp 0.2 \
    --pisa_density 0.15 \
    --pisa_threshold 1.7 \
    --pisa_block_size 64 \
    --pisa_reorder_order morton3d \
    --output_file "$OUT_ROOT/pisa2_sm100_colmask.mp4" \
    >"$OUT_ROOT/run.log" 2>&1

test -s "$OUT_ROOT/pisa2_sm100_colmask.mp4"
echo "WAN_720P_SM100_OK OUT=$OUT_ROOT"
