#!/usr/bin/env bash
#SBATCH --job-name=lbv-b2-baseline
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/%x_%j.out
# End-to-end product latency baseline: base (40 steps @480p) + refiner (8 steps @1080p), single GPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh

export CUDA_VISIBLE_DEVICES=0

PROMPT_JSON="${PROMPT_JSON:-assets/cases/t2v/example_1/prompt.json}"
OUT_DIR="${OUT_DIR:-outputs/baseline_b2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-832}"; STEPS="${STEPS:-40}"
REFINER_HEIGHT="${REFINER_HEIGHT:-1088}"; REFINER_WIDTH="${REFINER_WIDTH:-1920}"; REFINER_STEPS="${REFINER_STEPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"; REFINER_GUIDANCE_SCALE="${REFINER_GUIDANCE_SCALE:-3}"
SHIFT="${SHIFT:-3}"; REFINER_SHIFT="${REFINER_SHIFT:-3}"
REFINER_T_THRESH="${REFINER_T_THRESH:-0.85}"; REFINER_SIGMA_TAIL_STEPS="${REFINER_SIGMA_TAIL_STEPS:-2}"
FPS="${FPS:-24}"; DURATION="${DURATION:-5}"; SEED="${SEED:-42}"

# batch_cfg (base+refiner) needs the FA3 varlen path; gate it (default off = sequential CFG + SDPA).
CFG_FLAGS=""
[ "${BATCH_CFG:-0}" = "1" ] && CFG_FLAGS="--batch_cfg --refiner_batch_cfg"

# Sample peak GPU memory in the background while the job runs.
( peak=0; while true; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null && peak=$u
    echo "$peak" > "$OUT_DIR/gpu_peak_mib.txt"; sleep 3
  done ) &
SAMPLER_PID=$!

echo "[B2] base+refiner single-GPU baseline  backend=diffusers moe=${LINGBOT_MOE_EXPERT_BACKEND} batch_cfg=${BATCH_CFG:-0}"
/usr/bin/time -v "$PYTHON_BIN" scripts/inference.py \
  --backend diffusers \
  --model_dir "$MOE_MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "$PROMPT_JSON" \
  --duration "$DURATION" \
  --output "$OUT_DIR/t2v_base.mp4" \
  --refiner_output "$OUT_DIR/t2v_refined.mp4" \
  --height "$HEIGHT" --width "$WIDTH" --steps "$STEPS" \
  --refiner_height "$REFINER_HEIGHT" --refiner_width "$REFINER_WIDTH" --refiner_steps "$REFINER_STEPS" \
  --guidance_scale "$GUIDANCE_SCALE" --refiner_guidance_scale "$REFINER_GUIDANCE_SCALE" \
  --shift "$SHIFT" --refiner_shift "$REFINER_SHIFT" \
  --refiner_t_thresh "$REFINER_T_THRESH" --refiner_sigma_tail_steps "$REFINER_SIGMA_TAIL_STEPS" \
  --seed "$SEED" --fps "$FPS" \
  --transformer_dtype bf16 --text_encoder_dtype bf16 --vae_dtype fp32 --refiner_vae_dtype fp32 \
  $CFG_FLAGS \
  --reuse_condition_features \
  2>&1 | tee "$OUT_DIR/run.log"

kill $SAMPLER_PID 2>/dev/null
echo "[B2] peak GPU mem: $(cat "$OUT_DIR/gpu_peak_mib.txt" 2>/dev/null) MiB"
echo "[B2] done -> $OUT_DIR"
