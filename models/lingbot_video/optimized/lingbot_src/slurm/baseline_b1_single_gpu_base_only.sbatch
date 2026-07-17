#!/usr/bin/env bash
#SBATCH --job-name=lbv-b1-baseline
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/%x_%j.out
# NOTE: QOS requires a full node (gpu:4); the job itself uses only 1 GPU (CUDA_VISIBLE_DEVICES=0).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh

export CUDA_VISIBLE_DEVICES=0

PROMPT_JSON="${PROMPT_JSON:-assets/cases/t2v/example_1/prompt.json}"
OUT_DIR="${OUT_DIR:-outputs/baseline_b1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-832}"
STEPS="${STEPS:-40}"; GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"; SHIFT="${SHIFT:-3}"
FPS="${FPS:-24}"; DURATION="${DURATION:-5}"; SEED="${SEED:-42}"

# batch_cfg batches cond+uncond (B=2), which requires the FA3 varlen attention path.
# The pure baseline uses sequential CFG (B=1) + SDPA. Set BATCH_CFG=1 to enable (needs flash_attn).
BATCH_CFG_FLAG=""
[ "${BATCH_CFG:-0}" = "1" ] && BATCH_CFG_FLAG="--batch_cfg"

echo "[B1] base-only single-GPU baseline  backend=diffusers moe=${LINGBOT_MOE_EXPERT_BACKEND} batch_cfg=${BATCH_CFG:-0}"
/usr/bin/time -v "$PYTHON_BIN" scripts/inference.py \
  --backend diffusers \
  --model_dir "$MOE_MODEL_DIR" \
  --mode t2v \
  --prompt_json "$PROMPT_JSON" \
  --duration "$DURATION" \
  --output "$OUT_DIR/t2v.mp4" \
  --height "$HEIGHT" --width "$WIDTH" \
  --steps "$STEPS" --guidance_scale "$GUIDANCE_SCALE" --shift "$SHIFT" \
  --seed "$SEED" --fps "$FPS" \
  --transformer_dtype bf16 --text_encoder_dtype bf16 --vae_dtype fp32 \
  $BATCH_CFG_FLAG \
  2>&1 | tee "$OUT_DIR/run.log"

echo "[B1] done -> $OUT_DIR"
