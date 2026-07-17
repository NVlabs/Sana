#!/usr/bin/env bash
#SBATCH --job-name=lbv-megafuse
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/%x_%j.out
# Approach 1 (megafusion) end-to-end: single-GPU 480p base, cuDNN attention,
# LINGBOT_COMPILE_MOE off vs on. Compares base_denoise steady-state s/step.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh
export CUDA_VISIBLE_DEVICES=0
export LINGBOT_ATTN_KERNEL=cudnn
export LINGBOT_PHASE_TIMING=1

run () {
  local tag=$1 compile=$2
  local out="outputs/megafuse_${tag}_$(date +%H%M%S)"
  mkdir -p "$out"
  echo "########## RUN tag=$tag LINGBOT_COMPILE_MOE=$compile ##########"
  LINGBOT_COMPILE_MOE=$compile "$PYTHON_BIN" scripts/inference.py \
    --backend diffusers --model_dir "$MOE_MODEL_DIR" --mode t2v \
    --prompt_json assets/cases/t2v/example_1/prompt.json --duration 5 \
    --output "$out/t2v.mp4" --height 480 --width 832 --steps 40 \
    --guidance_scale 3 --shift 3 --seed 42 --fps 24 \
    --transformer_dtype bf16 --text_encoder_dtype bf16 --vae_dtype fp32 \
    --batch_cfg 2>&1 | tee "$out/run.log" | grep -iE "PHASE base_denoise|denoising:|compiled .* MoE|saved" | tr '\r' '\n' | tail -6
  echo
}

run baseline 0
run compiled 1
echo "DONE — compare the denoising s/it (steady-state, ignore step-1 compile) and PHASE base_denoise"
