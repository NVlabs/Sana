#!/usr/bin/env bash
#SBATCH --job-name=lbv-bc
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/%x_%j.out
# Official multi-GPU config adapted to a single 4-GPU node:
# CP4 (Ulysses context parallel) + FSDP + batch_cfg. This is the proper multi-GPU
# baseline AND the first real acceleration for the refiner (CP shards the 1080p
# attention sequence across GPUs, unlike FSDP-only which gave 0 speedup).
# Requires FA2 (flash_attn_interface shim) -> use the lingbot-fa2 env.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh   # CONDA_ENV defaults to lingbot-fa2 (FA2)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LINGBOT_PHASE_TIMING=1
export LINGBOT_OVERLAP_REFINER_LOAD=1
export LINGBOT_BCAST_WEIGHTS=1
NPROC="${NPROC:-4}"; CP="${CP:-4}"

PROMPT_JSON="${PROMPT_JSON:-assets/cases/t2v/example_1/prompt.json}"
OUT_DIR="${OUT_DIR:-outputs/c3_bcast_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-832}"; STEPS="${STEPS:-40}"
REFINER_HEIGHT="${REFINER_HEIGHT:-1088}"; REFINER_WIDTH="${REFINER_WIDTH:-1920}"; REFINER_STEPS="${REFINER_STEPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"; REFINER_GUIDANCE_SCALE="${REFINER_GUIDANCE_SCALE:-3}"
SHIFT="${SHIFT:-3}"; REFINER_SHIFT="${REFINER_SHIFT:-3}"
REFINER_T_THRESH="${REFINER_T_THRESH:-0.85}"; REFINER_SIGMA_TAIL_STEPS="${REFINER_SIGMA_TAIL_STEPS:-2}"
FPS="${FPS:-24}"; DURATION="${DURATION:-5}"; SEED="${SEED:-42}"

( peak=0; while true; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null && peak=$u
    echo "$peak" > "$OUT_DIR/gpu_peak_mib.txt"; sleep 3
  done ) & SAMPLER_PID=$!

echo "[CP4] base+refiner  CP=${CP} + FSDP + batch_cfg  env=${CONDA_ENV} moe=${LINGBOT_MOE_EXPERT_BACKEND}"
/usr/bin/time -v "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  scripts/inference.py \
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
  --context_parallel_degree "$CP" \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
  --batch_cfg --refiner_batch_cfg \
  --reuse_condition_features \
  2>&1 | tee "$OUT_DIR/run.log"

kill $SAMPLER_PID 2>/dev/null
echo "[CP4] peak GPU mem (max over 4): $(cat "$OUT_DIR/gpu_peak_mib.txt" 2>/dev/null) MiB"
echo "[CP4] done -> $OUT_DIR"
