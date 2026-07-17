#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/rewriter:${PYTHONPATH:-}"
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"
export LINGBOT_MOE_PAD_BACKEND="${LINGBOT_MOE_PAD_BACKEND:-vectorized}"
export LINGBOT_MOE_EXPERT_BACKEND="${LINGBOT_MOE_EXPERT_BACKEND:-grouped_mm}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-}"
PROMPT_JSON="${PROMPT_JSON:-assets/cases/t2v/example_1/prompt.json}"
OUT_DIR="${OUT_DIR:-outputs/moe_t2v_refiner_fsdp_cp8_$(date +%Y%m%d_%H%M%S)}"

NPROC="${NPROC:-8}"
CP="${CP:-8}"
BACKEND="${BACKEND:-diffusers}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
STEPS="${STEPS:-40}"
REFINER_HEIGHT="${REFINER_HEIGHT:-1088}"
REFINER_WIDTH="${REFINER_WIDTH:-1920}"
REFINER_STEPS="${REFINER_STEPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"
REFINER_GUIDANCE_SCALE="${REFINER_GUIDANCE_SCALE:-3}"
SHIFT="${SHIFT:-3}"
REFINER_SHIFT="${REFINER_SHIFT:-3}"
REFINER_T_THRESH="${REFINER_T_THRESH:-0.85}"
REFINER_SIGMA_TAIL_STEPS="${REFINER_SIGMA_TAIL_STEPS:-2}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ -z "$MODEL_DIR" ]]; then
  echo "Set MODEL_DIR to a converted LingBot-Video MoE model directory with refiner weights." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  scripts/inference.py \
  --backend "$BACKEND" \
  --model_dir "$MODEL_DIR" \
  --run_refiner \
  --mode t2v \
  --prompt_json "$PROMPT_JSON" \
  --output "$OUT_DIR/t2v_base.mp4" \
  --refiner_output "$OUT_DIR/t2v_refined.mp4" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS" \
  --refiner_height "$REFINER_HEIGHT" \
  --refiner_width "$REFINER_WIDTH" \
  --refiner_steps "$REFINER_STEPS" \
  --guidance_scale "$GUIDANCE_SCALE" \
  --refiner_guidance_scale "$REFINER_GUIDANCE_SCALE" \
  --shift "$SHIFT" \
  --refiner_shift "$REFINER_SHIFT" \
  --refiner_t_thresh "$REFINER_T_THRESH" \
  --refiner_sigma_tail_steps "$REFINER_SIGMA_TAIL_STEPS" \
  --seed "$SEED" \
  --fps "$FPS" \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --refiner_vae_dtype fp32 \
  --context_parallel_degree "$CP" \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
  --batch_cfg \
  --refiner_batch_cfg \
  --reuse_condition_features

echo "Saved: $OUT_DIR/t2v_base.mp4"
echo "Saved: $OUT_DIR/t2v_refined.mp4"
