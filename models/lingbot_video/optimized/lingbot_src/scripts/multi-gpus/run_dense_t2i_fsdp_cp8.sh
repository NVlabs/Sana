#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/rewriter:${PYTHONPATH:-}"
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-}"
PROMPT_JSON="${PROMPT_JSON:-assets/cases/t2i/example_1/prompt.json}"
OUT_DIR="${OUT_DIR:-outputs/dense_t2i_fsdp_cp8_$(date +%Y%m%d_%H%M%S)}"

NPROC="${NPROC:-8}"
CP="${CP:-8}"
BACKEND="${BACKEND:-diffusers}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
STEPS="${STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"
SHIFT="${SHIFT:-3}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ -z "$MODEL_DIR" ]]; then
  echo "Set MODEL_DIR to a converted LingBot-Video Dense model directory." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  scripts/inference.py \
  --backend "$BACKEND" \
  --model_dir "$MODEL_DIR" \
  --mode t2i \
  --prompt_json "$PROMPT_JSON" \
  --output "$OUT_DIR/t2i.png" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS" \
  --guidance_scale "$GUIDANCE_SCALE" \
  --shift "$SHIFT" \
  --seed "$SEED" \
  --fps "$FPS" \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32 \
  --context_parallel_degree "$CP" \
  --context_parallel_ulysses_anything \
  --enable_fsdp_inference \
  --batch_cfg

echo "Saved: $OUT_DIR/t2i.png"
