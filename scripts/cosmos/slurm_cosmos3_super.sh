#!/bin/bash
#SBATCH --job-name=cosmos3-super
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=cosmos3-super-%x-%j.out
#SBATCH --error=cosmos3-super-%x-%j.out

set -euo pipefail
# Cosmos3-Super (64b) inference entry — two clean modes, OFFICIAL settings,
# 4-GPU sequence parallel (T2V; set IMAGE_PATH + NUM_FRAMES=189 for the I2V
# cascade stage):
#   baseline : no acceleration (dense reference)
#   fullopt  : TeaCache 1.15/start10/max3 + step-selective NVFP4 (first/last 3 steps dense)
# Usage: sbatch scripts/cosmos/slurm_cosmos3_super.sh [baseline|fullopt]   (default fullopt)
#
# Required env: MODEL_REPO (e.g. nvidia/Cosmos3-Super), ROOT, PROMPT_FILE, PROMPT_TAG
# Optional env: WARMUP, NUM_FRAMES, IMAGE_PATH, NUM_GPUS

MODE="${1:-fullopt}"
case "$MODE" in
  baseline)
    VARIANT="baseline"
    ;;
  fullopt)
    VARIANT="teacache_c115_s10_m3"   # TeaCache thr1.15 / start10 / max3
    export SGLANG_COSMOS3_FP4_LINEAR=1
    export SGLANG_COSMOS3_FP4_TARGETS="${SGLANG_COSMOS3_FP4_TARGETS:-gate_up,down,qkv,out}"
    export SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS="${SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS:-3}"
    export SGLANG_COSMOS3_FP4_SKIP_LAST_STEPS="${SGLANG_COSMOS3_FP4_SKIP_LAST_STEPS:-3}"
    ;;
  *)
    echo "Usage: sbatch $0 [baseline|fullopt]" >&2; exit 2
    ;;
esac

: "${MODEL_REPO:?}" ; : "${ROOT:?}" ; : "${PROMPT_FILE:?}" ; : "${PROMPT_TAG:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.conda/ltx23/bin/python}"
CACHE="${SGLANG_CACHE_ROOT:-$REPO_ROOT/outputs/.cache}"
mkdir -p "$ROOT/logs" "$CACHE"/{xdg,torch,triton,torchinductor,torch_extensions,cuda,sgl_diffusion} "$REPO_ROOT/outputs/.tmp"
cd "$REPO_ROOT"
echo "[$(date)] Node $(hostname)  MODE=$MODE  MODEL=$MODEL_REPO  TAG=$PROMPT_TAG  frames=${NUM_FRAMES:-189}  image=${IMAGE_PATH:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
SP="$REPO_ROOT/.conda/ltx23/lib/python3.12/site-packages/nvidia"
if [[ -d "$SP/cu13" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$SP/cu13}"; export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
export XDG_CACHE_HOME=$CACHE/xdg TORCH_HOME=$CACHE/torch TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor TORCH_EXTENSIONS_DIR=$CACHE/torch_extensions
export CUDA_CACHE_PATH=$CACHE/cuda SGLANG_DIFFUSION_CACHE_ROOT=$CACHE/sgl_diffusion TMPDIR=$REPO_ROOT/outputs/.tmp

REPO_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_REPO/\//--}"
HASH=$(cat "$REPO_CACHE_DIR/refs/main")
LOCAL="$REPO_CACHE_DIR/snapshots/$HASH"
test -f "$LOCAL/model_index.json" || { echo "ERROR model_index.json missing for $MODEL_REPO"; exit 1; }

# negative prompt: honor $NEGATIVE_PROMPT, else a file ($NEGATIVE_PROMPT_FILE), else
# let run_cosmos3_cache_matrix.sh use its built-in default.
NEG="${NEGATIVE_PROMPT:-}"
[[ -z "$NEG" && -n "${NEGATIVE_PROMPT_FILE:-}" && -f "$NEGATIVE_PROMPT_FILE" ]] && NEG="$(cat "$NEGATIVE_PROMPT_FILE")"
PROMPT_STR="$(cat "$PROMPT_FILE")"

ROOT="$ROOT" \
MODEL_SIZES=64b \
VARIANTS="$VARIANT" \
PROMPT_COUNT=1 \
PROMPT_0="$PROMPT_STR" \
NEGATIVE_PROMPT="$NEG" \
HEIGHT=720 WIDTH=1280 NUM_FRAMES="${NUM_FRAMES:-189}" FPS=24 \
NUM_INFERENCE_STEPS=35 GUIDANCE_SCALE=6.0 FLOW_SHIFT=10.0 MAX_SEQUENCE_LENGTH=4096 \
PYTHON_BIN="$PYTHON_BIN" \
COSMOS3_64B_MODEL_PATH="$LOCAL" \
DIT_CPU_OFFLOAD=false \
COSMOS3_64B_NUM_GPUS="${NUM_GPUS:-4}" \
WARMUP="${WARMUP:-true}" WARMUP_STEPS=1 \
FORCE="${FORCE:-1}" \
MAKE_COMPARE=0 \
ALLOW_PARTIAL=1 \
bash scripts/cosmos/run_cosmos3_cache_matrix.sh

echo "[$(date)] DONE $MODE $MODEL_REPO $PROMPT_TAG -> $ROOT"
