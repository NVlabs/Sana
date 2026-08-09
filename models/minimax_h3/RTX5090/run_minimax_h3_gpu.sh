#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_transfeat.py}"
: "${H3_MODEL_PATH:?H3_MODEL_PATH must point to the MiniMax-H3 FL2VA directory}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H3_ROOT="${H3_ROOT:-${HOME}/minimax_h3_5090}"
PYTHON_BIN="${PYTHON_BIN:-${H3_ROOT}/.venv/bin/python}"
FFMPEG_BIN_DIR="${H3_FFMPEG_BIN_DIR:-${H3_ROOT}/tools/ffmpeg-static}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is not available: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${H3_ROOT}/cache/huggingface"
mkdir -p "${H3_ROOT}/cache/triton" "${H3_ROOT}/cache/torchinductor-sm120"

export CUDA_VISIBLE_DEVICES="${H3_CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
if [[ -d "${FFMPEG_BIN_DIR}" ]]; then
  export PATH="${FFMPEG_BIN_DIR}:${PATH}"
fi
export PYTHONPATH="${HERE}:${HERE}/../../../techniques/sparse_backends${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${H3_ROOT}/cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${H3_ROOT}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${H3_ROOT}/cache/torchinductor-sm120}"
export SGLANG_USE_RUNAI_MODEL_STREAMER=false

for media_tool in ffmpeg ffprobe; do
  if ! command -v "${media_tool}" >/dev/null 2>&1; then
    echo "${media_tool} is required in ${FFMPEG_BIN_DIR}, $(dirname "${PYTHON_BIN}"), or PATH" >&2
    exit 2
  fi
done

export SOL_ATTN_STRICT=1
export SOL_ATTN_TAU="${SOL_ATTN_TAU:-1.0}"
export SOL_ATTN_THRESH_TYPE="${SOL_ATTN_THRESH_TYPE:-diag}"
export SOL_ATTN_FIRST_DENSE_STEPS="${SOL_ATTN_FIRST_DENSE_STEPS:-10}"
export SOL_ATTN_FIRST_DENSE_LAYER_RATIO="${SOL_ATTN_FIRST_DENSE_LAYER_RATIO:-0.03}"
export SOL_ATTN_FIRST_DENSE_LAYERS="${SOL_ATTN_FIRST_DENSE_LAYERS:-2}"
export SOL_ATTN_CORRECTNESS_GATE="${SOL_ATTN_CORRECTNESS_GATE:-1}"
if [[ -z "${SOL_ATTN_EVENT_LOG:-}" ]]; then
  export SOL_ATTN_EVENT_LOG="${OUT_DIR}/sol_events_rank{rank}.jsonl"
fi
export SOL_ATTN_REQUEST_EPOCH_FILE="${SOL_ATTN_REQUEST_EPOCH_FILE:-${OUT_DIR}/request_epoch.txt}"

export H3_TEACACHE_THRESHOLD="${H3_TEACACHE_THRESHOLD:-0.10}"
export H3_TEACACHE_RETAIN_STEPS="${H3_TEACACHE_RETAIN_STEPS:-5}"
export H3_TEACACHE_COOLDOWN_STEPS="${H3_TEACACHE_COOLDOWN_STEPS:-1}"
export H3_TEACACHE_NUM_FORWARDS="${H3_TEACACHE_NUM_FORWARDS:-49}"
export H3_TEACACHE_COEFFICIENTS="${H3_TEACACHE_COEFFICIENTS:-1.0,0.0}"
export H3_FULL_VAE_DTYPE="${H3_FULL_VAE_DTYPE:-bfloat16}"
export H3_FULL_VAE_TILE_BATCH_SIZE="${H3_FULL_VAE_TILE_BATCH_SIZE:-0}"

case "${H3_RTX5090_PROFILE:-dense}" in
  dense)
    export H3_TEACACHE_ENABLED=0
    export H3_FULL_VAE_AFTER_DENOISE=0
    ;;
  sol)
    export H3_TEACACHE_ENABLED=0
    export H3_FULL_VAE_AFTER_DENOISE=0
    ;;
  fullopt)
    export H3_TEACACHE_ENABLED="${H3_TEACACHE_ENABLED:-1}"
    export H3_FULL_VAE_AFTER_DENOISE="${H3_FULL_VAE_AFTER_DENOISE:-1}"
    export H3_FULL_VAE_TORCH_COMPILE="${H3_FULL_VAE_TORCH_COMPILE:-0}"
    ;;
  *)
    echo "H3_RTX5090_PROFILE must be dense, sol, or fullopt" >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" "${HERE}/gpu_infer.py"
