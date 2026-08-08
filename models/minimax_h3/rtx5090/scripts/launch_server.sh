#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RUNTIME_ROOT}/../../.." && pwd)"

H3_ROOT="${H3_ROOT:-${HOME}/minimax_h3_5090}"
SGLANG_ROOT="${H3_SGLANG_ROOT:-${H3_ROOT}/sglang}"
MODEL_PATH="${H3_MODEL_PATH:-${H3_ROOT}/model/FL2VA}"
PROFILE="${H3_RTX5090_PROFILE:-dense}"
PORT="${H3_PORT:-30010}"
MASTER_PORT="${H3_MASTER_PORT:-30005}"
VENV_ROOT="${H3_VENV_ROOT:-${H3_ROOT}/.venv}"
SGLANG_BIN="${H3_SGLANG_BIN:-${VENV_ROOT}/bin/sglang}"

UPSTREAM_H3="${SGLANG_ROOT}/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
UPSTREAM_DENOISING="${SGLANG_ROOT}/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py"
UPSTREAM_DECODING="${SGLANG_ROOT}/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py"
EXPECTED_H3_SHA256="5f87319969c446685ee93d422fc34a7c040defb238eff2274d664f2f8310e997"
EXPECTED_DENOISING_SHA256="2325b039c055d2db8c320a00f65e2880816888fd5fa107b72cfd6a85be104399"
EXPECTED_DECODING_SHA256="5e2cf87da11e0c744d6c7703d8151abd7da7937e1ad67d576fdbf6c678380954"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_source() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
  actual="$(sha256_file "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Refusing to patch unexpected ${label}: ${actual}" >&2
    exit 2
  fi
}

if [[ ! -x "${SGLANG_BIN}" ]]; then
  echo "SGLang executable is not available: ${SGLANG_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MiniMax-H3 FL2VA model directory is not available: ${MODEL_PATH}" >&2
  exit 2
fi

verify_source "${UPSTREAM_H3}" "${EXPECTED_H3_SHA256}" "MiniMax-H3 DiT source"

compile="false"
regional_compile="false"
server_warmup="false"
patch_denoising=0
patch_decoding=0
case "${PROFILE}" in
  dense)
    export H3_TEACACHE_ENABLED=0
    export H3_FULL_VAE_AFTER_DENOISE=0
    ;;
  sol)
    export H3_TEACACHE_ENABLED=0
    export H3_FULL_VAE_AFTER_DENOISE=0
    ;;
  fullopt)
    verify_source \
      "${UPSTREAM_DENOISING}" \
      "${EXPECTED_DENOISING_SHA256}" \
      "SGLang denoising source"
    verify_source \
      "${UPSTREAM_DECODING}" \
      "${EXPECTED_DECODING_SHA256}" \
      "MiniMax-H3 decoding source"
    compile="${H3_FULL_OPT_COMPILE:-true}"
    regional_compile="${H3_FULL_OPT_REGIONAL_COMPILE:-true}"
    server_warmup="${H3_SERVER_WARMUP:-true}"
    patch_denoising=1
    patch_decoding=1
    export H3_TEACACHE_ENABLED="${H3_TEACACHE_ENABLED:-1}"
    export H3_FULL_VAE_AFTER_DENOISE="${H3_FULL_VAE_AFTER_DENOISE:-1}"
    ;;
  *)
    echo "H3_RTX5090_PROFILE must be dense, sol, or fullopt" >&2
    exit 2
    ;;
esac

python_root="${SGLANG_ROOT}/python"
if [[ "${PROFILE}" != "dense" ]]; then
  patched_h3_sha="$(sha256_file "${RUNTIME_ROOT}/patches/minimax_h3.py")"
  overlay_key="${patched_h3_sha:0:12}"
  if [[ "${patch_denoising}" == "1" ]]; then
    patched_denoising_sha="$(sha256_file "${RUNTIME_ROOT}/patches/denoising.py")"
    overlay_key="${overlay_key}-${patched_denoising_sha:0:12}"
  fi
  if [[ "${patch_decoding}" == "1" ]]; then
    patched_decoding_sha="$(sha256_file "${RUNTIME_ROOT}/patches/minimax_h3_decoding.py")"
    overlay_key="${overlay_key}-${patched_decoding_sha:0:12}"
  fi
  overlay="${H3_OVERLAY_ROOT:-${H3_ROOT}/cache/overlays}/sglang-h3-${PROFILE}-${overlay_key}"
  if [[ ! -s "${overlay}/.ready" ]]; then
    if [[ -e "${overlay}" ]]; then
      echo "Incomplete source overlay already exists: ${overlay}" >&2
      exit 2
    fi
    mkdir -p "${overlay}"
    cp -a "${SGLANG_ROOT}/python/sglang" "${overlay}/"
    install -m 0644 \
      "${RUNTIME_ROOT}/patches/minimax_h3.py" \
      "${overlay}/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
    if [[ "${patch_denoising}" == "1" ]]; then
      install -m 0644 \
        "${RUNTIME_ROOT}/patches/denoising.py" \
        "${overlay}/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py"
    fi
    if [[ "${patch_decoding}" == "1" ]]; then
      install -m 0644 \
        "${RUNTIME_ROOT}/patches/minimax_h3_decoding.py" \
        "${overlay}/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py"
    fi
    printf '%s\n' "${overlay_key}" >"${overlay}/.ready"
  fi
  python_root="${overlay}"
fi

mkdir -p "${H3_ROOT}/cache/huggingface" "${H3_ROOT}/cache/xdg"
mkdir -p "${H3_ROOT}/cache/triton" "${H3_ROOT}/cache/torchinductor-sm120"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="${VENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${python_root}:${RUNTIME_ROOT}:${REPO_ROOT}/techniques/sparse_backends${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${H3_ROOT}/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${H3_ROOT}/cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${H3_ROOT}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${H3_ROOT}/cache/torchinductor-sm120}"
export SGLANG_USE_RUNAI_MODEL_STREAMER=false

export H3_TEACACHE_THRESHOLD="${H3_TEACACHE_THRESHOLD:-0.10}"
export H3_TEACACHE_RETAIN_STEPS="${H3_TEACACHE_RETAIN_STEPS:-5}"
export H3_TEACACHE_COOLDOWN_STEPS="${H3_TEACACHE_COOLDOWN_STEPS:-1}"
export H3_TEACACHE_NUM_FORWARDS="${H3_TEACACHE_NUM_FORWARDS:-49}"
export H3_TEACACHE_COEFFICIENTS="${H3_TEACACHE_COEFFICIENTS:-1.0,0.0}"
export H3_FULL_VAE_TORCH_COMPILE="${H3_FULL_VAE_TORCH_COMPILE:-0}"
export H3_FULL_VAE_DTYPE="${H3_FULL_VAE_DTYPE:-bfloat16}"
export H3_FULL_VAE_TILE_BATCH_SIZE="${H3_FULL_VAE_TILE_BATCH_SIZE:-0}"

export SOL_ATTN_STRICT=1
export SOL_ATTN_TAU="${SOL_ATTN_TAU:-1.0}"
export SOL_ATTN_THRESH_TYPE="${SOL_ATTN_THRESH_TYPE:-diag}"
export SOL_ATTN_FIRST_DENSE_STEPS="${SOL_ATTN_FIRST_DENSE_STEPS:-10}"
export SOL_ATTN_FIRST_DENSE_LAYER_RATIO="${SOL_ATTN_FIRST_DENSE_LAYER_RATIO:-0.03}"
export SOL_ATTN_FIRST_DENSE_LAYERS="${SOL_ATTN_FIRST_DENSE_LAYERS:-2}"
export SOL_ATTN_CORRECTNESS_GATE="${SOL_ATTN_CORRECTNESS_GATE:-1}"
export SOL_ATTN_FORCE_DENSE=0

exec "${SGLANG_BIN}" serve \
  --model-path "${MODEL_PATH}" \
  --model-subfolder . \
  --num-gpus 1 \
  --tp-size 1 \
  --ulysses-degree 1 \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --dit-offload-prefetch-size "${H3_DIT_OFFLOAD_PREFETCH_SIZE:-1}" \
  --dit-layerwise-resident-layers "${H3_DIT_RESIDENT_LAYERS:-0}" \
  --pin-cpu-memory false \
  --enable-torch-compile "${compile}" \
  --regional-compile "${regional_compile}" \
  --server-warmup "${server_warmup}" \
  --master-port "${MASTER_PORT}" \
  --host 127.0.0.1 \
  --port "${PORT}"
