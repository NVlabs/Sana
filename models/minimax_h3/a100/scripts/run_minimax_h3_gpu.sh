#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_candidate.py}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
RUNTIME_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)
REPO_ROOT=$(cd "${RUNTIME_ROOT}/../../.." && pwd -P)

export H3_CONTAINER_RUNTIME=${H3_CONTAINER_RUNTIME:-none}
requested_runtime=${H3_CONTAINER_RUNTIME}
export H3_CONTAINER_IMAGE=${H3_CONTAINER_IMAGE:-docker://lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86}
export H3_MODEL_PATH=${H3_MODEL_PATH:-MiniMaxAI/MiniMax-H3}
export H3_MODEL_REVISION=${H3_MODEL_REVISION:-bfc8ed0353f5a9733be73e6b2c98ec0948195b86}
export H3_PROMPT_FILE=${H3_PROMPT_FILE:-${REPO_ROOT}/models/minimax_h3/prompts/t2va_example_1.json}
export H3_WARMUP_NUM_STEPS=${H3_WARMUP_NUM_STEPS:-50}
export H3_MEASURED_NUM_STEPS=${H3_MEASURED_NUM_STEPS:-50}
export H3_DURATION_SECONDS=${H3_DURATION_SECONDS:-5.166667}
export H3_SEED=${H3_SEED:-0}
export H3_WARMUP_SEED=${H3_WARMUP_SEED:-10000}
export H3_MASTER_PORT=${H3_MASTER_PORT:-30005}
export H3_MODEL_SUBFOLDER=${H3_MODEL_SUBFOLDER:-}
export H3_EXPECTED_TORCH=${H3_EXPECTED_TORCH:-2.11.0+cu130}
export H3_EXPECTED_TRITON=${H3_EXPECTED_TRITON:-3.6.0}
export H3_STORAGE_ROOT=${H3_STORAGE_ROOT:-${REPO_ROOT}}
export H3_CACHE_ROOT=${H3_CACHE_ROOT:-${H3_STORAGE_ROOT}/cache}
export H3_SGLANG_PYTHON_ROOT=${H3_SGLANG_PYTHON_ROOT:-/sgl-workspace/sglang/python}
export H3_PYTHON_BIN=${H3_PYTHON_BIN:-python3}
export OUT_DIR

if [[ "${H3_PROMPT_FILE}" != /* ]]; then
  export H3_PROMPT_FILE=${REPO_ROOT}/${H3_PROMPT_FILE}
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${H3_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${H3_CACHE_ROOT}/triton}
export TORCH_HOME=${TORCH_HOME:-${H3_CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${H3_CACHE_ROOT}/xdg}
export TMPDIR=${TMPDIR:-/tmp}
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/techniques/sparse_backends:${H3_SGLANG_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${OUT_DIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

if [[ "${H3_CONTAINER_RUNTIME}" == none ]]; then
  "${H3_PYTHON_BIN}" "${RUNTIME_ROOT}/gpu_infer.py" 2>&1 | tee "${OUT_DIR}/run.log"
  exit "${PIPESTATUS[0]}"
fi

H3_STORAGE_ROOT=$(cd "${H3_STORAGE_ROOT}" && pwd -P)
host_storage_root=${H3_STORAGE_ROOT}
OUT_DIR=$(cd "${OUT_DIR}" && pwd -P)
H3_CACHE_ROOT=$(cd "${H3_CACHE_ROOT}" && pwd -P)
prompt_dir=$(cd "$(dirname "${H3_PROMPT_FILE}")" && pwd -P)
H3_PROMPT_FILE=${prompt_dir}/$(basename "${H3_PROMPT_FILE}")

require_below_storage() {
  local label=$1
  local path=$2
  case "${path}" in
    "${H3_STORAGE_ROOT}"|"${H3_STORAGE_ROOT}"/*) ;;
    *)
      echo "${label} must be below H3_STORAGE_ROOT for container runs: ${path}" >&2
      exit 2
      ;;
  esac
}

to_inside_path() {
  local path=$1
  if [[ "${path}" == "${H3_STORAGE_ROOT}" ]]; then
    printf '/h3'
  else
    printf '/h3/%s' "${path#${H3_STORAGE_ROOT}/}"
  fi
}

require_below_storage REPO_ROOT "${REPO_ROOT}"
require_below_storage OUT_DIR "${OUT_DIR}"
require_below_storage H3_CACHE_ROOT "${H3_CACHE_ROOT}"
require_below_storage H3_PROMPT_FILE "${H3_PROMPT_FILE}"
inside_repo=$(to_inside_path "${REPO_ROOT}")
inside_output=$(to_inside_path "${OUT_DIR}")
inside_cache=$(to_inside_path "${H3_CACHE_ROOT}")
inside_prompt=$(to_inside_path "${H3_PROMPT_FILE}")
inside_model=${H3_MODEL_PATH}
if [[ "${H3_MODEL_PATH}" == /* ]]; then
  model_dir=$(cd "${H3_MODEL_PATH}" && pwd -P)
  require_below_storage H3_MODEL_PATH "${model_dir}"
  inside_model=$(to_inside_path "${model_dir}")
fi

export H3_CONTAINER_RUNTIME=none
export H3_STORAGE_ROOT=/h3
export H3_CACHE_ROOT=${inside_cache}
export H3_PROMPT_FILE=${inside_prompt}
export H3_MODEL_PATH=${inside_model}
export OUT_DIR=${inside_output}
export PYTHONPATH=${inside_repo}:${inside_repo}/techniques/sparse_backends:${H3_SGLANG_PYTHON_ROOT}
export HF_HOME=${inside_cache}/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRITON_CACHE_DIR=${inside_cache}/triton
export TORCH_HOME=${inside_cache}/torch
export XDG_CACHE_HOME=${inside_cache}/xdg
export TMPDIR=/tmp

case "${requested_runtime}" in
  pyxis)
    container_env=OUT_DIR,PYTHONPATH,H3_CONTAINER_RUNTIME,H3_STORAGE_ROOT,H3_CACHE_ROOT,H3_SGLANG_PYTHON_ROOT,H3_PYTHON_BIN,H3_MODEL_PATH,H3_MODEL_REVISION,H3_MODEL_SUBFOLDER,H3_PROMPT_FILE,H3_WARMUP_NUM_STEPS,H3_MEASURED_NUM_STEPS,H3_DURATION_SECONDS,H3_SEED,H3_WARMUP_SEED,H3_MASTER_PORT,H3_SOL_PROFILE,H3_EXPECTED_TORCH,H3_EXPECTED_TRITON,HF_HOME,HUGGINGFACE_HUB_CACHE,HF_HUB_DISABLE_XET,HF_HUB_DOWNLOAD_TIMEOUT,HF_HUB_OFFLINE,TRITON_CACHE_DIR,TORCH_HOME,XDG_CACHE_HOME,TMPDIR,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,NUMEXPR_NUM_THREADS,TOKENIZERS_PARALLELISM,PYTHONUNBUFFERED
    exec srun \
      --ntasks=1 \
      --nodes=1 \
      --container-image="${H3_CONTAINER_IMAGE}" \
      --container-mounts="${host_storage_root}:/h3" \
      --container-env="${container_env}" \
      --no-container-mount-home \
      --container-workdir="${inside_output}" \
      --no-container-entrypoint \
      bash "${inside_repo}/models/minimax_h3/a100/scripts/run_minimax_h3_gpu.sh"
    ;;
  apptainer|singularity)
    if ! command -v "${requested_runtime}" >/dev/null 2>&1; then
      for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -f "${init_script}" ]]; then
          # shellcheck disable=SC1090
          source "${init_script}"
          break
        fi
      done
      module load "${H3_CONTAINER_MODULE:-singularity/4.4.1}"
    fi
    exec "${requested_runtime}" exec \
      --nv \
      --bind "${host_storage_root}:/h3" \
      "${H3_CONTAINER_IMAGE}" \
      bash "${inside_repo}/models/minimax_h3/a100/scripts/run_minimax_h3_gpu.sh"
    ;;
  *)
    echo "H3_CONTAINER_RUNTIME must be none, pyxis, apptainer, or singularity" >&2
    exit 2
    ;;
esac
