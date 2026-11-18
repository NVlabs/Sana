#!/usr/bin/env bash
# Host interpreter adapter for the pipeline's persistent Qwen JSONL worker.
set -euo pipefail
: "${SOL_H3_SPARK_RUNTIME_ROOT:?Set a dedicated absolute directory containing code, dependencies, weights and request outputs}"
: "${SOL_H3_SPARK_QWEN_IMAGE:?Set the built Qwen image or an existing verified image digest}"
readonly sol_spark_package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${SOL_H3_SPARK_RUNTIME_ROOT}" in
  /*) ;;
  *) printf '%s\n' 'SOL_H3_SPARK_RUNTIME_ROOT must be absolute' >&2; exit 2 ;;
esac
if [[ ! -d "${SOL_H3_SPARK_RUNTIME_ROOT}" || "${SOL_H3_SPARK_RUNTIME_ROOT}" == / ]]; then
  printf '%s\n' 'Use an existing dedicated runtime directory' >&2
  exit 2
fi
readonly sol_spark_user="$(id -un)"
readonly sol_spark_home="${SOL_H3_SPARK_RUNTIME_ROOT}/.qwen-home"
readonly sol_spark_cache="${sol_spark_home}/.cache"
mkdir -p "${sol_spark_cache}/torchinductor"
mounts=(--mount "type=bind,src=${SOL_H3_SPARK_RUNTIME_ROOT},dst=${SOL_H3_SPARK_RUNTIME_ROOT}")
mounts+=(--mount "type=bind,src=${sol_spark_package},dst=${sol_spark_package},readonly")
for readonly_root in "${SOL_H3_SPARK_QWEN_WEIGHTS_ROOT:-}" "${SOL_H3_SPARK_COMFY_ROOT:-}"; do
  if [[ -n "${readonly_root}" ]]; then
    if [[ "${readonly_root}" != /* || ! -d "${readonly_root}" || "${readonly_root}" == / ]]; then
      printf '%s\n' 'Optional weight and Comfy roots must be existing absolute directories' >&2
      exit 2
    fi
    mounts+=(--mount "type=bind,src=${readonly_root},dst=${readonly_root},readonly")
  fi
done
readonly sol_spark_container="sol-h3-qwen-$$-${RANDOM}-${RANDOM}"
if docker container inspect "${sol_spark_container}" >/dev/null 2>&1; then
  printf '%s\n' 'Generated container name already exists; refusing to reuse it' >&2
  exit 2
fi
cleanup() {
  local result=$?
  trap - EXIT TERM INT
  if docker container inspect "${sol_spark_container}" >/dev/null 2>&1; then
    docker stop --time=5 "${sol_spark_container}" >/dev/null 2>&1 || true
  fi
  exit "${result}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
docker run --rm --name "${sol_spark_container}" --pull=never --gpus device=0 --ipc=host --network=none -i \
  --user "$(id -u):$(id -g)" \
  "${mounts[@]}" \
  --workdir "${SOL_H3_SPARK_RUNTIME_ROOT}" \
  -e "USER=${sol_spark_user}" -e "LOGNAME=${sol_spark_user}" \
  -e "HOME=${sol_spark_home}" -e "XDG_CACHE_HOME=${sol_spark_cache}" \
  -e "TORCHINDUCTOR_CACHE_DIR=${sol_spark_cache}/torchinductor" \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "PYTHONPATH=${sol_spark_package}" \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint "${SOL_H3_SPARK_QWEN_PYTHON:-/usr/bin/python3}" \
  "${SOL_H3_SPARK_QWEN_IMAGE}" "$@" <&0 &
sol_spark_docker_pid=$!
wait "${sol_spark_docker_pid}"
