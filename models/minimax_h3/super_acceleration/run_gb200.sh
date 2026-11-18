#!/usr/bin/env bash
# Two independent serial pipeline pairs on one four-GB200 node.  Every request
# uses one Stage-1 GPU and one Stage-2 GPU; no request uses TP, CP, or FSDP.
#
# Site account/partition/QoS are intentionally not pinned here.  Supply them to
# sbatch, for example:
#   sbatch -A <account> -p batch --qos=interactive run_gb200.sh
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 03:45:00
#SBATCH -J h3-super-2pair
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?submit this launcher with sbatch}"
: "${SLURM_JOB_NUM_NODES:?}" "${SLURM_NTASKS:?}"
: "${H3_SUPER_RUN_NAME:?set a fresh H3_SUPER_RUN_NAME}"
: "${H3_SUPER_RUNTIME_ROOT:?set an absolute writable H3_SUPER_RUNTIME_ROOT outside the source tree}"
: "${H3_SUPER_STAGE1_ASSET_ROOT:?set the host root containing H3/LoRA/TAEH3 assets}"
: "${H3_SUPER_STAGE1_DATASET_ROOT:?set the host root containing inputs/i2v_talking}"
: "${H3_SUPER_LTX_WEIGHTS:?set the LTX-2.5-public weight root}"

if [[ -n ${H3_SUPER_SOL_ENGINE_ROOT:-} ]]; then
  readonly sol_engine_root=$(cd -- "${H3_SUPER_SOL_ENGINE_ROOT}" && pwd -P)
elif [[ -f ${SLURM_SUBMIT_DIR:-}/models/minimax_h3/super_acceleration/run_gb200.sh ]]; then
  readonly sol_engine_root=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)
else
  echo "set H3_SUPER_SOL_ENGINE_ROOT, or submit from the Sol-Engine repository root" >&2
  exit 2
fi
readonly super_root=${sol_engine_root}/models/minimax_h3/super_acceleration
readonly runtime_root=${H3_SUPER_RUNTIME_ROOT}
readonly asset_root=${H3_SUPER_STAGE1_ASSET_ROOT}
readonly dataset_root=${H3_SUPER_STAGE1_DATASET_ROOT}
readonly ltx_weights=${H3_SUPER_LTX_WEIGHTS}
readonly run_name=${H3_SUPER_RUN_NAME}
readonly host_run=${runtime_root}/runs/${run_name}
readonly hot_repeats=${H3_SUPER_HOT_REPEATS:-10}
readonly transport=${H3_SUPER_HANDOFF_TRANSPORT:-tcp}
readonly handoff_mode=${H3_SUPER_HANDOFF_MODE:-direct_tensor}
readonly temporal_tile=${H3_SUPER_INPUT_VAE_TEMPORAL_TILE:-full}
readonly ephemeral_runtime=${H3_SUPER_EPHEMERAL_RUNTIME:-0}
readonly refiner_first_frame_root=${H3_SUPER_REFINER_FIRST_FRAME_ROOT:-${dataset_root}}
readonly stage2_python=${H3_SUPER_STAGE2_PYTHON:-${sol_engine_root}/.venv/bin/python}
readonly container_image=${H3_SUPER_SGLANG_IMAGE:-docker://lmsysorg/sglang@sha256:71145ca99ebc458265e93cebd00b52bb9f419f052e7d0de09a54fa0f72fed888}
readonly socket_dir=$(mktemp -d "/tmp/h3-super-${SLURM_JOB_ID}.XXXXXX")
runtime_created=0

cleanup() {
  rm -f "${socket_dir}"/pair_*.sock 2>/dev/null || true
  rmdir "${socket_dir}" 2>/dev/null || true
  if [[ ${ephemeral_runtime} == 1 && ${runtime_created} == 1 ]]; then
    rm -rf -- "${runtime_root}"
  fi
}
trap cleanup EXIT

[[ ${runtime_root} == /* ]] || { echo "H3_SUPER_RUNTIME_ROOT must be absolute" >&2; exit 2; }
[[ ${runtime_root} != "${sol_engine_root}"* ]] || {
  echo "H3_SUPER_RUNTIME_ROOT must be outside the source checkout" >&2
  exit 2
}
[[ ${ephemeral_runtime} == 0 || ${ephemeral_runtime} == 1 ]] || {
  echo "H3_SUPER_EPHEMERAL_RUNTIME must be 0 or 1" >&2
  exit 2
}
if [[ ${ephemeral_runtime} == 1 ]]; then
  [[ ${runtime_root} == /tmp/h3-super-* ]] || {
    echo "ephemeral runtime cleanup is restricted to /tmp/h3-super-*" >&2
    exit 2
  }
  [[ ! -e ${runtime_root} ]] || {
    echo "refusing pre-existing ephemeral runtime root ${runtime_root}" >&2
    exit 2
  }
fi
[[ ${hot_repeats} == 1 || ${hot_repeats} == 10 ]] || {
  echo "H3_SUPER_HOT_REPEATS must be 1 or 10" >&2
  exit 2
}
[[ ${transport} == tcp || ${transport} == unix ]] || {
  echo "H3_SUPER_HANDOFF_TRANSPORT must be tcp or unix" >&2
  exit 2
}
[[ ${handoff_mode} == direct_tensor || ${handoff_mode} == mp4 ]] || {
  echo "H3_SUPER_HANDOFF_MODE must be direct_tensor or mp4" >&2
  exit 2
}
if [[ ${handoff_mode} == direct_tensor && ${transport} != tcp ]]; then
  echo "direct_tensor currently requires TCP loopback" >&2
  exit 2
fi
[[ ${temporal_tile} == default || ${temporal_tile} == full ]] || {
  echo "H3_SUPER_INPUT_VAE_TEMPORAL_TILE must be default or full" >&2
  exit 2
}
(( SLURM_JOB_NUM_NODES == 1 && SLURM_NTASKS == 4 )) || {
  echo "requires one exclusive four-GPU node and four tasks" >&2
  exit 2
}

for path in   "${stage2_python}"   "${super_root}/stage1/stage1_producer.py"   "${super_root}/stage2/stage2_server.py"   "${super_root}/assets/refiner_manifest.json"   "${refiner_first_frame_root}"   "${asset_root}"   "${dataset_root}"   "${ltx_weights}"; do
  test -e "${path}" || { echo "missing ${path}" >&2; exit 1; }
done
[[ ! -e ${host_run} ]] || {
  echo "refusing existing run root ${host_run}" >&2
  exit 2
}
mkdir -p "${host_run}/logs" "${runtime_root}/cache"
runtime_created=1

export E2E_SUPER_ROOT=/opt/sol-engine/models/minimax_h3/super_acceleration
export E2E_HOST_SUPER_ROOT=${super_root}
export E2E_HOST_RUNTIME=${runtime_root}
export E2E_STAGE1_RUN_ROOT=/super-runtime/runs/${run_name}
export E2E_STAGE2_RUN_ROOT=${host_run}
export E2E_STAGE1_CACHE_ROOT=/super-runtime/cache/stage1/gb200_sm100_torch211_sglang12eadf86
export E2E_STAGE2_CACHE_ROOT=${runtime_root}/cache/stage2/gb200_sm100_torch211_sol
export E2E_SOL_REPO=${sol_engine_root}
export E2E_STAGE2_PYTHON=${stage2_python}
export E2E_LTX_WEIGHTS=${ltx_weights}
export E2E_REFINER_TEMPLATE_MANIFEST=${super_root}/assets/refiner_manifest.json
export E2E_REFINER_FIRST_FRAME_ROOT=${refiner_first_frame_root}
export E2E_HOT_REPEATS=${hot_repeats}
export E2E_INPUT_VAE_TEMPORAL_TILE=${temporal_tile}
export E2E_HANDOFF_TRANSPORT=${transport}
export E2E_HANDOFF_MODE=${handoff_mode}
export E2E_SOCKET_DIR=${socket_dir}
export E2E_HANDOFF_PORT_BASE=$((31000 + SLURM_JOB_ID % 1000 * 2))
export E2E_H3_PORT_BASE=$((20000 + SLURM_JOB_ID % 1000 * 8))
export E2E_AUTH_TOKEN=${SLURM_JOB_ID}-${SLURM_JOB_UID:-unknown}-${run_name}

export E2E_H3_MANIFEST=${E2E_SUPER_ROOT}/assets/talking8_fbcache49_5s.json
export E2E_H3_SOURCE_ASSET_ROOT=/talking_dataset
export E2E_H3_MODEL=${H3_SUPER_H3_MODEL_IN_CONTAINER:-/assets/h3_run/cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/6818f6c32d12b210915e44ad56a4228c2608f160}
export E2E_H3_LORA=${H3_SUPER_H3_LORA_IN_CONTAINER:-/assets/models/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v0.1.safetensors}
export E2E_TAEH3_SOURCE=${E2E_SUPER_ROOT}/vendor/taeh3/taehv.py
export E2E_TAEH3_CHECKPOINT=${H3_SUPER_TAEH3_CHECKPOINT_IN_CONTAINER:-/assets/third_party/taehv/taeh3.pth}
export HF_HOME=/super-runtime/cache/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
export SGLANG_VAE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH=${E2E_SUPER_ROOT}:${E2E_SUPER_ROOT}/stage1:/sgl-workspace/sglang/python

readonly container_env=E2E_SUPER_ROOT,E2E_HOST_RUNTIME,E2E_STAGE1_RUN_ROOT,E2E_STAGE1_CACHE_ROOT,E2E_HOT_REPEATS,E2E_INPUT_VAE_TEMPORAL_TILE,E2E_HANDOFF_TRANSPORT,E2E_HANDOFF_MODE,E2E_SOCKET_DIR,E2E_HANDOFF_PORT_BASE,E2E_H3_PORT_BASE,E2E_AUTH_TOKEN,E2E_H3_MANIFEST,E2E_H3_SOURCE_ASSET_ROOT,E2E_H3_MODEL,E2E_H3_LORA,E2E_TAEH3_SOURCE,E2E_TAEH3_CHECKPOINT,HF_HOME,HUGGINGFACE_HUB_CACHE,HF_HUB_OFFLINE,HF_HUB_DISABLE_XET,SGLANG_TORCH_COMPILE_MODE,SGLANG_VAE_TORCH_COMPILE_MODE,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,TOKENIZERS_PARALLELISM,PYTHONUNBUFFERED,PYTHONPATH

echo "launch topology: two independent 1-GPU H3 -> 1-GPU LTX pairs; hot=${hot_repeats}; tile=${temporal_tile}; handoff=${handoff_mode}/${transport}"
srun --exclusive --exact --kill-on-bad-exit=1   --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-node=2 --gpus-per-task=1 --cpus-per-task=32   --cpu-bind=cores --gpu-bind=verbose,single:1   bash "${super_root}/stage2/run_worker.sh" &
stage2_step=$!

srun --exclusive --exact --kill-on-bad-exit=1   --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-node=2 --gpus-per-task=1 --cpus-per-task=32   --cpu-bind=cores --gpu-bind=verbose,single:1   --container-image="${container_image}"   --container-mounts="${sol_engine_root}:/opt/sol-engine:ro,${asset_root}:/assets:ro,${dataset_root}:/talking_dataset:ro,${runtime_root}:/super-runtime,${socket_dir}:${socket_dir}"   --container-env="${container_env}"   --no-container-mount-home --no-container-entrypoint   bash "${E2E_SUPER_ROOT}/stage1/run_worker.sh" &
stage1_step=$!

set +e
wait "${stage1_step}"
stage1_status=$?
if (( stage1_status != 0 )); then
  kill "${stage2_step}" 2>/dev/null || true
fi
wait "${stage2_step}"
stage2_status=$?
set -e
if (( stage1_status != 0 || stage2_status != 0 )); then
  echo "pipeline step failure: Stage1=${stage1_status} Stage2=${stage2_status}" >&2
  exit 1
fi

"${stage2_python}" "${super_root}/summarize.py" "${host_run}" --hot-repeats "${hot_repeats}"
echo "complete: ${host_run}/summary.json"
