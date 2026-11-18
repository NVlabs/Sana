#!/usr/bin/env bash
set -euo pipefail

readonly pair_id=${SLURM_PROCID:?launch as the two-task Stage-1 step}
(( pair_id >= 0 && pair_id < 2 )) || { echo "invalid pair ${pair_id}" >&2; exit 2; }
: "${E2E_SUPER_ROOT:?}" "${E2E_HOST_RUNTIME:?}" "${E2E_STAGE1_RUN_ROOT:?}"
: "${E2E_STAGE1_CACHE_ROOT:?}" "${E2E_HOT_REPEATS:?}"
: "${E2E_AUTH_TOKEN:?}" "${E2E_HANDOFF_TRANSPORT:?}" "${E2E_HANDOFF_PORT_BASE:?}"

case ${E2E_HANDOFF_TRANSPORT} in
  tcp) endpoint="tcp://127.0.0.1:$((E2E_HANDOFF_PORT_BASE + pair_id))" ;;
  unix) endpoint="unix://${E2E_SOCKET_DIR:?}/pair_${pair_id}.sock" ;;
  *) echo "invalid transport ${E2E_HANDOFF_TRANSPORT}" >&2; exit 2 ;;
esac

export H3_DIRECT_HANDOFF_ACTIVE=0
if [[ ${E2E_HANDOFF_MODE:?} == direct_tensor ]]; then
  export H3_DIRECT_HANDOFF_ACTIVE=1
  export H3_DIRECT_HANDOFF_ENDPOINT=${endpoint}
  export H3_DIRECT_HANDOFF_AUTH_TOKEN=${E2E_AUTH_TOKEN}
  export H3_DIRECT_HANDOFF_PAIR_ID=${pair_id}
fi

readonly cell=${E2E_STAGE1_RUN_ROOT}/pair_${pair_id}/stage1
mkdir -p "$(dirname "${cell}")"
[[ ! -e ${cell} ]] || { echo "refusing existing Stage-1 cell ${cell}" >&2; exit 2; }

unset H3_FF_ARM H3_FF_LORA_PROFILE H3_FF_STUDENT_FORWARDS H3_LORA_PATH || true
export H3_GRID_ACTIVE=1
export H3_GRID_MODEL_PROFILE=lx2v_4s_v01_544p
export H3_GRID_CACHE_MODE=none
export H3_GRID_WIDTH=896 H3_GRID_HEIGHT=512 H3_GRID_COMPILE=1
export H3_GRID_TELEMETRY=${cell}/denoise_telemetry.jsonl
export H3_GRID_LORA_PATH=${E2E_H3_LORA}
export H3_DELIVERY_BENCH_ACTIVE=1 H3_DELIVERY_DECODER=taeh3
export H3_DELIVERY_DECODER_TELEMETRY=${cell}/decoder_telemetry.jsonl
export H3_DELIVERY_ENCODE_TELEMETRY=${cell}/encode_mux_telemetry.jsonl
export H3_DELIVERY_TAE_SOURCE=${E2E_TAEH3_SOURCE}
export H3_DELIVERY_TAE_CHECKPOINT=${E2E_TAEH3_CHECKPOINT}
export SGLANG_CACHE_DIT_ENABLED=0
unset \
  SGLANG_CACHE_DIT_FN \
  SGLANG_CACHE_DIT_BN \
  SGLANG_CACHE_DIT_WARMUP \
  SGLANG_CACHE_DIT_RDT \
  SGLANG_CACHE_DIT_MC \
  SGLANG_CACHE_DIT_TAYLORSEER \
  SGLANG_CACHE_DIT_SCM_PRESET \
  SGLANG_CACHE_DIT_SCM_POLICY \
  SGLANG_CACHE_DIT_SCM_COMPUTE_BINS \
  SGLANG_CACHE_DIT_SCM_CACHE_BINS || true
export SGLANG_H3_VAE_DECODER_TILE_BATCH_SIZE=1
export SGLANG_H3_VAE_DECODER_FULL_DTYPE=stock
export MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM=1

readonly port=$((E2E_H3_PORT_BASE + pair_id * 4))
export H3_HTTP_PORT=${port}
export H3_SCHEDULER_PORT=$((port + 1))
export H3_MASTER_PORT=$((port + 2))
export H3_NCCL_PORT=$((port + 3))
readonly profile=${E2E_STAGE1_CACHE_ROOT}/pair_${pair_id}/896x512_124f_lora4_taeh3
export TRITON_CACHE_DIR=${profile}/triton
export TORCHINDUCTOR_CACHE_DIR=${profile}/inductor
export TORCH_HOME=${profile}/torch
export XDG_CACHE_HOME=${profile}/xdg
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"
exec 9>"${profile}.lock"
flock -n 9 || {
  echo "Stage-1 compile cache is in use: ${profile}" >&2
  exit 3
}

exec python3 "${E2E_SUPER_ROOT}/stage1/stage1_producer.py" \
  --endpoint "${endpoint}" \
  --auth-token "${E2E_AUTH_TOKEN}" \
  --pair-id "${pair_id}" \
  --handoff-mode "${E2E_HANDOFF_MODE}" \
  --path-map "/super-runtime=${E2E_HOST_RUNTIME}" \
  --pair-metadata "${E2E_STAGE1_RUN_ROOT}/pair_${pair_id}/benchmark.json" \
  --duration 5 \
  --decoder taeh3 \
  --manifest "${E2E_H3_MANIFEST}" \
  --source-asset-root "${E2E_H3_SOURCE_ASSET_ROOT}" \
  --prompt-index 3 \
  --out "${cell}" \
  --model-path "${E2E_H3_MODEL}" \
  --lora-path "${E2E_H3_LORA}" \
  --http-port "${port}" \
  --scheduler-port "$((port + 1))" \
  --master-port "$((port + 2))" \
  --nccl-port "$((port + 3))" \
  --compile-prime-requests 1 \
  --warmup-requests 1 \
  --hot-repeats "${E2E_HOT_REPEATS}"
