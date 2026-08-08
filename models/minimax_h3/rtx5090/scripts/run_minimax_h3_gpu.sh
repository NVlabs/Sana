#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${H3_RTX5090_PROFILE:-dense}"
H3_ROOT="${H3_ROOT:-${HOME}/minimax_h3_5090}"
VENV_ROOT="${H3_VENV_ROOT:-${H3_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
PORT="${H3_PORT:-30010}"
GPU="${H3_CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-${RUNTIME_ROOT}/outputs/${PROFILE}-${STAMP}}"
MEASURED_STEPS="${H3_MEASURED_NUM_STEPS:-50}"

if [[ "${PROFILE}" == "dense" ]]; then
  WARMUP_STEPS="${H3_WARMUP_NUM_STEPS:-5}"
else
  WARMUP_STEPS="${H3_WARMUP_NUM_STEPS:-50}"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is not available: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/warmup" "${OUT_DIR}/measured"
server_pid=""
monitor_pid=""

stop_scoped_processes() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap stop_scoped_processes EXIT TERM INT

export CUDA_VISIBLE_DEVICES="${GPU}"
export H3_PORT="${PORT}"
export SOL_ATTN_EVENT_LOG="${OUT_DIR}/sol_events_rank0.jsonl"
export SOL_ATTN_REQUEST_EPOCH_FILE="${OUT_DIR}/request_epoch.txt"

setsid bash "${SCRIPT_DIR}/launch_server.sh" >"${OUT_DIR}/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" >"${OUT_DIR}/server.pid"

(
  while kill -0 "${server_pid}" 2>/dev/null; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi -i "${GPU}" \
      --query-gpu=memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 1
  done
) >"${OUT_DIR}/resources.csv" 2>&1 &
monitor_pid=$!

healthy=0
for _ in $(seq 1 1440); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "${PROFILE} server exited during startup" >&2
    tail -200 "${OUT_DIR}/server.log" >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "${healthy}" != "1" ]]; then
  echo "${PROFILE} server did not become healthy" >&2
  exit 1
fi

printf 'warmup-%s\n' "${STAMP}" >"${SOL_ATTN_REQUEST_EPOCH_FILE}"
H3_SERVER_URL="http://127.0.0.1:${PORT}" \
H3_ROOT="${H3_ROOT}" \
H3_NUM_STEPS="${WARMUP_STEPS}" \
H3_DURATION_SECONDS="${H3_DURATION_SECONDS:-5}" \
H3_SEED="${H3_WARMUP_SEED:-1100}" \
H3_OUTPUT_DIR="${OUT_DIR}/warmup" \
  "${PYTHON_BIN}" "${RUNTIME_ROOT}/request.py" \
  >"${OUT_DIR}/warmup_request.log" 2>&1

printf 'measured-%s\n' "${STAMP}" >"${SOL_ATTN_REQUEST_EPOCH_FILE}"
H3_SERVER_URL="http://127.0.0.1:${PORT}" \
H3_ROOT="${H3_ROOT}" \
H3_NUM_STEPS="${MEASURED_STEPS}" \
H3_DURATION_SECONDS="${H3_DURATION_SECONDS:-5}" \
H3_SEED="${H3_SEED:-1101}" \
H3_OUTPUT_DIR="${OUT_DIR}/measured" \
  "${PYTHON_BIN}" "${RUNTIME_ROOT}/request.py" \
  >"${OUT_DIR}/measured_request.log" 2>&1

stop_scoped_processes
trap - EXIT TERM INT
"${PYTHON_BIN}" "${RUNTIME_ROOT}/summarize.py" "${OUT_DIR}" \
  >"${OUT_DIR}/summary.json"
printf '%s\n' "${OUT_DIR}"
