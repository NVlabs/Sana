#!/usr/bin/env bash
set -euo pipefail
runtime_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$runtime_root:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export FLASH_ATTN_4_AVAILABLE=${FLASH_ATTN_4_AVAILABLE:-1}
export FLEX_FLASH_ATTN_AVAILABLE=0
export H3_FA4_NUM_SPLITS=${H3_FA4_NUM_SPLITS:-1}
export H3_CACHE_QUERY_CAPACITY=${H3_CACHE_QUERY_CAPACITY:-12288}
export NCCL_NVLS_ENABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export STREAMRENDER_SERVER=${STREAMRENDER_SERVER:-http://127.0.0.1:8765}
python_bin=${PYTHON_BIN:-python3}
backend=${BACKEND:-h3}
if [[ $backend == passthrough ]]; then
  "$python_bin" -m runtime.worker --backend passthrough "$@" &
else
  "$python_bin" -m torch.distributed.run --standalone --nproc-per-node="${GPUS:-8}" -m runtime.worker "$@" &
fi
worker_pid=$!
cleanup() { kill "$worker_pid" "${frontend_pid:-$worker_pid}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
npm --prefix "$runtime_root/coding/threejs-racing" run dev &
frontend_pid=$!
wait -n "$worker_pid" "$frontend_pid"
