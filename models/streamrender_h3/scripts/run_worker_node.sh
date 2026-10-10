#!/usr/bin/env bash
set -euo pipefail
runtime_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$runtime_root:${PYTHONPATH:-}"
export FLASH_ATTN_4_AVAILABLE=${FLASH_ATTN_4_AVAILABLE:-1}
export FLEX_FLASH_ATTN_AVAILABLE=0
export H3_FA4_NUM_SPLITS=${H3_FA4_NUM_SPLITS:-1}
export H3_CACHE_QUERY_CAPACITY=${H3_CACHE_QUERY_CAPACITY:-12288}
export NCCL_NVLS_ENABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
node_cache="$RUNTIME_ROOT/.cache/${SLURM_JOB_ID:-local}/node${SLURM_NODEID:-0}"
export XDG_CACHE_HOME="$node_cache"
export TMPDIR="$node_cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$node_cache/torchinductor"
export TRITON_CACHE_DIR="$node_cache/triton"
export CUDA_CACHE_PATH="$node_cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME/torch/kernels" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
validation_args=()
if [[ ${VALIDATE_BROWSER:-0} == 1 ]]; then
  validation_args+=(--validate-browser)
fi
exec "${PYTHON_BIN:-python3}" -m torch.distributed.run \
  --nnodes="${SLURM_NNODES:-1}" --node-rank="${SLURM_NODEID:-0}" \
  --nproc-per-node="${GPUS_PER_NODE:-4}" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m runtime.worker --assets="$ASSETS_FILE" "${validation_args[@]}"
