#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 10000 + 20000))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG_SPEC="${CONFIG_SPEC:-configs/sol_rl/flux1.py:flux1_diffusionnft_pickscore}"

cmd=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  train_scripts/sol_rl/train_flux1.py
  --config="$CONFIG_SPEC"
)

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf -v cmd_str '%q ' "${cmd[@]}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $cmd_str"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
