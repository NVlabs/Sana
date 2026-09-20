#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 10000 + 20000))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG_SPEC="${CONFIG_SPEC:-configs/sol_rl/sana.py:sana_diffusionnft_pickscore}"
NATIVE_CONFIG="${NATIVE_CONFIG:-}"
DISABLE_XFORMERS="${DISABLE_XFORMERS:-1}"
SANA_NATIVE_MODEL_PATH="${SANA_NATIVE_MODEL_PATH:-$ROOT_DIR/output/pretrained_models/SANA_LinearFFN.pth}"
SANA_NATIVE_MODEL_SOURCE="${SANA_NATIVE_MODEL_SOURCE:-hf://yitongl/SANA_LinearFFN/SANA_LinearFFN.pth}"

export DISABLE_XFORMERS

if [[ ! -f "$SANA_NATIVE_MODEL_PATH" ]]; then
  echo "Missing Sana native checkpoint at $SANA_NATIVE_MODEL_PATH, downloading from $SANA_NATIVE_MODEL_SOURCE"
  mkdir -p "$(dirname "$SANA_NATIVE_MODEL_PATH")"
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  SANA_NATIVE_MODEL_PATH="$SANA_NATIVE_MODEL_PATH" \
  SANA_NATIVE_MODEL_SOURCE="$SANA_NATIVE_MODEL_SOURCE" \
  python - <<'PY'
import os
import shutil

from sana.tools import hf_download_or_fpath

dst = os.environ["SANA_NATIVE_MODEL_PATH"]
src = os.environ["SANA_NATIVE_MODEL_SOURCE"]

resolved = hf_download_or_fpath(src)
if resolved is None:
    raise RuntimeError(f"Failed to resolve Sana native checkpoint source: {src}")

if os.path.realpath(resolved) != os.path.realpath(dst):
    tmp_dst = f"{dst}.tmp"
    if os.path.lexists(tmp_dst):
        os.remove(tmp_dst)
    try:
        os.symlink(resolved, tmp_dst)
    except OSError:
        shutil.copy2(resolved, tmp_dst)
    os.replace(tmp_dst, dst)

print(f"Resolved Sana native checkpoint: {dst}")
PY
fi

cmd=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  train_scripts/sol_rl/train_sana.py
  --config="$CONFIG_SPEC"
)

if [[ -n "$NATIVE_CONFIG" ]]; then
  cmd+=(--native_config="$NATIVE_CONFIG")
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf -v cmd_str '%q ' "${cmd[@]}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES DISABLE_XFORMERS=$DISABLE_XFORMERS $cmd_str"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
DISABLE_XFORMERS="$DISABLE_XFORMERS" \
SANA_NATIVE_MODEL_PATH="$SANA_NATIVE_MODEL_PATH" \
SANA_NATIVE_MODEL_SOURCE="$SANA_NATIVE_MODEL_SOURCE" \
"${cmd[@]}"
