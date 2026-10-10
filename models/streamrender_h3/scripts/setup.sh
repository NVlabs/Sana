#!/usr/bin/env bash
set -euo pipefail
runtime_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export XDG_CACHE_HOME="$runtime_root/.cache/xdg"
export npm_config_cache="$runtime_root/.cache/npm"
export PIP_CACHE_DIR="$runtime_root/.cache/pip"
export UV_CACHE_DIR="$runtime_root/.cache/uv"
export HF_HOME="${HF_HOME:-$runtime_root/.cache/huggingface}"
export TMPDIR="$runtime_root/.cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$npm_config_cache" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TMPDIR"
node --version
npm --prefix "$runtime_root/coding/threejs-racing" ci
if [[ ${1:-} == --python-extras ]]; then
  "${PYTHON_BIN:-python3}" -m pip install -r "$runtime_root/requirements.txt"
fi
echo "Frontend ready. Configure assets.local.json and validate the GPU environment before launching H3."
