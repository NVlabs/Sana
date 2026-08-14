#!/usr/bin/env bash
# Rebuild the repository-local LTX-2.5 environment from the vendored uv
# workspace. Run this on a Slurm compute node, not on a login node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
ENV_PROJECT="$HERE/environment/LTX-2"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "[error] uv is required; set UV_BIN=/absolute/path/to/uv" >&2
  exit 1
fi
if [[ ! -f "$ENV_PROJECT/pyproject.toml" || ! -f "$ENV_PROJECT/uv.lock" ]]; then
  echo "[error] incomplete vendored environment workspace: $ENV_PROJECT" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/.cache/uv" "$REPO_ROOT/.cache/tmp"
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"
export UV_CACHE_DIR="$REPO_ROOT/.cache/uv"
export TMPDIR="$REPO_ROOT/.cache/tmp"
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

"$UV_BIN" sync \
  --project "$ENV_PROJECT" \
  --frozen \
  --group kernels

PYTHONPATH="$HERE/ltx_src:$ENV_PROJECT/packages/ltx-kernels/src" \
  "$REPO_ROOT/.venv/bin/python" "$HERE/validate_env.py"

echo "[ltx25] repository-local environment ready: $REPO_ROOT/.venv"
