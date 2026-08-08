#!/usr/bin/env bash
# Registered MiniMax-H3 runtime entrypoint.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_candidate.py}"
: "${H3_MODEL_PATH:?H3_MODEL_PATH must point to the converted diffusers checkpoint}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT_DIR"
export H3_OUTPUT_DIR="$OUT_DIR"

# The vendored diffusers (PR #14355) and the huggingface_hub >= 1.23 it needs have to shadow
# whatever the environment ships; `gpu_infer.py` also puts both on `sys.path`, so this only matters
# for subprocesses.
export PYTHONPATH="${HERE}/vendor_site:${HERE}/diffusers_src/src:${PYTHONPATH:-}"

if [[ -n "${H3_CONDA_ROOT:-}" && -n "${H3_CONDA_ENV:-}" && \
      -f "${H3_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  _h3_had_nounset=0
  [[ $- == *u* ]] && _h3_had_nounset=1
  set +u
  # shellcheck disable=SC1090
  source "${H3_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${H3_CONDA_ENV}"
  [[ "${_h3_had_nounset}" == 1 ]] && set -u
fi

PYBIN="${PYTHON_BIN:-python}"
exec "$PYBIN" "$HERE/gpu_infer.py"
