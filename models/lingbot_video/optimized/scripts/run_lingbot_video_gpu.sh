#!/usr/bin/env bash
# Registered LingBot-Video runtime entrypoint.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_config.py}"
: "${LINGBOT_MODEL_DIR:?LINGBOT_MODEL_DIR must point to the external MoE checkpoint}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT_DIR"

# Activation is optional when PYTHON_BIN is already an absolute environment
# interpreter, but it preserves CUDA/library activation hooks on the cluster.
if [[ -n "${LINGBOT_CONDA_ROOT:-}" && -n "${LINGBOT_CONDA_ENV:-}" && \
      -f "${LINGBOT_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  _lingbot_had_nounset=0
  [[ $- == *u* ]] && _lingbot_had_nounset=1
  set +u
  # shellcheck disable=SC1090
  source "${LINGBOT_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${LINGBOT_CONDA_ENV}"
  [[ "${_lingbot_had_nounset}" == 1 ]] && set -u
fi

PYBIN="${PYTHON_BIN:-python}"
exec "$PYBIN" "$HERE/gpu_infer.py"
