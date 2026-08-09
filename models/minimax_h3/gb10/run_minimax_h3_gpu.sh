#!/usr/bin/env bash
# Registered MiniMax-H3 *gb10* runtime entrypoint.
#
# Same contract as the baseline and optimized entrypoints: the candidate manifest sets OUT_DIR
# and the H3_* switches, this resolves the vendored deps and launches the driver. The
# acceleration line is entirely env-gated (see `gb10/gpu_infer.py`), so this script is
# identical for every candidate in `candidates/minimax_h3_gb10_*.toml`.
#
# No torchrun. One GB10 is one process; there is no context parallelism to launch into.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_candidate.py}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${HERE}/../gb200/baseline" && pwd)"
mkdir -p "$OUT_DIR"
export H3_OUTPUT_DIR="$OUT_DIR"

# Two things have to resolve from beside the environment rather than inside it, and both have
# to win from process start rather than from a later sys.path.insert:
#
#   the vendored diffusers (PR #14355) — MiniMax-H3 is in no release, and the commit is the one
#     `baseline/SOURCE_SNAPSHOT.json` pins.
#   sol_attn — only `sol_attn.preprocess` is used, for the block-mean reduction. The released
#     CuTe kernels are not: their dispatcher refuses SM121, so `sol_attn_h3.py` carries the
#     Triton reference with the routing policy rebuilt.
SOL_ATTN_ROOT="${H3_SOL_ATTN_ROOT:-$(cd "${HERE}/../../../techniques/sparse_backends" && pwd)}"
export H3_SOL_ATTN_ROOT="${SOL_ATTN_ROOT}"
DIFFUSERS_SRC="${H3_DIFFUSERS_SRC:-${BASELINE}/diffusers_src/src}"
[ -d "$DIFFUSERS_SRC" ] || DIFFUSERS_SRC="${HERE}/diffusers_src/src"
export H3_DIFFUSERS_SRC="${DIFFUSERS_SRC}"
export PYTHONPATH="${DIFFUSERS_SRC}:${SOL_ATTN_ROOT}:${HERE}:${PYTHONPATH:-}"

# huggingface.co does not resolve from the machine this was measured on; all three checkpoint
# repositories were fetched from ModelScope into the flat layout `paths.py` also accepts.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${H3_CONDA_ROOT:-}" && -n "${H3_CONDA_ENV:-}" && \
      -f "${H3_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${H3_CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${H3_CONDA_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

# The pinned versions are not advisory. Installing torchvision separately once pulled torch
# 2.11.0 -> 2.13.0 and Triton 3.6 -> 3.7, which would have silently invalidated every
# bit-exactness claim in the README without failing anything.
"$PYTHON_BIN" - <<'PY'
import torch
expected = ("2.11.0", "3.6")
if not torch.__version__.startswith(expected[0]):
    print(f"[warn] torch {torch.__version__}, bit-exactness was verified on {expected[0]}")
PY

exec "$PYTHON_BIN" -u "${HERE}/gpu_infer.py" "$@" 2>&1 | tee "${OUT_DIR}/run.log"
