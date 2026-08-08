#!/usr/bin/env bash
# Registered MiniMax-H3 *optimized* runtime entrypoint.
#
# Same contract as the baseline entrypoint: the candidate manifest sets OUT_DIR and the H3_* switches,
# this resolves the vendored deps and launches the driver under torchrun. The acceleration line is
# entirely env-gated (see `optimized/gpu_infer.py`), so this script is identical for every candidate
# in `candidates/minimax_h3_*.toml`.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_candidate.py}"
: "${H3_MODEL_PATH:?H3_MODEL_PATH must point to the converted diffusers checkpoint}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$(cd "${HERE}/../baseline" && pwd)"
mkdir -p "$OUT_DIR"
export H3_OUTPUT_DIR="$OUT_DIR"

# Three things have to resolve from beside the environment rather than inside it, and all three have
# to win from process start rather than from a later sys.path.insert:
#
#   nvidia-cutlass-dsl >= 4.5 — Sol-Attn's CuTe code reads `cutlass.CUDA_VERSION` to pick the nvvm
#     fmax API, and 4.3.x does not define it.
#   sol_attn — the released kernel package. It vendors its CuTe dependencies under
#     `sol_attn._vendor.flash_attn`, a private namespace, so no separate flash-attn shim is needed.
#   the vendored diffusers (PR #14355) plus the huggingface_hub >= 1.23 it needs.
SOL_ATTN_ROOT="${H3_SOL_ATTN_ROOT:-$(cd "${HERE}/../../../techniques/sparse_backends" && pwd)}"
export H3_SOL_ATTN_ROOT="${SOL_ATTN_ROOT}"
export PYTHONPATH="${H3_CUTLASS_DSL:+${H3_CUTLASS_DSL}:}${SOL_ATTN_ROOT}:${BASELINE}/vendor_site:${BASELINE}/diffusers_src/src:${PYTHONPATH:-}"

# A sparse configuration that silently fell back to dense would be a dense measurement wearing a
# sparse label, which is the one failure this stack must not ship.
export SOL_ATTN_STRICT="${SOL_ATTN_STRICT:-1}"

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
NPROC="${H3_ULYSSES_DEGREE:-8}"

# Under Slurm the launcher is srun and RANK/LOCAL_RANK/WORLD_SIZE come from SLURM_*; standalone,
# torchrun supplies them. `gpu_infer.py` reads the same three variables either way.
if [[ -n "${SLURM_PROCID:-}" ]]; then
  export RANK="${SLURM_PROCID}" LOCAL_RANK="${SLURM_LOCALID}" WORLD_SIZE="${SLURM_NTASKS}"
  exec "$PYBIN" "$HERE/gpu_infer.py"
fi
exec "$PYBIN" -m torch.distributed.run --nproc_per_node="${NPROC}" --standalone "$HERE/gpu_infer.py"
