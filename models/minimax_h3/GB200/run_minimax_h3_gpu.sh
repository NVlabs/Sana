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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# H3_PROMPT_FILE may arrive repo-relative (that is how every candidate and config
# writes it). Whether that resolves depends on who launched us -- scripts/run.py
# uses the repo root as cwd, launch_candidate.py cds to the runtime dir first --
# so pin it to the repo root here rather than leaving it to the caller. h100 and
# a100 already do this; gb10 and gb200 did not, which made the same config value
# work under one launcher and not the other.
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
if [[ -n "${H3_PROMPT_FILE:-}" && "${H3_PROMPT_FILE}" != /* ]]; then
  export H3_PROMPT_FILE="${REPO_ROOT}/${H3_PROMPT_FILE}"
fi
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
export PYTHONPATH="${H3_CUTLASS_DSL:+${H3_CUTLASS_DSL}:}${SOL_ATTN_ROOT}:${HERE}/vendor_site:${HERE}/diffusers_src/src:${PYTHONPATH:-}"

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

# Two ways the ranks can arrive, and the test has to be for all three variables, not one of them.
# Under `srun` the step exports SLURM_PROCID, SLURM_LOCALID and SLURM_NTASKS together, and
# `gpu_infer.py` reads RANK/LOCAL_RANK/WORLD_SIZE from them. Under a plain `sbatch` batch step
# there is no srun, yet SLURM_PROCID is still set while SLURM_NTASKS is not -- testing only
# SLURM_PROCID took that branch and then died on `unbound variable` under `set -u`, which is how a
# `sbatch` wrapper calling `scripts/run.py` failed while the same command outside Slurm worked.
# Anything short of the full set means we are not inside an srun step: fall through to torchrun.
if [[ -n "${SLURM_PROCID:-}" && -n "${SLURM_LOCALID:-}" && -n "${SLURM_NTASKS:-}" ]]; then
  export RANK="${SLURM_PROCID}" LOCAL_RANK="${SLURM_LOCALID}" WORLD_SIZE="${SLURM_NTASKS}"
  exec "$PYBIN" "$HERE/gpu_infer.py"
fi
exec "$PYBIN" -m torch.distributed.run --nproc_per_node="${NPROC}" --standalone "$HERE/gpu_infer.py"
