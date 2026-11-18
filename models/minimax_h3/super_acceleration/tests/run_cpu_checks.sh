#!/usr/bin/env bash
# CPU-only validation; submit to Slurm rather than running test discovery on a
# login node.
#SBATCH -p cpu_short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -J h3-super-cpu

set -euo pipefail

readonly repo_root=$(cd -- "${SLURM_SUBMIT_DIR:?submit from the repository root}" && pwd -P)
readonly super_root=${repo_root}/models/minimax_h3/super_acceleration

cd "${repo_root}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

bash -n "${super_root}/run_gb200.sh" "${super_root}/stage1/run_worker.sh" "${super_root}/stage2/run_worker.sh"
python3 -m unittest discover -s "${super_root}/tests" -p "test_*.py" -v
