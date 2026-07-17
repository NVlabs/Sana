#!/usr/bin/env bash
#SBATCH --job-name=lbv-groupedbench
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:12:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh
export CUDA_VISIBLE_DEVICES=0
"$PYTHON_BIN" slurm/bench_grouped_moe.py 2>&1 | grep -viE "warning|cuda init|return torch"
