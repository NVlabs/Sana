#!/usr/bin/env bash
#SBATCH --job-name=lbv-benchattn
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[benchattn] start"
"$PYTHON_BIN" slurm/bench_attn_kernel.py
echo "[benchattn] done"
