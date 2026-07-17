#!/usr/bin/env bash
#SBATCH --job-name=lbv-nccl2n
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:12:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source slurm/env.sh

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=29677
export SLURM_GPUS_ON_NODE=4

echo "nodes: $SLURM_JOB_NODELIST"
srun --ntasks-per-node=4 bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  "'"$PYTHON_BIN"'" slurm/bench_nccl_2node.py
' 2>&1 | grep -viE "warning|cuda init|return torch"
