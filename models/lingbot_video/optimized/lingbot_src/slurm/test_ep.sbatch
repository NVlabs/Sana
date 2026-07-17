#!/usr/bin/env bash
#SBATCH --job-name=lbv-ep-test
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
echo "[ep-test] running 4-rank EP correctness test"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 4 \
  slurm/test_ep_correctness.py 2>&1 | grep -viE "warning|cuda init|return torch"
