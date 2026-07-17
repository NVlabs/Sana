#!/usr/bin/env bash
#SBATCH --job-name=lbv-fa3-wheel
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/%x_%j.out
# Try to obtain FlashAttention-3 (flash_attn_interface) via prebuilt wheels before
# resorting to a source build. Verified against the actual GB200 (sm_100).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source /lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/etc/profile.d/conda.sh
conda activate lingbot-video

PY=python; PIP=pip
echo "=== GPU / arch ==="; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
echo "=== torch ==="; $PY -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"

check() { $PY -c "import flash_attn_interface as f; from flash_attn_interface import flash_attn_varlen_func; print('FA3 OK:', f.__file__)" 2>&1 | tail -1; }

echo "=== already present? ==="; check

echo "=== attempt 1: flash-attn (may include FA3 interface in 3.x) ==="
$PIP install --no-cache-dir --only-binary=:all: "flash-attn" 2>&1 | tail -3
echo "-> "; check

echo "=== attempt 2: flash-attn-3 / flashattn-hopper style packages ==="
for pkg in flash-attn-3 flashattn-hopper flash_attn_3; do
  echo "--- try $pkg ---"; $PIP install --no-cache-dir --only-binary=:all: "$pkg" 2>&1 | tail -2
done
echo "-> "; check

echo "=== pip index versions (for planning a build) ==="
$PIP index versions flash-attn 2>&1 | head -3
echo "=== DONE ==="