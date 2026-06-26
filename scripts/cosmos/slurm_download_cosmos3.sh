#!/bin/bash
#SBATCH --job-name=cosmos3-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=cosmos3-download-%j.out
#SBATCH --error=cosmos3-download-%j.out
# Download the Cosmos3 model used by the inference entry. Override the repo with
# COSMOS3_REPO=... (default nvidia/Cosmos3-Super, which slurm_cosmos3_super.sh runs).
# Honors a pre-set HF_HOME / HF_TOKEN; otherwise uses sane fallbacks.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COSMOS3_REPO="${COSMOS3_REPO:-nvidia/Cosmos3-Super}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}" 2>/dev/null || true)}"
# hf-xet / hf_transfer buffers the whole shard in RAM and OOM-killed this job at
# 32G (Cosmos3-Super is ~124GB / 27 shards). Use plain streaming LFS + a timeout.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.conda/ltx23/bin/python}"
mkdir -p "$HF_HUB_CACHE"

echo "[$(date)] Node: $(hostname)  repo: $COSMOS3_REPO  cache: $HF_HUB_CACHE"

"$PYTHON" - "$COSMOS3_REPO" << 'PYEOF'
from huggingface_hub import snapshot_download
import os, sys
repo = sys.argv[1]
path = snapshot_download(
    repo,
    token=os.environ.get('HF_TOKEN') or None,
    cache_dir=os.environ['HF_HUB_CACHE'],
    ignore_patterns=['*.mp4', '*.png', '*.jpg', 'assets/*', 'images/*'],
    max_workers=8,
)
print('Snapshot path:', path)
PYEOF

echo "[$(date)] Download finished. Total cache size:"
du -sh "$HF_HUB_CACHE/models--${COSMOS3_REPO/\//--}" 2>/dev/null
echo "[$(date)] DOWNLOAD_COMPLETE_MARKER"
