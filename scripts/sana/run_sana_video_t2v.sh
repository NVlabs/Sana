#!/bin/bash
#SBATCH --job-name=sana-video-t2v
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=sana-video-t2v-%j.out
#SBATCH --error=sana-video-t2v-%j.out

# SANA-Video 2B text-to-video via diffusers SanaVideoPipeline (single-GPU; the
# 4-GPU exclusive request is a cluster QOS convention, override on the CLI).
# Honors pre-set HF_HOME / HF_TOKEN / PYTHON_BIN; otherwise repo-relative defaults.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}" 2>/dev/null || true)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/outputs/.cache/xdg}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/outputs/.tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$REPO_ROOT/outputs/sana_video"

PYTHON="${PYTHON_BIN:-$REPO_ROOT/.conda/ltx23/bin/python}"
echo "[$(date)] Node: $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

"$PYTHON" scripts/sana/run_sana_video_t2v.py "$@"
rc=$?
echo "[$(date)] EXIT_RC=$rc"
exit $rc
