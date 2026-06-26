#!/bin/bash
#SBATCH --job-name=sana-video-sglang
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=sana-video-sglang-%j.out
#SBATCH --error=sana-video-sglang-%j.out

# Validate + run the SANA-Video port in the sglang multimodal_gen runtime.
# Single GPU (the model is single-GPU); the 4-GPU exclusive request is a cluster
# QOS convention — override with `sbatch --gpus-per-node=...` / `-A <account>`.
# Honors pre-set HF_HOME / HF_TOKEN / PYTHON_BIN; otherwise repo-relative defaults.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}" 2>/dev/null || true)}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

CACHE="${SGLANG_CACHE_ROOT:-$REPO_ROOT/outputs/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE/xdg}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/outputs/.tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE/torch_extensions}"
export SGLANG_DIFFUSION_CACHE_ROOT="${SGLANG_DIFFUSION_CACHE_ROOT:-$CACHE/sgl_diffusion}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE/cuda_cache}"

# CUDA toolkit (pip nvidia cu13) so sglang JIT kernels can compile/link.
SP="$REPO_ROOT/.conda/ltx23/lib/python3.12/site-packages/nvidia"
if [[ -d "$SP/cu13" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$SP/cu13}"; export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$SP/cublas/lib:$SP/cudnn/lib:$SP/nccl/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$SGLANG_DIFFUSION_CACHE_ROOT" "$CUDA_CACHE_PATH"

PY="${PYTHON_BIN:-$REPO_ROOT/.conda/ltx23/bin/python}"
echo "[$(date)] node=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

"$PY" scripts/sana/sana_video_sglang_run.py "$@"
rc=$?
echo "[$(date)] EXIT_RC=$rc"
exit $rc
