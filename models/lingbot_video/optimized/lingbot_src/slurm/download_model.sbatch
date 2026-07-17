#!/usr/bin/env bash
#SBATCH --job-name=lbv-download
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs

HF_BIN="/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/hf"
MODELDIR="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/models/lingbot-video-moe-30b-a3b"
REPO="robbyant/lingbot-video-moe-30b-a3b"

# IMPORTANT: hf_transfer DISABLED — it hangs on this cluster's network without
# erroring and does not resume partial files. Plain hub downloader resumes
# incomplete .safetensors via byte-range and is reliable here.
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)"
mkdir -p "$MODELDIR"

# Priority order: transformer (needed by B1) first, then the rest.
echo "[download] node=$(hostname) phase 1/2: transformer + text_encoder + vae + configs"
"$HF_BIN" download "$REPO" --local-dir "$MODELDIR" --max-workers 8 \
  --include "transformer/*" "text_encoder/*" "vae/*" "processor/*" "scheduler/*" \
            "model_index.json" "*.py" "*.json"
echo "[download] phase1 EXIT=$?"

echo "[download] phase 2/2: refiner (needed by B2)"
"$HF_BIN" download "$REPO" --local-dir "$MODELDIR" --max-workers 8 --include "refiner/*"
echo "[download] phase2 EXIT=$?"

du -sh "$MODELDIR"
echo "[download] transformer shards: $(ls "$MODELDIR"/transformer/*.safetensors 2>/dev/null | wc -l)"
