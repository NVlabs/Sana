#!/usr/bin/env bash
#SBATCH --job-name=lbv-build-env
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs

CONDA_ROOT="/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

ENV=lingbot-video
ENVDIR="${CONDA_ROOT}/envs/${ENV}"

echo "[build-env] node=$(hostname)  removing any corrupted env"
conda env remove -n "$ENV" -y 2>/dev/null || true
rm -rf "$ENVDIR" 2>/dev/null || true

echo "[build-env] cloning hunyuanvideo15 -> ${ENV}"
conda create --clone hunyuanvideo15 --name "$ENV" -y
echo "[build-env] clone exit=$?"

PIP="${ENVDIR}/bin/pip"
PY="${ENVDIR}/bin/python"

echo "[build-env] installing pinned deps"
"$PIP" install \
  "diffusers==0.39.0" "transformers==5.8.1" "peft==0.19.1" \
  "decord>=0.6.0" "json_repair>=0.30" "imageio-ffmpeg>=0.5.1" \
  "accelerate>=1.8.0" "safetensors>=0.4.5" "scipy>=1.11" "hf_transfer"
echo "[build-env] pip exit=$?"

echo "[build-env] installing repo (editable)"
"$PIP" install -e . --no-deps
echo "[build-env] repo install exit=$?"

echo "[build-env] === verification ==="
"$PY" - <<'EOF'
import torch, diffusers, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "grouped_mm", hasattr(torch,"_grouped_mm"))
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
from diffusers import ContextParallelConfig
from diffusers.hooks.context_parallel import apply_context_parallel
from diffusers.models._modeling_parallel import ParallelConfig
print("CP APIs: OK")
from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL: OK")
import decord, json_repair
print("decord/json_repair: OK")
EOF
echo "[build-env] DONE"
