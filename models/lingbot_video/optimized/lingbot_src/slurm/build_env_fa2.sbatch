#!/usr/bin/env bash
#SBATCH --job-name=lbv-env-fa2
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Clone nunchaku_blackwell (py3.11, torch2.10+cu130, FA2 2.8.3 prebuilt) -> lingbot-fa2,
# then install LingBot-Video deps. FA2 is reused as the flash_attn_interface backend
# (via slurm/shims/flash_attn_interface.py) to unlock batch_cfg + context parallel.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
CONDA_ROOT="/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

ENV=lingbot-fa2
ENVDIR="${CONDA_ROOT}/envs/${ENV}"

echo "[env-fa2] node=$(hostname) removing any stale ${ENV}"
conda env remove -n "$ENV" -y 2>/dev/null || true
rm -rf "$ENVDIR" 2>/dev/null || true

echo "[env-fa2] cloning nunchaku_blackwell -> ${ENV}"
conda create --clone nunchaku_blackwell --name "$ENV" -y
echo "[env-fa2] clone exit=$?"

PIP="${ENVDIR}/bin/pip"
PY="${ENVDIR}/bin/python"

echo "[env-fa2] installing LingBot deps (no decord; keep torch/FA2 intact)"
"$PIP" install \
  "diffusers==0.39.0" "transformers==5.8.1" "peft==0.19.1" \
  "json_repair>=0.30" "imageio-ffmpeg>=0.5.1" "accelerate>=1.8.0" \
  "safetensors>=0.4.5" "scipy>=1.11" "hf_transfer"
echo "[env-fa2] pip exit=$?"

echo "[env-fa2] installing repo editable"
"$PIP" install -e . --no-deps
echo "[env-fa2] repo install exit=$?"

echo "[env-fa2] === CPU-side verification (no GPU needed) ==="
"$PY" - <<'EOF'
import torch, diffusers, transformers, peft
print("python", __import__("sys").version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "grouped_mm", hasattr(torch,"_grouped_mm"))
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
from diffusers import ContextParallelConfig
from diffusers.hooks.context_parallel import apply_context_parallel
print("CP APIs: OK")
from transformers import Qwen3VLForConditionalGeneration
print("Qwen3VL: OK")
EOF
echo "[env-fa2] DONE"
