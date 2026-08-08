#!/bin/bash
#SBATCH --job-name=h3-baseline
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/home/yitongl/code/models/logs/h3_baseline_%j.out

# MiniMax-H3 pristine diffusers baseline on one GB200 node.
#
# hsg's QOSMinGRES makes a whole 4-GPU node the smallest allocation, so the node is taken
# exclusively. The baseline itself is not context-parallel: `MiniMaxH3Transformer3DModel` declares
# no `_cp_plan`, so diffusers cannot shard the packed sequence.
#
# Everything is resident on GPU 0 — the DiT, the Qwen3-VL conditioner and both VAEs, 134 GB of
# weights in 186 GB. Splitting the conditiner onto a second card is what upstream documents for
# 80 GB cards, but it cannot work here: `_execution_device` returns the device of the first
# hooked component, `text_encoder` is first in the component map, and a `device_map` on it drags
# the whole pipeline onto that card while the DiT stays behind (job 5812896 died exactly there).
#
# Workload matches the published SGLang H3 benchmark cell so the numbers are comparable:
# t2va, 1344x768, 124 frames @ 24 fps (5.167 s), 50 steps, shift 12.0 / 3.0, 1 warmup + 1 measured.
set -uo pipefail

readonly ROOT=/home/yitongl/code/agent_deploy/Sol-LTX-Infer/models/minimax_h3
readonly RUN_ID="${SLURM_JOB_ID:-manual}"

export PYTHON_BIN=/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/lingbot-fa2/bin/python
export H3_CONDA_ROOT=/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3
export H3_CONDA_ENV=lingbot-fa2

export H3_MODEL_PATH=/home/yitongl/code/models/MiniMax-H3-diffusers
export H3_PROMPT_FILE="${ROOT}/prompts/t2va_example_1.json"
export OUT_DIR="${ROOT}/runs/baseline_${RUN_ID}"

export H3_HEIGHT=768
export H3_WIDTH=1344
export H3_NUM_FRAMES=124
export H3_STEPS=50
export H3_SEED=0
export H3_WARMUP=1
export H3_MEASURE=1
export H3_DIT_DEVICE=0
export H3_ENCODER_DEVICE=0
export H3_TAG="baseline_${RUN_ID}"

export HF_HUB_OFFLINE=1
export HF_HOME=/home/yitongl/code/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${RUN_ID} start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

bash "${ROOT}/baseline/scripts/run_minimax_h3_gpu.sh"
status=$?

echo "end=$(date -Is) status=${status}"
ls -la "${OUT_DIR}" 2>/dev/null
exit ${status}
