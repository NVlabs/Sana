#!/usr/bin/env bash
# Shared environment for LingBot-Video Slurm jobs on the GB200 cluster.
# Source this from every job script.

# --- Conda env (cloned from hunyuanvideo15, torch 2.10+cu130 with torch._grouped_mm) ---
CONDA_ROOT="/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3"
# lingbot-fa2: py3.11 clone of nunchaku_blackwell — has FA2 2.8.3 prebuilt, reused as the
# flash_attn_interface backend (slurm/shims) to unlock batch_cfg + context parallel.
# lingbot-video (py3.12) has no FA2 → SDPA-only (sequential CFG). Switch via CONDA_ENV.
CONDA_ENV="${CONDA_ENV:-lingbot-fa2}"
# conda's aarch64 binutils activate.d hooks reference unbound vars (ADDR2LINE, ...),
# which trip `set -u` in job scripts. Relax nounset only around activation, then restore.
case $- in *u*) _lbv_had_u=1;; *) _lbv_had_u=0;; esac
set +u
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
[ "${_lbv_had_u}" = 1 ] && set -u

# --- Repo + model paths ---
_LINGBOT_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR="${REPO_DIR:-${_LINGBOT_SRC_DIR}}"
export MOE_MODEL_DIR="${MOE_MODEL_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/models/lingbot-video-moe-30b-a3b}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# --- MoE / attention backends (overridable per experiment) ---
export LINGBOT_MOE_EXPERT_BACKEND="${LINGBOT_MOE_EXPERT_BACKEND:-grouped_mm}"
export LINGBOT_MOE_PAD_BACKEND="${LINGBOT_MOE_PAD_BACKEND:-vectorized}"
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"
# Qwen3-VL text encoder defaults to flash_attention_3 (not installed on this stack).
# sdpa is torch-native and works on Blackwell; text-encoder attn cost is negligible vs DiT.
export LINGBOT_QWEN_ATTN_IMPLEMENTATION="${LINGBOT_QWEN_ATTN_IMPLEMENTATION:-sdpa}"

# slurm/shims provides an imageio-backed `decord` drop-in (no aarch64 wheel exists).
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/rewriter:${REPO_DIR}/slurm/shims:${PYTHONPATH:-}"

echo "[env] conda_env=${CONDA_ENV} moe_backend=${LINGBOT_MOE_EXPERT_BACKEND} model=${MOE_MODEL_DIR}"
