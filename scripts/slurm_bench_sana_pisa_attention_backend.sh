#!/usr/bin/env bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 00:45:00
#SBATCH -J sana-pisa-attn-bench
#SBATCH -o output/benchmarks/sana_pisa_attention/slurm-%j.out
#SBATCH -e output/benchmarks/sana_pisa_attention/slurm-%j.err

set -euo pipefail

ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/agent_deploy/Sol-LTX-Infer"
SGL_ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer"
PYTHON="$SGL_ROOT/.conda/ltx23/bin/python"
OUT_DIR="${OUT_DIR:-$ROOT/output/benchmarks/sana_pisa_attention/$SLURM_JOB_ID}"
CACHE_ROOT="$ROOT/output/benchmarks/sana_pisa_attention/cache"

mkdir -p "$OUT_DIR" "$CACHE_ROOT"
cd "$ROOT"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SGL_ROOT/python:${PYTHONPATH:-}"
export LTX23_CACHE_ROOT="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR"

CU13="$SGL_ROOT/.conda/ltx23/lib/python3.12/site-packages/nvidia/cu13"
if [[ -d "$CU13" ]]; then
  export CUDA_HOME="$CU13"
  export CUDA_PATH="$CU13"
  export PATH="$CU13/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$SGL_ROOT/.conda/ltx23/lib/python3.12/site-packages/nvidia/cublas/lib:$SGL_ROOT/.conda/ltx23/lib/python3.12/site-packages/nvidia/cudnn/lib:$SGL_ROOT/.conda/ltx23/lib/python3.12/site-packages/nvidia/nccl/lib:$CU13/lib:$CU13/lib64:${LD_LIBRARY_PATH:-}"
fi

"$PYTHON" scripts/bench_sana_pisa_attention_backend.py \
  --out "$OUT_DIR/result.json" \
  --warmup "${WARMUP:-5}" \
  --iterations "${ITERATIONS:-15}" \
  --densities "${DENSITIES:-0.1,0.125,0.25,0.5,0.75}"

echo "[done] $OUT_DIR/result.json"
