#!/usr/bin/env bash
#SBATCH -J sol-wan-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:20:00

set -euo pipefail

SOL_REPO=${SOL_REPO:?set SOL_REPO to this Sol-Attn checkout}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to an output directory}
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  source "$ENV_ACTIVATE"
fi

mkdir -p "$OUT_ROOT"
cd "$SOL_REPO"
export PYTHONPATH="$SOL_REPO${PYTHONPATH:+:$PYTHONPATH}"
export SOL_ATTN_SM100_CORRECTNESS_GATE=1
export SOL_ATTN_SM100_EVENT_LOG="$OUT_ROOT/sm100_events.jsonl"

python -m integrations.wan.smoke \
  --tokens 8192 \
  --batch 1 \
  --heads 1 \
  --tau 1.7 \
  --seed 37 \
  --output "$OUT_ROOT/smoke.json" \
  >"$OUT_ROOT/run.log" 2>&1

test -s "$OUT_ROOT/smoke.json"
echo "WAN_SM100_SMOKE_OK OUT=$OUT_ROOT"
