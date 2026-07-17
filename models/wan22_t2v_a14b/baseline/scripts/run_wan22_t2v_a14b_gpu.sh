#!/usr/bin/env bash
# run_script for the Wan2.2-T2V-A14B (MoE) baseline. launch.sh cd's to the
# runtime root, exports the model [env] + OUT_DIR, then calls this shim.
set -euo pipefail
: "${OUT_DIR:?OUT_DIR must be set (launch.sh sets it)}"
: "${WAN22_WEIGHTS:?WAN22_WEIGHTS must be set (models/wan22_t2v_a14b.toml [env])}"
PYBIN="${PYTHON_BIN:-/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/sparse_attn_training/.venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # runtime/wan22_t2v_a14b_baseline
mkdir -p "$OUT_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
echo "[run] host=$(hostname) weights=$WAN22_WEIGHTS out=$OUT_DIR"
nvidia-smi -L || true
"$PYBIN" "$HERE/gpu_infer.py" 2>&1 | tee "$OUT_DIR/run.log"
exit "${PIPESTATUS[0]}"
