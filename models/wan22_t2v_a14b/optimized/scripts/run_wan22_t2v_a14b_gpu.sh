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
if [[ "${WAN22_CONTEXT_PARALLEL:-0}" == "1" ]]; then
  nvidia-smi topo -m || true
  NPROC="${WAN22_CP_DEGREE:-4}"
  set +e
  if [[ "${WAN22_LAUNCHER:-torchrun}" == "srun" ]]; then
    MASTER_ADDR="${WAN22_MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
    MASTER_PORT="${WAN22_MASTER_PORT:-29500}"
    export MASTER_ADDR MASTER_PORT
    srun --ntasks="$NPROC" --ntasks-per-node="$NPROC" --gpus-per-task=1 \
      --cpus-per-task="${WAN22_SRUN_CPUS_PER_TASK:-16}" --cpu-bind=none \
      bash -c 'export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS" LOCAL_RANK=0; exec "$1" "$2"' \
      _ "$PYBIN" "$HERE/gpu_infer.py" 2>&1 | tee "$OUT_DIR/run.log"
  else
    "$PYBIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
      "$HERE/gpu_infer.py" 2>&1 | tee "$OUT_DIR/run.log"
  fi
  run_status="${PIPESTATUS[0]}"
  set -e
else
  set +e
  "$PYBIN" "$HERE/gpu_infer.py" 2>&1 | tee "$OUT_DIR/run.log"
  run_status="${PIPESTATUS[0]}"
  set -e
fi
exit "$run_status"
