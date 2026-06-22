#!/usr/bin/env bash
# Reconstructed vanilla HunyuanVideo (diffusers) baseline launcher.
#
# Invoked by scripts/launch_candidate.py's generated launch.sh, which has already:
#   - cd'd into the runtime root (this dir's parent),
#   - exported the model profile [env] (MODEL_REPO, HUNYUAN_*, HF_*, PYTHON_BIN, SEED, ...),
#   - exported OUT_DIR (the run's outputs/ dir).
# We run gpu_infer.py under $PYTHON_BIN and tee everything to $OUT_DIR/run.log so
# collect_run.determine_status() can read out.mp4 + run.log.
set -uo pipefail

: "${OUT_DIR:?OUT_DIR must be set by the harness}"
mkdir -p "$OUT_DIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # runtime root (has gpu_infer.py)
PYBIN="${PYTHON_BIN:-python}"

run() {
    echo "[run] host=$(hostname) pybin=$PYBIN runtime=$HERE out=$OUT_DIR"
    echo "[run] HF_HOME=${HF_HOME:-<unset>} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
    "$PYBIN" "$HERE/gpu_infer.py"
    local rc=$?
    echo "[run] gpu_infer.py exit=$rc"
    return $rc
}

run 2>&1 | tee "$OUT_DIR/run.log"
exit "${PIPESTATUS[0]}"
