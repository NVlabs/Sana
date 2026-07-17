#!/usr/bin/env bash
set -uo pipefail

: "${OUT_DIR:?OUT_DIR must be set by the harness}"
mkdir -p "$OUT_DIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh ]]; then
    CODE_STORAGE_ENV_QUIET=1 source /home/yitongl/.codex/skills/code-storage-env/scripts/code_storage_env.sh
fi
PYBIN="${PYTHON_BIN:-python3}"

run() {
    echo "[run] host=$(hostname) pybin=$PYBIN runtime=$HERE out=$OUT_DIR"
    echo "[run] asset_root=${SANA_VIDEO_ASSET_ROOT:-<default>} prepare_assets=${SANA_VIDEO_PREPARE_ASSETS:-0}"
    "$PYBIN" "$HERE/gpu_infer.py"
    local rc=$?
    echo "[run] gpu_infer.py exit=$rc"
    return $rc
}

run 2>&1 | tee "$OUT_DIR/run.log"
exit "${PIPESTATUS[0]}"
