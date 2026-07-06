#!/usr/bin/env bash
# Sol-LTX-Infer run_script for the Bernini t2v baseline.
#
# launch.sh cd's into runtime/bernini_baseline and exports the model [env] +
# OUT_DIR, then calls this shim. We hand off to gpu_infer.py, which cd's into
# $BERNINI_ROOT and orchestrates the 4-way torchrun warmup+hot t2v run, then
# normalizes outputs into the standard run bundle. All stdout is teed to run.log.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by launch.sh}"
: "${BERNINI_WEIGHTS:?BERNINI_WEIGHTS must be set in models/bernini.toml [env]}"
PYBIN="${PYTHON_BIN:-/usr/bin/python3.12}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # runtime/bernini_baseline
mkdir -p "$OUT_DIR"

"$PYBIN" "$HERE/gpu_infer.py" 2>&1 | tee "$OUT_DIR/run.log"
exit "${PIPESTATUS[0]}"
