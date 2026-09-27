#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 MODEL_DIR H3_VIDEO PROMPT OUTPUT_MP4" >&2
  exit 2
fi
example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$example_dir/../infer.py" --model "$1" --input "$2" --prompt "$3" --output "$4" \
  --width 1920 --height 1080 --seed 303000 --decoder-seed 303000
