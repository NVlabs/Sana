#!/bin/bash
#SBATCH --job-name=sana-video-dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sana-video-dl-%j.out
#SBATCH --error=sana-video-dl-%j.out

# Download SANA-Video weights (default Efficient-Large-Model/SANA-Video_2B_480p_diffusers;
# override as $1). Honors pre-set HF_HOME / HF_TOKEN / PYTHON_BIN.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}" 2>/dev/null || true)}"
# Xet CAS stream has stalled indefinitely on large shards; force the classic LFS
# CDN and fail fast on a hung connection so the built-in retry actually fires.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.conda/ltx23/bin/python}"

REPO=${1:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}
mkdir -p "$HF_HUB_CACHE"

echo "[$(date)] Node: $(hostname)  Python: $PYTHON"
echo "[$(date)] Downloading $REPO (max_workers=16) -> $HF_HUB_CACHE"

if "$PYTHON" - "$REPO" << 'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

repo = sys.argv[1]
path = snapshot_download(
    repo,
    token=os.environ.get("HF_TOKEN") or None,
    cache_dir=os.environ["HF_HUB_CACHE"],
    max_workers=16,
)
print("Snapshot path:", path, flush=True)
total = 0
print("---- files ----", flush=True)
for root, _, files in os.walk(path):
    for f in sorted(files):
        fp = os.path.join(root, f)
        try:
            sz = os.path.getsize(fp)  # resolves symlink -> real blob size
        except OSError:
            sz = 0
        total += sz
        print(f"  {sz/1e6:10.1f} MB  {os.path.relpath(fp, path)}", flush=True)
print(f"---- total: {total/1e9:.2f} GB ----", flush=True)
PYEOF
then
    echo "[$(date)] du of cache entry:"
    du -sh "$HF_HUB_CACHE/models--$(echo "$REPO" | sed 's:/:--:g')" 2>/dev/null
    echo "[$(date)] DOWNLOAD_COMPLETE_MARKER"
else
    rc=$?
    echo "[$(date)] DOWNLOAD_FAILED_MARKER rc=$rc"
    exit $rc
fi
