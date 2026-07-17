#!/usr/bin/env bash
#SBATCH --job-name=lbv-fetch-te
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu_datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs

MODELDIR="/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/models/lingbot-video-moe-30b-a3b"
REPO="robbyant/lingbot-video-moe-30b-a3b"
REL="text_encoder/model-00001-of-00002.safetensors"
URL="https://huggingface.co/${REPO}/resolve/main/${REL}"
DEST="${MODELDIR}/${REL}"
TOKEN="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)"
EXPECTED=4967229296

echo "[fetch-te] node=$(hostname) -> $DEST"
# -c: resume; --retry-connrefused + tries + waitretry: survive reset-near-end;
# --timeout guards against silent hangs so a reset triggers a retry instead of freezing.
wget -c \
  --header="Authorization: Bearer ${TOKEN}" \
  --tries=50 --waitretry=5 --retry-connrefused \
  --timeout=60 --read-timeout=60 \
  -O "$DEST" "$URL"
echo "[fetch-te] wget exit=$?"

SZ=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
echo "[fetch-te] size=$SZ expected=$EXPECTED"
if [ "$SZ" = "$EXPECTED" ]; then echo "[fetch-te] OK complete"; else echo "[fetch-te] INCOMPLETE"; fi
