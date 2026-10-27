#!/usr/bin/env bash
# Environment for the MiniMax-H3 GB200 server. Used verbatim by the Dockerfile and by the pyxis
# validation run on hsg, so the image and the validated environment cannot drift apart.
#   install.sh <sol-engine checkout> <prefix>
set -euo pipefail
SRC=${1:?sol-engine checkout}
PREFIX=${2:-/opt/sol-engine}
DIFFUSERS_COMMIT=abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc   # huggingface/diffusers PR #14355 (minimax-h3)

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  apt-get update
  apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev \
    build-essential git ca-certificates curl
  rm -rf /var/lib/apt/lists/*
fi

python3.12 -m venv $PREFIX/venv
PIP="$PREFIX/venv/bin/python -m pip"
$PIP install --no-cache-dir --upgrade pip
$PIP install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0 torchvision==0.25.0
$PIP install --no-cache-dir -r $SRC/models/minimax_h3/GB200/docker/requirements.txt

# diffusers MiniMax-H3 branch, kept as a checkout so the checkpoint converter ships with it
if [ ! -d $PREFIX/diffusers/.git ]; then
  git clone --filter=blob:none https://github.com/huggingface/diffusers.git $PREFIX/diffusers
fi
git -C $PREFIX/diffusers fetch --depth 1 origin $DIFFUSERS_COMMIT 2>/dev/null || true
git -C $PREFIX/diffusers checkout -q $DIFFUSERS_COMMIT
$PIP install --no-cache-dir --no-deps -e $PREFIX/diffusers

mkdir -p $PREFIX/models/minimax_h3 $PREFIX/techniques
cp -r $SRC/models/minimax_h3/GB200 $PREFIX/models/minimax_h3/
cp -r $SRC/techniques/sparse_backends $PREFIX/techniques/
$PREFIX/venv/bin/python -c "import torch, triton, cutlass, diffusers, transformers, fastapi; \
print('torch', torch.__version__, 'triton', triton.__version__, 'cutlass-dsl ok', 'diffusers', diffusers.__version__, \
'transformers', transformers.__version__)"
