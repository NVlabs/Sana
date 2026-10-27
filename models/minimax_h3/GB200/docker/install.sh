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
  # Only the toolchain Triton/Inductor JIT needs. Python comes from uv below, not apt: apt's
  # python3.12 drags in tzdata, whose postinst fails where /etc/localtime is a bind mount (pyxis).
  apt-get update
  apt-get install -y --no-install-recommends build-essential git ca-certificates curl
  rm -rf /var/lib/apt/lists/*
fi

# Self-contained CPython 3.12 + venv under $PREFIX, managed by uv.
mkdir -p $PREFIX/bin
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | env UV_INSTALL_DIR=$PREFIX/bin UV_NO_MODIFY_PATH=1 sh
export UV_PYTHON_INSTALL_DIR=$PREFIX/python UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
$PREFIX/bin/uv python install 3.12
$PREFIX/bin/uv venv --python 3.12 --seed $PREFIX/venv
PIP="$PREFIX/bin/uv pip install --python $PREFIX/venv/bin/python"
$PIP --index-url https://download.pytorch.org/whl/cu130 torch==2.10.0 torchvision==0.25.0
$PIP -r $SRC/models/minimax_h3/GB200/docker/requirements.txt

# diffusers MiniMax-H3 branch, kept as a checkout so the checkpoint converter ships with it
if [ ! -d $PREFIX/diffusers/.git ]; then
  git clone --filter=blob:none https://github.com/huggingface/diffusers.git $PREFIX/diffusers
fi
git -C $PREFIX/diffusers fetch --depth 1 origin $DIFFUSERS_COMMIT 2>/dev/null || true
git -C $PREFIX/diffusers checkout -q $DIFFUSERS_COMMIT
$PIP --no-deps -e $PREFIX/diffusers
rm -rf ${UV_CACHE_DIR}

mkdir -p $PREFIX/models/minimax_h3 $PREFIX/techniques
cp -r $SRC/models/minimax_h3/GB200 $PREFIX/models/minimax_h3/
cp -r $SRC/techniques/sparse_backends $PREFIX/techniques/
$PREFIX/venv/bin/python -c "import torch, triton, cutlass, diffusers, transformers, fastapi; \
print('torch', torch.__version__, 'triton', triton.__version__, 'cutlass-dsl ok', 'diffusers', diffusers.__version__, \
'transformers', transformers.__version__)"
