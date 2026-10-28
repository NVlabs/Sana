#!/usr/bin/env bash
# Environment for the Sol-Engine GB200 servers (MiniMax-H3 + Wan2.2). Used by the Dockerfile; the
# versions are the ones the 2026-10-28 measurements ran on.
#   install.sh <dir with requirements.txt> <prefix>      (the code itself is copied by the Dockerfile)
set -euo pipefail
SRC=${1:?directory holding requirements.txt}
PREFIX=${2:-/opt/sol-engine}
DIFFUSERS_COMMIT=abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc   # huggingface/diffusers PR #14355 (minimax-h3)

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  # only the toolchain Triton/Inductor JIT needs; Python comes from uv (apt's drags in tzdata)
  apt-get update
  apt-get install -y --no-install-recommends build-essential git ca-certificates curl
  rm -rf /var/lib/apt/lists/*
fi

mkdir -p $PREFIX/bin
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | env UV_INSTALL_DIR=$PREFIX/bin UV_NO_MODIFY_PATH=1 sh
export UV_PYTHON_INSTALL_DIR=$PREFIX/python UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
$PREFIX/bin/uv python install 3.12
$PREFIX/bin/uv venv --python 3.12 --seed $PREFIX/venv
PIP="$PREFIX/bin/uv pip install --python $PREFIX/venv/bin/python"
$PIP --index-url https://download.pytorch.org/whl/cu130 torch==2.10.0 torchvision==0.25.0
$PIP -r $SRC/requirements.txt
# packages pinned outside their own dependency metadata:
#   cuDNN 9.24 (torch's wheel pins an older 9.x; 9.24 is what the Wan numbers were measured with)
#   FlashAttention-4 4.0.0b36 for the fa4sol sparse backend (its cutlass-dsl / tvm-ffi are in requirements)
$PIP --no-deps nvidia-cudnn-cu13==9.24.0.43 flash-attn-4==4.0.0b36 quack-kernels==0.6.5 torch-c-dlpack-ext==0.1.5

if [ ! -d $PREFIX/diffusers/.git ]; then
  git clone --filter=blob:none https://github.com/huggingface/diffusers.git $PREFIX/diffusers
fi
git -C $PREFIX/diffusers fetch --depth 1 origin $DIFFUSERS_COMMIT 2>/dev/null || true
git -C $PREFIX/diffusers checkout -q $DIFFUSERS_COMMIT
$PIP --no-deps -e $PREFIX/diffusers
rm -rf ${UV_CACHE_DIR}

$PREFIX/venv/bin/python -c "import torch, triton, diffusers, transformers, fastapi
from diffusers import WanPipeline
print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'triton', triton.__version__,
      'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
# CuTe DSL / FA4 may need the driver (libcuda) at import; a build host has none, so only warn here.
$PREFIX/venv/bin/python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('fa4 import ok')" \
  || echo "WARNING: FA4 import not checked at build time (no GPU driver); verified at run time"
