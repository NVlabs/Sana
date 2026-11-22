#!/usr/bin/env bash
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

set -e
MODE=${1:-}
TARGET=${2:-}
# Jetson Thor (aarch64, sm_110) needs CUDA 13: CUDA 12.8 and the cu128 wheels of torch have no sm_110.
if [ "$(uname -m)" = aarch64 ] && [ -e /etc/nv_tegra_release ]; then
    THOR=1
    TORCH_INDEX=cu130
else
    THOR=0
    TORCH_INDEX=cu128
fi
if [ "$MODE" = conda ]; then
    eval "$(conda shell.bash hook)"
    conda create -y -n "${TARGET:-sana-robot}" python=3.11
    conda activate "${TARGET:-sana-robot}"
    if [ "$THOR" = 1 ]; then
        conda install -y -c nvidia/label/cuda-13.0.2 cuda-toolkit
    else
        conda install -y -c nvidia cuda-toolkit=12.8
    fi
elif [ "$MODE" = venv ]; then
    "${PYTHON:-python3.11}" -m venv "${TARGET:-.venv}"
    source "${TARGET:-.venv}/bin/activate"
fi
if [ "$THOR" = 1 ]; then
    # Triton 3.5.1 ships the ptxas of CUDA 12.8, which rejects sm_110a: point it, and the builds, at the CUDA 13
    # toolkit, now and at every activation of the environment.
    CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
    nvcc --version | grep -q "release 13\." || { echo "Jetson Thor needs the CUDA 13 toolkit's nvcc on PATH" >&2; exit 1; }
    HOOK="export CUDA_HOME=\"$CUDA_ROOT\"; export TRITON_PTXAS_PATH=\"$CUDA_ROOT/bin/ptxas\""
    if [ -n "${CONDA_PREFIX:-}" ] && [ "$MODE" != venv ]; then
        mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
        echo "$HOOK" > "$CONDA_PREFIX/etc/conda/activate.d/sana_robot_thor.sh"
    elif [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "$HOOK" >> "$VIRTUAL_ENV/bin/activate"
    else
        echo "Set these in every shell that runs sana_robot: $HOOK" >&2
    fi
    eval "$HOOK"
fi
pip install --upgrade pip
pip install torch==2.9.1 --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
pip install ninja packaging psutil wheel "setuptools<80"
if [ "$THOR" = 1 ]; then
    # flash-attn's setup.py knows sm 80 / 90 / 100 / 120 only: add the sm_110 target and compile for it alone.
    WORK=$(mktemp -d)
    pip download --no-deps --no-binary :all: --no-build-isolation flash-attn==2.8.3.post1 -d "$WORK"
    tar xzf "$WORK/flash_attn-2.8.3.post1.tar.gz" -C "$WORK"
    python - "$WORK/flash_attn-2.8.3.post1/setup.py" <<'EOF'
import sys
path = sys.argv[1]
source = open(path).read()
anchor = '            cc_flag.append("arch=compute_120,code=sm_120")\n'
target = (
    '        if bare_metal_version >= Version("13.0") and "110" in cuda_archs():\n'
    '            cc_flag.append("-gencode")\n'
    '            cc_flag.append("arch=compute_110,code=sm_110")\n'
)
assert source.count(anchor) == 1, "unexpected flash-attn setup.py"
open(path, "w").write(source.replace(anchor, anchor + target))
EOF
    (cd "$WORK/flash_attn-2.8.3.post1" && FLASH_ATTN_CUDA_ARCHS=110 MAX_JOBS=${MAX_JOBS:-8} pip install --no-build-isolation .)
    rm -rf "$WORK"
else
    pip install flash-attn==2.8.3.post1 --no-build-isolation
fi
pip install -e "$(dirname "$0")"
if [ "$THOR" = 1 ]; then
    # Compile the NVFP4 matrix product now, while no policy is loaded (about 4 minutes).
    python -c "from sana_robot.kernels.inference.ops.nvfp4.build import load_extension; load_extension('sm110')"
fi
