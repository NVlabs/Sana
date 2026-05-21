#!/usr/bin/env bash
set -e

# Check if we should skip environment setup entirely
if [ "${SKIP_ENV_SETUP}" = "true" ]; then
    echo "SKIP_ENV_SETUP is set to true. Skipping all environment setup steps."
    echo "Using default conda environment. Make sure it has all required packages installed."
    exit 0
fi

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit=12.8 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# update pip to latest version for pyproject.toml setup.
pip install -U pip

# Pin PyTorch to a version compatible with flash-attn==2.8.2.
# PyTorch>=2.8.0 does not yet have prebuilt flash-attn wheels; pinning to 2.6.0 ensures
# a successful install. See: https://github.com/NVlabs/Sana/issues/375
pip install torch==2.6.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# for fast attn
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128

# install sana
pip install -e .

pip install flash-attn==2.8.2 --no-build-isolation

# install torchprofile
# pip install git+https://github.com/zhijian-liu/torchprofile
