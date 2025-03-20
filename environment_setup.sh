#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit=12.4 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# init a raw torch to avoid installation errors.
# pip install torch

# update pip to latest version for pyproject.toml setup.
pip install -U pip

# for fast attn
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# install sana
pip install -e .

# install torchprofile
# pip install git+https://github.com/zhijian-liu/torchprofile
