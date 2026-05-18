#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SANA-WM environment installer.
#
# This script is *separate* from the upstream ``environment_setup.sh`` because
# SANA-WM needs a newer toolchain (Python 3.11, CUDA 12.8, torch 2.9.1,
# triton 3.5.1, xformers 0.0.33.post2) that would break the other configs
# in this repo (Sana, Sana-Sprint, Sana-Video, LongSana). The default
# installer is intentionally untouched.
#
# Usage:
#   bash ./environment_setup_sana_wm.sh sana-wm        # create a fresh conda env
#   bash ./environment_setup_sana_wm.sh                # install into the currently
#                                                     # activated env (no conda
#                                                     # create)
#
# Idempotent: re-running on an existing env will reconcile versions instead of
# failing.
# -----------------------------------------------------------------------------
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # Required to activate conda environments inside a non-interactive shell.
    eval "$(conda shell.bash hook)"

    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "[sana-wm] conda env '$CONDA_ENV' already exists; reusing it."
    else
        # Python 3.11 is required: triton 3.5.0's JIT decorator does
        # ``inspect.getsource(fn)`` and regex-searches for ``^def\s+\w+\s*\(``.
        # On 3.10 the source returned for fla's ``@triton.jit`` kernels begins
        # after the decorator line, so the regex returns None and import-time
        # blows up with
        # ``AttributeError: 'NoneType' object has no attribute 'start'``.
        conda create -n "$CONDA_ENV" python=3.11 -y
    fi
    conda activate "$CONDA_ENV"

    # Match the CUDA major/minor of the torch wheels installed below
    # (``+cu128``). Cross-major (e.g., toolkit 12.4 + torch cu128) works for
    # most inference paths because the torch wheel ships its own CUDA libs,
    # but flash-attn and other from-source builds need a matching nvcc.
    conda install -c nvidia cuda-toolkit=12.8 -y
else
    echo "[sana-wm] Skipping conda env creation. Make sure the target env is activated."
fi

# pip + wheel + setuptools<80. setuptools must be pinned to <80 because
# mmcv 1.7.2 still imports ``pkg_resources`` at build time, and setuptools
# 80+ has removed the implicit shim that exposes ``pkg_resources`` as an
# importable module. Without this pin, ``pip install -e .`` fails inside
# mmcv with ``ModuleNotFoundError: No module named 'pkg_resources'``.
pip install -U pip wheel
pip install "setuptools<80"

# Install the torch + companions matching the working SANA-WM env before mmcv,
# which imports torch from setup.py.
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
pip install --upgrade triton==3.5.1
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    xformers==0.0.33.post2

# mmcv==1.7.2 must be built without PEP 517 build isolation so its setup.py
# can resolve ``pkg_resources`` from the env's setuptools<80.
pip install --no-build-isolation mmcv==1.7.2

# Editable install of this repo without resolving pyproject.toml dependencies.
# SANA-WM has its own requirements file because Python extras can only add to
# base dependencies; they cannot replace the older default Sana runtime pins.
pip install -e . --no-deps
pip install -r requirements/sana_wm.txt

# Pi3X for camera intrinsics estimation from a single image.
# Install with --no-deps to avoid overwriting torch/torchvision/numpy.
pip install git+https://github.com/yyfz/Pi3.git --no-deps
pip install huggingface_hub opencv-python plyfile

pip install --no-build-isolation "flash-attn>=2.7.0"

echo
echo "[sana-wm] Done. Activate with:  conda activate ${CONDA_ENV:-<your-env>}"
echo "[sana-wm] Next: see docs/sana_wm.md for data paths and pretrained weights."
