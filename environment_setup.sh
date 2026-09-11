#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SANA environment installer using uv. Single source of truth for deps is
# pyproject.toml; this script handles environment creation, Python 3.11,
# OS-aware PyTorch installation (CUDA vs. Apple Silicon/MLX), and packages
# that need special install flags.
#
# Usage:
#   bash ./environment_setup.sh .venv  # create a fresh uv env in .venv
#   bash ./environment_setup.sh        # install into the active env or default .venv
#
# Idempotent: re-running on an existing env will reconcile versions.
# -----------------------------------------------------------------------------
set -e

if [ "${SKIP_ENV_SETUP}" = "true" ]; then
    echo "SKIP_ENV_SETUP is set to true. Skipping all environment setup steps."
    exit 0
fi

if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first."
    exit 1
fi

VENV_DIR=${1:-".venv"}

if [ -n "$1" ] || [ ! -d "$VENV_DIR" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "[sana] Creating virtual environment '$VENV_DIR' with Python 3.11..."
        uv venv "$VENV_DIR" --python 3.11
    fi
    source "$VENV_DIR/bin/activate"
else
    if [ -z "$VIRTUAL_ENV" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    fi
fi

# Detect Apple Silicon
OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)
IS_APPLE_SILICON=false

if [ "$OS_NAME" = "Darwin" ] && [ "$ARCH_NAME" = "arm64" ]; then
    IS_APPLE_SILICON=true
    echo "🍎 Detected Apple Silicon (Mac ARM64). Adapting for MPS/MLX..."
else
    echo "🐧 Detected standard/CUDA target environment."
    if ! command -v nvcc &> /dev/null; then
        echo "WARNING: 'nvcc' not found in PATH. Source extensions will fail to build."
    fi
fi

# setuptools<80: mmcv 1.7.2's setup.py imports ``pkg_resources``
uv pip install "setuptools<80" wheel

if [ "$IS_APPLE_SILICON" = true ]; then
    # Standard PyTorch has native MPS support on Mac
    uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
    uv pip install mlx
else
    # Pre-install the torch stack from the cu128 index so uv doesn't pull PyPI defaults
    uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
fi

# mmcv must build without PEP 517 isolation
uv pip install --no-build-isolation mmcv==1.7.2

# Editable install resolves everything else from pyproject.toml
# (Triton, xformers, etc. will be automatically skipped on macOS thanks to PEP 508 markers)
uv pip install -e .

# Pi3X (camera intrinsics)
uv pip install git+https://github.com/yyfz/Pi3.git --no-deps

if [ "$IS_APPLE_SILICON" = true ]; then
    echo "[sana] Skipping flash-attn installation (CUDA-only)."
else
    MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-2} \
        uv pip install --no-build-isolation "flash-attn>=2.7.0"
fi

# NVIDIA Transformer Engine: enables fp8 / fp4 quantized SANA-WM streaming
# inference (--stage1_precision / --refiner_precision). Built from source against
# the env's CUDA toolkit; best-effort -- a build failure here does not abort the
# install (bf16 inference works without it). Skip explicitly with SANA_SKIP_TE=1.
if [ "${SANA_SKIP_TE:-0}" != "1" ]; then
    echo "[sana] Installing Transformer Engine (fp8/fp4 inference); set SANA_SKIP_TE=1 to skip."
    if ! MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-2} \
        pip install --no-build-isolation "transformer_engine[pytorch]>=2.0"; then
        echo "[sana] WARNING: Transformer Engine install failed; bf16 inference still works."
        echo "[sana]          fp8/fp4 need it -- retry with: pip install --no-build-isolation 'transformer_engine[pytorch]>=2.0'"
    fi
fi

echo
echo "[sana] Done. Activate with:  source ${VENV_DIR}/bin/activate"
