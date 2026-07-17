#!/usr/bin/env bash
#SBATCH --job-name=lbv-fa3-build
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Build FlashAttention-3 (flash_attn_interface) from source for GB200 (sm_100).
# No prebuilt wheel exists for aarch64+Blackwell+cu130.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm/logs
source /lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/etc/profile.d/conda.sh
conda activate lingbot-video

CUDA_PREFIX=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/cuda13
BUILD_DIR=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/build/flash-attention

echo "=== GPU ==="; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
echo "=== torch ==="; python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"

# --- 1. CUDA 13 toolkit in a dedicated prefix (does not touch torch's bundled libs) ---
if [ ! -x "$CUDA_PREFIX/bin/nvcc" ]; then
  echo "=== installing cuda-toolkit 13 into $CUDA_PREFIX ==="
  conda create -y -p "$CUDA_PREFIX" -c nvidia "cuda-toolkit=13.*" 2>&1 | tail -5
fi
export CUDA_HOME="$CUDA_PREFIX"
export PATH="$CUDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "=== nvcc ==="; "$CUDA_PREFIX/bin/nvcc" --version | tail -2

# --- 2. build deps ---
pip install -q ninja packaging wheel setuptools 2>&1 | tail -2
echo "ninja: $(python -c 'import ninja,sys;print(ninja.__version__)' 2>/dev/null)"

# --- 3. source ---
if [ ! -d "$BUILD_DIR/.git" ]; then
  mkdir -p "$(dirname "$BUILD_DIR")"
  git clone https://github.com/Dao-AILab/flash-attention.git "$BUILD_DIR" 2>&1 | tail -3
fi
cd "$BUILD_DIR"
git log --oneline -1

# --- 4. build FA3 (hopper tree also targets Blackwell in recent commits) ---
export TORCH_CUDA_ARCH_LIST="10.0"
export MAX_JOBS="${MAX_JOBS:-48}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
echo "=== building FA3 (hopper) arch=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS ==="
cd "$BUILD_DIR/hopper"
python setup.py install 2>&1 | tail -40
echo "BUILD_EXIT=${PIPESTATUS[0]}"

# --- 5. verify on the GB200 ---
echo "=== verify ==="
python - <<'EOF'
try:
    import flash_attn_interface as f
    from flash_attn_interface import flash_attn_varlen_func
    import torch
    print("FA3 import OK:", f.__file__)
    # tiny varlen smoke test
    q=torch.randn(8,4,64,device='cuda',dtype=torch.bfloat16)
    cu=torch.tensor([0,8],device='cuda',dtype=torch.int32)
    o=flash_attn_varlen_func(q=q,k=q,v=q,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=8,max_seqlen_k=8,causal=False)
    print("FA3 varlen smoke OK:", (o[0] if isinstance(o,tuple) else o).shape)
except Exception as e:
    import traceback; traceback.print_exc(); print("FA3 VERIFY FAILED:", e)
EOF
echo "=== DONE ==="
