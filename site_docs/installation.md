# Installation

## Prerequisites

- NVIDIA GPU and driver support for CUDA 13.
- `conda` or miniforge, `git`, and a Hugging Face account/token.
- Cosmos3-Super 64B uses 4xB200. LTX-2.3 and SANA-Video use 1xB200.

## Environment

```bash
git clone https://github.com/NVlabs/Sol-Video-Inference-Engine.git Sol-Video-Inference-Engine
cd Sol-Video-Inference-Engine

PYTHON_VERSION=3.12 bash scripts/create_code_conda_env.sh
source "$PWD/scripts/use_code_storage_env.sh"
conda activate "$PWD/.conda/ltx23"

uv pip install -e "$PWD/python[diffusion]" --prerelease=allow

PYTHON_BIN=.conda/ltx23/bin/python bash scripts/postinstall_cuda_jit.sh
```

`scripts/create_code_conda_env.sh` creates `.conda/ltx23` by default, installs
`pip` and `uv`, and installs conda activation hooks for `scripts/use_code_storage_env.sh`.
The script defaults to Python 3.11 when `PYTHON_VERSION` is unset; the agent guide
sets `PYTHON_VERSION=3.12` for this stack.

Add `--with-te` to `scripts/postinstall_cuda_jit.sh` when using the NVFP4 path for Cosmos3-Super or LTX-2.3.

## Verify

```bash
.conda/ltx23/bin/python -c "import torch, diffusers, sglang; print(torch.__version__, diffusers.__version__, torch.cuda.is_available())"
```

Expected versions are torch 2.11.0+cu130 and diffusers 0.38.0. `torch.cuda.is_available()` is false on CPU-only setup hosts.

## Model downloads

```bash
export HF_HOME="$PWD/.hf_cache"
huggingface-cli login

huggingface-cli download nvidia/Cosmos3-Super
huggingface-cli download Lightricks/LTX-2.3
huggingface-cli download Efficient-Large-Model/SANA-Video_2B_480p_diffusers
```

Convenience scripts are available under `scripts/cosmos/`, `scripts/ltx/`, and `scripts/sana/` for resumable or Slurm-based downloads.
