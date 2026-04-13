<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Sol-RL/asset/sol-rl_logo.png" width="82%" alt="Sol-RL Logo"/>
</p>

# Sol-RL: FP4 Explore, BF16 Train for SANA, FLUX.1, and SD3.5-L

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Sol-RL/"><img src="https://img.shields.io/static/v1?label=Project&message=Homepage&color=blue&logo=github-pages" alt="Sol-RL Homepage"></a> &ensp;
  <a href="https://arxiv.org/abs/2604.06916"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sol-RL&color=red&logo=arxiv" alt="Sol-RL Arxiv"></a> &ensp;
  <a href="https://github.com/NVlabs/Sana/blob/main/train_scripts/sol_rl/README.md"><img src="https://img.shields.io/static/v1?label=Guide&message=Repository&color=black&logo=github" alt="Repository Guide"></a>
</div>

Sol-RL is a two-stage diffusion reinforcement learning framework that uses **FP4 / NVFP4 for exploration** and **BF16 for regeneration and optimization**. In this repository, we provide Sol-RL-style post-training entrypoints for **SANA**, **FLUX.1**, and **SD3.5-L**.

<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Sol-RL/asset/teaser.png" width="95%" alt="Sol-RL Teaser"/>
</p>

## Overview

This Sol-RL stack in `Sana` includes:

- Single-node multi-GPU launchers based on `torchrun`
- Training entrypoints for **SANA**, **FLUX.1**, and **SD3.5-L**
- Preset config families for **DiffusionNFT**, **Naive Scaling**, **Naive Quant**, and **Sol-RL**
- Built-in reward integration for **PickScore**, **CLIPScore**, **HPSv2**, and **ImageReward**
- Bundled prompt datasets under `diffusion/post_training/dataset`

Repository entrypoints:

- `train_scripts/sol_rl/run_sana_single_node_8gpu.sh`
- `train_scripts/sol_rl/run_sd3_single_node_8gpu.sh`
- `train_scripts/sol_rl/run_flux1_single_node_8gpu.sh`
- `train_scripts/sol_rl/train_sana.py`
- `train_scripts/sol_rl/train_sd3.py`
- `train_scripts/sol_rl/train_flux1.py`

## Environment Setup

### Basic Environment

Use this first. This tier does **not** require `transformer-engine`, and it does **not** use NVFP4 quantization.

Recommended config families:

- `*_diffusionnft_*`
- `*_naive_scaling_*`

Typical setup:

```bash
./environment_setup.sh sana-sol-rl
```

Equivalent manual setup:

```bash
pip install -U pip
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install flash-attn==2.8.2 --no-build-isolation
```

If you already activated your own environment, you can run `./environment_setup.sh` without the env-name argument.

### Advanced Environment

Use this only if you want the NVFP4 path.

Required config families:

- `*_naive_quant_*`
- `*_sol_rl_*`

Install `transformer-engine` with the same Python interpreter that will be used by `torchrun`:

```bash
python -m pip install --no-build-isolation "transformer-engine[pytorch]"
```

If the build fails with missing `nccl.h`, install NCCL development headers:

```bash
conda install -c nvidia nccl nccl-devel
```

## Quick Start

Default single-node launchers:

```bash
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

Examples:

```bash
CONFIG_SPEC=configs/sol_rl/sana.py:sana_diffusionnft_pickscore \
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
```

```bash
CONFIG_SPEC=configs/sol_rl/sd3.py:sd3_naive_scaling_hpsv2 \
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
```

```bash
CONFIG_SPEC=configs/sol_rl/flux1.py:flux1_sol_rl_imagereward \
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

SANA note:

- The SANA launcher automatically resolves the required native checkpoint if it is missing locally.
- The launcher also defaults to `DISABLE_XFORMERS=1` to avoid broken local xFormers CUDA builds.

## Configuration Families

Config naming pattern:

```text
<model>_<family>_<reward>
```

Examples:

- `sana_diffusionnft_pickscore`
- `sd3_naive_scaling_hpsv2`
- `flux1_sol_rl_imagereward`

There are 48 preset configs in total:

- 3 model families
- 4 rollout families
- 4 reward suffixes

| Family | Meaning | Rollout shape | TE / NVFP4 needed |
|---|---|---|---|
| `diffusionnft` | PEFT-only baseline | 24-in-24 | No |
| `naive_scaling` | BF16 compiled brute-force scaling | 24-in-96 | No |
| `naive_quant` | Direct NVFP4 compiled rollout | 24-in-96 | Yes |
| `sol_rl` | Two-stage decoupled rollout | 24-in-96 | Yes |

In this repository:

- `diffusionnft`: `preview_model="peft"`, `fullrollout_model="peft"`
- `naive_scaling`: `fullrollout_model="compile"`
- `naive_quant`: `fullrollout_model="compile_nvfp4"`
- `sol_rl`: `preview_step=6`, `preview_model="compile_nvfp4"`, `fullrollout_model="compile"`

## Reward Models

Current online reward suffixes:

- `pickscore`
- `clipscore`
- `hpsv2`
- `imagereward`

### Manual Reward Checkpoints

`HPSv2` expects local files under `reward_ckpts/`:

```bash
mkdir -p reward_ckpts
cd reward_ckpts

wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt

cd ..
```

### Auto-Downloaded Reward Models

The other reward models are downloaded automatically on first use:

- `clipscore`: `openai/clip-vit-large-patch14`
- `pickscore`: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` and `yuvalkirstain/PickScore_v1`
- `imagereward`: `ImageReward-v1.0`

## Datasets

Bundled prompt datasets live under:

```text
diffusion/post_training/dataset/
```

Included datasets:

- `pickscore`
- `pickscore_sfw`
- `ocr`
- `drawbench`
- `geneval`
- `geneval_unseen_objects`
- `counting_edit`

Reader behavior:

- `general_ocr` expects `train.txt` and `test.txt`
- `geneval` expects `train_metadata.jsonl` and `test_metadata.jsonl`

Default preset behavior:

- All current preset reward configs are built with `dataset="pickscore"`
- Reward suffix does not automatically switch dataset

If `dataset/<name>` is missing, the loader falls back to `diffusion/post_training/dataset/<name>`.

## Links

- [Sol-RL homepage](https://nvlabs.github.io/Sana/Sol-RL/)
- [Sol-RL paper](https://arxiv.org/abs/2604.06916)
- [Repository guide](https://github.com/NVlabs/Sana/blob/main/train_scripts/sol_rl/README.md)
- [Sana documentation home](https://nvlabs.github.io/Sana/docs/)
