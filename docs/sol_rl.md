<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Sol-RL/asset/sol-rl_logo.png" width="82%" alt="Sol-RL Logo"/>
</p>

# Sol-RL: FP4 Explore, BF16 Train for SANA, FLUX.1, and SD3.5-L

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Sol-RL/"><img src="https://img.shields.io/static/v1?label=Project&message=Homepage&color=blue&logo=github-pages" alt="Sol-RL Homepage"></a> &ensp;
  <a href="https://arxiv.org/abs/2604.06916"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sol-RL&color=red&logo=arxiv" alt="Sol-RL Arxiv"></a>
</div>

This guide covers Sol-RL post-training in `Sana`, including single-node launchers, config families, reward setup, and model-specific notes for **SANA**, **FLUX.1**, and **SD3.5-L**.

Base installation is shared with the rest of the repo and is documented in [Installation](installation.md).

If you want the NVFP4 path (`*_naive_quant_*` or `*_sol_rl_*`), also install `transformer-engine` with the same Python interpreter used by `torchrun`:

```bash
python -m pip install --no-build-isolation "transformer-engine[pytorch]"
```

## How to Train

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
CONFIG_SPEC=configs/sol_rl/sd3.py:sd3_compile_hpsv2 \
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
```

```bash
CONFIG_SPEC=configs/sol_rl/flux1.py:flux1_sol_rl_imagereward \
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

## Configuration Families

Config naming pattern:

```text
<model>_<family>_<reward>
```

Examples:

- `sana_diffusionnft_pickscore`
- `sd3_compile_hpsv2`
- `flux1_sol_rl_imagereward`

| Family | Meaning | Rollout shape | TE / NVFP4 needed |
|---|---|---|---|
| `diffusionnft` | PEFT-only baseline | 24-in-24 | No |
| `naive_scaling` | PEFT brute-force scaling | 24-in-96 | No |
| `compile` | BF16 compiled brute-force scaling | 24-in-96 | No |
| `naive_quant` | Direct NVFP4 compiled rollout | 24-in-96 | Yes |
| `sol_rl` | Two-stage decoupled rollout | 24-in-96 | Yes |

In this repository:

- `diffusionnft`: `preview_model="peft"`, `fullrollout_model="peft"`
- `naive_scaling`: `preview_model="peft"`, `fullrollout_model="peft"`
- `compile`: `fullrollout_model="compile"`
- `naive_quant`: `fullrollout_model="compile_nvfp4"`
- `sol_rl`: `preview_step=6`, `preview_model="compile_nvfp4"`, `fullrollout_model="compile"`

Recommended first runs:

- `sana_diffusionnft_pickscore`
- `sd3_diffusionnft_pickscore`
- `flux1_diffusionnft_pickscore`

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

## Acknowledgements

- Sol-RL training recipes in this repo draw on [Advantage Weighted Matching](https://github.com/scxue/advantage_weighted_matching) and [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT).
