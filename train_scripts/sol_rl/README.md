<p align="center" style="border-radius: 10px">
  <img src="../../asset/sol_rl_logo.png" width="56%" alt="Sol-RL logo"/>
</p>

<h2 align="center">⚡ Sol-RL Training Guide</h2>

<h4 align="center">
<a href="https://nvlabs.github.io/Sana/Sol-RL/"><b>🏠 Homepage</b></a> |
<a href="https://nvlabs.github.io/Sana/docs/"><b>📚 Docs</b></a> |
<a href="https://arxiv.org/abs/2604.06916"><b>📄 arXiv</b></a> |
<a href="https://github.com/NVlabs/DiffusionNFT"><b>🧩 DiffusionNFT Reference</b></a>
</h4>

<p align="center">
  <a href="https://nvlabs.github.io/Sana/Sol-RL/"><img src="https://img.shields.io/static/v1?label=Convergence&message=4.64x%20faster&color=brightgreen"></a> &ensp;
  <a href="https://nvlabs.github.io/Sana/Sol-RL/"><img src="https://img.shields.io/static/v1?label=Precision&message=~1%20gap%20vs%20BF16&color=blue"></a> &ensp;
  <a href="https://nvlabs.github.io/Sana/Sol-RL/"><img src="https://img.shields.io/static/v1?label=Validated&message=3x4%20models%20x%20rewards&color=orange"></a>
</p>

<p align="center" border-radius="10px">
  <img src="../../asset/sol_rl_teaser.png" width="100%" alt="Sol-RL teaser"/>
</p>

**Sol-RL** (Speed-of-light RL) is a two-stage diffusion RL framework that uses **FP4 / NVFP4 for exploration** and **BF16 for regeneration and optimization**. This directory contains the single-node multi-GPU launchers and training entrypoints needed to run that workflow on top of **SANA**, **SD3**, and **FLUX.1**.

## ✨ About Sol-RL

Inspired by the official [Sol-RL homepage](https://nvlabs.github.io/Sana/Sol-RL/) and [paper](https://arxiv.org/abs/2604.06916), this folder is organized around one simple idea:

- use cheap FP4 / NVFP4 rollouts to explore a much larger candidate pool
- keep BF16 regeneration only for the most contrastive samples that actually drive the update
- share the same post-training interface across SANA, SD3, and FLUX.1

## 🧩 Scope

- Launchers: `train_scripts/sol_rl/run_sana_single_node_8gpu.sh`, `train_scripts/sol_rl/run_sd3_single_node_8gpu.sh`, `train_scripts/sol_rl/run_flux1_single_node_8gpu.sh`
- Training entrypoints: `train_scripts/sol_rl/train_sana.py`, `train_scripts/sol_rl/train_sd3.py`, `train_scripts/sol_rl/train_flux1.py`
- Shared helpers: `train_scripts/sol_rl/train_utils.py`
- Configs: `configs/sol_rl/base.py`, `configs/sol_rl/sana.py`, `configs/sol_rl/sd3.py`, `configs/sol_rl/flux1.py`

## 🏗️ Model Stacks

| Model | Base model source | Extra native config | Notes |
|---|---|---|---|
| SANA | diffusers text encoder + VAE, native Sana transformer | Yes | Uses the new LinearFFN native architecture and an extra native checkpoint |
| SD3 | `stabilityai/stable-diffusion-3.5-large` | No | Pure diffusers pipeline |
| FLUX.1 | `black-forest-labs/FLUX.1-dev` | No | Pure diffusers pipeline |

## 🧠 SANA New Architecture

The SANA branch in this folder is not using the original native model directly.

- Native model class: `SanaMSLinearFFN_1600M_P1_D20`
- Native FFN type: `glumbconv_linear`
- Native YAML: `configs/sol_rl/Sana1.0_1600M_linear.yaml`
- Default native checkpoint cache: `output/pretrained_models/SANA_LinearFFN.pth`
- Default download source: [yitongl/SANA_LinearFFN](https://huggingface.co/yitongl/SANA_LinearFFN/tree/main)

`run_sana_single_node_8gpu.sh` will automatically check whether `output/pretrained_models/SANA_LinearFFN.pth` exists. If it does not exist, the launcher resolves the Hugging Face checkpoint and places it at that path before starting `torchrun`.

## ⚙️ Environment Tiers

### Basic Environment

Use this first. This tier does not require `transformer-engine`, and it does not use NVFP4 quantization.

Recommended usage:

- `*_diffusionnft_*`
- `*_naive_scaling_*`

Not recommended in this tier:

- `*_naive_quant_*`
- `*_sol_rl_*`

Suggested setup:

```bash
conda create -n sana-sol-rl python=3.10 -y
conda activate sana-sol-rl

./environment_setup.sh
```

Equivalent manual setup:

```bash
pip install -U pip
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install flash-attn==2.8.2 --no-build-isolation
```

Notes:

- `transformer-engine` is optional in the codebase now. `train_utils.py` only requires it when NVFP4 is actually requested.
- If your local `xformers` build is importable but broken at runtime, SANA can still run with `DISABLE_XFORMERS=1`. The SANA launcher already exports this by default.
- SD3 and FLUX.1 do not need the extra SANA native YAML/checkpoint path.

### Advanced Environment

Use this only if you want the NVFP4 path.

Required config families:

- `*_naive_quant_*`
- `*_sol_rl_*`

Extra dependency:

```bash
python -m pip install --no-build-isolation "transformer-engine[pytorch]"
```

If the build fails with missing `nccl.h`, install NCCL development headers in the same environment:

```bash
conda install -c nvidia nccl nccl-devel
```

Important notes:

- Always install `transformer-engine` with the same `python` that will be used by `torchrun`.
- This path is for newer NVIDIA GPU/toolchain combinations. If TE is not healthy on your node, stay on the basic configs.
- In code, TE/NVFP4 is only touched when `preview_model` or `fullrollout_model` becomes `compile_nvfp4`, or when linear layers are replaced by TE wrappers.

## 🚀 Quick Start

Default launchers:

```bash
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

Each launcher is single-node and multi-GPU by default:

- launcher: `torchrun`
- topology: single node
- default GPU count: `8`
- env knobs: `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `MASTER_PORT`

Examples:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash train_scripts/sol_rl/run_sd3_single_node_8gpu.sh
```

```bash
CONFIG_SPEC=configs/sol_rl/flux1.py:flux1_naive_scaling_hpsv2 \
bash train_scripts/sol_rl/run_flux1_single_node_8gpu.sh
```

```bash
CONFIG_SPEC=configs/sol_rl/sana.py:sana_diffusionnft_pickscore \
SANA_NATIVE_MODEL_PATH=/path/to/SANA_LinearFFN.pth \
bash train_scripts/sol_rl/run_sana_single_node_8gpu.sh
```

Direct `torchrun` usage:

```bash
torchrun --standalone --nproc_per_node=8 --master_port=29501 \
  train_scripts/sol_rl/train_sd3.py \
  --config=configs/sol_rl/sd3.py:sd3_diffusionnft_pickscore
```

```bash
torchrun --standalone --nproc_per_node=8 --master_port=29501 \
  train_scripts/sol_rl/train_flux1.py \
  --config=configs/sol_rl/flux1.py:flux1_diffusionnft_pickscore
```

```bash
DISABLE_XFORMERS=1 \
torchrun --standalone --nproc_per_node=8 --master_port=29501 \
  train_scripts/sol_rl/train_sana.py \
  --config=configs/sol_rl/sana.py:sana_diffusionnft_pickscore \
  --native_config=configs/sol_rl/Sana1.0_1600M_linear.yaml
```

## 🧭 Config Naming Rule

There are 48 runnable preset configs in total:

- 3 model families
- 4 rollout families
- 4 reward suffixes

Pattern:

```text
<model>_<family>_<reward>
```

Examples:

- `sana_diffusionnft_pickscore`
- `sd3_naive_scaling_hpsv2`
- `flux1_sol_rl_imagereward`

## 🗂️ Config Families

| Family | Meaning | Rollout shape | TE / NVFP4 needed | Typical use |
|---|---|---|---|---|
| `diffusionnft` | PEFT-only baseline | 24-in-24 | No | Safest baseline, best first run |
| `naive_scaling` | BF16 compiled brute-force scaling | 24-in-96 | No | Stronger rollout budget without NVFP4 |
| `naive_quant` | Direct NVFP4 compiled rollout | 24-in-96 | Yes | Quantized advanced path |
| `sol_rl` | Two-stage decoupled rollout | 24-in-96 | Yes | Preview with NVFP4 draft model, regenerate with BF16 compiled model |

In this repo:

- `diffusionnft`: `preview_model="peft"`, `fullrollout_model="peft"`
- `naive_scaling`: `fullrollout_model="compile"`
- `naive_quant`: `fullrollout_model="compile_nvfp4"`
- `sol_rl`: `preview_step=6`, `preview_model="compile_nvfp4"`, `fullrollout_model="compile"`

## 🏅 Reward Suffixes

The suffix controls which reward scorer is active inside `config.reward_fn`.

| Suffix | Reward |
|---|---|
| `pickscore` | PickScore |
| `clipscore` | CLIP score |
| `hpsv2` | HPSv2 |
| `imagereward` | ImageReward |

Reward implementations live in `diffusion/post_training/rewards.py`.

## 🧪 Reward Model Setup

This section follows the same spirit as [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), but is scoped to the reward stack actually wired into this Sol-RL folder.

### Current Online Rewards

The current `configs/sol_rl/*.py` presets only use these online reward models:

- `pickscore`
- `clipscore`
- `hpsv2`
- `imagereward`

Important distinction:

- `geneval` and `ocr` exist in this repo as datasets or evaluation utilities
- they are not current reward suffixes in `configs/sol_rl/sana.py`, `configs/sol_rl/sd3.py`, or `configs/sol_rl/flux1.py`

### Python Packages

For the four current online rewards, the base environment already installs most Python dependencies through `pip install -e .`.

Relevant packages already included in `pyproject.toml`:

- `image-reward`
- `hpsv2`
- `open_clip_torch`
- `transformers`
- `clip`

So for normal Sol-RL training, the main extra setup is downloading the HPSv2 checkpoint files into the expected local folder.

### Manual Checkpoint Downloads

`diffusion/post_training/rewards.py` expects local reward checkpoints under:

```text
reward_ckpts/
```

Create that folder and download the required HPSv2 files:

```bash
mkdir -p reward_ckpts
cd reward_ckpts

# HPSv2 backbone
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin

# HPSv2 scoring head
wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt

cd ..
```

These two files are loaded directly by `diffusion/post_training/rewards.py`.

### Auto-Downloaded Reward Models

The other reward models are downloaded automatically on first use:

| Reward | Download behavior | Source | Cache location |
|---|---|---|---|
| `clipscore` | auto | `openai/clip-vit-large-patch14` | Hugging Face cache |
| `pickscore` | auto | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` and `yuvalkirstain/PickScore_v1` | Hugging Face cache |
| `imagereward` | auto | `ImageReward-v1.0` | `${HF_HOME:-~/.cache}/ImageReward` |
| `hpsv2` | manual | files above | `reward_ckpts/` |

If you want to pre-download the auto-fetched models before launching a long run, you can warm them once:

```bash
python - <<'PY'
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")

import ImageReward as RM
RM.load("ImageReward-v1.0")
PY
```

### Optional Evaluation Extras

If you also want DiffusionNFT-style evaluation helpers such as GenEval or OCR, treat them as optional extras rather than required training-time rewards.

For pure Sol-RL training on the current preset configs, you can skip this subsection.

#### GenEval object detector weights

```bash
mkdir -p output/pretrained_models/geneval
bash tools/metrics/geneval/evaluation/download_models.sh output/pretrained_models/geneval
```

#### GenEval environment

```bash
pip install -U openmim
mim install mmengine

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout 1.x
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 2.x
pip install -e . -v
cd ..

pip install open-clip-torch clip-benchmark
```

#### OCR environment

```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```

## 📌 Important Config Fields

| Field | Meaning |
|---|---|
| `config.dataset` | Prompt dataset path |
| `config.prompt_fn` | Prompt reader type, either `general_ocr` or `geneval` |
| `config.reward_fn` | Active reward dictionary |
| `config.sample.num_image_per_prompt` | Number of rollout samples per prompt |
| `config.sample.best_of_n` | Number of candidates kept in best-of-n |
| `config.sample.rollout_batch_size` | Per-iteration rollout batch size |
| `config.sample.per_gpu_to_process_prompts` | Number of prompt groups each GPU handles |
| `config.train.batch_size` | Per-GPU training batch size |
| `config.train.gradient_accumulation_steps` | Accumulation steps after rollout |
| `config.preview_step` | Number of preview-stage steps before the second-stage rollout |
| `config.preview_model` | Preview model type: `peft`, `compile`, or `compile_nvfp4` |
| `config.fullrollout_model` | Full rollout model type: `peft`, `compile`, or `compile_nvfp4` |
| `config.compile_mode` | `torch.compile` mode for compiled inference models |
| `config.nvfp4_skip_modules` | Modules excluded from TE/NVFP4 replacement |
| `config.nvfp4_min_dim` | Minimum linear dimension before TE replacement is attempted |
| `config.resume` | Whether to resume automatically |
| `config.resume_from` | Resume path |
| `config.save_dir` | Output path |

SANA-only fields:

| Field | Meaning |
|---|---|
| `config.native_config` | Native Sana YAML path |
| `config.native_model_path` | Preferred local path for the LinearFFN checkpoint |
| `config.native_model_source` | Fallback HF source when the local file is missing |

## 📦 Datasets

Bundled prompt datasets now live under:

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

Current default behavior:

- All preset reward configs are built with `dataset="pickscore"` by default
- Reward suffix does not automatically switch dataset

Loader fallback behavior:

- Configs still point to `dataset/<name>` by convention
- If that path does not exist, `diffusion/post_training/prompt_dataset.py` falls back to `diffusion/post_training/dataset/<name>`

If you override to a JSONL-style dataset, also change `prompt_fn`:

```bash
torchrun --standalone --nproc_per_node=8 --master_port=29501 \
  train_scripts/sol_rl/train_sd3.py \
  --config=configs/sol_rl/sd3.py:sd3_diffusionnft_pickscore \
  --config.dataset=/abs/path/to/diffusion/post_training/dataset/geneval \
  --config.prompt_fn=geneval
```

## 💾 Output and Resume Semantics

By default, every preset config writes to:

```text
logs/nft_slurm/<config_name>
```

Each preset also sets:

- `config.run_name = <config_name>`
- `config.resume_from = logs/nft_slurm/<config_name>`
- `config.save_dir = logs/nft_slurm/<config_name>`
- `config.resume = True`

If you want a truly new run, do not only change `run_name`. Also change the output path fields.

Example:

```bash
torchrun --standalone --nproc_per_node=8 --master_port=29501 \
  train_scripts/sol_rl/train_flux1.py \
  --config=configs/sol_rl/flux1.py:flux1_diffusionnft_pickscore \
  --config.run_name=flux1_exp_a \
  --config.save_dir=logs/nft_slurm/flux1_exp_a \
  --config.resume_from=logs/nft_slurm/flux1_exp_a \
  --config.resume=False
```

## 🛠️ Common Issues

### `transformer-engine` import or build failure

Symptom:

- import errors
- empty meta-package error
- `nccl.h` missing during build

Meaning:

- Your environment is not ready for NVFP4

What to do:

- stay on `*_diffusionnft_*` or `*_naive_scaling_*`
- only install TE if you really need `*_naive_quant_*` or `*_sol_rl_*`

### `xformers` imports but fails at runtime

Symptom:

- `No operator found for memory_efficient_attention_forward`
- `xFormers wasn't build with CUDA support`

Meaning:

- The Python package exists, but its CUDA operator build does not match your local environment

What to do:

- for SANA, use `DISABLE_XFORMERS=1`
- `run_sana_single_node_8gpu.sh` already does this by default

### SANA native checkpoint missing

Symptom:

- missing `output/pretrained_models/SANA_LinearFFN.pth`

What to do:

- just run `run_sana_single_node_8gpu.sh`
- or set `SANA_NATIVE_MODEL_PATH` and `SANA_NATIVE_MODEL_SOURCE` yourself

## ✅ Recommended First Runs

If you only want a stable first run:

- `sana_diffusionnft_pickscore`
- `sd3_diffusionnft_pickscore`
- `flux1_diffusionnft_pickscore`

If the basic environment is already stable and you want more rollout budget without TE:

- `sana_naive_scaling_pickscore`
- `sd3_naive_scaling_pickscore`
- `flux1_naive_scaling_pickscore`

Only move to these after TE is healthy:

- `*_naive_quant_*`
- `*_sol_rl_*`
