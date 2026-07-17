# Quick Start

This page gives the shortest public path from installation to a generated
video. For detailed prompt and inference options, use the specialized guides:
[Prompt Preparation](prompt_preparation.md), [DiT Inference](dit_inference.md),
and [SGLang Diffusion](sglang_diffusion.md).

## 1. Install

```bash
git clone <repo_url>
cd <repo_name>

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

pip install -r requirements.txt
pip install -e .
```

The base `requirements.txt` already covers two default paths, with no extra install needed:

- diffusers inference (`scripts/inference.py --backend diffusers`);
- the prompt rewriter transformers backend (`rewriter/inference.py --backend transformers`).

### Optional acceleration dependencies

Install these only when you need the corresponding acceleration path.

> **💡 Rewriter deployment**: the bundled rewriter ships only the single-process
> `transformers` backend. For higher throughput, deploy the VLM yourself and call
> it through an OpenAI-compatible API. Keep step 1 on the base VLM without LoRA,
> and step 2 on the same base VLM with the rewriter LoRA enabled. Follow the
> official serving docs for adapter routing: [vLLM](https://docs.vllm.ai) /
> [SGLang](https://docs.sglang.ai).

SGLang Diffusion or FP8 MoE acceleration:

```bash
python -m pip install --no-deps -r requirements-sglang.txt
```

Recommended package versions:

| Package | Version |
| --- | --- |
| `torch` | `2.12.0.dev20260220+cu130` |
| `torchvision` | `0.26.0.dev20260220+cu130` |
| `transformers` | `5.8.1` |
| `diffusers` | `0.39.0` |
| `peft` | `0.19.1` |

## 2. Model Directory Configuration

Download the DiT weight package and the rewriter weights from
[Model Download](../../README.md#-model-download), then point the three
environment variables below at the corresponding directories. The rewriter
scripts read the weights from these environment variables automatically:

```bash
export MODEL_DIR="<path_to_lingbot-video-model>"          # DiT weight package root
export REWRITER_BASE_MODEL="<path_to_rewriter_base_vlm>"  # rewriter base VLM
export REWRITER_ADAPTER="<path_to_rewriter_lora>"         # rewriter LoRA
```

## 3. End-To-End T2V Example

First rewrite a plain prompt into the structured JSON prompt consumed by DiT
(see [Prompt Preparation](prompt_preparation.md) for details):

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode t2v \
  --prompt "<plain_user_prompt>" \
  --duration 5 \
  --output prompt.json
```

Then create a per-sample negative prompt:

```bash
python rewriter/auto_negative.py \
  --backend transformers \
  --mode t2v \
  --caption prompt.json \
  --output negative.json
```

Run base video generation through diffusers:

```bash
python scripts/inference.py \
  --backend diffusers \
  --model_dir "$MODEL_DIR" \
  --mode t2v \
  --prompt_json prompt.json \
  --negative_prompt_json negative.json \
  --output "outputs/base.mp4" \
  --height 480 \
  --width 832 \
  --fps 24 \
  --steps 40 \
  --guidance_scale 3 \
  --shift 3 \
  --transformer_dtype bf16 \
  --text_encoder_dtype bf16 \
  --vae_dtype fp32
```

The runner creates the directory given to `--output` automatically; there is no
need to `mkdir` it first.

If auto negative is skipped, remove `--negative_prompt_json negative.json` and
the runner uses the built-in default negative prompt for the selected mode. For
detailed configuration and refiner usage, see [DiT Inference](dit_inference.md).
