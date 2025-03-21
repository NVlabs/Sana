<p align="center" style="border-radius: 10px">
  <img src="asset/logo.png" width="35%" alt="logo"/>
</p>

# ⚡️Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer

### <div align="center"> ICLR 2025 Oral Presentation <div>

<div align="center">
  <a href="https://nvlabs.github.io/Sana/"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://hanlab.mit.edu/projects/sana/"><img src="https://img.shields.io/static/v1?label=Page&message=MIT&color=darkred&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2410.10629"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href="https://nv-sana.mit.edu/"><img src="https://img.shields.io/static/v1?label=Demo:6x3090&message=MIT&color=yellow"></a> &ensp;
  <a href="https://nv-sana.mit.edu/4bit/"><img src="https://img.shields.io/static/v1?label=Demo:1x3090&message=4bit&color=yellow"></a> &ensp;
  <a href="https://nv-sana.mit.edu/ctrlnet/"><img src="https://img.shields.io/static/v1?label=Demo:1x3090&message=ControlNet&color=yellow"></a> &ensp;
  <a href="https://replicate.com/chenxwh/sana"><img src="https://img.shields.io/static/v1?label=API:H100&message=Replicate&color=pink"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>

<p align="center" border-radius="10px">
  <img src="asset/Sana.jpg" width="90%" alt="teaser_page1"/>
</p>

## 💡 Introduction

We introduce Sana, a text-to-image framework that can efficiently generate images up to 4096 × 4096 resolution.
Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU.
Core designs include:

(1) [**DC-AE**](https://hanlab.mit.edu/projects/dc-ae): unlike traditional AEs, which compress images only 8×, we trained an AE that can compress images 32×, effectively reducing the number of latent tokens. \
(2) **Linear DiT**: we replace all vanilla attention in DiT with linear attention, which is more efficient at high resolutions without sacrificing quality. \
(3) **Decoder-only text encoder**: we replaced T5 with a modern decoder-only small LLM as the text encoder and designed complex human instruction with in-context learning to enhance the image-text alignment. \
(4) **Efficient training and sampling**: we propose **Flow-DPM-Solver** to reduce sampling steps, with efficient caption labeling and selection to accelerate convergence.

As a result, Sana-0.6B is very competitive with modern giant diffusion models (e.g. Flux-12B), being 20 times smaller and 100+ times faster in measured throughput. Moreover, Sana-0.6B can be deployed on a 16GB laptop GPU, taking less than 1 second to generate a 1024 × 1024 resolution image. Sana enables content creation at low cost.

<p align="center" border-raduis="10px">
  <img src="asset/model-incremental.jpg" width="90%" alt="teaser_page2"/>
</p>

## 🔥🔥 News

- (🔥 New) \[2025/3/21\] 🚀Sana + Inference Scaling is released. [\[Guidance\]](asset/docs/inference_scaling/inference_scaling.md)
- (🔥 New) \[2025/3/16\] 🔥**SANA-1.5 code & weights are released!** 🎉 Include: [DDP/FSDP](#3-train-with-tar-file) | [TAR file WebDataset](#3-train-with-tar-file) | [Multi-Scale](#3-train-with-tar-file) Training code and [Weights](asset/docs/model_zoo.md) | [HF](https://huggingface.co/collections/Efficient-Large-Model/sana-15-67d6803867cb21c230b780e4) are all released.
- (🔥 New) \[2025/3/14\] 🏃SANA-Sprint is coming out!🎉 A new one/few-step generator of Sana. 0.1s per 1024px image on H100, 0.3s on RTX 4090. Find out more details: [\[Page\]](https://nvlabs.github.io/Sana/Sprint/) | [\[Arxiv\]](https://arxiv.org/abs/2503.09641). Code is coming very soon along with `diffusers`
- (🔥 New) \[2025/2/10\] 🚀Sana + ControlNet is released. [\[Guidance\]](asset/docs/sana_controlnet.md) | [\[Model\]](asset/docs/model_zoo.md) | [\[Demo\]](https://nv-sana.mit.edu/ctrlnet/)
- (🔥 New) \[2025/1/30\] Release CAME-8bit optimizer code. Saving more GPU memory during training. [\[How to config\]](https://github.com/NVlabs/Sana/blob/main/configs/sana_config/1024ms/Sana_1600M_img1024_CAME8bit.yaml#L86)
- (🔥 New) \[2025/1/29\] 🎉 🎉 🎉**SANA 1.5 is out! Figure out how to do efficient training & inference scaling!** 🚀[\[Tech Report\]](https://arxiv.org/abs/2501.18427)
- (🔥 New) \[2025/1/24\] 4bit-Sana is released, powered by [SVDQuant and Nunchaku](https://github.com/mit-han-lab/nunchaku) inference engine. Now run your Sana within **8GB** GPU VRAM [\[Guidance\]](asset/docs/4bit_sana.md) [\[Demo\]](https://svdquant.mit.edu/) [\[Model\]](asset/docs/model_zoo.md)
- (🔥 New) \[2025/1/24\] DCAE-1.1 is released, better reconstruction quality. [\[Model\]](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1) [\[diffusers\]](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers)
- (🔥 New) \[2025/1/23\] **Sana is accepted as Oral by ICLR-2025.** 🎉🎉🎉

<details>
  <summary>Click to show all updates</summary>

- (🔥 New) \[2025/1/12\] DC-AE tiling makes Sana-4K inferences 4096x4096px images within 22GB GPU memory. With model offload and 8bit/4bit quantize. The 4K Sana run within **8GB** GPU VRAM. [\[Guidance\]](asset/docs/model_zoo.md#-3-4k-models)
- (🔥 New) \[2025/1/11\] Sana code-base license changed to Apache 2.0.
- (🔥 New) \[2025/1/10\] Inference Sana with 8bit quantization.[\[Guidance\]](asset/docs/8bit_sana.md#quantization)
- (🔥 New) \[2025/1/8\] 4K resolution [Sana models](asset/docs/model_zoo.md) is supported in [Sana-ComfyUI](https://github.com/Efficient-Large-Model/ComfyUI_ExtraModels) and [work flow](asset/docs/ComfyUI/Sana_FlowEuler_4K.json) is also prepared. [\[4K guidance\]](asset/docs/ComfyUI/comfyui.md)
- (🔥 New) \[2025/1/8\] 1.6B 4K resolution [Sana models](asset/docs/model_zoo.md) are released: [\[BF16 pth\]](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16) or [\[BF16 diffusers\]](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers). 🚀 Get your 4096x4096 resolution images within 20 seconds! Find more samples in [Sana page](https://nvlabs.github.io/Sana/). Thanks [SUPIR](https://github.com/Fanghua-Yu/SUPIR) for their wonderful work and support.
- (🔥 New) \[2025/1/2\] Bug in the `diffusers` pipeline is solved. [Solved PR](https://github.com/huggingface/diffusers/pull/10431)
- (🔥 New) \[2025/1/2\] 2K resolution [Sana models](asset/docs/model_zoo.md) is supported in [Sana-ComfyUI](https://github.com/Efficient-Large-Model/ComfyUI_ExtraModels) and [work flow](asset/docs/ComfyUI/Sana_FlowEuler_2K.json) is also prepared.
- ✅ \[2024/12\] 1.6B 2K resolution [Sana models](asset/docs/model_zoo.md) are released: [\[BF16 pth\]](https://huggingface.co/Efficient-Large-Model/Sana_1600M_2Kpx_BF16) or [\[BF16 diffusers\]](https://huggingface.co/Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers). 🚀 Get your 2K resolution images within 4 seconds! Find more samples in [Sana page](https://nvlabs.github.io/Sana/). Thanks [SUPIR](https://github.com/Fanghua-Yu/SUPIR) for their wonderful work and support.
- ✅ \[2024/12\] `diffusers` supports Sana-LoRA fine-tuning! Sana-LoRA's training and convergence speed is super fast. [\[Guidance\]](asset/docs/sana_lora_dreambooth.md) or  [\[diffusers docs\]](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sana.md).
- ✅ \[2024/12\] `diffusers` has Sana! [All Sana models in diffusers safetensors](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e) are released and diffusers pipeline `SanaPipeline`, `SanaPAGPipeline`, `DPMSolverMultistepScheduler(with FlowMatching)` are all supported now. We prepare a [Model Card](asset/docs/model_zoo.md) for you to choose.
- ✅ \[2024/12\] 1.6B BF16 [Sana model](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16) is released for stable fine-tuning.
- ✅ \[2024/12\] We release the [ComfyUI node](https://github.com/Efficient-Large-Model/ComfyUI_ExtraModels) for Sana. [\[Guidance\]](asset/docs/ComfyUI/comfyui.md)
- ✅ \[2024/11\] All multi-linguistic (Emoji & Chinese & English) SFT models are released: [1.6B-512px](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_MultiLing), [1.6B-1024px](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_MultiLing), [600M-512px](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px), [600M-1024px](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px). The metric performance is shown [here](#performance)
- ✅ \[2024/11\] Sana Replicate API is launching at [Sana-API](https://replicate.com/chenxwh/sana).
- ✅ \[2024/11\] 1.6B [Sana models](https://huggingface.co/collections/Efficient-Large-Model/sana-673efba2a57ed99843f11f9e) are released.
- ✅ \[2024/11\] Training & Inference & Metrics code are released.
- ✅ \[2024/11\] Working on [`diffusers`](https://github.com/huggingface/diffusers/pull/9982).
- \[2024/10\] [Demo](https://nv-sana.mit.edu/) is released.
- \[2024/10\] [DC-AE Code](https://github.com/mit-han-lab/efficientvit/blob/master/applications/dc_ae/README.md) and [weights](https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b) are released!
- \[2024/10\] [Paper](https://arxiv.org/abs/2410.10629) is on Arxiv!

</details>

## Performance

| Methods (1024x1024)                                                                                 | Throughput (samples/s) | Latency (s) | Params (B) | Speedup | FID 👇      | CLIP 👆      | GenEval 👆  | DPG 👆      |
|-----------------------------------------------------------------------------------------------------|------------------------|-------------|------------|---------|-------------|--------------|-------------|-------------|
| FLUX-dev                                                                                            | 0.04                   | 23.0        | 12.0       | 1.0×    | 10.15       | 27.47        | _0.67_      | 84.0        |
| **Sana-0.6B**                                                                                       | 1.7                    | 0.9         | 0.6        | 39.5×   | _5.81_      | 28.36        | 0.64        | 83.6        |
| **[Sana-0.6B-MultiLing](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px)**            | 1.7                    | 0.9         | 0.6        | 39.5×   | **5.61**    | <u>28.80</u> | <u>0.68</u> | _84.2_      |
| **Sana-1.6B**                                                                                       | 1.0                    | 1.2         | 1.6        | 23.3×   | <u>5.76</u> | _28.67_      | 0.66        | **84.8**    |
| **[Sana-1.6B-MultiLing](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_MultiLing)** | 1.0                    | 1.2         | 1.6        | 23.3×   | 5.92        | **28.94**    | **0.69**    | <u>84.5</u> |

<details>
  <summary><h4>Click to show all performance</h4></summary>

| Methods                      | Throughput (samples/s) | Latency (s) | Params (B) | Speedup   | FID 👆      | CLIP 👆      | GenEval 👆  | DPG 👆      |
|------------------------------|------------------------|-------------|------------|-----------|-------------|--------------|-------------|-------------|
| _**512 × 512 resolution**_   |                        |             |            |           |             |              |             |             |
| PixArt-α                     | 1.5                    | 1.2         | 0.6        | 1.0×      | 6.14        | 27.55        | 0.48        | 71.6        |
| PixArt-Σ                     | 1.5                    | 1.2         | 0.6        | 1.0×      | _6.34_      | _27.62_      | <u>0.52</u> | _79.5_      |
| **Sana-0.6B**                | 6.7                    | 0.8         | 0.6        | 5.0×      | <u>5.67</u> | <u>27.92</u> | _0.64_      | <u>84.3</u> |
| **Sana-1.6B**                | 3.8                    | 0.6         | 1.6        | 2.5×      | **5.16**    | **28.19**    | **0.66**    | **85.5**    |
| _**1024 × 1024 resolution**_ |                        |             |            |           |             |              |             |             |
| LUMINA-Next                  | 0.12                   | 9.1         | 2.0        | 2.8×      | 7.58        | 26.84        | 0.46        | 74.6        |
| SDXL                         | 0.15                   | 6.5         | 2.6        | 3.5×      | 6.63        | _29.03_      | 0.55        | 74.7        |
| PlayGroundv2.5               | 0.21                   | 5.3         | 2.6        | 4.9×      | _6.09_      | **29.13**    | 0.56        | 75.5        |
| Hunyuan-DiT                  | 0.05                   | 18.2        | 1.5        | 1.2×      | 6.54        | 28.19        | 0.63        | 78.9        |
| PixArt-Σ                     | 0.4                    | 2.7         | 0.6        | 9.3×      | 6.15        | 28.26        | 0.54        | 80.5        |
| DALLE3                       | -                      | -           | -          | -         | -           | -            | _0.67_      | 83.5        |
| SD3-medium                   | 0.28                   | 4.4         | 2.0        | 6.5×      | 11.92       | 27.83        | 0.62        | <u>84.1</u> |
| FLUX-dev                     | 0.04                   | 23.0        | 12.0       | 1.0×      | 10.15       | 27.47        | _0.67_      | _84.0_      |
| FLUX-schnell                 | 0.5                    | 2.1         | 12.0       | 11.6×     | 7.94        | 28.14        | **0.71**    | **84.8**    |
| **Sana-0.6B**                | 1.7                    | 0.9         | 0.6        | **39.5×** | <u>5.81</u> | 28.36        | 0.64        | 83.6        |
| **Sana-1.6B**                | 1.0                    | 1.2         | 1.6        | **23.3×** | **5.76**    | <u>28.67</u> | <u>0.66</u> | **84.8**    |

</details>

## Contents

- [Env](#-1-dependencies-and-installation)
- [Demo](#-2-how-to-play-with-sana-inference)
- [Model Zoo](asset/docs/model_zoo.md)
- [Training](#-3-how-to-train-sana)
- [Testing](#-4-metric-toolkit)
- [TODO](#to-do-list)
- [Citation](#bibtex)

# 🔧 1. Dependencies and Installation

- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.1+cu12.1](https://pytorch.org/)

```bash
git clone https://github.com/NVlabs/Sana.git
cd Sana

./environment_setup.sh sana
# or you can install each components step by step following environment_setup.sh
```

# 💻 2. How to Play with Sana (Inference)

## 💰Hardware requirement

- 9GB VRAM is required for 0.6B model and 12GB VRAM for 1.6B model. Our later quantization version will require less than 8GB for inference.
- All the tests are done on A100 GPUs. Different GPU version may be different.

## 🔛 Choose your model: [Model card](asset/docs/model_zoo.md)

## 🔛 Quick start with [Gradio](https://www.gradio.app/guides/quickstart)

```bash
# official online demo
DEMO_PORT=15432 \
python app/app_sana.py \
    --share \
    --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
    --image_size=1024
```

### 1. How to use `SanaPipeline` with `🧨diffusers`

> \[!IMPORTANT\]
> Upgrade your `diffusers>=0.32.0.dev` to make the `SanaPipeline` and `SanaPAGPipeline` available!
>
> ```bash
> pip install git+https://github.com/huggingface/diffusers
> ```
>
> Make sure to specify `pipe.transformer` to default `torch_dtype` and `variant` according to [Model Card](asset/docs/model_zoo.md).
>
> Set `pipe.text_encoder` to BF16 and `pipe.vae` to FP32 or BF16. For more info, [docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana#sanapipeline) are here.

```python
# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

image[0].save("sana.png")
```

### 2. How to use `SanaPAGPipeline` with `🧨diffusers`

```python
# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPAGPipeline

pipe = SanaPAGPipeline.from_pretrained(
  "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
  variant="fp16",
  torch_dtype=torch.float16,
  pag_applied_layers="transformer_blocks.8",
)
pipe.to("cuda")

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
image = pipe(
    prompt=prompt,
    guidance_scale=5.0,
    pag_scale=2.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]
image[0].save('sana.png')
```

<details>
<summary><h3>3. How to use Sana in this repo</h3></summary>

```python
import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana_config/1024ms/Sana_1600M_img1024.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/Sana_1600M_1024px_BF16/checkpoints/Sana_1600M_1024px_BF16.pth")
prompt = 'a cyberpunk cat with a neon sign that says "Sana"'

image = sana(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
    generator=generator,
)
save_image(image, 'output/sana.png', nrow=1, normalize=True, value_range=(-1, 1))
```

</details>

<details>
<summary><h3>4. Run Sana (Inference) with Docker</h3></summary>

```
# Pull related models
huggingface-cli download google/gemma-2b-it
huggingface-cli download google/shieldgemma-2b
huggingface-cli download mit-han-lab/dc-ae-f32c32-sana-1.0
huggingface-cli download Efficient-Large-Model/Sana_1600M_1024px

# Run with docker
docker build . -t sana
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache:/root/.cache \
    sana
```

</details>

## 🔛 Run inference with TXT or JSON files

```bash
# Run samples in a txt file
python scripts/inference.py \
      --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
      --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
      --txt_file=asset/samples/samples_mini.txt

# Run samples in a json file
python scripts/inference.py \
      --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
      --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
      --json_file=asset/samples/samples_mini.json
```

where each line of [`asset/samples/samples_mini.txt`](asset/samples/samples_mini.txt) contains a prompt to generate

# 🔥 3. How to Train Sana

## 💰Hardware requirement

- 32GB VRAM is required for both 0.6B and 1.6B model's training

### 1). Train with image-text pairs in directory

We provide a training example here and you can also select your desired config file from [config files dir](configs/sana_config) based on your data structure.

To launch Sana training, you will first need to prepare data in the following formats. [Here](asset/example_data) is an example for the data structure for reference.

```bash
asset/example_data
├── AAA.txt
├── AAA.png
├── BCC.txt
├── BCC.png
├── ......
├── CCC.txt
└── CCC.png
```

Then Sana's training can be launched via

```bash
# Example of training Sana 0.6B with 512x512 resolution from scratch
bash train_scripts/train.sh \
  configs/sana_config/512ms/Sana_600M_img512.yaml \
  --data.data_dir="[asset/example_data]" \
  --data.type=SanaImgDataset \
  --model.multi_scale=false \
  --train.train_batch_size=32

# Example of fine-tuning Sana 1.6B with 1024x1024 resolution
bash train_scripts/train.sh \
  configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
  --data.data_dir="[asset/example_data]" \
  --data.type=SanaImgDataset \
  --model.load_from=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
  --model.multi_scale=false \
  --train.train_batch_size=8
```

### 2). Train with image-text pairs in directory

We also provide conversion scripts to convert your data to the required format. You can refer to the [data conversion scripts](asset/data_conversion_scripts) for more details.

```bash
python tools/convert_ImgDataset_to_WebDatasetMS_format.py
```

Then Sana's training can be launched via

```bash
# Example of training Sana 0.6B with 512x512 resolution from scratch
bash train_scripts/train.sh \
  configs/sana_config/512ms/Sana_600M_img512.yaml \
  --data.data_dir="[asset/example_data_tar]" \
  --data.type=SanaWebDatasetMS \
  --model.multi_scale=true \
  --train.train_batch_size=32
```

### 3). Train with TAR file

We prepared a toy TAR dataset containing 100 random images from Journey-DB, duplicated for testing purposes. Note that this dataset is not intended for training.

```bash
huggingface-cli download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data --local-dir-use-symlinks False
```

Then, you are ready to run with FSDP or DDP:

```bash
# DDP
# Example of training Sana 1.6B with 512x512 resolution from scratch
bash train_scripts/train.sh \
      configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml \
      --data.data_dir="[data/toy_data]" \
      --data.type=SanaWebDatasetMS \
      --model.multi_scale=true \
      --data.load_vae_feat=true \
      --train.train_batch_size=2
```

```bash
# FSDP
# Example of training Sana 1.6B with 512x512 resolution from scratch
bash train_scripts/train.sh \
      configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml \
      --data.data_dir="[data/toy_data]" \
      --data.type=SanaWebDatasetMS \
      --model.multi_scale=true \
      --data.load_vae_feat=true \
      --train.use_fsdp=true \
      --train.train_batch_size=2
```

# 💻 4. Metric toolkit

Refer to [Toolkit Manual](asset/docs/metrics_toolkit.md).

# 🚀 5. Inference Scaling

We trained a specialized [NVILA-2B](https://huggingface.co/Efficient-Large-Model/NVILA-Lite-2B-Verifier) model to score images, which we named VISA (VIla as SAna verifier). By selecting the top 4 images from 2,048 candidates, we enhanced the GenEval performance of SD1.5 and SANA-1.5-4.8B v2, increasing their scores from 42 to 87 and 81 to 96, respectively.

| Method                         | Overall | Single | Two  | Counting | Colors | Position | Color Attribution |
|--------------------------------|---------|--------|------|----------|--------|----------|------------------|
| SD1.5                          | 0.42    | 0.98   | 0.39 | 0.31     | 0.72   | 0.04     | 0.06             |
| **+ Inference Scaling**        | **0.87** | **1.00** | **0.97** | **0.93** | **0.96** | **0.75** | **0.62** |
| SANA-1.5 4.8B v2              | 0.81    | 0.99   | 0.86 | 0.86     | 0.84   | 0.59     | 0.65             |
| **+ Inference Scaling**        | **0.96** | **1.00** | **1.00** | **0.97** | **0.94** | **0.96** | **0.87** |

Details refer to [Inference Scaling Manual](asset/docs/inference_scaling/inference_scaling.md).

# 💪To-Do List

We will try our best to release

- \[✅\] Training code
- \[✅\] Inference code
- \[✅\] Model zoo
- \[✅\] ComfyUI
- \[✅\] DC-AE Diffusers
- \[✅\] Sana merged in Diffusers(https://github.com/huggingface/diffusers/pull/9982)
- \[✅\] LoRA training by [@paul](https://github.com/sayakpaul)(`diffusers`: https://github.com/huggingface/diffusers/pull/10234)
- \[✅\] 2K/4K resolution models.(Thanks [@SUPIR](https://github.com/Fanghua-Yu/SUPIR) to provide a 4K super-resolution model)
- \[✅\] 8bit / 4bit Laptop development
- \[✅\] ControlNet (train & inference & models)
- \[✅\] FSDP Training
- \[✅\] **SANA-1.5 (Larger model size / Inference Scaling)**
- \[💻\] Better re-construction F32/F64 VAEs.
- \[💻\] **SANA-Sprint: Few-step generator**

# 🤗Acknowledgements

**Thanks to the following open-sourced codebase for their wonderful work and codebase!**

- [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha)
- [PixArt-Σ](https://github.com/PixArt-alpha/PixArt-sigma)
- [Efficient-ViT](https://github.com/mit-han-lab/efficientvit)
- [ComfyUI_ExtraModels](https://github.com/city96/ComfyUI_ExtraModels)
- [SVDQuant and Nunchaku](https://github.com/mit-han-lab/nunchaku)
- [diffusers](https://github.com/huggingface/diffusers)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/Sana&type=Date)](https://star-history.com/#NVlabs/sana&Date)

# 📖BibTeX

```
@misc{xie2024sana,
      title={Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer},
      author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2410.10629},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10629},
    }
```
