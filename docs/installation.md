# Installation

## Requirements

- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.5.1+cu12.4](https://pytorch.org/)

## Quick Install

```bash
git clone https://github.com/NVlabs/Sana.git
cd Sana

bash ./environment_setup.sh sana
# or you can install each components step by step following environment_setup.sh
```

## Hardware Requirements

| Model | VRAM Required |
|-------|---------------|
| Sana-0.6B | 9GB |
| Sana-1.6B | 12GB |
| 4-bit Quantized | < 8GB |

!!! Note
All the tests are done on A100 GPUs. Different GPU versions may vary.

## Diffusers Installation

To use Sana with `diffusers`, make sure to upgrade to the latest version:

```bash
pip install git+https://github.com/huggingface/diffusers
```

## Quick Start with Diffusers

```python
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
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

## Optional: Docker

```bash
# Build Docker image
docker build -t sana .

# Run inference with Docker
docker run --gpus all -it sana python scripts/inference.py
```

## Next Steps

- [Model Zoo](model_zoo.md) - Choose your model
- [SANA-Sprint](sana_sprint.md) - Fast inference mode with 1-4 steps generations
- [SANA-Video](sana_video.md) - Video Gen with Linear Attention and Linear Block KV-Cache
