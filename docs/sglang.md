<p align="center" style="border-radius: 10px">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="40%" alt="Sana Logo"/>
</p>

# 🚀 SGLang: High-Performance Serving for SANA

<div align="center">
  <a href="https://github.com/sgl-project/sglang"><img src="https://img.shields.io/static/v1?label=Project&message=SGLang&color=blue&logo=github"></a> &ensp;
  <a href="https://github.com/NVlabs/Sana"><img src="https://img.shields.io/static/v1?label=Project&message=Sana&color=blue&logo=github-pages"></a> &ensp;
</div>

[SGLang](https://github.com/sgl-project/sglang) is an inference framework for accelerated image/video generation. SANA models are natively supported in SGLang, providing high-performance serving with OpenAI-compatible API, CLI, and Python SDK.

## Supported Models

| Model | HuggingFace ID |
|-------|----------------|
| Sana 0.6B (512px) | `Efficient-Large-Model/Sana_600M_512px_diffusers` |
| Sana 0.6B (1024px) | `Efficient-Large-Model/Sana_600M_1024px_diffusers` |
| Sana 1.6B (512px) | `Efficient-Large-Model/Sana_1600M_512px_diffusers` |
| Sana 1.6B (1024px) | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` |
| SANA-1.5 1.6B (1024px) | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` |
| SANA-1.5 4.8B (1024px) | `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers` |

______________________________________________________________________

## Installation

```bash
uv pip install 'sglang[diffusion]' --prerelease=allow
```

For more installation methods (e.g. Docker, ROCm/AMD), check the [SGLang installation guide](https://github.com/sgl-project/sglang/tree/main/docs/diffusion/installation.md).

______________________________________________________________________

## Quick Start

### 1. CLI

The simplest way to generate an image:

```bash
sglang generate \
    --model-path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --prompt 'a cyberpunk cat with a neon sign that says "Sana"' \
    --save-output
```

### 2. Python SDK

```python
from sglang.multimodal_gen import DiffGenerator

generator = DiffGenerator.from_pretrained(
    model_path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    num_gpus=1,
)

image = generator.generate(
    sampling_params_kwargs=dict(
        prompt='a cyberpunk cat with a neon sign that says "Sana"',
        height=1024,
        width=1024,
        num_inference_steps=20,
        guidance_scale=4.5,
        seed=42,
        save_output=True,
        output_path="outputs/",
    )
)
```

______________________________________________________________________

## Server Mode (OpenAI-Compatible API)

### Launch the Server

```bash
sglang serve --model-path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --host 0.0.0.0 --port 30000
```

### Send Requests

Once the server is running, use the OpenAI-compatible image generation API:

```python
import requests

response = requests.post(
    "http://127.0.0.1:30000/v1/images/generations",
    json={
        "prompt": 'a cyberpunk cat with a neon sign that says "Sana"',
        "size": "1024x1024",
        "num_inference_steps": 20,
        "guidance_scale": 4.5,
        "seed": 42,
        "response_format": "b64_json",
        "n": 1,
    },
)

result = response.json()
```

______________________________________________________________________

## Memory Optimization

For GPUs with limited VRAM, SGLang provides CPU offloading options:

```bash
sglang generate \
    --model-path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --pin-cpu-memory \
    --prompt "A beautiful landscape" \
    --save-output
```

| Option | Description |
|--------|-------------|
| `--dit-cpu-offload` | Offload DiT model to CPU |
| `--text-encoder-cpu-offload` | Offload text encoder to CPU |
| `--vae-cpu-offload` | Offload VAE to CPU |
| `--pin-cpu-memory` | Pin CPU memory for faster transfer |

______________________________________________________________________

## LoRA Support

Apply LoRA adapters during inference:

```bash
sglang generate \
    --model-path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --lora-path <your-lora-path> \
    --prompt "A beautiful landscape" \
    --save-output
```

______________________________________________________________________

## Related

- [SGLang Diffusion Documentation](https://github.com/sgl-project/sglang/tree/main/docs/diffusion)
- [Model Zoo](model_zoo.md) - All available Sana models
- [SANA Inference & Training](sana.md) - Native inference pipeline

______________________________________________________________________

## Citation

```bibtex
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
