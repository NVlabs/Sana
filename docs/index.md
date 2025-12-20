<p align="center" style="border-radius: 10px">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="50%" alt="Sana Logo"/>
</p>

<h3 align="center"><b>⚡️ Efficient High-Resolution Image & Video Generation</b></h3>

<h4 align="center">ICLR 2025 Oral | ICML 2025 | ICCV 2025 Spotlight</h4>

<p align="center">
  <a href="https://nvlabs.github.io/Sana/"><img src="https://img.shields.io/static/v1?label=Project&message=Sana&color=blue&logo=github-pages" alt="Sana"></a>
    <a href="https://nvlabs.github.io/Sana/Sana-1.5/"><img src="https://img.shields.io/static/v1?label=Project&message=Sana&color=blue&logo=github-pages" alt="Sana"></a>
  <a href="https://nvlabs.github.io/Sana/Sprint/"><img src="https://img.shields.io/static/v1?label=Project&message=Sprint&color=blue&logo=github-pages" alt="Sprint"></a>
  <a href="https://nvlabs.github.io/Sana/Video/"><img src="https://img.shields.io/static/v1?label=Project&message=Video&color=blue&logo=github-pages" alt="Video"></a>
</p>

<p align="center">
  <a href="https://hanlab.mit.edu/blog/infinite-context-length-with-global-but-constant-attention-memory"><img src="https://img.shields.io/static/v1?label=Blog&message=MIT&color=darkred&logo=github-pages" alt="Blog"></a>
  <a href="https://replicate.com/chenxwh/sana"><img src="https://img.shields.io/static/v1?label=API:H100&message=Replicate&color=pink" alt="Replicate"></a>
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://nv-sana.mit.edu/"><img src="https://img.shields.io/static/v1?label=Demo:6x3090&message=SANA&color=green" alt="Demo"></a>
  <a href="https://nv-sana.mit.edu/4bit/"><img src="https://img.shields.io/static/v1?label=Demo:1x3090&message=4bit&color=green" alt="4bit"></a>
  <a href="https://nv-sana.mit.edu/ctrlnet/"><img src="https://img.shields.io/static/v1?label=Demo:1x3090&message=ControlNet&color=green" alt="ControlNet"></a>
  <a href="https://nv-sana.mit.edu/sprint/"><img src="https://img.shields.io/static/v1?label=Demo:1x3090&message=Sprint&color=green" alt="Sprint"></a>
  <a href="https://huggingface.co/spaces/Efficient-Large-Model/SanaSprint"><img src="https://img.shields.io/static/v1?label=HF Demo&message=Sprint&color=green" alt="HF Sprint"></a>
</p>

______________________________________________________________________

## Introduction

**SANA** is an efficiency-oriented codebase for high-resolution image and video generation, providing complete training and inference pipelines.

### Models

| Model | Description |
|-------|-------------|
| **Sana** | Efficient text-to-image generation with Linear DiT, up to 4K resolution |
| **Sana-1.5** | Training-time and inference-time compute scaling |
| **Sana-Sprint** | Few-step generation via sCM (Consistency Model) distillation |
| **Sana-Video** | Efficient video generation with Block Linear Attention |
| **LongSana** | Minute-length real-time video generation (with LongLive) |

### Key Techniques

- **Linear Attention**: Replace vanilla attention with linear attention for efficiency at high resolutions
- **DC-AE**: 32× image compression (vs. traditional 8×) to reduce latent tokens
- **Block Causal Linear Attention**: Efficient attention for video generation
- **Causal Mix-FFN**: Memory-efficient feedforward for long videos
- **Flow-DPM-Solver**: Reduce sampling steps with efficient training and sampling
- **sCM Distillation**: One/few-step generation with continuous-time consistency distillation

## Highlights

- 🚀 **20× smaller, 100× faster** than Flux-12B
- 🖼️ **Up to 4K resolution** image generation
- ⚡ **One-step inference** with Sana-Sprint
- 💻 **< 8GB VRAM** with 4-bit quantization
- 🎬 **Efficient video generation** with Sana-Video
- ⏱️ **27 FPS real-time** minute-length video with LongSana
- 📦 **Full training & inference codebase**

## Quick Start

```bash
git clone https://github.com/NVlabs/Sana.git
cd Sana
bash ./environment_setup.sh sana
```

```python
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
"Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("a cyberpunk cat").images[0]
image.save("sana.png")
```

## Links

<p align="center">
  <a href="https://arxiv.org/abs/2410.10629"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv" alt="Sana"></a>
  <a href="https://arxiv.org/abs/2501.18427"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana-1.5&color=red&logo=arxiv" alt="Sana-1.5"></a>
  <a href="https://arxiv.org/abs/2503.09641"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sprint&color=red&logo=arxiv" alt="Sprint"></a>
  <a href="https://arxiv.org/abs/2509.24695"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Video&color=red&logo=arxiv" alt="Video"></a>
</p>

<p align="center">
  <a href="https://hanlab.mit.edu/projects/sana/"><img src="https://img.shields.io/static/v1?label=MIT&message=SANA&color=darkred&logo=github-pages" alt="SANA"></a>
  <a href="https://hanlab.mit.edu/projects/sana-1-5"><img src="https://img.shields.io/static/v1?label=MIT&message=SANA-1.5&color=darkred&logo=github-pages" alt="SANA-1.5"></a>
  <a href="https://hanlab.mit.edu/projects/sana-sprint/"><img src="https://img.shields.io/static/v1?label=MIT&message=SANA-Sprint&color=darkred&logo=github-pages" alt="SANA-Sprint"></a>
  <a href="https://hanlab.mit.edu/projects/sana-video/"><img src="https://img.shields.io/static/v1?label=MIT&message=SANA-Video&color=darkred&logo=github-pages" alt="SANA-Video"></a>

</p>
