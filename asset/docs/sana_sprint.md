<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Sprint/asset/SANA-Sprint.png" width="70%" alt="SANA-Sprint Logo"/>
</p>

# 🏃SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Sprint"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2503.09641"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>

[//]: # (<a href="https://hanlab.mit.edu/projects/sana/"><img src="https://img.shields.io/static/v1?label=Page&message=MIT&color=darkred&logo=github-pages"></a> &ensp;)
[//]: # (  <a href="https://nv-sana.mit.edu/"><img src="https://img.shields.io/static/v1?label=Demo:6x3090&message=MIT&color=yellow"></a> &ensp;)
[//]: # (  <a href="https://nv-sana.mit.edu/ctrlnet/"><img src="https://img.shields.io/static/v1?label=Demo:1x3090&message=ControlNet&color=yellow"></a> &ensp;)

<div style="text-align: center">
  <video width="80%" controls muted loop autoplay>
    <source src="https://nvlabs.github.io/Sana/Sprint/asset/video/sana-sprint-speed-video-4images.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

## How to Train
Working on
```bash
bash 
```

## How to Inference

### diffusers
Under Construction [PR](https://github.com/huggingface/diffusers/pull/11074)

```python
# test sana sprint
from diffusers import SanaSprintPipeline
import torch

device = "cuda:0"

repo = "Efficient-Large-Model/SANA_Sprint_1.6B_1024px_diffusers"

pipeline = SanaSprintPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
pipeline.to(device)

prompt = "a tiny astronaut hatching from an egg on the moon"

image = pipeline(prompt=prompt, num_inference_steps=2).images[0]
image.save("test_out.png")
```

### In this code base