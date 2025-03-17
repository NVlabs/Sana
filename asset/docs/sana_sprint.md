<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Sprint/asset/SANA-Sprint.png" width="70%" alt="SANA-Sprint Logo"/>
</p>

# 🏃SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Sprint"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2503.09641"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>

<div align="center">
  <a href="https://www.youtube.com/watch?v=nI_Ohgf8eOU" target="_blank">
    <img src="https://img.youtube.com/vi/nI_Ohgf8eOU/0.jpg" alt="Demo Video of SANA-Sprint" style="width: 49%; display: block; margin: 0 auto;">
  </a>
  <a href="https://www.youtube.com/watch?v=OOZzkirgsAc" target="_blank">
    <img src="https://img.youtube.com/vi/OOZzkirgsAc/0.jpg" alt="Demo Video of SANA-Sprint" style="width: 49%; display: block; margin: 0 auto;">
  </a>
</div>

## How to Inference

### 1. How to use `SanaSprintPipeline` with `🧨diffusers`

> \[!IMPORTANT\]
> It is now under construction [PR](https://github.com/huggingface/diffusers/pull/11074)
>
> ```bash
> pip install git+https://github.com/huggingface/diffusers
> ```

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

### 2. How to use `SanaSprintPipeline`  in this repo

```python
import torch
from app.sana_sprint_pipeline import SanaSprintPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaSprintPipeline("configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/SANA_Sprint_1.6B_1024px/checkpoints/SANA_Sprint_1.6B_1024px.pth")

prompt = "a tiny astronaut hatching from an egg on the moon",

image = sana(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=2,
    generator=generator,
)
save_image(image, 'sana_sprint.png', nrow=1, normalize=True, value_range=(-1, 1))
```

## How to Train

Working on it

```bash
bash train_scripts/train_scm_ladd.sh \
      configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml
      --data.data_dir="[data/toy_data]" \
      --data.type=SanaWebDatasetMS \
      --model.multi_scale=true \
      --data.load_vae_feat=true \
      --train.train_batch_size=2
```
