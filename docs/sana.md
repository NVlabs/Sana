<p align="center" style="border-radius: 10px">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="50%" alt="Sana Logo"/>
</p>

# ⚡️ Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2410.10629"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href="https://hanlab.mit.edu/blog/infinite-context-length-with-global-but-constant-attention-memory"><img src="https://img.shields.io/static/v1?label=Blog&message=MIT&color=darkred&logo=github-pages"></a> &ensp;
  <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana"><img src="https://img.shields.io/static/v1?label=diffusers&message=SanaPipeline&color=yellow"></a> &ensp;
  <a href="https://nv-sana.mit.edu/"><img src="https://img.shields.io/static/v1?label=Demo&message=SANA&color=green"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;
</div>

This guide covers training and inference for Sana text-to-image models.

## Hardware Requirements

| Task | VRAM |
|------|------|
| Inference (0.6B) | 9GB |
| Inference (1.6B) | 12GB |
| Inference (4-bit) | < 8GB |
| Training | 32GB |

!!! Note
All tests are done on A100 GPUs. Different GPU versions may vary.

______________________________________________________________________

## Inference

### Using Diffusers (Recommended)

```bash
pip install git+https://github.com/huggingface/diffusers
```

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

### Using SanaPAGPipeline

```python
import torch
from diffusers import SanaPAGPipeline

pipe = SanaPAGPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
    pag_applied_layers="transformer_blocks.8",
)
pipe.to("cuda")

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

image = pipe(
    prompt='a cyberpunk cat with a neon sign that says "Sana"',
    guidance_scale=5.0,
    pag_scale=2.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]
image[0].save('sana.png')
```

### Using Native Pipeline

```python
import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth")

image = sana(
    prompt='a cyberpunk cat with a neon sign that says "Sana"',
    height=1024,
    width=1024,
    guidance_scale=4.5,
    pag_guidance_scale=1.0,
    num_inference_steps=20,
    generator=generator,
)
save_image(image, 'output/sana.png', nrow=1, normalize=True, value_range=(-1, 1))
```

### Gradio Demo

```bash
DEMO_PORT=15432 \
python app/app_sana.py \
    --share \
    --config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_1600M_1024px_BF16/checkpoints/Sana_1600M_1024px_BF16.pth \
    --image_size=1024
```

### Batch Inference

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

______________________________________________________________________

## Training

### Data Preparation

Prepare image-text pairs in the following format:

```
asset/example_data
├── AAA.txt
├── AAA.png
├── BCC.txt
├── BCC.png
├── CCC.txt
└── CCC.png
```

### Train from Scratch

```bash
# Train Sana 0.6B with 512x512 resolution
bash train_scripts/train.sh \
    configs/sana_config/512ms/Sana_600M_img512.yaml \
    --data.data_dir="[asset/example_data]" \
    --data.type=SanaImgDataset \
    --model.multi_scale=false \
    --train.train_batch_size=32
```

### Fine-tuning

```bash
# Fine-tune Sana 1.6B with 1024x1024 resolution
bash train_scripts/train.sh \
    configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    --data.data_dir="[asset/example_data]" \
    --data.type=SanaImgDataset \
    --model.load_from=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
    --model.multi_scale=false \
    --train.train_batch_size=8
```

### Multi-Scale WebDataset

Convert data to WebDataset format:

```bash
python tools/convert_scripts/convert_ImgDataset_to_WebDatasetMS_format.py
```

Then train:

```bash
bash train_scripts/train.sh \
    configs/sana_config/512ms/Sana_600M_img512.yaml \
    --data.data_dir="[asset/example_data_tar]" \
    --data.type=SanaWebDatasetMS \
    --model.multi_scale=true \
    --train.train_batch_size=32
```

### Training with FSDP

```bash
# Download toy dataset
huggingface-cli download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data

# DDP training
bash train_scripts/train.sh \
    configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml \
    --data.data_dir="[data/toy_data]" \
    --data.type=SanaWebDatasetMS \
    --model.multi_scale=true \
    --data.load_vae_feat=true \
    --train.train_batch_size=2

# FSDP training
bash train_scripts/train.sh \
    configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml \
    --data.data_dir="[data/toy_data]" \
    --data.type=SanaWebDatasetMS \
    --model.multi_scale=true \
    --data.load_vae_feat=true \
    --train.use_fsdp=true \
    --train.train_batch_size=2
```

______________________________________________________________________

## Related

- [Model Zoo](model_zoo.md) - All available models
- [4-bit Sana](4bit_sana.md) - Memory-efficient inference
- [LoRA & DreamBooth](sana_lora_dreambooth.md) - Fine-tuning methods

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

@misc{xie2025sana,
      title={SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer},
      author={Xie, Enze and Chen, Junsong and Zhao, Yuyang and Yu, Jincheng and Zhu, Ligeng and Lin, Yujun and Zhang, Zhekai and Li, Muyang and Chen, Junyu and Cai, Han and others},
      year={2025},
      eprint={2501.18427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.18427},
}
```
