<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Video/logo.svg" width="70%" alt="SANA-Sprint Logo"/>
</p>

# 🎬 SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Video"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2509.24695"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/cover.png" alt="SANA-Video Cover" style="width: 90%; margin: 0 auto; display: inline-block">
</div>

## 🏃 How to Inference

### 1. How to use `SanaVideoPipeline` with `🧨diffusers`

> \[!IMPORTANT\]
> It is now under construction [PR](https://github.com/huggingface/diffusers/pull/12584)
>
> ```bash
> pip install git+https://github.com/huggingface/diffusers
> ```

```python
# test SANA-Video
import torch
from diffusers import SanaPipeline, SanaVideoPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
pipe = SanaVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")
model_score = 30

prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_prompt = f" motion score: {model_score}."
prompt = prompt + motion_prompt

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "sana_video.mp4", fps=16)
```

### 2. Inference with TXT file

```bash
bash inference_video_scripts/inference_sana_video.sh \
      --np 1 \
      --config configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
      --model_path hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
      --txt_file=asset/samples/video_prompts_samples.txt \
      --cfg_scale 6 \
      --motion_score 30 \
      --flow_shift 8 \
      --work_dir output/sana_video_results
```

## 💻 How to Train

```bash
# 5s Video Model Pre-Training
bash train_video_scripts/train_video_ivjoint.sh \
      configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
      --data.data_dir="[data/toy_data]" \
      --train.train_batch_size=1 \
      --work_dir=output/sana_video \
      --train.num_workers=10 \
      --train.visualize=true
```

## Convert pth to diffusers safetensor

```bash
python scripts/convert_sana_video_to_diffusers.py --dump_path output/SANA_Video_2B_480p_diffusers --save_full_pipeline
```

## Performance

### VBench Results - 480p Resolution

#### Text-to-Video

| Methods | Latency (s) | Speedup | #Params (B) | Total ↑ | Quality ↑ | Semantic / I2V ↑ |
|---------|-------------|---------|-------------|---------|-----------|------------------|
| Open-Sora-2.0 | 465 | 1.0× | 14 | 84.34 | 85.4 | 80.72 |
| Wan2.1-14B | 484 | 1.0× | 14 | 83.69 | 85.59 | 76.11 |
| Wan2.1-1.3B | 103 | 4.7× | 1.3 | 83.31 | 85.23 | 75.65 |
| **SANA-Video** | **60** | **8.0×** | **2** | **83.71** | **84.35** | **81.35** |

<details>
<summary>Click to expand full comparison table</summary>

| Methods | Latency (s) | Speedup | #Params (B) | Total ↑ | Quality ↑ | Semantic / I2V ↑ |
|---------|-------------|---------|-------------|---------|-----------|------------------|
| MAGI-1 | 435 | 1.1× | 4.5 | 79.18 | 82.04 | 67.74 |
| Step-Video | 246 | 2.0× | 30 | 81.83 | 84.46 | 71.28 |
| CogVideoX1.5 | 111 | 4.4× | 5 | 82.17 | 82.78 | 79.76 |
| SkyReels-V2 | 132 | 3.7× | 1.3 | 82.67 | 84.70 | 74.53 |
| Open-Sora-2.0 | 465 | 1.0× | 14 | 84.34 | 85.4 | 80.72 |
| Wan2.1-14B | 484 | 1.0× | 14 | 83.69 | 85.59 | 76.11 |
| Wan2.1-1.3B | 103 | 4.7× | 1.3 | 83.31 | 85.23 | 75.65 |
| **SANA-Video** | **60** | **8.0×** | **2** | **83.71** | **84.35** | **81.35** |

</details>

#### Image-to-Video

| Methods | Latency (s) | Speedup | #Params (B) | Total ↑ | Quality ↑ | Semantic / I2V ↑ |
|---------|-------------|---------|-------------|---------|-----------|------------------|
| MAGI-1 | 435 | 1.1× | 4.5 | 89.28 | 82.44 | 96.12 |
| Step-Video-TI2V | 246 | 2.0× | 30 | 88.36 | 81.22 | 95.50 |
| CogVideoX-5b-I2V | 111 | 4.4× | 5 | 86.70 | 78.61 | 94.79 |
| HunyuanVideo-I2V | 210 | 2.3× | 13 | 86.82 | 78.54 | 95.10 |
| Wan2.1-14B | 493 | 1.0× | 14 | 86.86 | 80.82 | 92.90 |
| **SANA-Video** | **60** | **8.2×** | **2** | **88.02** | **79.65** | **96.40** |

### VBench Results - 720p Resolution

| Models | Latency (s) | Total ↑ | Quality ↑ | Semantic ↑ |
|--------|-------------|---------|-----------|------------|
| Wan-2.1-14B | 1897 | 83.73 | 85.77 | 75.58 |
| Wan-2.1-1.3B | 400 | 83.38 | 85.67 | 74.22 |
| Wan-2.2-5B | 116 | 83.28 | 85.03 | 76.28 |
| **SANA-Video-2B** | **36** | **84.05** | **84.63** | **81.73** |

**Summary**: Compared with the current SOTA small video models, SANA's performance is very competitive and speed is much faster. SANA provides 83.71 VBench overall performance with only 2B model parameters, **16× acceleration** at 480p, and achieves 84.05 total score with only **36s latency** at 720p resolution.
