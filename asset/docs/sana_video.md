<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Video/logo.svg" width="70%" alt="SANA-Sprint Logo"/>
</p>

# 🎬 SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Video"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2509.24695"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_video"><img src="https://img.shields.io/static/v1?label=diffusers&message=SANAVideoPipeline&color=yellow"></a> &ensp;
</div>

## 🎬 Demos of SANA-Video

<div align="center">
  <a href="https://www.youtube.com/watch?v=ztdkfIMkdJ4" target="_blank">
    <img src="https://img.youtube.com/vi/ztdkfIMkdJ4/0.jpg" alt="Demo Video of SANA-Sprint" style="width: 32.5%; margin: 0 auto; display: inline-block">
  </a>
  <a href="https://www.youtube.com/watch?v=7eNfDzA4yBs" target="_blank">
    <img src="https://img.youtube.com/vi/7eNfDzA4yBs/0.jpg" alt="Demo Video of SANA-Sprint" style="width: 32.5%; margin: 0 auto; display: inline-block">
  </a>
  <a href="https://www.youtube.com/watch?v=A9PnJ0y1DHY" target="_blank">
    <img src="https://img.youtube.com/vi/A9PnJ0y1DHY/0.jpg" alt="Demo Video of SANA-Video speed" style="width: 32.5%; margin: 0 auto; display: inline-block">
  </a>
</div>

## 📽️ About SANA-Video

**SANA-Video** is a small diffusion model designed for **efficient video generation**, capable of synthesizing high-resolution videos (up to $720 \\times 1280$) and **minute-length duration** with strong text-video alignment, while maintaining a remarkably fast speed.It enables low-cost, high-quality video generation and can be deployed efficiently on consumer GPUs like the RTX 5090.

SANA-Video's Core Contributions:

- **Efficient Architecture (Linear DiT)**: Leverages **linear attention** as the core operation, which is significantly more efficient than vanilla attention for video generation due to the large number of tokens processed.
- **Long-Sequence Capability (Constant-Memory KV Cache)**: Introduces a **Constant-Memory KV cache for Block Linear Attention**. This block-wise autoregressive approach uses a fixed-memory state derived from the cumulative properties of linear attention, which eliminates the need for a traditional KV cache, enabling **efficient minute-long video generation**.
- **Low Training Cost**: Achieved effective data filters and model training strategies, narrowing the training cost to only **12 days on 64 H100 GPUs**, which is just **1%** of the cost of MovieGen.
- **State-of-the-Art Speed and Performance**: Achieves competitive performance compared to modern SOTA small diffusion models (e.g., Wan 2.1-1.3B) while being **$16\\times$ faster** in measured latency。Deployment Acceleration: Can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (**$2.4\\times$ speedup**).

In summary, SANA-Video enables high-quality video synthesis at an unmatched speed and low operational cost.

## 💻 Block Causal Linear Attention && Causal Mix-FFN Mechanism

<div align="center">
  <a href="https://www.youtube.com/watch?v=-vuCn_d9Qjk" target="_blank">
    <img src="https://img.youtube.com/vi/-vuCn_d9Qjk/0.jpg" alt="Demo Video of SANA-Sprint" style="width: 49%; margin: 0 auto; display: inline-block">
  </a>
  <a href="https://www.youtube.com/watch?v=r347mG1rKqk" target="_blank">
    <img src="https://img.youtube.com/vi/r347mG1rKqk/0.jpg" alt="Demo Video of SANA-Sprint" style="width: 49%; margin: 0 auto; display: inline-block">
  </a>
</div>

## 🏃 How to Inference

### 1. How to use `SanaVideoPipeline` with `🧨diffusers`

> \[!IMPORTANT\]
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
motion_score = 30

prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_prompt = f" motion score: {motion_score}."
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
# Text to Video
bash inference_video_scripts/inference_sana_video.sh \
      --np 1 \
      --config configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
      --model_path hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
      --txt_file=asset/samples/video_prompts_samples.txt \
      --cfg_scale 6 \
      --motion_score 30 \
      --flow_shift 8 \
      --work_dir output/sana_t2v_video_results

# Text-Image to Video
bash inference_video_scripts/inference_sana_video.sh \
      --np 1 \
      --config configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
      --model_path hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
      --txt_file=asset/samples/sample_i2v.txt \
      --task=ltx \
      --cfg_scale 6 \
      --motion_score 30 \
      --flow_shift 8 \
      --work_dir output/sana_ti2v_video_results
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
