<p align="center" style="border-radius: 10px">
  <img src="https://nvlabs.github.io/Sana/Video/logo.svg" width="70%" alt="SANA-Sprint Logo"/>
</p>

# 🎬 LongSANA

<div align="center">
  <a href="https://nvlabs.github.io/Sana/Video"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2509.24695"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Sana&color=red&logo=arxiv"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=diffusers&message=LongSANAPipeline&color=yellow"></a> &ensp;
</div>

## 📽️ About LongSANA

**LongSANA** is the specialized long-video variant of the SANA-Video framework. It is designed to create **minute-long, high-resolution (720p)** videos in real time.

LongSANA's Core Contributions:

- **Constant-Memory KV Cache**: LongSANA addresses the memory explosion typical of long-context generation by reformulating causal linear attention. Instead of storing a growing history of tokens (which scales linearly or quadratically), it maintains a **compact, fixed-size recurrent state** (comprising the cumulative sum of states and keys). This reduces the memory complexity to **$O(1)$** (constant), allowing the model to generate arbitrarily long videos without increasing GPU memory usage.

- **Block-Wise Autoregressive Training**: To effectively learn long-term temporal dependencies, the model employs a novel autoregressive training paradigm with **Monotonically Increasing SNR Sampler** and **Improved Self-Forcing**. 

- **Performance**: LongSANA generates 1-minute length video with 35 seconds, achieving 27 FPS generation speed.

## 🏃 How to Inference

### 1. How to use LongSANA Pipelines in `🧨diffusers`

The diffusers version will release soon.


### 2. Inference with TXT file

```bash
# Text to Video
accelerate launch --mixed_precision=bf16 \
    inference_video_scripts/inference_sana_video.py \
    --config=configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml \
    --model_path=hf://Efficient-Large-Model/SANA-Video_2B_480p_LongLive/checkpoints/SANA_Video_2B_480p_LongLive.pth \
    --work_dir=output/inference/longsana_480p \
    --txt_file=asset/samples/video_prompts_samples.txt \
    --dataset=samples --cfg_scale=1.0 --num_frames 961
```

## 💻 How to Train

### Data Preparation
Please follow Self-Forcing to download training prompts:
```bash
mkdir -p data/longsana
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir data/longsana
```

### Launch Training
LongSANA is trained in three stages: ODE Initialization, Self-Forcing Training and LongSANA Training. For ODE initialization, we directly provide the [ODE initialization checkpoint](https://huggingface.co/Efficient-Large-Model/LongSANA_2B_480p_ode). If you are interested in training this stage by yourself, you may follow the process described in the [CausVid](https://github.com/tianweiy/CausVid) repo to generate trajectories and train the model with: 
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train_video_scripts/train_longsana.py \
  --config_path configs/sana_video_config/longsana/480ms/ode.yaml \
  --wandb_name debug_480p_ode --logdir output/debug_480p_ode
```

Self-Forcing Training and LongSANA Training can be implemented with:
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train_video_scripts/train_longsana.py \
  --config_path configs/sana_video_config/longsana/480ms/self_forcing.yaml \
  --wandb_name debug_480p_self_forcing --logdir output/debug_480p_self_forcing


torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train_video_scripts/train_longsana.py \
  --config_path configs/sana_video_config/longsana/480ms/longsana.yaml \
  --wandb_name debug_480p_longsana --logdir output/debug_480p_longsana
```
