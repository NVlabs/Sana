# SANA-Video 2.0

SANA-Video 2.0 provides efficient text-to-video and text-image-to-video models
built with hybrid linear/softmax attention and Attention Residuals. This
release includes the model architecture, training and inference code, 480p
reference configs for the 5B and 14B variants, and a post-trained 5B checkpoint
for 720p, 8-second generation.

> **Release status:** The 5B 720p checkpoint is available on
> [Hugging Face](https://huggingface.co/Efficient-Large-Model/SANA-Video_2.0_5B_720p).
> The 14B checkpoint is coming soon.

## Architecture

| Model | Layers | Hidden size | Linear attention | Softmax anchors | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SanaVideo2_5B` | 32 | 2,560 | 20 heads × 128 | 8 layers, 10 heads × 256 | 4,466,980,960 |
| `SanaVideo2_14B` | 40 | 4,096 | 32 heads × 128 | 10 layers, 16 heads × 256 | 14,246,716,224 |

Both variants use:

- 75% gated bidirectional linear-attention layers and 25% dense softmax anchor
  layers;
- SwiGLU feed-forward networks with a 4× expansion ratio;
- Wan 3D rotary position embeddings;
- shared, timestep-independent Attention Residual aggregation in blocks of
  eight layers;
- `(1, 1, 1)` latent patches; and
- the LTX 2.3 VAE contract: 128 latent channels with `(8, 32, 32)`
  temporal/spatial strides.

## Model zoo

| Model | Resolution | Checkpoint | Precision |
| --- | --- | --- | --- |
| SANA-Video 2.0 5B | 720p, 193 frames at 24 FPS | [SANA-Video_2.0_5B_720p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2.0_5B_720p) | BF16 inference |
| SANA-Video 2.0 14B | 480p | Coming soon | BF16 |

The released 5B checkpoint was jointly post-trained for text-to-video (T2V)
and text-image-to-video (TI2V) generation with ReFL. It contains only the
merged model `state_dict`; optimizer, scheduler, and standalone LoRA state are
not included. The checkpoint preserves the source EMA tensor values and is
cast to BF16 by the inference entry point.

## Code layout

- Model: `diffusion/model/nets/sana_video2.py`
- Transformer blocks: `diffusion/model/nets/sana_video2_blocks.py`
- Training: `train_video_scripts/train_video_ivjoint_chunk.py`
- Inference: `inference_video_scripts/inference_sana_video.py`
- Configs: `configs/sana_video2/`

SANA-Video 2.0 reuses the repository's existing video training and inference
entry points. No separate version-specific runner is required.

## Setup

Install the repository environment:

```bash
bash environment_setup.sh sana
conda activate sana
```

Place the Diffusers-format LTX 2.3 VAE under:

```text
output/pretrained_models/LTX-2.3-Diffusers/
```

Alternatively, update `vae.vae_pretrained` in the selected YAML. The example
configs use `data/video_toy_data`; replace `data.data_dir` with a
`SanaZipDataset`-compatible dataset for training.

## Inference

The prompt file contains one prompt per line. The released 720p model generates
193 frames at 24 FPS (about 8 seconds) in a 736×1280 bucket. It was evaluated
with classifier-free guidance 8, flow shift 12, and 50 sampling steps.

### Text-to-video

```bash
bash inference_video_scripts/inference_sana_video.sh \
  --np 1 \
  --config configs/sana_video2/SanaVideo2_5B_720p.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Video_2.0_5B_720p/checkpoints/SANA_Video_2.0_5B_720p.pth \
  --txt_file=asset/samples/video_prompts_samples.txt \
  --cfg_scale 8 \
  --flow_shift 12 \
  --step 50 \
  --fps 24 \
  --motion_score 20 \
  --work_dir output/sana_video2_t2v_720p
```

### Text-image-to-video

Each line in `asset/samples/sample_i2v.txt` contains a prompt and an input-image
path separated by `<image>`.

```bash
bash inference_video_scripts/inference_sana_video.sh \
  --np 1 \
  --config configs/sana_video2/SanaVideo2_5B_720p.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Video_2.0_5B_720p/checkpoints/SANA_Video_2.0_5B_720p.pth \
  --txt_file=asset/samples/sample_i2v.txt \
  --task=ltx \
  --cfg_scale 8 \
  --flow_shift 12 \
  --step 50 \
  --fps 24 \
  --motion_score 20 \
  --work_dir output/sana_video2_ti2v_720p
```

Height and width must be divisible by 32, and frame counts must satisfy
`(num_frames - 1) % 8 == 0`. Select the 14B YAML when a compatible 14B
checkpoint becomes available.

## Training

The reference configs run video-only FSDP training with gradient checkpointing,
online Gemma 2 caption encoding, causal LTX 2.3 VAE encoding, and flow matching.

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  train_video_scripts/train_video_ivjoint_chunk.py \
  --config_path configs/sana_video2/SanaVideo2_5B_480p.yaml
```

For 14B:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  train_video_scripts/train_video_ivjoint_chunk.py \
  --config_path configs/sana_video2/SanaVideo2_14B_480p.yaml
```

The released 720p config can also be used as the starting point for 5B
fine-tuning:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  train_video_scripts/train_video_ivjoint_chunk.py \
  --config_path configs/sana_video2/SanaVideo2_5B_720p.yaml
```

To resume, pass `--resume_from=<checkpoint>`. Set `model.load_from` when
initializing from a compatible model checkpoint; leave it as `null` to train
from scratch.
