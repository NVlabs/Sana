# SANA-Video 2.0

SANA-Video 2.0 provides efficient text-to-video models built with hybrid
linear/softmax attention and Attention Residuals. This release includes the
model architecture, training code, inference code, and fixed 480p reference
configs for the 5B and 14B variants.

> **Release status:** Code and configs are available. The 5B and 14B
> checkpoints are coming soon.

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
| SANA-Video 2.0 5B | 480p | Coming soon | BF16 |
| SANA-Video 2.0 14B | 480p | Coming soon | BF16 |

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

Until the public checkpoints are released, provide a local compatible
checkpoint through `--model_path`. The prompt file contains one prompt per
line.

```bash
accelerate launch --num_processes=1 \
  inference_video_scripts/inference_sana_video.py \
  --config=configs/sana_video2/SanaVideo2_5B_480p.yaml \
  --model_path=/path/to/sana_video2_5b.pth \
  --txt_file=asset/samples/video_prompts_samples.txt \
  --work_dir=output/sana_video2_samples \
  --dataset=sana_video2 \
  --motion_score=-1
```

Select the 14B YAML for a 14B checkpoint. Height and width must be divisible by
32, and frame counts must satisfy `(num_frames - 1) % 8 == 0`.

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

To resume, pass `--resume_from=<checkpoint>`. Set `model.load_from` when
initializing from a compatible model checkpoint; leave it as `null` to train
from scratch.
