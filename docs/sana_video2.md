# SANA-Video 2.0

SANA-Video 2.0 provides efficient text-to-video and text-image-to-video models
built with hybrid linear/softmax attention and Attention Residuals. This
release includes the model architecture, training and inference code, 480p
and 720p reference configs for the 5B variant, and a post-trained 5B checkpoint
for 720p, 8-second generation. The 14B config and checkpoint are not included
yet.

> **Release status:** Explore the [project page](https://nvlabs.github.io/Sana/Video2/),
> try the [4-step 5B 720p preview](https://huggingface.co/spaces/Efficient-Large-Model/sana-video2-5b-720p-demo),
> or download the [50-step](https://huggingface.co/Efficient-Large-Model/SANA-Video_2.0_5B_720p)
> and [4-step preview](https://huggingface.co/Efficient-Large-Model/SANA-Video_2.0_5B_720p_4step)
> checkpoints.
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
| SANA-Video 2.0 5B 4-step preview | 720p, 81 frames at 16 FPS | [SANA-Video_2.0_5B_720p_4step](https://huggingface.co/Efficient-Large-Model/SANA-Video_2.0_5B_720p_4step) | BF16 inference |
| SANA-Video 2.0 14B | 480p | Coming soon | BF16 |

The released 5B checkpoint was jointly post-trained for text-to-video (T2V)
and text-image-to-video (TI2V) generation with ReFL. It contains only the
merged model `state_dict`; optimizer, scheduler, and standalone LoRA state are
not included. The checkpoint preserves the source EMA tensor values and is
cast to BF16 by the inference entry point.

The 4-step research preview is a full-model DMD EMA export initialized from
the merged ReFL step-500 model. It is T2V-only and is not a LoRA adapter. The
release artifact contains model tensors only, without optimizer or scheduler
state.

## Code layout

- Model: `diffusion/model/nets/sana_video2.py`
- Transformer blocks: `diffusion/model/nets/sana_video2_blocks.py`
- Four-step sampler: `diffusion/scheduler/fastvideo_dmd_sampler.py`
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

### Verified 5B 720p release example

The following sample was generated from the public checkpoint with seed 4. The
encoded result is 1280 × 736, 193 frames, 24 FPS, and 8.04 seconds long.

[Try the SANA-Video 2.0 5B 720p 4-step preview online](https://huggingface.co/spaces/Efficient-Large-Model/sana-video2-5b-720p-demo), or reproduce the original 50-step sample below with its exact command.

<video controls muted loop playsinline poster="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Video2/assets/release-demo/sana_video2_5b_720p_rooster_poster.png" style="width: 100%;">
  <source src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Video2/assets/release-demo/sana_video2_5b_720p_rooster.mp4" type="video/mp4">
  <a href="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Video2/assets/release-demo/sana_video2_5b_720p_rooster.mp4">Open the generated video.</a>
</video>

[Open or download the generated MP4](https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Video2/assets/release-demo/sana_video2_5b_720p_rooster.mp4).

> **Prompt:** In a cozy, vintage room adorned with floral wallpaper, a cartoon
> rooster sits comfortably in a floral-patterned armchair, sipping from a bottle
> of beer. The rooster, with its vibrant red comb and wattle, displays a range of
> expressions—smiling, nodding, and opening its beak wide in a cheerful manner.
> The setting includes wooden furniture and another beer bottle on the table,
> adding to the relaxed atmosphere. The camera captures the rooster from a
> close-up angle, emphasizing its animated movements and lively demeanor.

### Text-to-video

This is the exact command used to generate the verified example above:

```bash
bash inference_video_scripts/inference_sana_video.sh \
  --np 1 \
  --config configs/sana_video2/SanaVideo2_5B_720p.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Video_2.0_5B_720p/checkpoints/SANA_Video_2.0_5B_720p.pth \
  --txt_file=asset/samples/sana_video2_5b_720p_demo.txt \
  --cfg_scale 8 \
  --flow_shift 12 \
  --step 50 \
  --fps 24 \
  --motion_score 20 \
  --seed 4 \
  --work_dir output/sana_video2_t2v_720p_demo
```

### Four-step text-to-video preview

The DMD preview uses four fixed stochastic stages, CFG 1, BF16 latent noise,
and the `sana_shift6_dpm` sigma profile. It generates 81 frames at 16 FPS
(about 5.06 seconds) and does not support first-frame conditioning. The
`flow_shift` argument is retained for CLI consistency but is not applied by
this sampler. Model construction stays at the source tower's 480 setting;
`custom_height_width` controls the actual 736×1280 latent and MP4 dimensions.

```bash
bash inference_video_scripts/inference_sana_video.sh \
  --np 1 \
  --config configs/sana_video2/SanaVideo2_5B_720p.yaml \
  --model_path hf://Efficient-Large-Model/SANA-Video_2.0_5B_720p_4step/checkpoints/SANA_Video_2.0_5B_720p_4step.pth \
  --txt_file=asset/samples/sana_video2_5b_720p_demo.txt \
  --task=t2v \
  --model.image_size=480 \
  --custom_height_width='[736,1280]' \
  --sampling_algo=fastvideo_dmd_4step \
  --generator_sigma_profile=sana_shift6_dpm \
  --cfg_scale=1.0 \
  --flow_shift=1.0 \
  --motion_score=0 \
  --negative_prompt=None \
  --num_frames=81 \
  --step=4 \
  --fps=16 \
  --seed=0 \
  --work_dir output/sana_video2_t2v_720p_4step_preview
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
`(num_frames - 1) % 8 == 0`.

## Training

The reference configs run video-only FSDP training with gradient checkpointing,
online Gemma 2 caption encoding, causal LTX 2.3 VAE encoding, and flow matching.

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  train_video_scripts/train_video_ivjoint_chunk.py \
  --config_path configs/sana_video2/SanaVideo2_5B_480p.yaml
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
