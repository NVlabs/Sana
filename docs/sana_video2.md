# Sana-Video 2.0

This release contains the text-to-video training, inference, and model architecture for the final Sana-Video 2.0 5B and 14B variants. It exposes the production architecture and a fixed 480p reference recipe.

## Released architectures

| Model | Layers | Hidden size | Linear attention | Softmax anchors | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SanaVideo2_5B` | 32 | 2,560 | 20 heads × 128 | 8 layers, 10 heads × 256 | 4,466,980,960 |
| `SanaVideo2_14B` | 40 | 4,096 | 32 heads × 128 | 10 layers, 16 heads × 256 | 14,246,716,224 |

Both variants use:

- 75% gated bidirectional linear-attention layers and 25% dense softmax anchor layers;
- SwiGLU feed-forward networks with a 4× expansion ratio;
- Wan 3D rotary position embeddings;
- shared, timestep-independent Attention Residual aggregation in blocks of eight layers;
- `(1, 1, 1)` latent patches; and
- the LTX 2.3 VAE contract: 128 latent channels with `(8, 32, 32)` temporal/spatial strides.

## Files

- Model: `diffusion/model/nets/sana_video2.py`
- Transformer blocks: `diffusion/model/nets/sana_video2_blocks.py`
- Training: `train_video_scripts/train_sana_video2.py`
- Inference: `inference_video_scripts/inference_sana_video2.py`
- Configs: `configs/sana_video2/`

## Prerequisites

Install the repository environment, then place the Diffusers-format LTX 2.3 VAE under:

```text
output/pretrained_models/LTX-2.3-Diffusers/
```

Alternatively, change `vae.vae_pretrained` in the selected YAML. Set `model.load_from` to a checkpoint when fine-tuning; leave it as `null` to initialize a new model.

The example configs use `data/video_toy_data`. Replace `data.data_dir` with a `SanaZipDataset`-compatible public or local dataset. Paths are resolved relative to the repository.

## Inference

Generate one 480×832, 81-frame video:

```bash
python inference_video_scripts/inference_sana_video2.py \
  --config configs/sana_video2/SanaVideo2_5B_480p.yaml \
  --model-path /path/to/sana_video2_5b.pth \
  --prompt "A red panda runs through a misty bamboo forest." \
  --output-dir output/sana_video2_samples
```

Use `--prompt-file prompts.txt` for one prompt per line. Select the 14B YAML for a 14B checkpoint. The checkpoint loader accepts regular `.pth`, `.safetensors`, sharded Safetensors index files, and `hf://` paths supported by the repository downloader.

Height and width must be divisible by 32. Frame counts must satisfy `(num_frames - 1) % 8 == 0`.

## Training

The public trainer is video-only and fixed-resolution. It uses FSDP, gradient checkpointing, Gemma 2 caption embeddings, causal LTX 2.3 encoding, flow matching, and strict checkpoint compatibility checks.

```bash
torchrun --nproc_per_node=8 train_video_scripts/train_sana_video2.py \
  --config configs/sana_video2/SanaVideo2_5B_480p.yaml
```

For 14B:

```bash
torchrun --nproc_per_node=8 train_video_scripts/train_sana_video2.py \
  --config configs/sana_video2/SanaVideo2_14B_480p.yaml
```

To resume an FSDP run, pass `--resume-from` with a checkpoint directory such as `output/sana_video2_5b/checkpoints/epoch_1_step_1000`. Use the merged sibling `.pth` file for inference.
