# Training Configs

Example training configurations for the LTX-2 trainer. Each file is a ready-to-run config for one training mode,
expressed through the unified **flexible** strategy (`name: "flexible"`). Pick the one closest to your use case and
adjust paths, dataset, and hyperparameters.

> 📖 For more information about using each training mode, see [Training Modes Guide](../docs/training-modes.md).

## Training Modes

| Mode                  | Video     | Audio     | Conditions          | Config |
|-----------------------|-----------|-----------|---------------------|--------|
| **T2V**               | Generated | Generated | —                   | [`t2v_lora.yaml`](./t2v_lora.yaml), [`t2v_lora_low_vram.yaml`](./t2v_lora_low_vram.yaml) (low VRAM) |
| **I2V**               | Generated | Generated | `first_frame`       | [`i2v_lora.yaml`](./i2v_lora.yaml) |
| **Video Extension**   | Generated | Generated | `prefix`/`suffix`   | [`video_extend_lora.yaml`](./video_extend_lora.yaml) (forward), [`video_suffix_lora.yaml`](./video_suffix_lora.yaml) (backward) |
| **V2V IC-LoRA**       | Generated | —         | `reference`         | [`v2v_ic_lora.yaml`](./v2v_ic_lora.yaml) |
| **A2V**               | Generated | Frozen    | —                   | [`a2v_lora.yaml`](./a2v_lora.yaml) |
| **V2A (Foley)**       | Frozen    | Generated | —                   | [`v2a_lora.yaml`](./v2a_lora.yaml) |
| **Video Inpainting**  | Generated | —         | `mask`              | [`video_inpainting_lora.yaml`](./video_inpainting_lora.yaml) |
| **Video Outpainting** | Generated | —         | `spatial_crop`      | [`video_outpainting_lora.yaml`](./video_outpainting_lora.yaml) |
| **T2A**               | —         | Generated | —                   | [`t2a_lora.yaml`](./t2a_lora.yaml) |
| **Audio Extension**   | —         | Generated | `prefix`/`suffix`   | [`audio_extend_lora.yaml`](./audio_extend_lora.yaml) (forward), [`audio_suffix_lora.yaml`](./audio_suffix_lora.yaml) (backward) |
| **Audio Inpainting**  | —         | Generated | `mask`              | [`audio_inpainting_lora.yaml`](./audio_inpainting_lora.yaml) |
| **A2A IC-LoRA**       | —         | Generated | `reference`         | [`a2a_ic_lora.yaml`](./a2a_ic_lora.yaml) |
| **AV2AV IC-LoRA**     | Generated | Generated | `reference` (both)  | [`av2av_ic_lora.yaml`](./av2av_ic_lora.yaml) |

The [`accelerate/`](./accelerate) directory holds the Accelerate launch configs (FSDP, DDP) for multi-GPU training.
