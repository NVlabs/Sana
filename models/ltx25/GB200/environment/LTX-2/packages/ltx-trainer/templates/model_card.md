---
tags:
  - ltx-2
  - ltx-video
  - text-to-video
  - audio-video
pinned: true
language:
  - en
license: other
pipeline_tag: text-to-video
library_name: diffusers
---

# {model_name}

This is a fine-tuned version of [`{base_model}`]({base_model_link}) trained on custom data.

## Model Details

- **Base Model:** [`{base_model}`]({base_model_link})
- **Training Type:** {training_type}
- **Training Steps:** {training_steps}
- **Learning Rate:** {learning_rate}
- **Batch Size:** {batch_size}

## Sample Outputs

| | | | |
|:---:|:---:|:---:|:---:|
{sample_grid}

## Usage

This model is designed to be used with the LTX-2 (Lightricks Audio-Video) pipeline.

### 🔌 Using Trained LoRAs in ComfyUI

In order to use the trained LoRA in ComfyUI, follow these steps:

1. Copy your trained LoRA checkpoint (`.safetensors` file) to the `models/loras` folder in your ComfyUI installation.
2. In your ComfyUI workflow:
    - Add the "Load LoRA" node to choose your LoRA file
    - Connect it to the "Load Checkpoint" node to apply the LoRA to the base model

You can find reference Text-to-Video (T2V) and Image-to-Video (I2V) workflows in the
official [LTX-2 repository](https://github.com/Lightricks/LTX-2).

### Example Prompts

{validation_prompts}


This model inherits the license of the base model ([`{base_model}`]({base_model_link})).

## Acknowledgments

- Base model: [Lightricks](https://huggingface.co/Lightricks/LTX-2)
- Trainer: [LTX-2](https://github.com/Lightricks/LTX-2)
