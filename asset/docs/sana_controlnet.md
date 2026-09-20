<!-- Copyright 2024 NVIDIA CORPORATION & AFFILIATES

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SPDX-License-Identifier: Apache-2.0 -->

## 🔥 ControlNet

We incorporate a ControlNet-like(https://github.com/lllyasviel/ControlNet) module enables fine-grained control over text-to-image diffusion models. We implement a ControlNet-Transformer architecture, specifically tailored for Transformers, achieving explicit controllability alongside high-quality image generation.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/page/asset/content/controlnet/sana_controlnet.jpg"  height=480>
</p>

## Inference of `Sana + ControlNet`

### 1). Gradio Interface

```bash
python app/app_sana_controlnet_hed.py \
        --config configs/sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml \
        --model_path hf://Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED/checkpoints/Sana_1600M_1024px_BF16_ControlNet_HED.pth
```

<p align="center" border-raduis="10px">
  <img src="https://nvlabs.github.io/Sana/asset/content/controlnet/controlnet_app.jpg" width="90%" alt="teaser_page2"/>
</p>

### 2). Inference with JSON file

```bash
python tools/controlnet/inference_controlnet.py \
        --config configs/sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml \
        --model_path hf://Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED/checkpoints/Sana_1600M_1024px_BF16_ControlNet_HED.pth \
        --json_file asset/controlnet/samples_controlnet.json
```

### 3). Inference code snap

```python
import torch
from PIL import Image
from app.sana_controlnet_pipeline import SanaControlNetPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = SanaControlNetPipeline("configs/sana_controlnet_config/Sana_1600M_1024px_controlnet_bf16.yaml")
pipe.from_pretrained("hf://Efficient-Large-Model/Sana_1600M_1024px_BF16_ControlNet_HED/checkpoints/Sana_1600M_1024px_BF16_ControlNet_HED.pth")

ref_image = Image.open("asset/controlnet/ref_images/A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a la.jpg")
prompt = "A transparent sculpture of a duck made out of glass. The sculpture is in front of a painting of a landscape."

images = pipe(
    prompt=prompt,
    ref_image=ref_image,
    guidance_scale=4.5,
    num_inference_steps=10,
    sketch_thickness=2,
    generator=torch.Generator(device=device).manual_seed(0),
)
```

## Training of `Sana + ControlNet`

### Coming soon
