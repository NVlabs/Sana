<!--Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# 4bit SanaPipeline

### 1. Environment setup

Follow the official [SVDQuant-Nunchaku](https://github.com/mit-han-lab/nunchaku) repository to set up the environment. The guidance can be found [here](https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation).

### 1-1. Quantize Sana with SVDQuant-4bit (Optional)

1. Convert pth to SVDQuant required safetensor

```
python tools/convert_sana_to_svdquant.py \
      --orig_ckpt_path Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth \
      --model_type SanaMS1.5_1600M_P1_D20 \
      --dtype bf16 \
      --dump_path output/SANA1.5_1.6B_1024px_svdquant_diffusers \
      --save_full_pipeline
```

2. follow the guidance to compress model
   [Quantization guidance](https://github.com/mit-han-lab/deepcompressor/tree/main/examples/diffusion)

### 2. Code snap for inference

Here we show the code snippet for SanaPipeline. For SanaPAGPipeline, please refer to the [SanaPAGPipeline](https://github.com/mit-han-lab/nunchaku/blob/main/examples/sana_1600m_pag.py) section.

```python
import torch
from diffusers import SanaPipeline

from nunchaku.models.transformer_sana import NunchakuSanaTransformer2DModel

transformer = NunchakuSanaTransformer2DModel.from_pretrained("mit-han-lab/svdq-int4-sana-1600m")
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    transformer=transformer,
    variant="bf16",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

image = pipe(
    prompt="A cute 🐼 eating 🎋, ink drawing style",
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("sana_1600m.png")
```

### 3. Online demo

1). Launch the 4bit Sana.

```bash
python app/app_sana_4bit.py
```

2). Compare with BF16 version

Refer to the original [Nunchaku-Sana.](https://github.com/mit-han-lab/nunchaku/tree/main/app/sana/t2i) guidance for SanaPAGPipeline

```bash
python app/app_sana_4bit_compare_bf16.py
```
