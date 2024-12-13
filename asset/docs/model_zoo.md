## 🔥 1. We provide all the links of Sana pth and diffusers safetensor below

| Model     | Reso   | pth link                                                                                                | diffusers                                                                                                                                         | Precision     | Description    |
|-----------|--------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------|----------------|
| Sana-0.6B | 512px  | [Sana_600M_512px](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px)                         | [Efficient-Large-Model/Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)                         | fp16/fp32     | Multi-Language |
| Sana-0.6B | 1024px | [Sana_600M_1024px](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px)                       | [Efficient-Large-Model/Sana_600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers)                       | fp16/fp32     | Multi-Language |
| Sana-1.6B | 512px  | [Sana_1600M_512px](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px)                       | [Efficient-Large-Model/Sana_1600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers)                       | fp16/fp32     | -              |
| Sana-1.6B | 512px  | [Sana_1600M_512px_MultiLing](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_MultiLing)   | [Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers)   | fp16/fp32     | Multi-Language |
| Sana-1.6B | 1024px | [Sana_1600M_1024px](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px)                     | [Efficient-Large-Model/Sana_1600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers)                     | fp16/fp32     | -              |
| Sana-1.6B | 1024px | [Sana_1600M_1024px_MultiLing](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_MultiLing) | [Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers) | fp16/fp32     | Multi-Language |
| Sana-1.6B | 1024px | [Sana_1600M_1024px_BF16](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16)           | [Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers)           | **bf16**/fp32 | Multi-Language |

## ❗ 2. Make sure to use correct precision(fp16/bf16/fp32) for training and inference.

### We provide two samples to use fp16 and bf16 weights, respectively.

❗️Make sure to set `variant` and `torch_dtype` in diffusers pipelines to the desired precision.

#### 1). For fp16 models

```python
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)

prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

image[0].save("sana.png")
```

#### 2). For bf16 models

```python
# run `pip install -U diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPAGPipeline

pipe = SanaPAGPipeline.from_pretrained(
  "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
  variant="bf16",
  torch_dtype=torch.bfloat16,
  pag_applied_layers="transformer_blocks.8",
)
pipe.to("cuda")

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
image = pipe(
    prompt=prompt,
    guidance_scale=5.0,
    pag_scale=2.0,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]
image[0].save('sana.png')
```
