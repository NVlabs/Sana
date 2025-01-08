## 🖌️ Sana-ComfyUI

[Original Repo](https://github.com/city96/ComfyUI_ExtraModels)

### Model info / implementation

- Uses Gemma2 2B as the text encoder
- Multiple resolutions and models available
- Compressed latent space (32 channels, /32 compression) - needs custom VAE

### Usage

1. All the checkpoints will be downloaded automatically.
1. KSampler(Flow Euler) is available for now; Flow DPM-Solver will be available soon.

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
git clone https://github.com/Efficient-Large-Model/ComfyUI_ExtraModels.git custom_nodes/ComfyUI_ExtraModels

python main.py
```

### A sample workflow for Sana

[Sana workflow](Sana_FlowEuler.json)

![Sana](https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/page/asset/content/comfyui/sana.jpg)

### A sample for T2I(Sana) + I2V(CogVideoX)

[Sana + CogVideoX workflow](Sana_CogVideoX.json)

[![Sample T2I + I2V](https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/page/asset/content/comfyui/sana-cogvideox.jpg)](https://nvlabs.github.io/Sana/asset/content/comfyui/Sana_CogVideoX_Fun.mp4)

### A sample workflow for Sana 4096x4096 image (18GB GPU is needed)

[Sana workflow](Sana_FlowEuler_4K.json)

![Sana](https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/page/asset/content/comfyui/Sana_4K_workflow.jpg)
