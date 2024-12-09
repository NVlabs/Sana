## 🖌️ Sana-ComfyUI

[Original Repo](https://github.com/city96/ComfyUI_ExtraModels)

### Model info / implementation
- Uses Gemma2 2B as the text encoder
- Multiple resolutions and models available
- Compressed latent space (32 channels, /32 compression) - needs custom VAE

### Usage
1. All the checkpoints will be downloaded automatically.
2. KSampler(Flow Euler) is available for now; Flow DPM-Solver will be available soon.
3. For more information, check the [original city96/ComfyUI_ExtraModels](https://github.com/city96/ComfyUI_ExtraModels).

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

![Sample T2I + I2V workflow](https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/page/asset/content/comfyui/sana-cogvideox.jpg)
