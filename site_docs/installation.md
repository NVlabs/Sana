# Installation

## Environment

```bash
conda create -n solengine python=3.12 -y
conda activate solengine

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install diffusers==0.37.1 transformers accelerate imageio-ffmpeg

git clone -b sol-engine https://github.com/NVlabs/Sana.git
cd Sana
pip install ./techniques/sparse_backends
```

`techniques/sparse_backends` is the Sol-Attn kernel package. It needs PyTorch
already installed — it does not pull one in, so it cannot override a build
matched to your CUDA.

Sol-Attn ships CuTe kernels for sm90 (H100), sm100 (B200/GB200) and sm120
(RTX 5090), and a Triton reference elsewhere. `benchmark.json` records which
backend a run used.

## Weights

```bash
export HF_HOME=/somewhere/with/room     # 30–140 GB per model
hf auth login

hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
hf download MiniMaxAI/MiniMax-H3
hf download Efficient-Large-Model/SANA-Video_2B_480p_diffusers
```
