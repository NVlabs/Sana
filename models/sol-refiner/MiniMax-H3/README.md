# SoL-Refiner for MiniMax-H3

Refine an H3 video with **one denoising step**, using Diffusers LTX-2.5 components.

[Project page](https://nvlabs.github.io/Sana/Sol-Refiner/) · [All variants](../)

## Install

```bash
pip install -r requirements.txt
```

Install the matching [NATTEN wheel](https://whl.natten.org/) for your PyTorch/CUDA
version. Tested on H100 80GB with Python 3.12, PyTorch 2.9.1+cu126 and NATTEN
0.21.5 (torch290cu126). Diffusers is pinned for its LTX-2.5 decoder APIs.

## Run

The official model repository is pending release. For now, use the complete local
Diffusers model package:

```bash
python infer.py \
  --model /path/to/sol-refiner-h3 \
  --input /path/to/h3-video.mp4 \
  --prompt 'A snow leopard walks along a snowy mountain ridge.' \
  --output outputs/refined.mp4 \
  --seed 303000 --decoder-seed 20260826
```

The Refiner performs one Transformer forward and one Euler update
(`0.9093750119 → 0`), without CFG. Text encoding, video conditioning, upsampling
and diffusion decoding use Diffusers. The two small compatibility adapters reuse
the upstream video encoder and locally installed NATTEN attention processor.

Output defaults to 1920×1080 at the input frame rate. Inputs are truncated to
`8k + 1` frames (124 → 121). The 1920×1088 internal canvas is center-cropped to
1080p. The CLI uses CPU offload and 768-pixel decoder tiles with a 512-pixel stride.
Tiled output can differ from untiled decoding. Output contains no audio.

## Validation

```bash
python -m unittest discover -s tests -v
```

Five CPU tests cover the one-step schedule, repeated invocation, geometry, invalid
inputs and encoder save/reload. Three complete H100 MP4-to-MP4 runs produced
121 frames at 24 fps and 1920×1080, each with exactly one Refiner call. Controlled
core comparisons sharing the reference codecs gave video SSIM of 0.9922, 0.9922
and 0.9852. These sample checks do not establish pixel equality or general quality
parity. Upstream H3 generation is separate from this refiner.
