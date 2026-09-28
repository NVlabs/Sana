# SoL-Refiner: Original LTX-2.3 Version

The original [SoL-Refiner](../) release supports both the **one-step distilled
refiner** and the **multi-step refiner**, with baseline and SoL-Engine execution.
It refines existing videos from SANA-Video, WAN and other base generators using
LTX-2.3's video encoder, spatial upsampler and VAE decoder.

[Project page](https://nvlabs.github.io/Sana/Sol-Refiner/) · [Method](../#method) · [Video demos](https://nvlabs.github.io/Sana/Sol-Refiner/#generator-samples)

The subsequent [MiniMax-H3 extension](../MiniMax-H3/) uses LTX-2.5 and a different
checkpoint and decoder. Keep these model packages separate.

## Models and execution modes

| Model package | Euler updates | Sigma interval | CFG | Baseline DiT calls |
| --- | ---: | --- | ---: | ---: |
| One-step | 1 | 0.725 → 0 | 1 | 1 |
| Multi-step | 19 | 0.909375 → 0 | 3 | 38 |

These are two trained checkpoints. The model package records its sampling variant;
changing the step count does not turn one checkpoint into the other. The multi-step
schedule truncates the original 32-step LTX schedule to the interval shown above.

| Execution | One-step | Multi-step |
| --- | --- | --- |
| `--engine baseline` | Standard Diffusers operations | Standard Diffusers operations |
| `--engine sol` | SOL-Attn + fused operations | SOL-Attn + fused operations |
| `--engine sol --teacache` | Unsupported | Also reuses eligible denoised predictions |

SoL-Engine reuses Sana's integrated SM90 SOL-Attn backend. Tokens follow Morton
ordering across the block stack; the first and last layers retain dense attention.
The default routing threshold is `tau=1.5`, matching the reference video-demo
configuration. Use `--sol-tau` to adjust it, or `--sol-density 0.15` to select the
reference long-sequence benchmark policy. Low target densities can be too
aggressive for short clips; these two options are mutually exclusive.
Four Triton fusions cover RMSNorm with modulation, weighted RMSNorm, split RoPE,
and the feed-forward gated residual addition. TeaCache keeps all 19 scheduler updates but reduces
DiT calls when its reuse criterion passes. SOL-Attn and TeaCache are approximate
accelerations; all modes use BF16 without weight quantization.

## Install

Use a Linux CUDA environment with Python 3.12 and PyTorch 2.9.1+cu126:

```bash
pip install -r requirements.txt
# Optional, for the H100/H200 SoL-Engine backend:
pip install -r requirements-sol.txt
```

Run from this directory inside the Sana checkout so the shared SOL-Attn backend
is available. Baseline inference does not require the optional CuTe dependencies.

## Refine a video

Pretrained weights: [One-step](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.3-One-Step) ·
[Multi-step](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.3-Multi-Step). The same weights support baseline
and SoL-Engine execution. The CLI defaults to the one-step Hugging Face model.

```bash
# One-step, baseline
python infer.py --model Efficient-Large-Model/SoL-Refiner-LTX-2.3-One-Step \
  --input input.mp4 --prompt 'A cinematic scene.' --seed 1234 \
  --engine baseline --output one-step-baseline.mp4

# One-step, SoL-Engine
python infer.py --model Efficient-Large-Model/SoL-Refiner-LTX-2.3-One-Step \
  --input input.mp4 --prompt 'A cinematic scene.' --seed 1234 \
  --engine sol --output one-step-sol.mp4

# Multi-step, baseline
python infer.py --model Efficient-Large-Model/SoL-Refiner-LTX-2.3-Multi-Step \
  --input input.mp4 --prompt 'A cinematic scene.' --seed 1234 \
  --engine baseline --output multi-step-baseline.mp4

# Multi-step, SoL-Engine with TeaCache
python infer.py --model Efficient-Large-Model/SoL-Refiner-LTX-2.3-Multi-Step \
  --input input.mp4 --prompt 'A cinematic scene.' --seed 1234 \
  --engine sol --teacache --output multi-step-sol.mp4
```

Output defaults to 2048×1152 at the input frame rate. Set `--width` and `--height`
to change the target size. The pipeline preserves the reference BF16 coordinate and rotary arithmetic.
It uses spatial 2× latent upsampling, keeps
`8k+1` frames, and outputs video without audio. It uses posterior-mode encoding;
the one-step preset applies AdaIN after upsampling, while the multi-step preset
leaves it disabled. The multi-step preset uses the original negative prompt;
`--negative-prompt` overrides it.

The CLI enables model CPU offload and VAE tiling. Large 4K/long-video workloads
can require further memory tuning. For performance comparisons, use repeated
calls in a resident process and exclude compilation and warmup.

## Checks

```bash
python -m unittest discover -s tests -v
```

The tests verify the sampling schedules, CFG call counts, cache policy, repeated
invocation, Morton round trips and the adapted video block against Diffusers.
The command prints actual scheduler and Transformer counts for each video.
