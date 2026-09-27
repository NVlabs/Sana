# SoL-Refiner

SoL-Refiner refines generated videos in one denoising step. This directory is the
shared code entry for its generator-specific variants.

[Project page](https://nvlabs.github.io/Sana/Sol-Refiner/)

| Variant | Entry | Status |
| --- | --- | --- |
| SANA / Wan | [wan](wan/) | Separate implementation and model release pending |
| MiniMax-H3 | [MiniMax-H3](MiniMax-H3/) | Diffusers LTX-2.5 one-step inference implementation; model packaging and full video validation in progress |

Each variant documents its own pretrained model and input contract. The H3
variant is a dedicated refiner, not a substitute checkpoint for the SANA / Wan
variant. Model weights are distributed separately from this source tree.
