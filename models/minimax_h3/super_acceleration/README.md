# MiniMax-H3 Super Acceleration

This directory owns a separate composite GB200 inference profile:

```text
MiniMax-H3 Stage 1 -> direct tensor/audio handoff -> LTX-2.5 Stage 2
```

It is not a MiniMax-H3 replacement model, a Diffusers-only arm, or a
Lightweight/YAML recipe. It combines two independently resident runtimes and
therefore stays outside `sol_engine`, the lightweight package API, and the YAML
configuration surface.

## Fixed v2 profile

- Stage 1 runs on one GB200: first-frame-conditioned MiniMax-H3 FL2VA at
  896x512, 124 frames, 24 FPS, with the LightX2V four-step LoRA merged at
  startup. It performs four Transformer forwards, keeps cache reuse disabled,
  compiles the fixed-shape DiT/VAE path, and decodes with TAEH3.
- The handoff is authenticated loopback TCP, not an intermediate MP4. It sends
  contiguous BF16 video `[1, 3, 121, 384, 672]` in the LTX VAE range and FP32
  stereo 32 kHz PCM `[1, 2, 161333]`. Stage 1 does not return until Stage 2 has
  acknowledged that both tensors are resident on its CUDA device.
- Stage 2 runs on a second GB200. It uses the original LTX-2.5 Video VAE input
  encoder, the learned x2 latent upsampler, original-VAE high-resolution
  first-frame conditioning, and the original Audio VAE. It performs three
  joint updates at sigma points `0.909375, 0.725, 0.421875, 0.0`.
- LTX Transformer layer 0 remains dense. Layers 1-47 use strict Sol-Attn with
  taus `1.0, 1.25, 1.5`; cross-attention and audio attention remain dense. Only
  the 48 Transformer blocks are compiled. TAEHV decodes the final video, and
  the original H3 PCM is muxed into the final H.264/AAC output.

One request therefore occupies two GPUs. The validated four-GPU launch runs two
independent one-Stage-1-GPU to one-Stage-2-GPU pairs concurrently; it is not
four-way tensor, context, or model parallelism.

The exact Stage-2 contract and its relationship to the earlier frozen handoff
are documented in [STAGE2_CONTRACT.md](STAGE2_CONTRACT.md).

## Validated latency, with claim limits

Formal Slurm job `6304303` completed 20 hot requests: ten on each of two
independent pairs. Model load, compile prime, and one complete warmup per stage
were excluded. The cross-pair medians were:

| Boundary | Median |
| --- | ---: |
| Complete H3 Stage 1 -> final Stage 2 MP4 | 6.760544632 s |
| Stage 1 wall | 4.2919342775 s |
| Stage 2 resident service | 2.446938 s |

These values are absolute hot latency evidence for this v2 profile. There is no
matched same-profile end-to-end baseline, so this directory makes no speedup
claim. A perceptual quality gate was not run, so successful media/telemetry
validation must not be described as a quality pass. See
[BENCHMARK_REFERENCE.json](BENCHMARK_REFERENCE.json) for the machine-readable
record.

The hardened integration subsequently passed a one-hot-per-pair GB200 smoke in
Slurm job `6449281` (`COMPLETED 0:0`). The two end-to-end samples were
`6.675905103 s` and `6.701609667 s`; the Stage-2 resident-service median was
`2.4162235 s`. Both requests passed the fail-closed `3` dense / `141` Sol /
`141` actual `cute_sm100` kernel-call contract and produced verified
1344x768, 121-frame, 24-FPS media. This two-sample smoke establishes runtime
functionality only. It is not a replacement for job `6304303`, a formal
benchmark, a matched speedup baseline, or a perceptual-quality gate.

## Reproducibility and release state

[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) records the exact imported v2 file
hashes and the intentional drift from the earlier `FINAL_REFINER_HANDOFF`
snapshot. That earlier contract began Stage 2 from an MP4 and used the default
input-VAE temporal tiling policy. The measured v2 path instead uses direct BF16
video plus PCM and one full temporal tile. The core Stage-2 model, conditioning,
schedule, attention routing, compile scope, and call counts are unchanged.

The checked-in integration was hardened after job `6304303`: it now derives
repository paths, locks writable compile caches, and verifies the opening-frame
SHA across both stages. Those formal-to-integration hash transitions are in
`SOURCE_SNAPSHOT.json`; lightweight CPU checks and the one-hot GPU smoke passed,
but the 20-hot GPU formal has not been rerun. The latency table is therefore
historical evidence bound to the formal hashes in `BENCHMARK_REFERENCE.json`,
not a bit-for-bit formal measurement of the hardened tree.

All model weights, LoRAs, checkpoints, first-frame media, datasets, containers,
environments, and compile caches are reference-only. They are not distributed
by this directory. The checked-in JSON files under `assets/` are identity
manifests, not the media assets themselves. Third-party terms and remaining
license checks are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The launcher preserves the formal topology but replaces the original personal
storage paths with explicit site inputs and repo-derived paths. Treat the whole
directory as an integration candidate until the source/license gates in
`SOURCE_SNAPSHOT.json` are closed and the hardened 20-hot formal is rerun.
