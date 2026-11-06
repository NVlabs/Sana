# Validation and full measurements

## Status

| Check | Status |
|---|---|
| CPU tests (`pip install -r requirements-test.txt && pytest tests`) | pass |
| Pinned checkpoint `9e9ed14b` vs the weights measured below | byte-identical shards and index files (sha256 in `PROVENANCE.json`) |
| Sol-H3 `h3_runtime` modules used vs the measured snapshot | identical at this branch's base commit |
| GPU numerics gate: `official` on the released remote code vs the measured configuration | in progress |
| 1 / 4 / 8 GPU smoke, t2v and ref2va | in progress |
| Re-timing of the tables below with the released remote code | in progress |

The released remote code renames the transformer and INT8 classes and moves low-rank padding and the INT8 kernel choice
into load-time options. `sol_hyperflow_flash/checkpoint.py` finds them by feature and applies the settings the
measurements used (below).

## Presets

| Preset | What runs |
|---|---|
| `official` | The checkpoint as released, on the MiniMax-H3 modular pipeline: dense attention (diffusers dispatch, BF16), diffusers VAE decode / encode on every rank. With more than one GPU the live DiT blocks run under Ulysses sequence parallelism. |
| `sol-engine` (default) | The Full-Opt list in the README. `--attention-backend dense` keeps it and turns Sol-Attn off. |

Both presets load the checkpoint with the same run-time settings, none of which changes the weights:

- low-rank ranks zero-padded to multiples of 64;
- the feed-forward runs over the whole sequence;
- INT8 `int8_gemm="auto"` (cuBLASLt on compute capability 10.x) and `rotate_quant="dense"`.

## Protocol

- **Hardware and output:** GB200, 1344 x 768 at 24 FPS, video and stereo audio.
- **Requests:**
  - t2v: one prompt (`../demo_prompt.json`), seed 0;
  - ref2va, one image: one reference image, seed 0;
  - ref2va, video + voice: MiniMax-H3's reproducible ref2va request (a 5 s reference video with its soundtrack, plus a voice clip).
- **Timing:** each cell is the median of two timed requests after one warmup. Official and Sol-Engine run in the same process on the same node(s). 8 GPUs = two 4-GPU nodes.
- **e2e:** host wall clock around the pipeline call, with device syncs and barriers. It covers conditioning, reference encoding, the six DiT forwards and VAE decoding. It excludes model loading, warmup and MP4 encoding.
- **DiT:** sum of the six transformer forwards on rank 0.
- **Peak:** maximum over ranks of `torch.cuda.max_memory_allocated`.
- **Measured:** 2026-11-05, with a pre-release copy of the checkpoint whose weight shards are byte-identical to `9e9ed14b`.
- **t2v:** only the `sol-engine` preset is listed; the earlier official t2v cells used pre-release INT8 kernels and are not reported.

## End-to-end time (s)

| Workload | Preset | 5 s<br>1 GPU | 5 s<br>4 GPU | 5 s<br>8 GPU | 10 s<br>1 GPU | 10 s<br>4 GPU | 10 s<br>8 GPU | 15 s<br>1 GPU | 15 s<br>4 GPU | 15 s<br>8 GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t2v | sol-engine | 13.0 | 3.4 | 1.9 | 35.3 | 9.2 | 4.6 | 66.6 | 16.1 | 8.5 |
| ref2va, one image | official | 33.2 | 15.8 | 12.2 | 74.6 | 31.8 | 23.3 | 132.1 | 52.5 | 37.1 |
| ref2va, one image | sol-engine | 23.1 | 6.8 | 4.0 | 49.5 | 13.6 | 7.5 | 85.1 | 23.4 | 12.1 |
| ref2va, video + voice | official | 73.9 | 35.1 | 28.6 | 145.6 | 59.6 | 46.0 | 209.9 | 85.0 | 61.9 |
| ref2va, video + voice | sol-engine | 46.7 | 13.2 | 7.8 | 93.0 | 24.3 | 13.7 | 135.1 | 37.3 | 19.9 |

## DiT time (s)

| Workload | Preset | 5 s<br>1 GPU | 5 s<br>4 GPU | 5 s<br>8 GPU | 10 s<br>1 GPU | 10 s<br>4 GPU | 10 s<br>8 GPU | 15 s<br>1 GPU | 15 s<br>4 GPU | 15 s<br>8 GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t2v | sol-engine | 9.7 | 2.7 | 1.5 | 28.6 | 7.9 | 3.8 | 53.1 | 14.2 | 7.4 |
| ref2va, one image | official | 24.9 | 7.2 | 3.7 | 60.7 | 17.0 | 8.7 | 111.6 | 31.8 | 15.9 |
| ref2va, one image | sol-engine | 18.3 | 5.2 | 2.7 | 41.3 | 11.3 | 5.9 | 73.6 | 20.5 | 10.2 |
| ref2va, video + voice | official | 54.2 | 15.3 | 8.3 | 117.7 | 31.1 | 16.5 | 175.6 | 50.4 | 25.9 |
| ref2va, video + voice | sol-engine | 37.6 | 10.4 | 5.6 | 79.7 | 20.6 | 11.0 | 115.9 | 32.7 | 16.7 |

## Peak GPU memory (GiB per GPU)

| Workload | Preset | 5 s<br>1 GPU | 5 s<br>4 GPU | 5 s<br>8 GPU | 10 s<br>1 GPU | 10 s<br>4 GPU | 10 s<br>8 GPU | 15 s<br>1 GPU | 15 s<br>4 GPU | 15 s<br>8 GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t2v | sol-engine | 99 | 95 | 94 | 107 | 100 | 98 | 115 | 105 | 102 |
| ref2va, video + voice | official | 106 | 96 | 96 | 114 | 99 | 98 | 122 | 102 | 101 |
| ref2va, video + voice | sol-engine | 105 | 96 | 95 | 113 | 99 | 98 | 121 | 104 | 101 |

## Stage split at 8 GPUs (s, min-max over the 8 ranks; one audited request)

Video + voice, 15 s output:

| Stage | Official | Sol-Engine |
|---|---:|---:|
| setup (reference prep) | 3.66-3.68 | 0.28-0.31 |
| text encode | 0.80-0.90 | 0.65-0.82 |
| reference video encode | 11.09-11.24 | 0.85-1.06 |
| reference audio encode | 0.03-0.04 | 0.03 |
| latent prep | 0.11-0.12 | 0.11 |
| DiT | 25.68-25.92 | 16.75-16.78 |
| of which Ulysses all-to-all (incl. waiting) | 1.96-4.00 | 1.38-2.52 |
| video VAE decode | 17.81-21.78 | 0.86-0.88 |
| audio VAE decode | 0.10 | 0.10 |
| rest | 0.00-3.97 | 0.00-0.02 |
| e2e | 63.52 | 19.89 |

One image, 5 s output:

| Stage | Official | Sol-Engine |
|---|---:|---:|
| setup (reference prep) | 0.07 | 0.07 |
| text encode | 0.76-0.79 | 0.67-0.68 |
| reference video encode | 1.25-1.27 | 0.17-0.18 |
| reference audio encode | 0.00 | 0.00 |
| latent prep | 0.05 | 0.05 |
| DiT | 3.74-3.78 | 2.72-2.74 |
| of which Ulysses all-to-all (incl. waiting) | 0.22-0.47 | 0.22-0.35 |
| video VAE decode | 5.82-6.27 | 0.28 |
| audio VAE decode | 0.06-0.07 | 0.06 |
| rest | 0.00-0.46 | 0.00-0.01 |
| e2e | 12.27 | 4.05 |

## Numerics

Exact (bitwise) relative to the same request without the switch:

- conditioner trimming;
- threaded reference-frame resize;
- the eager sharded reference encode.

Approximate:

- **Sol-Attn:** approximate by design. The sink rows match dense attention to BF16 rounding.
- **Compressed transport:** lossy on the Sol-Attn calls only. Dense calls keep the BF16 all-to-all.
- **Compiled VAE decoder:** within Sol-H3's own tolerance (rel L2 about 4e-4). One 4-GPU t2v case reached a maximum absolute deviation of 0.071 in decoder space at a single pixel cluster. That case is reproducible.
- **Compiled reference encoder:** moments rel L2 1.4e-4 vs the eager encoder.

## Layout

```
infer.py, benchmark.py        command line, timing
sol_hyperflow_flash/
  checkpoint.py               checkpoint resolution and loading (pins, feature detection, run-time options)
  config.py                   presets
  runtime.py                  resident pipeline, preset switching, timers
  sampler.py                  6-step two-time sampler as the pipeline's denoise loop
  attention.py                dense / Sol-Attn schedule, exact sinks, compressed transport
  sp.py                       Ulysses sequence parallelism for the live blocks
  vae_par.py                  Sol-H3 parallel VAE decode, sharded reference encode
  merge_tuned.py              tuned launch of Sol-H3's residual merge
  references.py, media.py     ref2va request files, MP4 output
  solh3.py                    imports ../Sol-H3/h3_runtime modules
tests/                        CPU tests
examples/ref2va_request.json  request format
```
