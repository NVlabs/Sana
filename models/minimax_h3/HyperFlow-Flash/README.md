# HyperFlow H3 Flash for Sol-Engine

Six-step MiniMax-H3 video and audio generation with
[HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) (Video Rebirth; a pruned, low-rank and
W8A8 INT8 MiniMax-H3 transformer conditioned on two times) and the existing [Sol-H3 runtime](../Sol-H3/). The runner
targets **NVIDIA GB200** at 1, 4 and 8 GPUs and provides text-to-video (`t2v`) and reference-to-video (`ref2va`), both
with synchronized audio, at 1344 x 768 and 24 FPS.

The checkpoint's own transformer code (shipped next to its weights) defines the model. This runner keeps it, drives it
with the model card's 6-step two-time sampler inside the official diffusers MiniMax-H3 modular pipeline, and adds the
Sol-Engine optimizations listed below. Text encoder, tokenizer, processors, video VAE and audio VAE are the base
model's.

## Scope

| Task | Input | Checkpoint partition | Output presets |
|---|---|---|---|
| `t2v` | Text | `fl2va_int8_convrot` | 5 / 10 / 15 s, with audio |
| `ref2va` | Text and ordered references (images, video with its soundtrack, voice clips) | `ref2va_int8_convrot` | Same |

The frame counts are 124 / 243 / 362 (the model's 17n + 5 rule). Every request runs six DiT forwards. Keyframe
conditioning, other step counts and other resolutions are not exposed by this runner.

## Presets

| Preset | What runs |
|---|---|
| `stock` | The checkpoint on the MiniMax-H3 modular pipeline: dense attention (diffusers dispatch, BF16), diffusers VAE decode / encode on every rank. With more than one GPU the live DiT blocks run under Ulysses sequence parallelism (`sol_hyperflow_flash/sp.py`). |
| `ours` | Sol-Engine recipe, below. |

`ours`:

- **Attention:** dense at steps 0-1 and in blocks 0, 1, 48 and 49. Every other call uses Sol-Attn (`sol_bsa`, tau 1.25) with an exact sink: the text + audio prefix for t2v; for ref2va, the 64-blocks holding text or audio rows (Sol-H3's ref2va policy). The residual merge uses a tuned launch.
- **Fusions:** Sol-H3 block fusions on all six steps (RMSNorm + modulate, residual-gate-norm, SwiGLU, QK-norm + partial RoPE).
- **Multi-GPU transport:** Sol-H3's compressed Ulysses transport (INT8 QKV, FP8 output), on the Sol-Attn calls only.
- **Conditioner:** the Qwen3-VL conditioner is trimmed to the layers MiniMax-H3 reads (51 of 64 language layers; bitwise).
- **VAE:** Sol-H3's parallel compiled video VAE decode at every GPU count (1 GPU: compiled decoder only).
- **ref2va only:**
  - the reference video encode is sharded over all ranks (every clip's tiles, one gather) and runs with a compiled encoder;
  - reference frames are resized on 32 host threads (bitwise).

`--attention-backend dense` keeps everything else in the preset and turns Sol-Attn off.

Both presets load the checkpoint with the same run-time settings; none of them changes the weights:

- low-rank ranks zero-padded to multiples of 64;
- the feed-forward over the whole sequence;
- INT8 kernel options `int8_gemm="auto"` (cuBLASLt on compute capability 10.x) and `rotate_quant="dense"`.

These live in `sol_hyperflow_flash/checkpoint.py`, the only file that knows the checkpoint layout.

## Setup

```bash
# From models/minimax_h3/HyperFlow-Flash of this repository; install the CUDA 13 PyTorch build first.
pip install -r requirements.txt
hf download MiniMaxAI/MiniMax-H3 --revision 6818f6c32d12b210915e44ad56a4228c2608f160 --local-dir checkpoints/MiniMax-H3
# Optional: the runner downloads the pinned checkpoint partition itself when --weights is a Hub id (the default).
hf download VideoRebirthAI/hyperflow-h3-flash --revision 9e9ed14b8be4b751d7517b8ca58dd199d509720c \
  --include "fl2va_int8_convrot/*" "ref2va_int8_convrot/*" --local-dir checkpoints/hyperflow-h3-flash
```

The weights are a Model Derivative of MiniMax-H3 under the MiniMax H3 Community License Agreement, which restricts
territory and includes an Acceptable Use Policy; read the checkpoint's model card before downloading.

## Usage

```bash
# t2v, 1 GPU
python infer.py --task t2v --prompt "A red fox trots through fresh snow at dawn" --duration 5 \
  --base-model checkpoints/MiniMax-H3 --weights checkpoints/hyperflow-h3-flash --out outputs/fox.mp4
# t2v, 8 GPUs (Ulysses sequence parallelism)
torchrun --standalone --nproc_per_node 8 infer.py --task t2v --prompt-file prompt.txt --duration 15 \
  --base-model checkpoints/MiniMax-H3 --weights checkpoints/hyperflow-h3-flash --out outputs/t2v.mp4
# ref2va: prompt and ordered references from a request file (see examples/ref2va_request.json)
torchrun --standalone --nproc_per_node 4 infer.py --task ref2va --request examples/ref2va_request.json \
  --media-dir /path/to/media --duration 10 --base-model checkpoints/MiniMax-H3 \
  --weights checkpoints/hyperflow-h3-flash --out outputs/ref2va.mp4
```

`--preset stock` runs the baseline, `--warmup` runs one untimed request first (kernel compilation), `--te-offload`
keeps the text encoder on the host between requests. `--sol-h3-dir` defaults to `../Sol-H3` of this checkout.

## Performance

End-to-end seconds per request on **NVIDIA GB200** (1344 x 768, 24 FPS, video and stereo audio, fixed seed). Each cell
is the median of two timed requests after one warmup; stock and ours ran in the same process on the same node(s).
Time covers conditioning, reference encoding, the six DiT forwards, video / audio VAE decoding and synchronization
across all GPUs; it excludes model loading, warmup and MP4 encoding. Measured on 2026-11-05 with a pre-release copy of
the checkpoint whose weight shards are byte-identical to the pinned revision, and with Sol-H3 `h3_runtime` sources
identical to the ones at this branch's base commit. Re-timing with the released remote code is in progress (see
[Validation status](#validation-status)).

**t2v** (one prompt):

| Preset | 5 s<br>1 GPU | 5 s<br>4 GPU | 5 s<br>8 GPU | 10 s<br>1 GPU | 10 s<br>4 GPU | 10 s<br>8 GPU | 15 s<br>1 GPU | 15 s<br>4 GPU | 15 s<br>8 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock\* | 25.1 | 6.6 | 3.6 | 66.0 | 17.9 | 8.6 | 117.5 | 31.0 | 16.3 |
| ours | **13.0** | **3.4** | **1.9** | **35.3** | **9.2** | **4.6** | **66.6** | **16.1** | **8.5** |

\* This t2v stock row ran the pre-release model code's INT8 kernels: the Triton INT8 GEMM and unpadded low-rank ranks.
It also decoded with Sol-H3's parallel compiled VAE, so that it isolates the DiT. The `stock` preset of this runner
differs on both counts: it uses the released code's cuBLASLt INT8 GEMM and 64-padded ranks, and the diffusers VAE on
every rank. This row will be replaced by the re-timing.

**ref2va, one reference image:**

| Preset | 5 s<br>1 GPU | 5 s<br>4 GPU | 5 s<br>8 GPU | 10 s<br>1 GPU | 10 s<br>4 GPU | 10 s<br>8 GPU | 15 s<br>1 GPU | 15 s<br>4 GPU | 15 s<br>8 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 33.2 | 15.8 | 12.2 | 74.6 | 31.8 | 23.3 | 132.1 | 52.5 | 37.1 |
| ours | **23.1** | **6.8** | **4.0** | **49.5** | **13.6** | **7.5** | **85.1** | **23.4** | **12.1** |

**ref2va, MiniMax-H3's reproducible ref2va request** (a 5 s reference video with its soundtrack, plus a voice clip):

| Preset | 5 s<br>1 GPU | 5 s<br>4 GPU | 5 s<br>8 GPU | 10 s<br>1 GPU | 10 s<br>4 GPU | 10 s<br>8 GPU | 15 s<br>1 GPU | 15 s<br>4 GPU | 15 s<br>8 GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 73.9 | 35.1 | 28.6 | 145.6 | 59.6 | 46.0 | 209.9 | 85.0 | 61.9 |
| ours | **46.7** | **13.2** | **7.8** | **93.0** | **24.3** | **13.7** | **135.1** | **37.3** | **19.9** |

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

## Validation status

Done:

- CPU tests: `pip install -r requirements-test.txt && pytest tests`.
- The pinned checkpoint's weight shards and index files are byte-identical to the copy used for the measurements above.
- Its remote code renames the transformer and INT8 classes and moves rank padding and kernel choice into load-time options; `checkpoint.py` detects these by feature.

In progress, on the pinned revision and this branch's Sol-H3:

- GPU numerics gate: stock preset vs the measured configuration;
- 1 / 4 / 8 GPU smoke runs of both tasks;
- re-timing of the tables above.

## Layout

```
infer.py                      command line
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

## License

The code is under Apache License 2.0 (`LICENSE`); see `THIRD_PARTY_NOTICES.md`. MiniMax-H3 and HyperFlow H3 Flash
weights are downloaded separately under the MiniMax H3 Community License Agreement. HyperFlow is not an official
MiniMax product. Powered by MiniMax H3.
