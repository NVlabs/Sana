# HyperFlow H3 Flash on Sol-Engine

## Overview

[HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash)
([code](https://github.com/Video-Rebirth/hyperflow-h3-flash)) is Video Rebirth's six-step MiniMax-H3: pruned, low-rank
and W8A8 INT8, conditioned on two times. This runner serves its `t2v` and `ref2va` partitions with synchronized audio at
1344 x 768 and 24 FPS (5 / 10 / 15 s) on one, four or eight GB200 GPUs. It loads the checkpoint's own transformer code and
drives it inside the diffusers MiniMax-H3 modular pipeline, with the Sol-H3 runtime of this repository.

## Performance

| GPUs | Workload | Official (s) | Sol-Engine (s) | Speedup |
|---:|---|---:|---:|---:|
| 1 | t2v 5 s | 25.1\* | 13.0 | **1.93x** |
| 4 | t2v 5 s | 6.6\* | 3.4 | **1.94x** |
| 8 | t2v 5 s / 10 s / 15 s | 3.6\* / 8.6\* / 16.3\* | 1.9 / 4.6 / 8.5 | **1.89x / 1.87x / 1.92x** |
| 1 | ref2va, one image, 5 s | 33.2 | 23.1 | **1.44x** |
| 4 | ref2va, one image, 5 s | 15.8 | 6.8 | **2.32x** |
| 8 | ref2va, one image, 5 s / 10 s / 15 s | 12.2 / 23.3 / 37.1 | 4.0 / 7.5 / 12.1 | **3.05x / 3.11x / 3.07x** |
| 8 | ref2va, video + voice, 5 s / 10 s / 15 s | 28.6 / 46.0 / 61.9 | 7.8 / 13.7 / 19.9 | **3.67x / 3.36x / 3.11x** |

Official is the checkpoint as released, on the diffusers MiniMax-H3 pipeline with dense attention. Times are end-to-end,
measured on GB200 on 2026-11-05 with byte-identical weights. \*The t2v official cells ran pre-release INT8 kernels and
are being re-timed. All cells, the protocol and the validation status are in [VALIDATION.md](VALIDATION.md).

## Full-Opt

- **Attention:** Sol-Attn with `tau=1.25` and diagonal thresholding, through the cuDNN block-sparse backend.
  - Dense attention at steps 0-1 and in blocks 0, 1, 48 and 49.
  - Exact sink: the text + audio prefix for t2v; for ref2va, the blocks holding text or audio rows.
- **Parallelism:** Ulysses across the packed sequence. Sol-H3's compressed transport (INT8 QKV, FP8 output) runs on the Sol-Attn calls only.
- **Fusions:** Sol-H3 block fusions on every step, plus a tuned launch of the Sol residual merge.
- **Conditioner:** Qwen3-VL trimmed to the layers MiniMax-H3 reads (bitwise).
- **VAE:** Sol-H3's sharded compiled video decode (1 GPU: compiled only).
- **ref2va:** sharded compiled reference encode and threaded reference resize.

## Usage

Run from this directory. The checkpoint is pinned to revision `9e9ed14b`; `--weights` may be the Hub id (the default) or a
local folder.

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 6818f6c32d12b210915e44ad56a4228c2608f160 --local-dir checkpoints/MiniMax-H3
hf download VideoRebirthAI/hyperflow-h3-flash --revision 9e9ed14b8be4b751d7517b8ca58dd199d509720c \
  --include "fl2va_int8_convrot/*" "ref2va_int8_convrot/*" --local-dir checkpoints/hyperflow-h3-flash

torchrun --standalone --nproc_per_node 8 infer.py --task t2v --preset sol-engine --duration 5 \
  --prompt "A red fox trots through fresh snow at dawn" \
  --base-model checkpoints/MiniMax-H3 --weights checkpoints/hyperflow-h3-flash --out outputs/t2v.mp4
torchrun --standalone --nproc_per_node 8 infer.py --task ref2va --preset sol-engine --duration 5 \
  --request examples/ref2va_request.json --media-dir /path/to/media \
  --base-model checkpoints/MiniMax-H3 --weights checkpoints/hyperflow-h3-flash --out outputs/ref2va.mp4
```

Use `--preset official` for the baseline and `--attention-backend dense` to turn Sol-Attn off. `benchmark.py` times both
presets in one process.

## Environment

- **Runtime:** `requirements.txt` (the Sol-H3 pins) and the sibling [`../Sol-H3`](../Sol-H3/) runtime.
- **Weights:** MiniMax-H3 and HyperFlow H3 Flash, downloaded separately under the MiniMax H3 Community License Agreement
  (see the model card and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **Placement:** one GB200 node (1 or 4 GPUs) or two (8 GPUs); every component resident, no offload.

## Outputs

`infer.py` writes one MP4 with audio; `benchmark.py` writes JSON lines (per-request times and a median summary).
