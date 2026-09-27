# Sol-H3 on NVIDIA GeForce RTX 5090

A two-stage video-and-audio pipeline for one RTX 5090 using CPU offload.
Generate a 384p draft, upscale and transfer its latents, then refine and decode
**1344 × 768, 121-frame video at 24 FPS** with original H3 audio.

[Setup](docs/setup.md) · [Validation](docs/validation.md) ·
[Component terms](THIRD_PARTY_NOTICES.md)

This is the dedicated Linux x86_64 entry, derived from the released
[Sol-H3 Spark pipeline](../Sol-H3-Spark/). Both the CLI and Python API default
to CPU offload. The Spark package remains a separate, unchanged entry.

> **T2VA validation:** one full warmup and one formal request completed on a
> single RTX 5090. The formal request took **39.80 s** through the completed MP4,
> including per-request Qwen loading, CPU/GPU transfers and muxing. This is one
> measurement, not a multi-run latency statistic. See [validation](docs/validation.md).

## The T2VA recipe

| Component | Fixed configuration |
| --- | --- |
| Prompt encoding | NVFP4 AWQ Qwen; loaded for each request and closed before H3 |
| Draft generation | W8A8 MiniMax-H3 with FastH3 VSA DataFree LoRA; four updates at 672 × 384 × 124 |
| Draft attention | VSA with 90% video sparsity and cuDNN BSA |
| Latent transfer | Learned H3 ×2 upscaler, then H3-to-LTX latent adapter |
| Refinement | Three BF16 LTX updates with Triton Sol attention and fixed generic AV context |
| Output | Original Conv VAE; 1344 × 768 × 121 at 24 FPS, with original H3 audio as stereo AAC |

The H3 body blocks, including nonpersistent FP8 buffers, stay on CPU between
forwards; one body block is transferred at a time. LTX uses its pinned source's
CPU streaming builder. The upscaler, adapter, audio encoder and video decoder
move to GPU for their active phases. The video stages execute serially.
H3 video decoding is bypassed; normalized H3 latents and PCM cross the stage
boundary through a local CPU tensor file.

## Prepare

Run from `models/minimax_h3/Sol-H3-RTX5090` on the GPU host. Follow
[setup](docs/setup.md) to reuse or prepare the pinned weights, sources,
native x86_64 interpreters and generic prompt cache, then write `paths-5090.json`.
The observed environments are documented; a fresh installation is not validated.

```bash
python prepare.py --plan
```

## Run

Expose exactly one GPU using a numeric index:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python infer.py --paths paths-5090.json \
  --prompt "A bird glides over a mountain lake. overall_soundscape: Soft wind and distant birds." \
  --seed 42 --output-dir outputs/lake-5090
```

CPU offload is enabled by default; `--offload cpu` is also accepted. The command
runs one full warmup and then the formal request. Choose a new output directory;
existing runs are not overwritten. `results.json` contains the continuous E2E
measurement and the final MP4 path.

For a batch, replace `--prompt` and `--seed` with
`--prompts examples/prompts.jsonl`. Video-model sessions are reused across
requests; Qwen is loaded and closed for each request.

The Python API has the same default:

```python
from runtime.config import load_paths
from runtime.pipeline import Pipeline

case = {"case_id": "lake", "prompt": "A bird glides over a mountain lake.", "seed": 42}
pipeline = Pipeline(load_paths("paths-5090.json"), "outputs/python-5090")
try:
    pipeline.start(case)
    result = pipeline.generate(case)
    pipeline.finish()
finally:
    pipeline.close()
```

The inherited FL2VA and Ref2VA selectors have CPU contract coverage, but
have not been GPU-validated on this offload profile. Memory fit and quality for
those tasks are not established by the T2VA result.

## Timing

E2E uses one same-host monotonic clock from request entry, before Qwen loading,
to the completed muxed MP4. It includes Qwen load/encoding/closure, transfers,
H3 generation, upscaling, refinement, decoding and muxing. It is not a sum of
component timers. Video-model initialization and full-chain warmup are separate.
HTTP transport, an external service queue and service cold start were not measured.

| Method | E2E latency |
| --- | --- |
| 4-step LoRA | 97 s |
| 4-step LoRA + quant | 51 s |
| Sol-H3 | 39.8 s |

See [validation](docs/validation.md) for exact timings, source provenance and
measurement limits.

## Code layout

```text
infer.py                    CLI; CPU offload by default
prepare.py                  Resolve paths and prepare pinned sources
download_checkpoints.py     Fetch the pinned model assets
configs/                    Recipe and source/checkpoint manifests
runtime/                    Request lifecycle, offload and inference operators
tests/                      CPU contracts and explicit small GPU offload check
validation/                 Sanitized 5090 measurement receipt
```

Sampling and attention reuse the Sana implementations in `super_acceleration`
and `techniques/sparse_backends`. Model weights and external runtime sources
are downloaded separately. See [component terms](THIRD_PARTY_NOTICES.md).
