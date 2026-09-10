# Sol-H3 on NVIDIA DGX Spark

A two-stage text-to-video-and-audio pipeline for one NVIDIA DGX Spark.
Generate a 384p draft, upscale and transfer its latent representation, then
refine and decode a **1344 × 768, 121-frame video at 24 FPS** with original H3 audio.

[Project page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/) ·
[Environment and weights](docs/setup.md) · [Component terms](THIRD_PARTY_NOTICES.md)

> **Validation:** this entry completed a full warmup and three consecutive
> requests on one DGX Spark. Offline prompt contexts and Stage1 latent/audio
> payloads match the reference. This used the existing runtime environments;
> a clean-environment installation has not yet been validated. See
> [validation](docs/validation.md).

## The recipe

| Component | Fixed configuration |
| --- | --- |
| Prompt encoding | Resident NVFP4 AWQ Qwen; fresh user-prompt features per request |
| Draft generation | MiniMax-H3, FastH3 VSA DataFree LoRA, W8A8 FP8 DiT; 4 updates at 672 × 384 × 124 |
| Draft attention | VSA, 90% video sparsity, 64-token blocks; optimized cuDNN BSA backend |
| Latent transfer | Learned H3 ×2 upscaler, then H3-to-LTX VAE latent adapter |
| Refinement context | Offline cached generic post-connector video **and** audio features |
| Refinement | BF16 LTX-2.5 dev + distilled LoRA 450 at strength 0.8; 3 joint audio/video updates |
| Refiner attention | Triton Sol-Attn, step thresholds 1 / 1.25 / 1.5; first layer dense |
| Output | Official Conv VideoVAE, H.264 video and original H3 audio as stereo AAC |

The implemented recipe is recorded in [configs/default.json](configs/default.json).
It is not a tuning interface: changing this record alone is rejected so that
the reported settings cannot disagree with the fixed implementation.
This entry exposes one recipe, not a collection of ablation switches. There
is no one-step refiner, TAE decoder, image anchor or intermediate video decode.

### Why the models can remain resident

The latent adapter replaces an intermediate H3 VAE decode and LTX VAE encode.
The H3 upscaler operates **before** the adapter, using its author's input and
output normalization. The transferred latent is already normalized for LTX;
do not normalize it again. Temporal conversion produces 17 latent frames and
the refiner consumes the first 16 for the 121-frame output.

The Stage1 latent supplies strong visual conditioning. Stage2 uses one generic
refinement prompt:

> 4K, refined, high quality, cinematic detail, clean textures, natural motion.

Its INT8 Gemma and INT8 dev-connector outputs are computed once, offline.
Inference retains only those small context tensors, so it never loads Gemma
or the connector. The original case-specific prompt is still freshly encoded
by Qwen. Cached generic conditioning is a quality/performance choice, not a
guarantee of unchanged identity in every generated scene.

Quantized Stage1 weights, latent-only transfer and removal of online Stage2
text models allow the required models to remain resident between requests.
The implementation also reuses compiled regions, fused VSA merging, direct
NHWC Conv upsampling and chunked video writing without changing the recipe.

## Run

Use Linux aarch64 on a single GB10/SM121 GPU with its full memory available.
The runtime uses separate Stage1, Stage2 and Qwen environments. Follow
[setup](docs/setup.md) to obtain the pinned dependencies and checkpoints,
prepare the generic context cache, and write `paths.json`.

From this directory:

```bash
python infer.py --paths paths.json \
  --prompt "A red fox walks through a sunlit meadow. A steady wide shot follows its movement. overall_soundscape: Soft wind and rustling grass." \
  --seed 42 --output-dir outputs/fox
```

For several prompts, keep the same model sessions alive:

```bash
python infer.py --paths paths.json \
  --prompts examples/prompts.jsonl --output-dir outputs/examples
```

Each JSONL row has `case_id`, `prompt` and `seed`. The output directory must
be new. Results are written after each completed request; no existing run is
overwritten.

```text
outputs/examples/
  results.json                 # Startup, per-request E2E and batch mean
  warmup/                      # Explicit full-chain warmup; excluded from mean
  <case_id>/
    qwen/                      # This request's prompt conditioning
    stage1/                    # Normalized H3 latent and original PCM
    stage2/                    # Final MP4 and component timings
  *-worker/                    # Session logs and local command receipts
```

These are local inference artifacts, not files to commit. The handoff uses a
CPU tensor file shared by the two local processes; it is not advertised as
zero-copy. Both stages and file transfer are included in request latency.

## Timing

One complete warmup precedes formal requests. Stage2's initial loading peak
is completed before permanent Qwen residency. Model loading, compilation,
warmup and initial Qwen residency are reported as startup, not hidden in a
formal-request average.

For every formal request, E2E is the continuous same-host monotonic interval
from request entry, **before fresh Qwen encoding**, to the completed, muxed
MP4. It includes scheduling and latent transfer. The first formal request is
retained. Stage timings may overlap and are **never summed to produce E2E**.

The [project page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/)
reports **56 s hot E2E** for the reference implementation. This is not a
throughput or cold-start claim; individual request latency varies.

## Code layout

```text
infer.py                     Batch/single-prompt entry
download_checkpoints.py      Pinned public weight downloads
prepare.py                   Resolve source, checkpoint and environment paths
configs/                    One recipe; source/checkpoint manifests
runtime/
  pipeline.py, worker.py     Persistent local sessions and request timing
  qwen.py, stage1.py         Resident quantized prompt and draft generation
  latent_transfer.py        Same-request H3 latent/PCM contract
  stage2.py                 Upscale, adapter, three-step refiner and Conv decode
  prompt_cache.py            Online feature-only cache loader
  cache_builder.py           Offline INT8 context generation
  *_ops/                    Required inference operators and model adapters
tests/                      CPU contracts; no model downloads
```

Sampling and attention reuse the existing Sana `super_acceleration` and
`techniques/sparse_backends/sol_attn` implementations. Run this package from
the Sana checkout; copying only this directory omits those shared modules.
FastVideo, ComfyUI, LTX and the H3 upscaler remain explicit pinned dependencies,
not private runtime overlays.
`prepare.py` applies the [required FastVideo loader patch](patches/README.md)
to its exact pinned source, or reuses an already matching patched checkout.

To run the CPU contracts without loading CUDA or model weights:

```bash
python -B -m unittest discover -s tests -v
```

## Acknowledgements

Built on MiniMax-H3, FastVideo/FastH3, LTX-2.5, Sol-Attn, ComfyUI and the
LBH-123-AI H3 latent upscaler. See [third-party notices](THIRD_PARTY_NOTICES.md)
for component provenance and terms. Model access permissions and licenses
remain separate from the license of this integration code.
