# Sol-H3 on NVIDIA DGX Spark

A two-stage video-and-audio pipeline for one NVIDIA DGX Spark.
Generate a 384p draft, upscale and transfer its latent representation, then
refine and decode a **1344 × 768, 121-frame video at 24 FPS** with original H3 audio.

[Project page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/) ·
[Environment and weights](docs/setup.md) · [Component terms](THIRD_PARTY_NOTICES.md)

> **T2VA validation:** this entry completed a full warmup and three consecutive
> text-only requests on one DGX Spark. Offline prompt contexts and Stage1 latent/audio
> payloads match the reference. This used the existing runtime environments;
> a clean-environment installation has not yet been validated. See
> [validation](docs/validation.md).

FL2VA has also completed first-only, last-only and first+last requests on
Spark. Ref2VA has completed image-reference and image-plus-audio requests
with its dedicated checkpoints, plus two three-image cases using the reference
budgets and Stage1 attention policies described below. This is bounded
functional coverage, not validation of every combination or reference type;
see [validation](docs/validation.md).

## The default T2VA recipe

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
Changing this record alone is rejected so that the reported settings cannot
disagree with the implementation. Ref2VA exposes only the two supported
reference-budget and draft-attention selections documented below; their
resolved settings are recorded in each run's `results.json`.
Task selection changes native input conditioning and, for Ref2VA, its required
model partition and LoRA. There is no one-step refiner, TAE decoder or
intermediate video decode.

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

### First/last-frame generation (FL2VA)

Prepare the additional input-encoding weights with `--task fl2va` as described
in [setup](docs/setup.md). Supply a first frame, a last frame, or both:

```bash
python infer.py --paths paths-fl2va.json --task fl2va \
  --first-frame first.png --last-frame last.png \
  --prompt "integrated_multimodal_description: A red fox walks from the woodland edge into the clearing. The camera stays wide. overall_soundscape: Soft wind and leaves rustling." \
  --seed 42 --output-dir outputs/fox-guided
```

Inputs use native H3 image preprocessing, Qwen visual conditioning and H3 VAE
encoding. FL2VA uses the same H3 model family as T2VA; the Spark adaptation
retains its four-update VSA draft. The VSA LoRA's upstream release claims T2VA,
so conditioned-image quality is a separate validation target, not an upstream
guarantee.

H3's first/last frames are **conditioning tokens**, not hard pixel replacement.
The downstream upscaler, adapter and three-step refinement remain unchanged.
The 124-frame draft is still converted to a 121-frame final video, so neither
pixel-exact endpoint reproduction nor unchanged identity is guaranteed.

### Reference generation (Ref2VA)

Use `--task ref2va` when downloading, preparing paths and running inference.
This selects **`transformer_ref` and the LightX2V Ref2VA four-step LoRA**, with
W8A8 FP8 DiT. The default is a **Stage1-sized reference budget and dense FA4
draft attention**. The T2VA VSA adapter is not reused for a different checkpoint
family. Stage2 stays on the same three-step Sol recipe under either selection.

```bash
python infer.py --paths paths-ref2va.json --task ref2va \
  --reference image:subject.png --reference audio:voice.wav \
  --prompt "subject_definitions: <Subject 1> is the person in <Picture 1>. <Audio 1> provides the voice reference. detailed_description: <Subject 1> faces the camera and says, <d>[English] Welcome to the garden.</d> overall_soundscape: Quiet outdoor ambience." \
  --seed 42 --output-dir outputs/reference
```

References retain command-line order. Supported input types are `image`,
`video` and `audio`; at least one image or video is required. Native limits
are nine images, three videos, three audio files and twelve references total.
Reference images use the selected per-image area budget below. Video
references use native frame sampling; embedded video audio contributes an
audio reference before that video's visual item. Keep the numbered media
references in the prompt aligned with this ordering. See the
[official H3 input and prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3).

The following options apply **only to Ref2VA** and remain fixed for the entire
batch or Python `Pipeline` session, not per JSONL row:

| Option | Default | Meaning |
| --- | --- | --- |
| `--ref-image-match {stage1,stage2}` | `stage1` | Per-image area budget from the 672 × 384 draft or 1344 × 768 final output, respectively |
| `--ref-stage1-attn {dense,sol}` | `dense` | Dense FA4 or the supported four-step Sol policy for the H3 draft only |

Each reference image is resized independently with scale
`min(1, sqrt(budget / (width * height)))`, then its dimensions are rounded to
the nearest multiple of 32. This preserves the image's aspect ratio up to
alignment, without cropping or intentional pre-alignment enlargement. It
does **not** force every image to 16:9; rounding can slightly exceed the
nominal area budget. Qwen and the H3 VAE receive matching prepared images.
These options do not change the draft/output dimensions or video-reference
sampling. Smaller references may lose fine identity or accessory details.

For example, select the larger reference budget with Sol draft attention:

```bash
python infer.py --paths paths-ref2va.json --task ref2va \
  --ref-image-match stage2 --ref-stage1-attn sol \
  --reference image:subject.png \
  --prompt "The subject in <Picture 1> turns toward the camera." \
  --seed 42 --output-dir outputs/reference-sol
```

The equivalent Python session selection is:

```python
from runtime.config import load_paths
from runtime.pipeline import Pipeline

pipeline = Pipeline(
    load_paths("paths-ref2va.json", task="ref2va"), "outputs/reference-sol",
    task="ref2va", ref_image_match="stage2", ref_stage1_attn="sol",
)
```

The Stage1 `sol` policy keeps the first of four updates fully dense. In the
remaining updates, body layer 0 stays dense and the other 49 use Sol with
thresholds 1 / 1.25 / 1.5. Only text and audio are forced sinks: neither H3
reference-image tokens nor Qwen visual tokens are sinks. The text/audio query
subset is recomputed with dense FA4. This is distinct from **Stage2**, which
always uses its original **three-update Sol** policy, regardless of either
Ref2VA option. Neither selection guarantees reference-identity preservation.

For batches, add `task` and either `first_frame` / `last_frame` or an ordered
`references` list to each row. Paths are relative to the JSONL file:

```json
{"case_id":"guided","task":"fl2va","prompt":"A fox walks into the clearing.","seed":42,"first_frame":"inputs/fox.png"}
```

```json
{"case_id":"reference","task":"ref2va","prompt":"The subject in <Picture 1> turns toward the camera.","seed":42,"references":[{"type":"image","path":"inputs/subject.png"}]}
```

Use one task per batch. Switching task families uses a separate invocation;
the entry does not swap model partitions mid-request. Input-conditioned tasks
also need the H3 VAE **encoder**, although H3 video decoding remains bypassed.
Their additional reference memory is not covered by the T2VA residency result.

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

One full-chain warmup uses the first case's input shape; it does not prewarm
every later reference layout. Stage2's initial loading peak is completed
before permanent Qwen residency. Work performed during this startup sequence
is reported separately from formal requests. The permanent Qwen worker is
then loaded, but its **first encoding is part of the first formal request**;
that request is not claimed to be fully warm. Later shapes may also incur
first-use work inside their measured requests.

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
