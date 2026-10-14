# StreamRender-H3: Code-driven Streaming Render Runtime

An end-to-end runtime connecting a controllable Three.js game engine to a
resident, multi-GPU H3 streaming renderer and browser video feedback.

[![Session setup and the interaction loop from player input to streaming video feedback](https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/assets/interaction_loop.webp)](https://nvlabs.github.io/Sana/Sol-Engine/StreamRender-H3/#2-3-a-real-time-interaction-loop-from-player-input-to-video-feedback)

## Runtime profile

The initial deployment configuration uses the **1008b step224** merged EMA
backbone, 50 DiT layers, 2 denoising evaluations, CFG=1, and the trial's
S=2/W=2/C=2 policy. Bootstrap is **7 scheduling units**, not 7 native latents.
Scheduling follows the H3 native temporal mapping rather than a hard-coded
number of RGB frames per chunk. There is no fixed final video duration.

- Engine input: 1344 x 768 RGB semantic frames, 24 fps.
- Reference encode: 832 x 480; nearest-neighbor resizing preserves label colors.
- Render output: 1344 x 768.
- Qwen: prompt and initial reference image, once per session.
- Encoder: native streaming TAE, causal per-frame dispatch with the H3
  17-frame/5-latent clock.
- Decoder: causal ViT24, two consecutive available latents per decoder
  publication, with four past latent positions in its attention cache.
  It never reads a later, ungenerated chunk.
- Static-context KV banks are separate for each diffusion timestep.
  Dynamic history/reference rows retain the checkpoint's recompute policy.
- Software optimizations include persistent final-layout KV, single-request
  cache reuse, exact guarded AdaLN projection reuse, attention metadata reuse,
  FA4 with the measured SplitKV profile, and the causal decoder CUDA Graph.
- No attention communication approximation, layer truncation, or extra
  quality-altering denoising shortcut is enabled.

Only one active interactive session is supported per worker allocation.
Stopping/restarting a session resets encoder memory, decoder memory, target
history, prefix snapshots/banks, and projection caches. The input queue is
bounded; the browser retries the same frame on backpressure.

## Layout

| Directory | Role |
| --- | --- |
| `coding/threejs-racing/` | Engine, controls, semantic rendering, browser UI, asset licenses |
| `render/pipeline.py` | Online encode / H3 step / decode, no reference MP4 |
| `render/optimizations.py` | Standalone projection and attention metadata reuse |
| `render/decoder/` | Fast causal decoder adapter and deployment implementation |
| `render/vendor/` | Local H3/common dependencies; no dependency on old experiment directories |
| `runtime/` | Session protocol, bounded bridge, resident distributed worker |
| `configs/` | Inference profile, asset manifest template, tested environment versions |
| `scripts/` | Preflight, local and Slurm startup |
| `benchmarks/browser_loop.mjs` | Actual engine-to-browser loop acceptance |
| `tests/` | CPU protocol/configuration tests |

## Assets and initialization

The public checkpoint bundle is
[Efficient-Large-Model/StreamRender-H3](https://huggingface.co/Efficient-Large-Model/StreamRender-H3).
Download it with the Hugging Face client in the configured Python environment:

```bash
hf download Efficient-Large-Model/StreamRender-H3 --local-dir /your/code/models/StreamRender-H3
python /your/code/models/StreamRender-H3/download_assets.py
```

The bundle is approximately 206.87 GB. The second command reconstructs the exact
merged backbone from its shards and verifies its SHA-256; it needs another
66.28 GB of disk space. Other weights need no conversion. The resulting
`assets.json` is the runtime weight manifest. All five model components,
including both Qwen storage formats, are supplied by this organization repository.

**60 curated examples** are included in [`examples/`](examples/README.md).
The default startup prompt is **B10_e-track-3_bluehour**. Semantic videos
and reference pictures are downloaded from the linked GitHub Release;
weights are downloaded from Hugging Face. To prepare the default example:

```bash
python scripts/prepare_example.py \
  --assets /your/code/models/StreamRender-H3/assets.json
GPUS=8 PYTHON_BIN=python bash scripts/launch_local.sh \
  --assets configs/example.local.json
```

See the example README for media download, alternative prompts and prerecorded
semantic-video playback. Existing local manifests retain their own prompts;
use the selection script to opt into the new default.

Weights are external assets, not Git contents. Copy
`configs/assets.example.json` to a local manifest and fill its paths. The
role of each path is specified in `asset_roles`. The required assets include:

1. Merged H3 EMA backbone.
2. Qwen model configuration, processor, tokenizer, and compatible DCP weights.
3. Native TAE weights.
4. Causal ViT24 decoder weights.
5. Audio VAE weights for the H3 model/codec contract.
6. A photorealistic reference image aligned with the engine opening scene,
   and its corresponding render prompt.

The causal decoder is built directly from its configuration and trained
weights. No original H3 video VAE checkpoint is loaded. Online reference
encoding uses TAE and online video decoding uses the causal ViT24 implementation.
Audio latents remain part of H3 inference, while this browser feedback path
publishes RGB video only.

The engine contains the opening-scene capture/export logic. Reference-image
creation/editing is a setup asset-production step: provide its finished image
in the manifest, then the worker encodes it once at session start. An image
generation API is not required during the interaction loop.


## Environment

Use Python 3.11 and a compatible CUDA PyTorch installation. The production
environment versions are recorded in `configs/environment-tested.json`;
the initial migration reuses that environment rather than replacing it.
The GPU runtime additionally needs FlashAttention including its FA4 CuTe
backend, CUTLASS DSL, Triton, MagiAttention/FlexAttention-compatible imports,
and the H3 codec dependencies. Merely installing `requirements.txt` is not
a complete CUDA-kernel environment installation.

Install Node 22 and frontend dependencies:

```bash
cd models/streamrender_h3
npm --prefix coding/threejs-racing ci
python scripts/preflight.py --assets configs/assets.local.json
python scripts/preflight.py --assets configs/assets.local.json --sha256
```

Do not silently use the top-level repository's different Torch/CUDA
environment for this profile. First validate the imported H3 modules and
FA4 availability in the selected environment.

## Local launch

On a GPU host with eight GPUs in one NVLink domain:

```bash
cd models/streamrender_h3
GPUS=8 PYTHON_BIN=python bash scripts/launch_local.sh \
  --assets configs/assets.local.json
```

The worker loads models, prepares initial conditioning, warms bootstrap and
continuation shapes, discards warmup stream state, then reports readiness.
Open `http://127.0.0.1:5191`, enable autopilot or drive with WASD, and click
the live streaming-render button. The engine and generated feedback appear
on the same page.

Transport-only diagnostic, explicitly **not H3 inference**:

```bash
BACKEND=passthrough bash scripts/launch_local.sh --assets configs/assets.local.json
```

## Inference with Reactor Runtime

The [Reactor integration](../../reactor/README.md) serves StreamRender-H3 with a
browser demo for recorded semantic videos, aligned reference images and prompts.
Click **Connect**, choose one of the 60 official input combinations or upload
your own inputs, then click **Start rendering**. Generated video streams one native
chunk at a time with 24 FPS playback; the prompt and reference stay fixed per render.

See the [Reactor guide](../../reactor/README.md) for weights, the YAML-defined
`reactor build` workflow, the required 2 GB of container shared memory, demo setup
and SSH/WebRTC access. The adapter and demo live in the repository's `reactor/`
directory; native inference code and launch paths are unchanged.

## Slurm launch

The provided cluster profile requests two nodes x four GPUs and checks their
NVL72 rack prefix before running. Supply the environment and deployment paths:

```bash
export RUNTIME_ROOT="$(pwd)/models/streamrender_h3"
export ASSETS_FILE="$RUNTIME_ROOT/configs/assets.local.json"
export PYTHON_BIN=/path/to/production/python
sbatch --output=/path/to/logs/runtime-%j.out \
       --error=/path/to/logs/runtime-%j.err \
       "$RUNTIME_ROOT/scripts/launch_slurm.sh"
```

The HTTP worker runs on allocation rank 0, bound to loopback. Run Vite on
that same GPU node to use its default loopback proxy:

```bash
npm --prefix /path/to/models/streamrender_h3/coding/threejs-racing run dev
```

From your laptop, where the cluster permits SSH to that node:

```bash
ssh -J hsg -L 5191:127.0.0.1:5191 GPU_NODE
```

Then open `http://127.0.0.1:5191` locally. Another site can use its equivalent
SSH tunnel or an authenticated reverse proxy. Both services default to
loopback; do not expose this development endpoint publicly.

## Provenance and licenses

The engine comes from `yitongl/CodeGameEngine`. Preserve the TORCS source
notices and Free Art License assets under `coding/threejs-racing/licenses/`.
The TAE adapter preserves its MIT notice and madebyollin/taehv provenance.
The H3/common snapshot comes from the previously validated Sana inference
workspace; original file-level notices are retained.

Deployment glue follows the repository's Apache-2.0 license. Model weights
retain their upstream permissions and licenses and are not redistributed here.

## Acknowledgements

We thank the following projects and their authors:

1. [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) for the base video
   generation model and native video/audio codecs. Our streaming renderer and
   distilled causal decoder build on these components.
2. [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for the text and visual
   conditioning model used to encode the prompt and initial reference image.
3. [TAE / TAeHV](https://github.com/madebyollin/taehv), by Ollin Boer Bohan,
   for the lightweight video autoencoder used by the reference-video encoder.

These acknowledgements do not replace the upstream licenses and attribution
notices, which remain applicable to the corresponding components.
