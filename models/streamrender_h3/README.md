# StreamRender-H3: Code-driven Streaming Render Runtime

An end-to-end runtime connecting a controllable Three.js game engine to a
resident, multi-GPU H3 streaming renderer and browser video feedback.

```text
player controls -> game engine -> semantic RGB frames
    -> streaming TAE encoder -> two full H3 denoising evaluations
    -> fast causal ViT24 decoder -> streaming RGB feedback in the browser
```

## Implementation and validation status

- The engine, online worker, ordered frame protocol, bounded queues, session
  reset, and browser feedback client are included here.
- CPU protocol/configuration tests and the frontend production build pass.
- The package-local H3 imports pass in the production Python environment.
- The actual same-rack 8-GB200 browser/H3 loop passed in job **7848438**.
  Final clean-exit verification **7848537** completed successfully (exit 0).
  Results and timing boundaries are recorded in `VALIDATION.md`.
- This runtime has its own entry point; it is not a SGLang `scripts/run.py`
  model adapter. It can be launched independently from this model folder.

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

Weights are external assets, not Git contents. Copy
`configs/assets.example.json` to a local manifest and fill its paths. The
role of each path is specified in `asset_roles`. The required assets include:

1. Merged H3 EMA backbone.
2. Qwen model configuration, processor, tokenizer, and compatible DCP weights.
3. Native TAE weights.
4. Causal ViT24 decoder weights.
5. Original H3 VAE weights used when constructing the decoder architecture.
6. Audio VAE weights for the H3 model/codec contract.
7. A photorealistic reference image aligned with the engine opening scene,
   and its corresponding render prompt.

The original H3 VAE is loaded for decoder construction; online reference
encoding uses TAE and online video decoding uses the causal ViT24 implementation.
Audio latents remain part of H3 inference, while this browser feedback path
publishes RGB video only.

The engine contains the opening-scene capture/export logic. Reference-image
creation/editing is a setup asset-production step: provide its finished image
in the manifest, then the worker encodes it once at session start. An image
generation API is not required during the interaction loop.

The current step224 merged backbone has SHA256:

```text
56c3ee3270db40354ef70fcbea16248809914cfa8f51acb57dae7d30fa4bb597
```

The runtime accepts paths relative to the manifest's directory as well as
absolute paths. `*.local.json`, weights, outputs and caches are Git-ignored.

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

## Validation and timing boundaries

```bash
PYTHONPATH="$PWD/models/streamrender_h3" python -m unittest discover \
  -s models/streamrender_h3/tests -v
```

Set `VALIDATE_BROWSER=1` when submitting the Slurm launcher to run a headless
browser producing real game-engine frames and consuming rendered feedback.
Provide `NODE_BIN` and, when necessary, `CHROMIUM_BIN`.
This validation run exits after completion. Normal workers stay resident.

Per-session output is written under `outputs/SESSION_ID/`:

- `controls.jsonl`: frame-indexed actions, simulation state and chunk pairing.
- `timings.jsonl`: chunk encode/render/decode, GPU RGB readiness, wall time
  and publication intervals.

There are no stage-level forced synchronizations; readiness is synchronized
once at the chunk boundary. GPU RGB-ready timing excludes input PNG transport,
PNG decode, H2D/resize, JPEG encoding and browser/network display.
Publication intervals include waiting and host-side work between chunks;
they are a different metric. The browser loop includes those transport
operations and must be evaluated separately.

An RGB-ready speed quoted for an older checkpoint/runner is not a performance
claim for this newly integrated worker.

## Provenance and licenses

The engine comes from `yitongl/CodeGameEngine`. Preserve the TORCS source
notices and Free Art License assets under `coding/threejs-racing/licenses/`.
The TAE adapter preserves its MIT notice and madebyollin/taehv provenance.
The H3/common snapshot comes from the previously validated Sana inference
workspace; original file-level notices are retained.

Deployment glue follows the repository's Apache-2.0 license. Model weights
retain their upstream permissions and licenses and are not redistributed here.
