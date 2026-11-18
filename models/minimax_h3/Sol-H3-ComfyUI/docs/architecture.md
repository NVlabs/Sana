# Architecture

The layout follows [FreeVideo](https://github.com/FlashML-org/FreeVideo/tree/7e506f4b1d46806a722adf08861009b4eb46f08e):
a thin root ComfyUI entry, an engine package, browser extensions, example
workflows, scripts and documentation. It keeps this project's existing H3
implementation and does not copy FreeVideo's VDN engine or installer.

```text
__init__.py                       ComfyUI node registration and web directory
sol_h3_engine/
  comfy_nodes.py                  Generate / Release GPU nodes
  bridge.py                       Serialized, cancellable worker lifecycle
  config.py                       Host-local profiles
  worker.py                       JSON request loop in the H3 environment
  fast_vae_adapter.py             Optional external noncausal decoder
  runtimes/h3_{fp8,bf16}/         Verified single-stage Sol-H3 runtime snapshots
  runtimes/h3_baseline/           Original-H3 50-point comparison runtime
example_workflows/                UI graphs (exposed as ComfyUI templates)
web/timing.js                     Live elapsed time and final timing display
scripts/                          Configure existing assets / run from CLI
tests/                            CPU subprocess and integration-path checks
docs/                             Architecture, measurements and demo assets
.github/workflows/                CPU checks on pushes and pull requests
```

## Execution

ComfyUI calls `SolH3Generate`, which asks the bridge for a persistent worker.
The bridge uses the Python interpreter selected in the host configuration.
The worker loads the requested runtime and owns its H3 and Qwen processes.
Completed MP4 paths return as native ComfyUI `VIDEO` values; the standard
SaveVideo node handles preview and saving. A separate text output holds timing JSON.

The worker environment isolates H3's CUDA allocator from ComfyUI, limits CPU
thread counts, and exposes optional VAE dependencies only to that worker.
Changing profiles, releasing the model or cancelling a request closes owned
workers. A shared GPU still needs sufficient free memory before generation.

## Runtime boundaries

The preserved snapshots implement four H3 updates with dense-datafree LoRA and
CPU offload on one 5090. Their numerical code and recipes are not altered by
this repository reorganization. `h3_baseline` is copied from `h3_bf16` and changes
only what defines the original model: no adapter, the official 50-point
MiniMaxH3Scheduler grid (49 forwards), and no four-update AdaLN lookup. Qwen,
VAE, offload, FlashAttention 4 and regional compile stay identical. `runtimes/*/SOURCE.json` inside the engine
records original source hashes and the earlier optional-decoder changes.

Fast VAE code and weights remain external. The adapter provides normalized
BTCHW latents and applies the delivered RGB conversion. Its per-request
upload, decode, conversion and offload all count toward measured request time.

The ComfyUI host does not need to import the model frameworks to register the
nodes. The runtime needs the existing patched dependencies and model assets;
this repository's CPU checks do not establish a clean GPU installation.
