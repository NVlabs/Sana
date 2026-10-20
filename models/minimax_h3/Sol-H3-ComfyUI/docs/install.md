# Installation

Sol-H3 in ComfyUI has two parts:

1. **The plugin** (this directory) — linked into `ComfyUI/custom_nodes/`.
   It uses only Python's standard library and never imports PyTorch in the
   ComfyUI process.
2. **The H3 runtime** — a separate Python environment that runs the model on
   the GPU. The plugin starts it as a worker process and talks to it over a
   small JSON channel.

## 1. Prepare the H3 runtime

The runtime is the single-GPU RTX 5090 port of the Sol-H3 engine from
[NVlabs/Sana `sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine). It needs:

| Path key | What it is |
|---|---|
| `stage1_python`, `qwen_python` | Interpreter(s) of the prepared H3 environment (PyTorch, Triton, FastVideo deps) |
| `fastvideo_root` | FastVideo checkout pinned in `sol_h3_engine/runtimes/*/configs/dependencies.json` |
| `fa4_root` | FlashAttention 4 runtime for SM120 |
| `ltx_root` | LTX-2 checkout, used only for its audio/video MP4 writer |
| `comfy_root` | ComfyUI checkout used by the NVFP4 Qwen loader |
| `h3_model` | `MiniMaxAI/MiniMax-H3` (Diffusers layout: `transformer/`, `vae/`, `audio_vae/`, …) |
| `qwen_checkpoint` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `stage1_dependencies` *(optional)* | Extra package overlay, e.g. the CUTLASS DSL packages |

Write these to a `paths.json`. Keep virtual-environment interpreter paths as they
are (do not resolve the `venv/bin/python` symlink). Upstream source revisions
(FastVideo, FlashAttention, LTX-2) are recorded in
`sol_h3_engine/runtimes/h3_fp8/configs/dependencies.json`; that file was captured
from the Spark port and is a provenance record, not a tested RTX 5090 lockfile.
The validated 5090 environment used Python 3.12.3, PyTorch 2.12.0+cu130 and
Triton 3.7.0 ([validation](validation.md)).

You also need the dense-datafree FastH3 adapter (`adapter_model.safetensors`).
The original-H3 baseline profile does not use it.

## 2. Install the plugin and write a host configuration

```bash
git clone -b sol-engine https://github.com/NVlabs/Sana.git
cd Sana/models/minimax_h3/Sol-H3-ComfyUI
ln -s "$PWD" /path/to/ComfyUI/custom_nodes/Sol-H3-ComfyUI
python3 scripts/configure.py \
  --reuse-paths /path/to/paths.json \
  --dense-lora /path/to/dense-datafree/adapter_model.safetensors \
  --output-dir ~/sol-h3-settings --gpu 0
```

`configure.py` checks every H3 checkpoint shard (broken links are rejected),
then writes `paths.json` and `config.json` to the output directory and prints
the `SOL_ENGINE_CONFIG=...` line to use. It creates these profiles:

| Profile | What runs |
|---|---|
| `H3 FP8 — single RTX 5090` | Sol-H3: 4 updates, W8A8 FP8 DiT, original H3 VAE |
| `H3 BF16 — single RTX 5090` | Sol-H3: 4 updates, BF16 DiT, original H3 VAE |
| `Original H3 50-step baseline — RTX 5090` | MiniMax-H3 without the adapter, official 50-point schedule (49 forwards), BF16 |
| `H3 FP8 + noncausal VAE — RTX 5090` | Sol-H3 FP8 with the distilled noncausal decoder *(only with the options below)* |
| `Sol-H3 HyperFlow · 8 steps — RTX 5090` | HyperFlow 8-step LoRA, BF16 *(with `--hyperflow-weights` and `--hyperflow-deps`)* |
| `Sol-H3 one stage — RTX 5090` | 8-step LoRA, FP8 + AdaLN table + Sol-Attn + compile + noncausal VAE; text, first frame, reference; 768p/480p *(see below)* |
| `Sol-H3 two stage — RTX 5090` | H3 draft + LTX refine from NVlabs/Sana *(see below)* |

All profiles share the NVFP4 Qwen encoder, CPU offload and FlashAttention 4.

### Optional noncausal VAE

For the FastH3 noncausal profile, add all three options to `configure.py`:

```bash
  --fast-vae-code /path/to/fast-vae-code \
  --fast-vae-weights /path/to/fast_h3_vae_noncausal_24l.safetensors \
  --fast-vae-diffusers-source /path/to/fast-vae-diffusers
```

`fast-vae-code` contains the color-corrected `spark_decoder.py`; the Diffusers
source overlay carries QuACK 0.5.3 (newer QuACK compiles asynchronously and is
incompatible with FastVideo's daemon workers). The overlay is visible only
inside the H3 worker. Decoder weights move to CPU between requests to fit 32 GB;
upload, decode, RGB conversion and offload all count toward the reported time.

### One stage

`Sol-H3 one stage — RTX 5090` is listed first (the Studio default) when these options are
given together with `--hyperflow-weights` and `--hyperflow-deps`; the last two are optional:

```bash
  --hyperflow-fp8-dir /path/to/transformer_hf8 \
  --hyperflow-adaln-table /path/to/transformer_hf8/adaln_table.pt \
  --hyperflow-fast-vae-weights /path/to/fast_h3_vae_noncausal_24l.safetensors \
  --fast-vae-code /path/to/fast-vae-code
```

It runs the HyperFlow DiT with the LoRA merged into per-tensor FP8 weights, an exact
AdaLN lookup table instead of the 26 GB `adaln_proj` weights, Sol-Attn sparse attention
(the `sol_attn` package in the HyperFlow overlay) with SageAttention 2.2 for the dense
steps, regional `torch.compile` of each block, a resident Qwen worker parked in host
RAM between requests, and — when given — the distilled noncausal decoder
[Sol-H3-VAE](https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE): pass its `fast_h3_vae_noncausal_24l.safetensors` and its
`code/` directory as `--fast-vae-code`. Without it the original H3 VAE decodes, about 10 s more
per 768p clip.
SageAttention must be built for SM120 in the runtime environment, and FlashAttention 4
must be importable by the worker (for example `PYTHONPATH=/path/to/fa4-runtime`).
Set `TORCHINDUCTOR_CACHE_DIR` to a persistent directory so later sessions reuse the
compiled blocks.

Build the FP8 checkpoint and the AdaLN table once, inside the runtime environment
on the GPU host, with `SOL_ENGINE_CONFIG` pointing at a configuration that has the BF16
`Sol-H3 HyperFlow · 8 steps — RTX 5090` profile (`configure.py` with `--hyperflow-weights`
and `--hyperflow-deps` only):

```bash
P='Sol-H3 HyperFlow · 8 steps — RTX 5090'
# 1. Merge the HyperFlow LoRA and quantise the 300 block linears to FP8 (shard by shard).
python scripts/hyperflow/convert_fp8.py /path/to/MiniMax-H3/transformer \
  /path/to/minimax_h3_hyperflow_8step_v1.0.safetensors /path/to/transformer_hf8
# 2. One conditioning payload per task (any prompt: the table depends only on the schedule).
python scripts/hyperflow/save_payload.py "$P" t2va.payload.pt "A bird over a lake."
python scripts/hyperflow/save_payload.py "$P" fl2va.payload.pt "The scene comes alive." frame.png
# 3. Record every forward's (timestep, endpoint) set and compute the table at the live batch
#    size, one task per run (host RAM), then merge.
python scripts/hyperflow/build_adaln_table.py "$P" /path/to/transformer_hf8 t2va.pt t2va.payload.pt
python scripts/hyperflow/build_adaln_table.py "$P" /path/to/transformer_hf8 fl2va.pt fl2va.payload.pt
python scripts/hyperflow/merge_adaln_tables.py /path/to/transformer_hf8/adaln_table.pt t2va.pt fl2va.pt
```

Optionally rerun step 1 into the same directory with `--drop-adaln`: it rewrites the shards
without the 26 GB of `adaln_proj` weights the table replaces (the table file is kept).

`build_adaln_table.py` checks the table against the official AdaLN module bit for bit.
A request whose schedule is not in the table fails instead of running with wrong values.
`scripts/hyperflow/e2e_comfy.py` times the same graph the Studio submits through a
running ComfyUI's `/prompt` API.

Reference images use MiniMax-H3's separate `transformer_ref` checkpoint. Convert it with the
same LoRA (its metadata lists `transformer_ref` as compatible); the script streams the 66 GB
checkpoint shard by shard, so it never sits on disk, and writes the AdaLN table next to it:

```bash
python scripts/hyperflow/convert_ref.py \
  https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/transformer_ref \
  /path/to/minimax_h3_hyperflow_8step_v1.0.safetensors \
  /path/to/transformer_hf8/adaln_table.pt /path/to/transformer_ref_hf8
```

Then add `--hyperflow-ref-fp8-dir /path/to/transformer_ref_hf8`. Only one FP8 DiT is kept in
host RAM: the first reference request swaps `transformer_ref` in (and a later text or
first-frame request swaps it back), which costs one checkpoint load.

### Two stage

Two stage is NVlabs/Sana `sol-engine`'s `models/minimax_h3/Sol-H3-RTX5090` (DGX Spark:
`Sol-H3-Spark`), used unchanged. Prepare it with its own `docs/setup.md` (`prepare.py`
writes the paths file), then:

```bash
  --two-stage-package /path/to/Sana/models/minimax_h3/Sol-H3-RTX5090 \
  --two-stage-paths /path/to/paths-5090.json
```

Add `--device spark` on a DGX Spark (no CPU offload). On an RTX 5090 the stage1 (H3) and
stage2 (LTX) workers stay loaded, about 72 GB of host RAM; the first session compiles LTX
once (about 15 minutes) into `TORCHINDUCTOR_CACHE_DIR`.

### Gallery

`--gallery DIR` shows showcase clips in the Studio's Gallery: `NAME.mp4` with `NAME.json`
(`{title, title_zh, prompt, mode, image, pipeline}`) and, for first-frame clips, the image.
`python3 scripts/fetch_assets.py gallery` downloads the clips shown in the film into
`assets/gallery/`; `scripts/build_gallery.py MANIFEST.json DIR` assembles a folder from your own.

### All together

The configuration the Studio was validated with: one stage (with reference images), two stage
and the gallery. Leave out the options for parts you did not build.

```bash
python3 scripts/fetch_assets.py gallery
python3 scripts/configure.py \
  --reuse-paths /path/to/paths.json \
  --dense-lora /path/to/dense-datafree/adapter_model.safetensors \
  --hyperflow-weights /path/to/minimax_h3_hyperflow_8step_v1.0.safetensors \
  --hyperflow-deps /path/to/hyperflow-overlay \
  --hyperflow-fp8-dir /path/to/transformer_hf8 \
  --hyperflow-adaln-table /path/to/transformer_hf8/adaln_table.pt \
  --hyperflow-ref-fp8-dir /path/to/transformer_ref_hf8 \
  --two-stage-package "$PWD/../Sol-H3-RTX5090" \
  --two-stage-paths /path/to/paths-5090.json \
  --gallery "$PWD/assets/gallery" \
  --output-dir ~/sol-h3-settings --gpu 0
```

With [Sol-H3-VAE](https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE), add `--hyperflow-fast-vae-weights` and `--fast-vae-code`.

## 3. Start ComfyUI

```bash
export SOL_ENGINE_CONFIG=~/sol-h3-settings/config.json
python main.py            # add --cpu if this ComfyUI should not use the GPU itself
```

The H3 worker uses the GPU named in its profile regardless of ComfyUI's own
device. ComfyUI must be recent enough to provide `VIDEO`, `SaveVideo` and
`comfy_api.latest.InputImpl`.

The Studio opens with ComfyUI; **Workflow → Browse Templates → Sol-H3-ComfyUI** also lists
`Sol-H3-Studio` and the plain node graph `t2v`.

### Remote GPU

Bind ComfyUI to loopback on the GPU machine and forward the port:

```bash
python main.py --listen 127.0.0.1 --port 8188          # on the GPU host
ssh -N -L 8188:127.0.0.1:8188 user@gpu-host           # on your computer
```

Host paths stay in `config.json`, never in shared workflows.

## Command-line runner

The same bridge works without the UI, which is useful for benchmarks:

```bash
SOL_ENGINE_CONFIG=~/sol-h3-settings/config.json python3 scripts/run_local.py \
  --profile 'H3 FP8 — single RTX 5090' \
  --prompt 'An eagle glides over a mountain lake. overall_soundscape: Soft wind.' \
  --seed 42 --output ./output --repeat 2
```

Each run writes the MP4, per-phase timings, the engine report and the worker log
to `output/sol_h3/<session>/`.

## Timing fields

| Field | Meaning |
|---|---|
| `e2e_s` | Request entry → completed audio/video MP4: fresh Qwen conditioning, denoising, decode and mux |
| `startup_and_warmup_s` | First-use model load, compilation and one warmup request (cold session only) |
| `node_wall_s` | Wall time inside the ComfyUI node, including startup if the session was cold |
| `h3_phases_s` | Denoise, video decode, audio decode and mux |

ComfyUI's own `Save Video` copy runs after these clocks stop.

## Troubleshooting

- **`Configure SOL_ENGINE_CONFIG first` in the profile list** — the variable is
  not set in the shell that started ComfyUI.
- **`Missing H3 checkpoint file`** — a transformer/VAE shard is absent or a
  broken link; re-download that shard.
- **CUDA out of memory** — another process is using the card. The worker needs
  an otherwise free 32 GB GPU.
- **Worker exited** — the error and the full worker log path are shown in the
  ComfyUI error; the log is in `output/sol_h3/<session>/worker.log`.
