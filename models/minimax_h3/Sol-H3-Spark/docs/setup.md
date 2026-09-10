# Prepare the Spark runtime

Use Linux aarch64 on a GB10/SM121 Spark. Preparation reuses separate Qwen,
Stage1 and Stage2 environments. `requirements/*-observed.txt` records versions
read from those environments; these files are inventories, not resolved pip
lockfiles. A clean environment installation has not been validated.

Keep model preparation and inference outside scheduler login nodes. The
filesystem-only commands below import no GPU framework. Model imports and
cache generation belong on the GPU host or in a compute allocation.

## Inspect and download weights

Run these commands from the `Sol-H3-Spark` directory:

```bash
python download_checkpoints.py --plan
python -m pip install -r requirements/download.txt
hf auth login
python download_checkpoints.py --output-dir checkpoints --include-offline
```

The default task is `t2va`. Pass the same `--task t2va|fl2va|ref2va` to the
downloader, preparation command and inference entry. The downloader uses
pinned Hugging Face revisions and selects these Stage1 resources:

| Task | Partition inside the shared `h3_model` root | Input video VAE | Stage1 adapter path key |
| --- | --- | --- | --- |
| `t2va` | `transformer/` | Not loaded | `vsa_lora` |
| `fl2va` | `transformer/` | Native H3 `vae/` | `vsa_lora` |
| `ref2va` | `transformer_ref/` | Native H3 `vae/` | `ref2va_lora` |

For example, add Ref2VA assets to the existing shared checkpoint directory:

```bash
python download_checkpoints.py --task ref2va --output-dir checkpoints
python prepare.py --task ref2va --plan
```

Common assets are reused in place; the Ref2VA download does not fetch another
copy of the FL2VA transformer. Both visual-input tasks require all three native
H3 VAE shards for input encoding, even though final video decoding still uses
the LTX Conv VAE. All tasks fetch H3 audio weights, NVFP4 Qwen, H3 upscaler,
H3-to-LTX adapter, LTX dev BF16 transformer, distilled LoRA 450, Conv VideoVAE and AudioVAE.

FL2VA reuses the VSA DataFree LoRA with native keyframe conditioning. The
[pinned FastVideo LoRA card](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/blob/f509e629374cac104e7f62daecce6d1488a3041d/README.md)
only declares text-to-audio-video generation; this FL2VA adaptation is not an
upstream-guaranteed checkpoint capability. Functional Spark checks are recorded
separately in [validation](validation.md). Ref2VA instead uses
the [task-specific LightX2V four-step v0.1 LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/2f015e66b37c585cea9dc4ae6f1850ea8788e742/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors)
with native dense attention, not VSA. Its expected SHA-256 is
`9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`.
Its checkpoint metadata specifies alpha 8 with rank 128, so the LoRA delta
multiplier is `8 / 128 = 0.0625` at unit adapter strength.

`--include-offline` adds INT8 Gemma and the INT8 dev transformer used only as
the context connector source. Omit that flag when reusing a valid prompt cache.
The Gemma tokenizer and processor assets are embedded in the Gemma checkpoint;
`gemma_tokenizer` therefore points to the same file as `offline_gemma`.

LTX-2.5 requires accepted access terms for the caller's Hugging Face account.
Authentication uses the normal library configuration; no token belongs in
the source tree, command arguments or generated path manifest.

`--verify-sha256` additionally reads complete single-file weights and compares
the recorded hashes. Download completion otherwise checks their byte counts;
it does not claim a complete SHA scan. Downloads never include native BF16
Qwen weights. The existing NVFP4 Qwen checkpoint includes its visual encoder;
visual references do not need a second Qwen checkpoint. No inference-time
downloads are performed.

The [public H3-to-LTX adapter](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter/tree/cfcd8a7cc36c135142287584728f7fe362df0867)
downloads as a directory containing `config.json` and `model.safetensors`.
Its expected model SHA-256 is
`170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac`.
An existing matching adapter can also be supplied with `--adapter-dir`.

Supply the exact fixed-prompt cache through `--prompt-cache`. It contains
post-connector video and audio contexts for:

```text
4K, refined, high quality, cinematic detail, clean textures, natural motion.
```

The runtime checks the prompt, INT8 encoder/connector recipe, tensor manifests
and values when loading it. An arbitrary text embedding or BF16-generated
cache does not satisfy this recipe. Cache preparation runs once before the
resident sessions and is outside request timing.

## Source and environment layout

`configs/dependencies.json` pins FastVideo, FA4, ComfyUI, LTX and the author's
H3 upscaler. `prepare.py --fetch-sources` fetches missing checkouts and refuses
to reset an existing checkout to a different revision. Existing checkouts can
be passed using a flat JSON file via `--paths`; key names are in the manifest.
It does not install or rebuild an environment.

Stage1 imports pinned FastVideo **with the declared Spark loader patch** and
its matching `fastvideo-kernel` 0.3.5 source. The Apache-2.0 patch in
`patches/fastvideo-spark-loader.patch` aligns dense LoRA deltas with the
destination device, uses concrete transformer mappings during lazy loading,
and avoids retaining an unused CPU master for constructor-merged inference.
`prepare.py` checks the patch SHA and all three file hashes before applying it
to a clean pinned checkout. It reuses a checkout whose three files exactly
match the patched hashes. Partial patches or other tracked modifications are
rejected without resetting user files. The unpatched commit alone does not
provide this release's loader behavior.

Its cuDNN BSA environment contains frontend 1.27.0 and CUDA13 binary
9.20.0.48. Dense text attention uses FA4 revision
`14c377950125c70b7a9dabf9c561fca53715ac7d`. That source declares CuTe
4.6.0.dev0, while the base BSA environment has CuTe 4.8.0.dev0. The measured
layout exposes FA4 and its matching dependencies from a separate import
directory; do not combine these constraints into one environment upgrade.

`fa4_source_root` names the pinned complete source checkout. Preparation
creates `fa4_root` as a separate namespace directory with only a
`flash_attn/cute` symlink and no `flash_attn/__init__.py`, matching the FA4-only
installation. An existing verified `fa4_root` may be reused without cloning
another source tree. Set `fa4_dependencies` to a directory containing the
packages in `requirements/fa4-isolated.txt`. The Stage1 runtime
prepends this directory and its `nvidia_cutlass_dsl/python_packages` directory.
The existing environment remains the verified route. Building a fresh
compatible isolated target remains to be validated on Spark.

Stage2 uses LTX source revision
`d151147788a9284cca791edc6ce898007e727fe6`, with `ltx-core`, `ltx-pipelines`
and `ltx-kernels` available to the Stage2 interpreter. The native
`all2all_cpp` extension from `ltx-kernels` must already be installed even on
one GPU because upstream imports it before the singleton path is selected.
`ltx_root` supplies the source paths; a source checkout alone does not supply
this compiled extension. Triton Sol attention comes from this Sana checkout.

The recorded Stage1 and Stage2 inventories include different Torch and
torchaudio release numbers. A fresh pip resolver may not reproduce that
combination. No installer here silently changes the existing Torch stack.

## Qwen container interpreter

The separate Qwen environment uses ComfyUI v0.30.0, comfy-kitchen 0.2.26 and
comfy-aimdo 0.4.11. `Dockerfile.qwen` pins the NGC base image digest and exact
ComfyUI commit while preserving NGC's Torch packages. This build recipe is
provided for review; a fresh image build has not been validated. Visual input
preprocessing also uses NumPy, Pillow and PyAV from that environment.

```bash
docker build -f Dockerfile.qwen -t sol-h3-spark-qwen .
export SOL_H3_SPARK_QWEN_IMAGE=sol-h3-spark-qwen
export SOL_H3_SPARK_RUNTIME_ROOT=/absolute/dedicated/spark-runtime
```

`qwen_python.sh` mounts the dedicated runtime root for request outputs and
mounts this package read-only at the same absolute path. It passes standard
input/output to the persistent Qwen worker. Set
`SOL_H3_SPARK_QWEN_WEIGHTS_ROOT` and `SOL_H3_SPARK_COMFY_ROOT` to mount existing
weight and Comfy source directories read-only at their original paths.
Use an existing verified image digest instead of the build tag
when one is available. The wrapper disables container network access during
inference and does not need Hugging Face credentials. `comfy_root` must name
the same mounted checkout in the host and container; its installed dependencies
come from the image. Symlink targets outside the mounted root are not visible.
Place first/last frames and reference images, videos or audio under
`SOL_H3_SPARK_RUNTIME_ROOT` or the explicitly mounted read-only
`SOL_H3_SPARK_QWEN_WEIGHTS_ROOT`. Arbitrary host paths outside these mounts
are not visible to the Qwen worker; use the same absolute input path in both
the host request and container.

The wrapper pins GPU 0 and refuses to pull an image during inference. Its
default entrypoint is `/usr/bin/python3`; set `SOL_H3_SPARK_QWEN_PYTHON` only
when the verified image uses a different interpreter.

The wrapper is passed as `qwen_python`, just as a native interpreter is passed
for the other two processes. The caller must provide writable output space in
the mounted directory; the container runs as the caller's UID/GID.

## Write the path manifest

Inspect the assembled paths before making downloads or source changes:

```bash
python prepare.py --plan
```

Then reuse the existing environments and required adapter/cache:

```bash
python prepare.py --fetch-sources \
  --python-stage1 /absolute/stage1-env/bin/python \
  --python-stage2 /absolute/stage2-env/bin/python \
  --python-qwen "$PWD/qwen_python.sh" \
  --prompt-cache /absolute/fixed-prompt.pt \
  --paths existing-paths.json \
  --output paths.json
```

`--paths` is optional and accepts existing model/source locations and
`fa4_dependencies`. It is a flat JSON object with string path values.
Preparation checks the selected H3 partition and any required input-VAE shard
index without reading tensor data. It refuses missing required inputs, mismatched source commits and
existing output-manifest files. It validates filesystem inputs only; model
and cache validation occurs in the runtime startup. Keep all generated paths,
weights, images, caches and environments outside Git.

For the first cache generation, add `--cache-pending` to the preparation
command. This permits the specified cache output to be absent, requires both
offline INT8 checkpoints, and writes the paths used by the cache builder.
Run the cache builder before starting inference. Existing-cache preparation
does not require the offline checkpoint files.

Offline context preparation additionally needs
`requirements/offline-context.txt` (comfy-kitchen 0.2.26, including
`TensorWiseINT8Layout`). This package can be exposed from a separate existing
dependency target; the online Stage2 process does not use it. Once the public
INT8 files are available, expose that package in the Stage2 environment and
generate the fixed cache. For a separate dependency target:

```bash
/absolute/stage2-env/bin/python -m pip install --no-deps \
  --target dependencies/offline-context -r requirements/offline-context.txt
PYTHONPATH="$PWD/dependencies/offline-context:$PWD" \
  /absolute/stage2-env/bin/python -m runtime.cache_builder \
  --paths paths.json --output /absolute/fixed-prompt.pt
```

Reuse an existing target when available. The fresh installation command is
provided for setup and has not been run as part of release validation.

Read `THIRD_PARTY_NOTICES.md` for each component's terms. The selected H3
upscaler source and model revisions do not declare an explicit license;
public accessibility does not establish redistribution rights.
