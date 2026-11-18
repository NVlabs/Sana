# Prepare the Spark runtime

Use Linux aarch64 on a single DGX Spark (GB10/SM121). Run the commands below
from `models/minimax_h3/Sol-H3-Spark`, on the GPU host or in a compute allocation.

The validated runtime uses separate Qwen, Stage1 and Stage2 environments.
Reuse them where available: the `requirements/*-observed.txt` files are version
inventories, not installable lockfiles. A clean installation has not yet been
validated; see [validation](validation.md).

## 1. Download weights

Accept the LTX-2.5 access terms with your Hugging Face account, then:

```bash
python download_checkpoints.py --plan
python -m pip install -r requirements/download.txt
hf auth login
python download_checkpoints.py --output-dir checkpoints --include-offline
```

The default task is `t2va`. Use the same `--task` for downloading, preparation
and inference. Common assets are shared between tasks.

| Task | H3 partition | Input VAE | Draft LoRA |
| --- | --- | --- | --- |
| `t2va` | `transformer/` | Not loaded | FastH3 VSA DataFree |
| `fl2va` | `transformer/` | Native H3 VAE | FastH3 VSA DataFree |
| `ref2va` | `transformer_ref/` | Native H3 VAE | LightX2V Ref2VA four-step v0.1 |

For example, add Ref2VA weights with
`python download_checkpoints.py --task ref2va --output-dir checkpoints`.
FL2VA is our adaptation of the T2VA LoRA; Ref2VA uses a task-specific LoRA
and dense attention by default. See [task usage](../README.md#run).

Downloads use pinned revisions, including the public
[H3-to-LTX adapter](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter).
Add `--verify-sha256` for full single-file hash checks. No weights are downloaded
during inference. Omit `--include-offline` if you already have a valid generic
prompt cache; it adds the INT8 Gemma and connector-source weights used only
to build that cache.

## 2. Prepare environments

[configs/dependencies.json](../configs/dependencies.json) pins the sources.
`prepare.py --fetch-sources` fetches missing checkouts and applies the verified
Spark loader patch; it does **not** install Python packages or reset existing
modified checkouts.

| Environment | Required setup |
| --- | --- |
| Stage1 | Patched FastVideo and matching `fastvideo-kernel` 0.3.5; cuDNN frontend 1.27.0 / CUDA 13 binary 9.20.0.48 for BSA. |
| FA4 import target | Use the pinned full checkout as `fa4_source_root`; preparation creates the `fa4_root` namespace. Point `fa4_dependencies` to the packages in [fa4-isolated.txt](../requirements/fa4-isolated.txt). Keep this target separate: FA4 and the BSA environment use different CuTe versions. |
| Stage2 | Pinned LTX source with `ltx-core`, `ltx-pipelines`, `ltx-kernels` and its compiled `all2all_cpp` extension, required even on one GPU. Sol-Attn comes from this Sana checkout. |
| Qwen | ComfyUI v0.30.0, comfy-kitchen 0.2.26 and comfy-aimdo 0.4.11. The container below preserves the NGC Torch stack. |

For Qwen, reuse a verified image or build the provided recipe:

```bash
docker build -f Dockerfile.qwen -t sol-h3-spark-qwen .
export SOL_H3_SPARK_QWEN_IMAGE=sol-h3-spark-qwen
export SOL_H3_SPARK_RUNTIME_ROOT=/absolute/dedicated/spark-runtime
```

The runtime root must already exist and be writable. `qwen_python.sh` mounts
it read/write and this package read-only at the same absolute paths. Set
`SOL_H3_SPARK_QWEN_WEIGHTS_ROOT` and `SOL_H3_SPARK_COMFY_ROOT` for additional
read-only mounts. The `comfy_root` path must be visible in both host and container.

Keep all input media under these mounts; outside paths and symlink targets
are not visible. The wrapper runs as your UID/GID on GPU 0, with networking
disabled and no image pulls. Its default interpreter is `/usr/bin/python3`;
override with `SOL_H3_SPARK_QWEN_PYTHON` if needed. Fresh image and dependency
builds have not been validated.

## 3. Write the path manifest

Inspect without downloads or writes:

```bash
python prepare.py --plan
```

Then supply your interpreters and prompt-cache path:

```bash
python prepare.py --fetch-sources \
  --python-stage1 /absolute/stage1-env/bin/python \
  --python-stage2 /absolute/stage2-env/bin/python \
  --python-qwen "$PWD/qwen_python.sh" \
  --prompt-cache /absolute/fixed-prompt.pt \
  --paths existing-paths.json \
  --output paths.json
```

`existing-paths.json` is an optional flat object of path overrides, including
`fa4_dependencies`; omit `--paths` when using the default locations. Use
`--adapter-dir` to reuse a matching adapter. For FL2VA or Ref2VA, add the
appropriate `--task` and choose a separate output manifest.

**First run without a cache:** add `--cache-pending` to the command above,
then complete step 4. This requires the offline weights from step 1.
With an existing valid cache, skip step 4.

Preparation checks filesystem inputs and source revisions, and refuses to
overwrite an existing output manifest. Runtime startup validates model/cache
contents. Keep generated manifests, weights, media and environments outside Git.

## 4. Build the generic prompt cache once

Expose [offline-context.txt](../requirements/offline-context.txt) to the Stage2
interpreter. Reuse an existing dependency target, or install a separate one:

```bash
/absolute/stage2-env/bin/python -m pip install --no-deps \
  --target dependencies/offline-context -r requirements/offline-context.txt
PYTHONPATH="$PWD/dependencies/offline-context:$PWD" \
  /absolute/stage2-env/bin/python -m runtime.cache_builder \
  --paths paths.json --output /absolute/fixed-prompt.pt
```

The cache contains post-connector video and audio contexts for:

> 4K, refined, high quality, cinematic detail, clean textures, natural motion.

Use the prescribed INT8 Gemma/connector recipe; arbitrary embeddings or a
BF16-generated cache are not interchangeable. `gemma_tokenizer` points to the
same embedded-assets checkpoint as `offline_gemma`. Cache generation runs
before resident sessions and is excluded from request timing.

Continue with [inference examples](../README.md#run). See
[component terms](../THIRD_PARTY_NOTICES.md) before redistribution; the selected
H3 upscaler source and weights do not declare an explicit license.
