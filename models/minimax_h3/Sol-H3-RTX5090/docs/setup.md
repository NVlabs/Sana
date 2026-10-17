# Prepare the RTX 5090 runtime

Use Linux x86_64 with one visible NVIDIA RTX 5090 (SM120). Run these commands
from `models/minimax_h3/Sol-H3-RTX5090` on the GPU host or a compute allocation.

Reuse verified environments and assets where available. The
[dependency manifest](../configs/dependencies.json) records the observed 5090
environments and pinned source revisions; it is not a fresh-install lockfile.
[Validation](validation.md) states the actual coverage.

## 1. Download weights

Accept the LTX-2.5 access terms with your Hugging Face account, then:

```bash
python download_checkpoints.py --plan
python -m pip install -r requirements/download.txt
hf auth login
python download_checkpoints.py --output-dir checkpoints
```

Add `--include-offline` only if the generic refinement-context cache must be
built. It adds the INT8 Gemma and connector-source weights used offline in
step 4. Existing caches can be reused. `--verify-sha256` checks the complete
single-file hashes. No weights are downloaded during inference.

The [checkpoint manifest](../configs/checkpoints.json) preserves the released
H3, LoRA, Qwen, learned upscaler, H3-to-LTX adapter and LTX source identities.
The default task is `t2va`; use the same task for preparation and inference.
Other task selectors are inherited but GPU-unvalidated on this profile.

## 2. Prepare environments

The validated deployment shares a native Python environment between H3 and
Qwen, and uses a separate LTX interpreter:

| Environment | Observed integration |
| --- | --- |
| H3 / Qwen | Python 3.12, Torch 2.12.0+cu130, Triton 3.7.0; patched pinned FastVideo, fastvideo-kernel 0.3.5, cuDNN frontend 1.27.0, ComfyUI v0.30.0, comfy-kitchen 0.2.26 and comfy-aimdo 0.4.11 |
| FA4 | Pinned full source checkout exposed through an isolated FA4 namespace and its matching dependencies |
| LTX | Python 3.13, Torch 2.13.0+cu132, Triton 3.7.1; pinned `ltx-core`, `ltx-pipelines` and `ltx-kernels`, including the compiled `all2all_cpp` extension |

Use native x86_64 interpreters. The Spark aarch64 container and environment
inventories are not part of this entry. Source fetching below does not install
Python environments or compile their extensions. Selected additional versions
are retained in the [measurement receipt](../validation/rtx5090-t2va.json).

Keep `fa4_source_root` at the pinned full checkout; preparation creates the
isolated `fa4_root` namespace. Use `fa4_dependencies` for its matching separate
dependency target; [fa4-isolated.txt](../requirements/fa4-isolated.txt) records
the inherited FA4 compatibility pins, not a complete 5090 environment lock.

When reusing a CuTe installation, inspect its actual `.pth` file. The tested
CuTe 4.6.0 installation used this path override:

```json
{
  "stage1_dependencies": "/absolute/cute-env/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages"
}
```

Adding a directory to `PYTHONPATH` does not execute that environment's `.pth`
files. Do not put its entire site-packages ahead of the selected Torch.

The test system reported driver 580.173.02, 32,607 MiB per GPU and 188 GiB
host RAM. Only one card was exposed to inference. These are observed system
properties, not a measured minimum RAM requirement.

## 3. Write the path manifest

Inspect paths without downloads or writes:

```bash
python prepare.py --plan
```

Then supply the native interpreters and prompt cache:

```bash
python prepare.py --fetch-sources \
  --python-stage1 /absolute/h3-env/bin/python \
  --python-qwen /absolute/h3-env/bin/python \
  --python-stage2 /absolute/ltx-env/bin/python \
  --prompt-cache /absolute/fixed-prompt.pt \
  --paths existing-paths.json --output paths-5090.json
```

`existing-paths.json` is an optional flat object with existing checkpoint and
dependency paths, including `stage1_dependencies` and `fa4_dependencies`.
Omit `--paths` when using the default locations.

`prepare.py` fetches missing pinned sources and applies the declared FastVideo
loader patch. It preserves existing modified checkouts, checks filesystem
inputs and source revisions, and refuses to overwrite an output manifest.
Runtime startup validates model and cache contents.

If the context cache does not exist, add `--cache-pending` and complete step 4.
Keep generated manifests, weights, media and environments outside Git.

## 4. Build the generic prompt cache once

Expose [offline-context.txt](../requirements/offline-context.txt) to the Stage2
interpreter. Reuse an existing dependency target or install a separate one:

```bash
/absolute/ltx-env/bin/python -m pip install --no-deps \
  --target dependencies/offline-context -r requirements/offline-context.txt
PYTHONPATH="$PWD/dependencies/offline-context:$PWD" \
  /absolute/ltx-env/bin/python -m runtime.cache_builder \
  --paths paths-5090.json --output /absolute/fixed-prompt.pt
```

The cache holds post-connector video and audio contexts for:

> 4K, refined, high quality, cinematic detail, clean textures, natural motion.

Use the prescribed INT8 Gemma/connector recipe. Arbitrary embeddings or a
BF16-generated cache are not interchangeable. `gemma_tokenizer` points to the
same embedded-assets checkpoint as `offline_gemma`. Cache creation runs once
outside online request timing.

Continue with [inference](../README.md#run) and [validation](validation.md).
See [component terms](../THIRD_PARTY_NOTICES.md); the selected H3 upscaler
source and weights do not declare an explicit license.
