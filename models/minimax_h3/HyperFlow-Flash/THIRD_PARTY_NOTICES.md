# Third-party notices

This directory contains source code written for it plus the attributed adaptations below. No third-party source tree is
vendored, and no model weights, checkpoint code or generated media are included.

## Code adapted here

| File | Origin | License |
|---|---|---|
| `sol_hyperflow_flash/attention.py`, `make_processor` | Follows the computation of `MiniMaxH3AttnProcessor` in [huggingface/diffusers](https://github.com/huggingface/diffusers) (`transformer_minimax_h3.py`), with the attention backend made switchable. | Apache License 2.0 (Copyright The HuggingFace Team) |
| `sol_hyperflow_flash/sampler.py` | Implements the 6-step two-time sampler of the HyperFlow H3 Flash model card as the denoise loop of the diffusers MiniMax-H3 modular pipeline. | Apache License 2.0 (sampler of [Video-Rebirth/hyperflow-h3-flash](https://github.com/Video-Rebirth/hyperflow-h3-flash), Copyright 2026 The HyperFlow authors) |
| `sol_hyperflow_flash/runtime.py`, `set_te_trim` | Follows the conditioner trimming of `../HyperFlow/sol_hyperflow/conditioner_memory.py`. | Apache License 2.0 |
| `sol_hyperflow_flash/references.py` | Builds requests for the diffusers MiniMax-H3 `ref2va` modular pipeline (`MiniMaxH3Ref2VABlocks`, `MiniMaxH3Reference`). | Apache License 2.0 (Copyright The MiniMax and HuggingFace Teams) |
| `sol_hyperflow_flash/vae_par.py`, `merge_tuned.py` | Call and re-launch kernels of `../Sol-H3/h3_runtime/` (`vae_parallel.py`, `sol_residual.py`); the global reference-encode split applies Sol-H3's tiling and blending rules to the encoder. | Apache License 2.0 |

## Runtime dependencies (installed separately)

| Dependency | Used for | License |
|---|---|---|
| `../Sol-H3/h3_runtime` of this repository | Sol-Attn block-sparse attention, fusions, compressed transport, parallel VAE, MP4 encoder | Apache-2.0 |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | MiniMax-H3 modular pipeline, VAEs, attention dispatch | Apache-2.0 |
| [NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) `[cutedsl]` | cuDNN block-sparse attention used by `sol_bsa` | MIT |
| PyTorch, Triton, Transformers, Accelerate, safetensors, PyAV, huggingface_hub | runtime | BSD-3 / MIT / Apache-2.0 / Apache-2.0 / Apache-2.0 / BSD-3 / Apache-2.0 |

## Models (downloaded separately, never redistributed here)

- **MiniMax-H3** (`MiniMaxAI/MiniMax-H3`): MiniMax H3 Community License Agreement, Copyright (c) 2026 MiniMax.
- **HyperFlow H3 Flash** (`VideoRebirthAI/hyperflow-h3-flash`, partitions `fl2va_int8_convrot` and
  `ref2va_int8_convrot`): a Model Derivative of MiniMax-H3 by Video Rebirth, distributed under the same agreement. Its
  transformer code (`hyperflow_h3_flash_transformer.py`, `hyperflow_h3_flash_int8.py`, Apache License 2.0, Copyright 2026
  The HyperFlow authors) is loaded at run time from the downloaded checkpoint folder.

HyperFlow is not an official MiniMax product. Powered by MiniMax H3.
