# Third-party notices

This file records provenance and license pointers for the MiniMax-H3 Super
Acceleration integration. It is not legal advice and does not replace any
upstream license or model-use terms.

This directory does not redistribute model weights, LoRA adapters, datasets,
first-frame media, generated videos, container images, Python environments, or
compiled kernels/caches. Those inputs are reference-only and remain subject to
their upstream terms.

## Source distributed in this repository

### madebyollin/taehv: Stage-1 TAEH3 source

The Stage-1 TAEH3 decoder vendors one source file from
[madebyollin/taehv](https://github.com/madebyollin/taehv):

- upstream commit: `e589fddc076e77f5ba8cd6baabe4ba3260b261cd`;
- vendored file: `vendor/taeh3/taehv.py`;
- source SHA-256:
  `28b452ad74f924d2d0922d6b3e805dfa7f504c523ced240435fefad5f6f650a7`;
- license: MIT, copyright (c) 2025 Ollin Boer Bohan;
- bundled license: `vendor/taeh3/LICENSE`, SHA-256
  `532f9e394518ffddecd294a517d5b41d79d3d3866c3fb95a6cb0e8bcc02370bf`.

The corresponding `taeh3.pth` checkpoint is not bundled. The runtime expects an
external file with SHA-256
`af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba`.

### madebyollin/taehv: Stage-2 LTX TAEHV source

Stage 2 reuses the existing repository copy at
`models/ltx2.5-refiner/GB200/vendor/taehv/` rather than duplicating it here. Its
`SOURCE.json` records:

- upstream repository: [madebyollin/taehv](https://github.com/madebyollin/taehv);
- branch: `2026_03_11_taeltx23_wide`;
- commit: `32ac0146b11007cda5a57b60a3b35653361fb8a4`;
- source SHA-256:
  `607c2a578bc2684e6cd21e96f8c1d024b1c32912e6f58c724f14131a3d4a2773`;
- license: MIT, preserved beside that source.

The wide `taeltx2_3_wide.pth` checkpoint remains external. Its required
SHA-256 is
`007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339`.

### LTX-2 source

The LTX runtime reused by Stage 2 is under `models/ltx25/GB200/`. It is derived
from [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2), upstream commit
`7954dcb0d986bdc36ef272564a9789ade07fcc65`, with the repository's optimized
snapshot at `ccedf8470c224181c65d146b66928d22bac04b22`.

LTX-2 is governed by the **LTX-2 Community License Agreement**, not the root
Apache-2.0 license. The complete license text is preserved at
`models/ltx25/GB200/environment/LTX-2/LICENSE`. Users and distributors must
review that agreement, including its commercial-use and use-based restrictions.
Release through a new channel requires project-owner/legal approval.

### Imported integration sources

The H3/LTX glue, SGLang overlays, and Stage-2 diagnostic sources were imported
from working trees in which they were untracked, and the Stage-1 LoRA overlay
also included a working-tree fix. Those files carried no per-file license
headers. Their exact imported hashes are recorded in `SOURCE_SNAPSHOT.json`.
Before public redistribution, the project owner must confirm that each file is
an authorized contribution under this repository's license. A source commit
alone is not sufficient provenance for these previously untracked files.

## Reference-only runtime and model inputs

| Component | Upstream identity | Distribution here |
| --- | --- | --- |
| MiniMax-H3 model | `MiniMaxAI/MiniMax-H3`; SGLang requested `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`, measured snapshot `6818f6c32d12b210915e44ad56a4228c2608f160` | Not bundled |
| LightX2V MiniMax-H3 Turbo LoRA | `lightx2v/Minimax-h3-Turbo` revision `050494d5fe05bd1b1140b8565ea51dc33a5085a5`; file `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | Not bundled |
| SGLang runtime/container | [sgl-project/sglang](https://github.com/sgl-project/sglang) commit `12eadf86f12aec2e6f81a6e38b61b964a4c6b529`; historical tag `lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86`; pinned image digest `sha256:71145ca99ebc458265e93cebd00b52bb9f419f052e7d0de09a54fa0f72fed888` | Not bundled; the launcher defaults to the content-addressed digest |
| LTX-2.5 Transformer, Gemma encoder, Video/Audio VAEs, x2 upsampler, Refiner LoRA | Lightricks LTX-2.5 distribution; exact filenames are listed in `SOURCE_SNAPSHOT.json` | Not bundled |
| Benchmark opening frame and source dataset | External benchmark asset; required image SHA-256 `0f41282b5101d1be9ef51ee2f0bb13d2c599f0a7139b7406d6534b678387f491` | Not bundled; checked-in JSON is identity metadata only |
| Sol-Attn compiled backend | Built from this repository plus its CUDA/CuTe dependencies | Compiled artifacts and caches are not bundled |
| FFmpeg/ffprobe and Python media packages | System/environment dependencies | Not bundled |

Model repositories and checkpoints may impose terms beyond their source-code
repositories. Consult each exact model card and license before downloading,
using, or redistributing an input.

## Generated and local-only material

Do not add any of the following to this source directory: model/checkpoint or
LoRA files, first-frame images, datasets, MP4 outputs, benchmark run roots,
telemetry logs, Slurm logs, `.venv`/Conda environments, Hugging Face caches,
TorchInductor/Triton/CUDA/CuTe caches, compiled shared objects, sockets,
`__pycache__`, or `.pyc` files.
