# Source notices

The six files under `hyperflow_h3/` are copied unchanged from
[Video-Rebirth/hyperflow](https://github.com/Video-Rebirth/hyperflow) at commit
`1dd2f342aba5ab51da02b62885939655e8e268da`, under Apache License 2.0.
They retain their original Copyright 2026 The HyperFlow authors notices.
Exact source hashes are recorded in `PROVENANCE.json`.

The complete upstream notices are reproduced in
[HYPERFLOW_THIRD_PARTY_NOTICES.md](HYPERFLOW_THIRD_PARTY_NOTICES.md), including
attributions for the derived Diffusers scheduler, modular pipeline, and attention
processor code. The source headers also acknowledge AnyFlow and NVIDIA Sol-Attn.

The optimized entry point reuses `../Sol-H3/h3_runtime/`, including LoRA consumer
fusion from NVlabs/Sana PR #503. Its kernels and third-party notices remain in
their existing locations. The guarded `packing` import compatibility fallback
is the sole change to that shared runtime in this integration.

`LICENSE` contains Apache License 2.0 for the code. Base model and adapter weights
are downloaded separately; their upstream model license is available from
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and
[the official HyperFlow weights repository](https://huggingface.co/videorebirth/hyperflow/blob/main/LICENSE).
