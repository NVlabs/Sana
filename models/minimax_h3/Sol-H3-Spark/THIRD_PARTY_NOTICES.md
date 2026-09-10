# External dependencies and model terms

This directory does not bundle model weights or external runtime installations.
Exact source revisions and model SHA-256 values are in
`configs/dependencies.json` and `configs/checkpoints.json`.

| Component | Source and terms |
| --- | --- |
| FastVideo and FastVideo kernels | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/3d8ac9d14bd697a89ede8f170cbfbca012a9edcc), Apache-2.0 |
| FlashAttention 4 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/14c377950125c70b7a9dabf9c561fca53715ac7d), BSD-3-Clause |
| ComfyUI | [Comfy-Org/ComfyUI v0.30.0](https://github.com/Comfy-Org/ComfyUI/tree/b1693ecba9f5b65f8c80ab36b195ab963ec92413), GPL-3.0; installed separately in the Qwen environment |
| LTX source | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2/blob/d151147788a9284cca791edc6ce898007e727fe6/LICENSE.md), its own Community License |
| MiniMax-H3, NVFP4 Qwen and VSA LoRA | Model-specific MiniMax-H3 Community License Agreement; follow each pinned model card |
| LTX-2.5 weights | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/tree/5e6e71018ee1756ed329b697a7b4aedc934dfce9), gated access and LTX-2.x Community License Agreement |
| H3 latent upscaler | [author source](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/tree/d7c01b9011f2e8439493f6c02c29995a27df276f) and [checkpoint](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/tree/13ccf95d85d120bdbc92c05b1247a6e147bf54bf); neither selected revision declares an explicit license. No redistribution permission is inferred from public accessibility. |

The Qwen container recipe fetches ComfyUI; its GPL terms continue to apply to
that installation and any redistributed image. The root repository license
does not relicense ComfyUI, LTX, model weights, CUDA libraries or container
components. The NGC base image has its own NVIDIA terms.

`patches/fastvideo-spark-loader.patch` modifies three Apache-2.0 FastVideo
loader files. The companion JSON records the upstream revision, patch SHA-256
and each pristine/patched file SHA-256. The public patch is required for the
resident constructor-merged LoRA route and does not include the unused decoder
experiment branch.

The [H3-to-LTX adapter weights](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter/tree/cfcd8a7cc36c135142287584728f7fe362df0867)
are downloaded separately with expected SHA-256
`170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac`.
The model card does not assign a new weight license.
