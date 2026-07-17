# LingBot-Video Documentation

This documentation is organized around the public inference workflow. Start with Quick Start, then open the detailed page for the step you need.

## Guides

| Guide | Purpose |
| --- | --- |
| [Quick Start](quickstart.md) | Install dependencies, set the model root, and run one end-to-end generation. |
| [Prompt Preparation](prompt_preparation.md) | Rewrite plain prompts into structured JSON captions and optionally create per-sample negative prompts. |
| [Diffusers Inference](dit_inference.md) | Run diffusers inference for T2I, T2V, TI2V, and Refinement. |
| [SGLang Diffusion Inference](sglang_diffusion.md) | Run SGLang Diffusion for parallel accelerated inference. |

## Recommended Workflow

1. Convert plain prompts into structured prompts with the prompt rewriter.
2. Optionally run auto negative to create a per-sample negative prompt, or use the default negative prompt.
3. Run DiT inference with `scripts/inference.py --prompt_json`.
4. Select `--backend diffusers` or `--backend sglang` for inference.
