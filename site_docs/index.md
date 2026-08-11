# Sol-Engine

Sol-Engine is an efficiency-oriented inference framework for high-resolution video diffusion. It wraps Cosmos3-Super, LTX-2.3, SANA-Video, Wan2.2 (TI2V-5B and A14B), and LingBot-Video with one explicit acceleration line per model.

!!! tip "Recently added"
    **2026/07/15** — three new models: [Wan2.2 TI2V-5B](https://github.com/NVlabs/Sana/blob/sol-engine/config/wan22_ti2v_5b/wan5b_kernel_easycache_pisa.toml) **~2.89×**, [Wan2.2-A14B](https://github.com/NVlabs/Sana/blob/sol-engine/config/wan22_t2v_a14b/singlegpu_opt.toml) **~2.17×**, and [LingBot-Video](https://github.com/NVlabs/Sana/blob/sol-engine/config/lingbot_video/cudnn_pisa_easycache_refiner.toml) **~2.60×** end-to-end.

    **2026/07/13** — refreshed the [agent workflow](agent-workflow.md): a master orchestrator driving per-technique executor sub-agents with automatic quality gates.

## Models and speedups

| Model | Params | Acceleration line | Speedup |
|---|:---:|---|:---:|
| [Cosmos3-Super (4xB200)](pipelines/cosmos3.md) | 64B | TeaCache + NVFP4 | ~2.26x |
| [LTX-2.3 (1xB200)](pipelines/ltx.md) | 22B | cache + PISA + NVFP4 + token-prune .. | 2.40x |
| [SANA-Video (1xB200)](pipelines/sana.md) | 2B | EasyCache + kernel fusion + compile | 2.77x |
| [Wan2.2 TI2V-5B (1xGB200)](pipelines/wan5b.md) | 5B | EasyCache + kernel fusion + compile | 2.89x |
| [Wan2.2-A14B (1xGB200)](pipelines/wan14b.md) | 14B | kernel fusion + EasyCache + PISA | 2.17x |
| [LingBot-Video (4xGB200)](pipelines/lingbot.md) | 30B-A3B | kernel fusion + refiner PISA + EasyCache | 2.60x |

## The five acceleration methods

Video diffusion inference exposes redundancy at three complementary levels: **Algorithm level**: adjacent denoising steps run structurally similar computation over slowly changing latents. **Model level**: long spatiotemporal sequences contain redundant tokens and attention interactions. **Kernel level**: DiT blocks repeatedly launch memory-bound work around GEMMs, layout movement, normalization, activation, and precision conversion.

| Method | Implemented entries |
|---|---|
| [Cache](techniques/cache.md) | TeaCache, EasyCache, fixed-step cache |
| [Quantization](techniques/quant.md) | NVFP4 |
| [Kernel fusion](techniques/kernel.md) | AdaLN gate fusion, GEMM epilogues, QKV merge, ... |
| [Sparse attention](techniques/sparse.md) | PISA, SpargeAttention, Sparse VideoGen, ... |
| [Token pruning](techniques/token_prune.md) | Feature-norm pruning, ToMe-SD |

## Quick start

In Claude Code or Codex, run:

```text
/goal Execute the inference code for the six models using both baseline and full-opt
settings with the following requirements. Refer to AGENTS.md for the environment creation,
model download, and inference guides. For the environment, you need to create a new
environment. For model weights, you are allowed to reuse existing weights if they are
locally available; otherwise, you need to download them. Regarding adaptability, be aware
that the provided guides for environment creation, download scripts, and inference may
contain system incompatibilities, so you are expected to troubleshoot and adapt them to
your specific machine.
```

## Start here

- [Installation](installation.md): environment creation, CUDA JIT fixups, and model downloads.
- [Pipelines](pipelines/cosmos3.md): optimized launch paths for all six models (Cosmos3-Super, LTX-2.3, SANA-Video, Wan-5B, Wan-14B, LingBot-Video).
- [Techniques](techniques/cache.md): the five acceleration methods and where they apply.
- [Agent workflow](agent-workflow.md): the orchestration (master/executors, quality gates) behind the agent-native quick start.

## Citation

```bibtex
@misc{li2026solvideoinferenceengine,
  title         = {Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation},
  author        = {Yitong Li and Junsong Chen and Haopeng Li and Haozhe Liu and Jincheng Yu and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
  year          = {2026},
  eprint        = {2606.23743},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.23743},
  url           = {https://arxiv.org/abs/2606.23743},
}
```
