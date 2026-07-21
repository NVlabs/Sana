# Cache-DiT / DBCache

Cache-DiT is a diffusion-cache family that includes block-level reuse policies such as DBCache and feature forecasting variants. The paper discusses it as part of the broader cross-step cache landscape.

## Design point

| Method | Skip scope | Decision signal |
|---|---|---|
| DBCache | selected DiT blocks | residual or feature difference |
| TaylorSeer integration | forecasted features | Taylor-series extrapolation |
| SCM-style masks | denoising steps | precomputed step mask |

## Role in Sol-Engine

Cache-DiT-style methods are useful baselines and candidates when a deployment benefits from block-level reuse instead of skipping an entire model path. They are not the selected default for the three reported full optimization stacks.

## Practical notes

- Block-level policies need careful quality validation.
- A cache hit count is not the same as end-to-end speedup.
- Decode, offload, LoRA switching, and stage transitions can hide denoising savings.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
