# PISA

PISA is the sparse attention method selected for LTX-2.3. It uses piecewise sparse attention: important blocks are computed exactly, while less important blocks can be approximated or skipped depending on configuration.

## Sol-Engine placement

Sol-Engine uses PISA in selected LTX-2.3 stage-2 video self-attention blocks
together with cache, token pruning, NVFP4, and kernel fusion.

## Tunable knobs

- sparsity.
- block size.
- route mode.
- dense fallback layers.
- approximate-remainder policy.
- stage and layer placement.

## Validation

Validate temporal coherence and fine detail. Sparse attention errors can be subtle in single frames but visible over motion.

## Wan / LingBot usage

- **LingBot refiner.** LingBot applies PISA to the 1080p refiner (density 0.10), where the spatiotemporal sequence is longest.
- **Single-GPU Wan-14B.** PISA runs on single-GPU Wan-14B through the `dispatch_attention_fn` entry (density 0.10), firing on real self-attention of hooked layers, with attention routed via `DIFFUSERS_ATTN_BACKEND=_native_cudnn`. The multi-GPU context-parallel path drives PISA through its `_native_attention_forward_op` hook.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Piecewise sparse attention](https://github.com/xie-lab-ml/piecewise-sparse-attention)
