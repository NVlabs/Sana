# NVFP4

NVFP4 is the production low-precision path used by the Sol-Engine full optimization stacks for Cosmos3-Super and LTX-2.3. The paper emphasizes that quantization is applied selectively, not uniformly.

## Sol-Engine placement

| Pipeline | Scope | Dense boundary |
|---|---|---|
| Cosmos3-Super | GEMM-heavy generation-path linear layers | first 3 and last 3 denoising steps |
| LTX-2.3 | video FFN linears | sensitive regions remain BF16 |

## Target modules

- FFN gate-up projection.
- FFN down projection.
- attention QKV projection.
- attention output projection.

## Requirements

NVFP4 requires Blackwell-class hardware and TransformerEngine support. On unsupported hardware the runtime should fall back to BF16 instead of failing hard.

## Validation

Boundary steps are kept dense because early steps affect global structure and late steps refine high-frequency detail. Any change to the dense boundary should be validated visually.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
