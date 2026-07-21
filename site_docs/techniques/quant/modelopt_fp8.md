# ModelOpt / FP8

ModelOpt and FP8 checkpoint flows are practical quantization families in the codebase. They are useful when a deployment has validated low-precision transformer components or when online quantization is acceptable for the target model.

## Role in Sol-Engine

The reported full optimization stacks focus on selective NVFP4 for Cosmos3-Super and LTX-2.3, but ModelOpt / FP8 remains part of the broader local quantization search space.

## Practical knobs

- transformer override path.
- transformer weights path.
- ignored layer patterns.
- layerwise offload compatibility.
- BF16 fallback modules.

## Validation

Quantized checkpoints should be validated in the same pipeline where they will be composed. A checkpoint that is acceptable alone may become unacceptable after cache or sparse attention is added.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
