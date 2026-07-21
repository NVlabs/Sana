# SVDQuant / Nunchaku

SVDQuant reduces low-bit quantization error by absorbing weight and activation outliers with low-rank components. Nunchaku is the practical inference family associated with SVDQuant-style 4-bit diffusion acceleration.

## Role in Sol-Engine

SVDQuant is part of the quantization design space described by the paper. It is a candidate when a deployment needs 4-bit execution but direct low-bit conversion causes outlier-driven quality loss.

## Practical path

- keep base model metadata separate from quantized transformer weights when possible.
- use explicit transformer override paths for quantized components.
- validate fallback layers and outlier handling before combining with cache or sparse attention.

## Trade-off

SVDQuant can preserve quality better than naive 4-bit quantization, but it adds method-specific checkpoint and runtime requirements.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Nunchaku](https://github.com/nunchaku-ai/nunchaku)
- [DeepCompressor](https://github.com/nunchaku-ai/deepcompressor)
