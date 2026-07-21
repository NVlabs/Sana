# AdaLN and residual gate fusion

AdaLN and residual gate fusion targets the normalization and modulation work that
surrounds DiT block compute. The fused path keeps AdaLN normalization, scale,
shift, and residual gate application close to the data they modify instead of
materializing each intermediate through separate memory-bound operators.

## Role in Sol-Engine

This is part of the LTX-2.3 kernel-fusion path. It is applied only where profiling
shows the AdaLN and gate sequence is a meaningful overhead at the deployment shape.

## Validation

The fused path should preserve the original block math within the expected BF16
rounding envelope. Compare outputs with the unfused path before combining it with
cache, sparse attention, quantization, or token pruning.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sol-Engine code](https://github.com/NVlabs/Sana/tree/sol-engine)
