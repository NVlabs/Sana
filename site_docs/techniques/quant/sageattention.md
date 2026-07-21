# SageAttention

SageAttention is an attention-specific quantization family. The paper cites it as a method line that targets 8-bit to 4-bit attention acceleration with outlier smoothing.

## Design point

Attention quantization differs from FFN quantization because attention scores and value aggregation can be sensitive to outliers and softmax behavior. SageAttention-style methods focus on this attention path rather than applying one uniform quantizer to the entire transformer.

## Role in Sol-Engine

SageAttention is a candidate for deployments where attention remains a major bottleneck after cache, sparse attention, or token pruning choices are fixed.

## Validation

Inspect both motion smoothness and temporal flicker. Attention quantization can preserve frame-level aesthetics while changing temporal consistency.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [SageAttention](https://github.com/thu-ml/SageAttention)
