# SpargeAttention

SpargeAttention is a training-free sparse attention family cited by the paper. It predicts near-zero attention entries with a block-wise similarity metric and skips low-value work.

## Role in Sol-Engine

SpargeAttention is part of the sparse-attention design space. It is a candidate when the attention map has clear block-level structure and the deployment can tolerate approximate attention entries.

## Search knobs

- block size.
- similarity threshold.
- dense fallback layers.
- per-stage sparsity budget.

## Validation

Compare against dense attention at the target resolution and duration; sparsity that works at shorter context can fail at longer temporal length.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [SpargeAttention](https://github.com/thu-ml/SpargeAttn)
