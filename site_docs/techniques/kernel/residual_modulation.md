# Residual and modulation glue fusion

Residual and modulation glue fusion targets the small operators that connect
normalization, gates, residual updates, and block outputs. These operators are
often individually cheap but collectively memory-bound when launched separately.

## Role in Sol-Engine

This fusion is a local kernel-level optimization for DiT block glue. It is kept
shape-specific and enabled only when profiling shows a real latency reduction.

## Validation

Confirm that residual accumulation order, dtype conversion, and modulation values
match the unfused path within the expected precision envelope.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sol-Engine code](https://github.com/NVlabs/Sana/tree/sol-engine)
