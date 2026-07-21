# QK-norm + RoPE fusion

QK-norm + RoPE fusion combines Q/K normalization and rotary-position work on the
attention Q/K path. This targets repeated small operators that otherwise launch
separately and move intermediate tensors through memory.

## Role in Sol-Engine

This is part of the DiT attention-path fusion design space and is especially relevant when attention remains active after sparse attention choices are fixed.

## Search knobs

- which attention blocks use fused Q/K preprocessing.
- tensor layout expected by the downstream attention backend.
- precision format used before attention.

## Validation

Confirm compatibility with the selected attention backend and with any quantization path applied to QKV projections.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [RoFormer / rotary position embedding](https://arxiv.org/abs/2104.09864)
- [Sol-Engine code](https://github.com/NVlabs/Sana/tree/sol-engine)
