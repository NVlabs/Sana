# Attention output gate fusion

Attention output gate fusion targets the path after attention has produced its
output tensor. It combines output projection, gate application, and nearby
layout or residual glue when the selected attention backend and tensor layout
make that profitable.

## Role in Sol-Engine

This fusion is useful when attention remains an active cost after sparse
attention choices are fixed. It reduces launch overhead and avoids writing
short-lived intermediates between the attention output and the block update.

## Validation

Validate the fused path with the exact attention backend, precision format, and
sequence shape used for deployment.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sol-Engine code](https://github.com/NVlabs/Sana/tree/sol-engine)
