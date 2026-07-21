# Sparse VideoGen

Sparse VideoGen exploits spatial-temporal sparsity patterns in video diffusion with online profiling and sparse execution.

## Role in Sol-Engine

It is part of the sparse-attention method family considered by the sparse-attention agent. It is most relevant when profiling shows persistent spatial-temporal sparsity across the target prompts and generation length.

## Practical knobs

- profiling window.
- selected sparse pattern.
- custom sparse-kernel layout.
- dense fallback schedule.

## Validation

Validate at the exact video duration and resolution. Sparse patterns can be configuration-specific.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sparse VideoGen](https://github.com/svg-project/Sparse-VideoGen)
