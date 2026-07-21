# Sparse VideoGen2

Sparse VideoGen2 improves sparse-token identification and GPU layout through semantic-aware permutation.

## Role in Sol-Engine

It is a candidate sparse backend when semantic grouping can make sparse execution more efficient or more stable than a purely geometric block pattern.

## Search knobs

- semantic grouping policy.
- sparse-token identification threshold.
- layout permutation.
- dense fallback layers.

## Validation

Inspect whether semantic grouping preserves object boundaries and motion continuity across frames.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sparse VideoGen](https://github.com/svg-project/Sparse-VideoGen)
