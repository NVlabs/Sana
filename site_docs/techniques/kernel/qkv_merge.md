# QKV merge

QKV merge combines separate query, key, and value projection work where the model and tensor layout allow it. In Sol-Engine, this is part of the SANA-Video full optimization path.

## Sol-Engine placement

Sol-Engine uses QKV merge in the SANA-Video attention projection path together
with EasyCache, BF16 linear-attention kernels, and compile.

## Why it helps

Projection paths often launch multiple similar kernels and materialize intermediate tensors. Merging QKV work reduces launch overhead and improves data movement locality.

## Validation

Keep the merged path equivalent to the original projections and verify output consistency before combining with EasyCache.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sol-Engine code](https://github.com/NVlabs/Sana/tree/sol-engine)
