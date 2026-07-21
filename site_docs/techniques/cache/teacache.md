# TeaCache

TeaCache is the cross-step cache policy selected for Cosmos3-Super. It estimates how much the denoising hidden state changes from one timestep to the next and reuses a cached residual when the accumulated change stays below a threshold.

## Sol-Engine placement

Sol-Engine uses TeaCache in the Cosmos3-Super generation-path transformer
blocks with threshold 1.15, start step 10, and max 3 continuous hits.

The first cache hit skips repeated generation-path transformer work and replays the cached residual. A recompute refreshes the residual when the accumulated change crosses the threshold or the max-hit cap is reached.

## Tunable knobs

- threshold: lower is safer and slower; higher is faster and riskier.
- start step: early steps are usually more structure-sensitive.
- max continuous hits: caps drift from repeated replay.
- cached state: residual replay is the default Sol-Engine path.

## Validation

Compare full videos, not only scalar similarity. Cache methods can preserve layout while shifting fine detail, so visual validation remains the acceptance criterion.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [TeaCache](https://github.com/ali-vilab/TeaCache)
