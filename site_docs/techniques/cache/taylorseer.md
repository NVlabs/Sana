# TaylorSeer

TaylorSeer is a cross-step feature forecasting method. Instead of only replaying cached features, it extrapolates features from previous timesteps with a Taylor-series approximation to reduce error when reuse intervals become larger.

## Role in Sol-Engine

TaylorSeer is part of the cache design space considered by the cache agent. It is useful as a candidate when plain residual replay produces too much drift at the desired skip interval.

## Search knobs

- forecast order.
- warmup steps before forecasting.
- maximum cached interval.
- fallback recompute cadence.

## When to use

Use TaylorSeer-style forecasting when the deployment needs more aggressive cross-step reuse than TeaCache or EasyCache can safely provide. Validate especially on motion consistency and high-frequency detail.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
