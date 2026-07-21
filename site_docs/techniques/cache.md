# Cache

Cache methods exploit algorithm-level redundancy across adjacent diffusion denoising steps. The latent state changes gradually, so selected intermediate work can be reused when a step is sufficiently similar to a prior step.

The paper frames cache as a cross-step family: the cache agent searches the policy, skip-step schedule, cached state, compensation rule, warmup, and maximum cached-step constraints for each deployment instance.

## In Sol-Engine

| Pipeline | Cache path | Role |
|---|---|---|
| Cosmos3-Super | TeaCache | Residual replay across denoising steps |
| LTX-2.3 | Stage-specific cache | Part of the full LTX optimization stack |
| SANA-Video | EasyCache | Main SANA acceleration component |

## Methods

| Method | Open-source status | Integration | Role in the design space |
|---|---|---|---|
| [TeaCache](cache/teacache.md) | open-source | local Cosmos runtime | timestep-conditioned residual replay, selected for Cosmos3-Super |
| [EasyCache](cache/easycache.md) | open-source | local SANA runtime | runtime-adaptive feature reuse, selected for SANA-Video |
| [TaylorSeer](cache/taylorseer.md) | open-source | cache-dit adapter | feature forecasting with Taylor-series extrapolation |
| [Fixed-step cache](cache/fixed_step.md) | open-source local | local LTX runtime | deterministic fixed-step schedule, selected for LTX-2.3 |
| [Cache-DiT / DBCache](cache/cache_dit.md) | open-source | cache-dit adapter | block-level diffusion cache family used as a related cache baseline |

## Practical notes

- Cache thresholds trade speed for visual fidelity.
- Timing should be read from warmup-excluded log lines.
- Quality checks should compare generated videos, not just total latency.
