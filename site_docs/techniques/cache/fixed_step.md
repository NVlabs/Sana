# Fixed-step cache

Fixed-step cache uses a deterministic fixed-step schedule rather than a learned or threshold-driven skip decision. The paper reports that the LTX-2.3 high-quality schedule required a custom fixed-step strategy because cache schedules tuned for Euler-style samplers did not transfer cleanly to the res_2s trajectory.

## Sol-Engine placement

Sol-Engine uses fixed-step cache in the LTX-2.3 stage-specific denoising path
as the selected cache component in the full optimization stack.

## Why it is useful

The LTX-2.3 pipeline has two stages and only a few high-resolution refinement steps. A deterministic schedule can be safer than threshold replay when the sampler trajectory is short, sensitive, or unlike the schedule assumed by generic cache heuristics.

## Tunable knobs

- skipped step indices.
- stage placement.
- recompute boundaries.
- residual compensation rule.

## Validation

Inspect stage-specific artifacts as well as final output. A schedule that helps stage 1 can still be unacceptable if it destabilizes stage 2 refinement.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [Sol-Engine code](https://github.com/NVlabs/Sana/tree/sol-engine)
