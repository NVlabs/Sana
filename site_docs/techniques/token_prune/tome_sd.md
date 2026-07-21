# ToMe-SD

ToMe-SD is a token merging method for diffusion inference. The paper cites it as an early example showing that redundant diffusion tokens can be merged during inference with limited quality loss.

## Role in Sol-Engine

ToMe-SD is part of the token-reduction design space. It is a candidate when merging similar tokens is safer than dropping tokens outright.

## Search knobs

- merge ratio.
- layer placement.
- timestep schedule.
- similarity metric.

## Validation

Token merging can blur fine detail. Inspect texture, object boundaries, and motion consistency.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [ToMe-SD](https://github.com/dbolya/tomesd)
- [Diffusers ToMe optimization guide](https://huggingface.co/docs/diffusers/en/optimization/tome)
