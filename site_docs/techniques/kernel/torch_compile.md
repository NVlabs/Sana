# torch.compile

`torch.compile` provides compiler-driven graph fusion and specialization for stable operator sequences. In Sol-Engine, it is used where compiler fusion is more practical than maintaining custom kernels for every shape.

## Sol-Engine placement

| Pipeline | Scope | Role |
|---|---|---|
| SANA-Video | compiled denoising path | part of the 2.77x full optimization stack |
| LTX-2.3 | selected graph regions | candidate when stable under target shapes |

## Practical notes

- cold compile time should be excluded from warmup-excluded latency numbers.
- compile mode and cache directory can change observed performance.
- compiler fusion should be disabled for regions that deadlock or regress.

## Validation

Run at least one warmup before quoting latency and compare generated video outputs under the same prompt and seed.

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [torch.compile documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
