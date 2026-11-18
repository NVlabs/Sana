# Separate LoRA branches with consumer fusion

Use `--lora-mode fused --compute-quant none` to keep separate LoRA branch
arithmetic and fuse its BF16 sums into the following consumer kernels.
`--lora-mode separate` uses native PEFT branches with the other Sol-H3
optimizations retained. The default `--lora-mode merged` preserves the existing
weight-merged path. The Python API accepts the same `lora_mode` argument.

The branch loader accepts the public PEFT LoRA and FastVideo v2 hybrid formats
supported by `h3_runtime/lora.py`. Hybrid parameter corrections still apply to
the base parameters in FP32; the low-rank A/B matrices remain separate.
It does not add or change a checkpoint's sampler, conditioning or model layout.

The kernel specialization requires one active scale-1 adapter in inference mode,
ordinary BF16 Linear projections, identity LoRA dropout and no projection hooks.
It supports arbitrary adapter names. Unsupported runtime states fall back to
native PEFT. Unsupported installation configurations fail before any consumer
patch is installed. Quantized linear compute is not supported in branch modes.

The implementation retains the same W, A, B and three GEMMs per projection.
Each fused consumer rounds `base + delta` to BF16 before applying its existing
arithmetic. Consumers cover QKV normalization/RoPE/packing, attention residual
modulation, SwiGLU, and the final FFN gate/residual operation. QKV consumer fusion
uses the multi-GPU Ulysses path; one-GPU dense inference retains native Q/K/V
projections. Projections outside the main transformer blocks remain native.

This preserves the separate branch's rounding rather than forming a rounded
merged weight. Exactness tests compare against native branches with the other
acceleration settings held constant. Enabling this mode does not disable
SOL/BSA sparse attention or low-precision attention transport, and does not imply
equality to an entirely unaccelerated model. Results across untested inputs,
GPUs or library versions require their own validation.

## Synthetic tests

The tests use generated tensors and tiny public-format adapters, without model
weights, generation prompts or application-specific checkpoints.

```bash
PYTHONPATH=. python -m pytest -q tests/test_lora_branches.py tests/test_lora_fusion_state.py
# Run only inside a GPU allocation. The large case uses tens of GiB.
H3_RUN_LARGE_GPU_TESTS=1 PYTHONPATH=. python -m pytest -q tests/test_lora_fusion.py
```

Coverage includes native PEFT branch arithmetic, public-format loading and
hybrid corrections, invalid-checkpoint rollback, non-default adapter names,
BF16 rounding boundaries, non-contiguous inputs, indexed gate tables, promoted
dtype fallback, BF16/INT8 QKV packet bytes and row addressing beyond 2^31 elements.
The final FFN fusion also combines gate lookup, multiplication and residual
addition while preserving the intermediate BF16 rounding. The LoRA GEMMs remain.
