# Wan2.2 T2V A14B follow-up optimization — 2026-07-12

## Result

The accepted follow-up stack is:

```text
context_parallel_ulysses4,
fused_qkv_projections,
compiled_block_glue,
compiled_qk_rope,
async_qkv_ulysses_a2a,
direct_ulysses_output_a2a,
reusable_ulysses_a2a_buffers,
invariant_rope_cache,
invariant_conditioning_cache
```

The new `reusable_ulysses_a2a_buffers` optimization reuses inference-only receive buffers for query, key, value, and direct output all-to-all transfers. It preserves source packing, message layout, and collective volume. Gradient-enabled execution falls back to fresh allocations.

## Full CP4 benchmark

All runs use 4 exclusive GB200 GPUs, 720x1280, 81 frames, 40 steps, seed 1024, CFG 4/3, flow shift 12, 5 prompts, and 2 warmup passes.

| Configuration | Total | Denoise | Speedup vs 129.01s | Quality evidence |
|---|---:|---:|---:|---|
| Existing cache t021 | 80.469s | 74.184s | 1.603x | LPIPS mean/max 0.176/0.205 |
| Existing fastest cache t030 | 77.239s | 70.789s | 1.670x | LPIPS mean/max 0.319/0.356 |
| Follow-up async/direct/reusable | **75.597s** | **69.268s** | **1.707x** | **81/81 frame SHA-256 matches t030** |

The follow-up is 1.642s, or 2.17%, faster than t030 at identical measured configuration. Prompt totals were 75.597, 69.933, 75.431, 75.613, and 75.639s; the aggregate is the median.

The run artifact is `runs/20260712-174030-wan22_t2v_a14b_async_direct_reusable_fused_invariant/outputs/benchmark.json`. Its Slurm job was `4847382`; it requested one exclusive node with all 4 GPUs and completed with an empty stderr log.

## Controlled rejection

The same stack with `native_flash_self_attention` added was tested on all 4 GPUs. It completed successfully but regressed to 174.088s total and 163.980s denoise with the same 35% EasyCache reuse. It is therefore excluded from the promoted stack; native Flash is not beneficial for this Wan CP4 layout on the tested GB200/Torch 2.11 environment.

## Quality and limitations

The follow-up output has 81 extracted frames. SHA-256 comparison against t030 found 81 identical frames and 0 differences, so this optimization did not change the generated video for the tested seed/prompts. The generic collector's NumPy/LPIPS judges remain deferred because the control Python environment lacks NumPy; the exact frame-hash audit is recorded as the additional deterministic quality evidence.

## Reproduction

- Candidate: `candidates/wan22_t2v_a14b_async_direct_reusable_fused_invariant.toml`
- Runtime implementation: `output/orchestrated/wan14b-20260706-155110/integrator/runtime/wan22_t2v_a14b_baseline/wan_kernel_optimizations.py`
- Output bundle: `runs/20260712-174030-wan22_t2v_a14b_async_direct_reusable_fused_invariant/outputs/`
