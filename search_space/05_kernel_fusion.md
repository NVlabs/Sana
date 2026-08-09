# Search Space: Kernel Fusion and Quality-Gated Operator Optimization

**Scope**: Find kernel-level and graph-level implementation optimizations that
preserve the same model algorithm. KWL transfeat may change floating-point
operation order, fused-multiply-add behavior, launch grouping, memory layout, or
custom-kernel implementation, but must not change the scheduler, step count,
token set, prompt/guidance state, LoRA state, resolution, frame count, attention
semantics, cache semantics, pruning semantics, quantization policy, or framework
attention/backend dispatch policy.

The `KWLFusions` transform is a build-time helper that emits
`SGLANG_HQ_KWL_*` flags. It is historical diagnostic scaffolding, not a valid
transfeat by itself. Subagents should inspect the target-model hot path directly
and may implement exact fused operators, custom kernel wrappers, module-local
microbenchmarks, or layout fixes in the execution repo.

## Lossless-First Implementation Direction

Start from repeated operator chains that are already present in the target
runtime and preserve the same mathematical program. Prefer exact or
lossless-within-existing-dtype optimizations before global compiler changes:

- fuse adjacent pointwise operations around existing matrix multiplies,
  normalizations, modulations, activations, gates, residual adds, and output
  conversions when the same tensors and parameters are consumed;
- remove redundant layout, allocation, copy, transpose, cast, or metadata work
  when aliasing and in-place safety are proven;
- pack or batch equivalent projections, small kernels, or repeated launch sites
  when this preserves the same inputs, outputs, masks, ordering constraints, and
  dtype contract;
- use compile, regional compile, CUDA graph capture, or launch batching only for
  stable repeated regions, with cold/warm/graph-replay timings separated;
- cache static descriptors, offsets, masks, layout metadata, and workspace
  allocations only when their values are shape/prompt invariant for the current
  benchmark contract.

Framework backend selection, SDPA backend swaps, FlashAttention/FlashInfer
dispatch switches, and env-flag-only bundles are out of scope for KWL agent
startup. Do not implement, resume, or re-run backend-selection probes. A KWL
transfeat must modify an operator, module, layout path, custom kernel, or
microbenchmarked DiT-level fusion boundary.
If `AGENT-STATUS.json`, `SEARCH_JOURNAL.md`, or a prior run directory contains
a partially completed backend-selection probe, mark that record stale/cancelled
and start a fresh module/DiT microbench transfeat instead of resuming it.

## Microbench-First Contract

Do not launch a full denoising/video generation run for a new KWL idea. First
write a module-level or DiT-block-level warm paired microbenchmark that:

- constructs representative tensors from the target model shape contract;
- runs OFF baseline and ON transfeat implementations in the same process, same
  Slurm allocation/GPU, same input tensors, same dtype, and same warmed cache
  state;
- uses explicit warmup iterations before timing and reports repeat statistics:
  median, p25/p75, min/max, iteration count, and OFF/ON ordering;
- measures latency, launch count or profiler evidence, peak memory when useful,
  and max/mean numerical difference;
- proves OFF identity for guarded code;
- records expected full contribution:
  `saved_ms_per_call * calls_per_step * steps`, percent of measured DiT/block
  time, and expected full-run speedup ceiling;
- records exact reproduction commands and writes a durable JSON result.

Promote a transfeat to full inference only when the microbench shows positive
latency or peak-memory movement under this warm paired DiT/module test and the
tensor difference is inside the declared tolerance. Full denoise is final
visual/quality sanity and gross-regression evidence; it is not the primary speed
authority for sub-percent KWL transfeat. If a full denoise later has visual
artifacts, suspect a kernel, layout, aliasing, masking, or module-boundary bug
first; do not dismiss it as small numeric drift accumulation without
module-level evidence.

## Hunyuan Diffusers DiT Fusion Reference

For `hunyuan_diffusers`, inspect the installed Diffusers transformer before
editing. The current structure has 20 dual-stream transformer blocks and 40
single-stream transformer blocks, with 24 attention heads, head dim 128, inner
dim 3072, patch size 2, and patch size t=1. Use these examples as concrete
starting points for microbench transfeat:

- attention-adjacent Q/K projection output + QK RMSNorm + RoPE application;
- packed latent QKV projection and packed text added-QKV projection, preserving
  the same dense mask and token order;
- single-stream block `cat(attn_output, mlp_output) -> proj_out -> gate ->
  residual`, replacing the materialized concatenation with equivalent split
  projection or fused epilogue;
- dual-stream attention residual gates: `x + gate * attn` for latent and text
  branches;
- dual-stream `LayerNorm -> scale/shift` before FFN for latent/text branches;
- FFN output epilogue `x + gate * ff(x)` for latent and text branches;
- attention output split/projection for latent/text outputs when the split
  shape is static;
- final `norm_out -> proj_out -> reshape/permute/flatten` output-layout path;
- attention mask and RoPE/layout descriptor construction when the values are
  invariant for the official benchmark shape.

Only after these module-local fusion transfeat are exhausted should an agent
try `torch.compile`, regional compile, CUDA graph capture, or other global graph
machinery.

## Quality-Gated Frontier Contract

KWL is an implementation optimization dimension with a strict semantic boundary.
Bit-exact or dtype-rounding-only transfeat are preferred. Non-bit-exact custom
kernel/operator paths are valid only when their module-level tensor drift is
declared, microbenchmarked, and then visually gated at the end.

- OFF path must be identity to baseline for guarded code paths.
- ON path may change floating-point order, FMA/epilogue behavior, custom kernel
  lowering, or use a declared approximate kernel path.
- Every transfeat must record its expected tolerance class: bit-exact,
  dtype-rounding-only, reduction-order drift, FMA/epilogue drift, fast-math
  drift, or approximate-kernel drift.
- Use the standard fixed-budget frontier rule: retain a transfeat when quality
  improves, latency improves, peak memory improves, or both quality and
  efficiency improve. Do not discard a speed/memory win only because it is not
  bit-exact; keep the aligned quality evidence for final tier selection.
- Final low/medium/high winners are selected after the 40-iteration budget by
  speed target and aligned quality ranking, the same as other dimensions.
- Any transfeat that changes sampling, denoising steps, token count, attention
  density, cache reuse, quantization policy, prompt handling, or output shape is
  not KWL. Route it to the appropriate dimension instead.
- Full denoising/video generation is a final validation step only, not the first
  measurement surface for kernel work.

## Required Preflight

Before proposing the first runnable transfeat, record:

- hot-path profile or code-inspection evidence: dominant kernel families,
  launch count, memory traffic, tensor shapes, dtype, and repeated operator
  chains;
- environment and kernel availability: PyTorch/CUDA versions, Triton/Inductor
  availability, cuBLASLt/CUTLASS capabilities, and project-local fused kernels
  when relevant;
- microbench plan: tensor shapes, warmup/iteration counts, paired OFF/ON
  ordering, median/p25/p75/min/max stats, diff metrics, profiler or
  launch-count collection, expected full contribution, and acceptance criterion;
- compile/graph state only after module-level transfeat are exhausted: cold
  compile cost, warm steady-state timing, graph breaks, dynamic-shape guards,
  CUDA graph compatibility, and whether timing is cold, warm, or cache-reused;
- identity proof: OFF flag leaves the baseline path byte-identical or otherwise
  proves no guarded code executes;
- risk list: shape polymorphism, dtype casts, aliasing/in-place writes,
  stream/event ordering, RNG use, host-device syncs, and fallback path behavior.

## Method Families

These are method families, not a fixed grid. Each transfeat should select one
family, prove why it is hot for the target model, implement one mechanism, and
record the expected numerical tolerance and aligned quality evidence.

### 1. GEMM Epilogue Fusion

Fuse post-GEMM work into the GEMM epilogue or into the closest available backend
primitive.

Possible targets:

- FFN `proj_out + bias + residual + gate` epilogues;
- `linear + bias + GELU/SwiGLU` epilogues;
- residual add as the GEMM `beta * C` operand when layout allows;
- cuBLASLt epilogues such as bias, ReLU, or GELU variants when the backend path
  exposes them;
- CUTLASS/Triton custom epilogues for patterns not covered by library enums.

Evidence to collect:

- GEMM shape and stride stability;
- whether output layout can avoid extra contiguous/copy kernels;
- separate timing for GEMM, epilogue elementwise kernels, and fused path;
- max/mean diff and aligned quality gate result.

### 2. Norm, Modulation, and Residual Fusion

Fuse exact transformer block elementwise chains around normalization and
modulation.

Possible targets:

- RMSNorm/LayerNorm with scale/shift;
- AdaLN scale, shift, and gate application;
- dual modulation and cross-attention modulation;
- residual add plus gate/multiply chains;
- repeated norm-factor reuse when the same input is modulated multiple ways.

Guardrails:

- reduction order may differ, but epsilon, dtype promotion, and affine
  parameters must match baseline semantics;
- in-place outputs must not alias tensors consumed later by the baseline graph;
- compare both module-level tensor diffs and full generated output quality when
  tensor diffs are practical; full aligned quality evidence is required for
  retained transfeat.

### 3. Attention-Adjacent Fusion

Optimize work around attention without changing which tokens attend to which
tokens.

Possible targets:

- Q/K RMSNorm plus RoPE fusion;
- QKV projection packing or batching when weights and bias layout permit;
- packed QKV layout conversion removal;
- fused attention output split plus output projection when latent/text split
  sizes are static;
- static mask, descriptor, or layout metadata preparation when it avoids repeated
  allocations and preserves the same dense attention semantics.

Hard boundary:

- changing attention sparsity, windowing, token dropping, or approximate masks
  belongs to sparse attention or token pruning, not KWL.

### 4. Compile and Graph Capture

Use compiler or capture mechanisms to reduce launch overhead while preserving
the eager graph. This family is lower priority than module-local kernel/operator
fusion; use it only after microbench evidence shows no viable local fusion
transfeat remains.

Possible targets:

- `torch.compile(..., mode="reduce-overhead")` for stable elementwise-heavy
  callables;
- `torch.compile` fullgraph or regional compile for stable submodules;
- CUDA graph capture for static-shape repeated denoising blocks;
- pre-warmed graph replay for repeated step shapes;
- graph-break repair around Python control flow, `.item()`, syncs, dynamic
  allocations, or shape-dependent branches.

Evidence to collect:

- graph break count and reason;
- cold compile time and warm timing after adequate warmup;
- memory pool or static-address constraints;
- fallback behavior when shape, dtype, or device changes.

### 5. Memory Layout and Copy Elimination

Remove exact no-op layout churn, dtype churn, and avoidable allocation/copy
kernels.

Possible targets:

- redundant `.contiguous()`, `reshape`, `permute`, `view`, and layout conversion
  chains;
- repeated dtype casts between equal-precision tensors;
- preallocated output/workspace buffers for stable shapes;
- fused transpose plus projection layout;
- pinned host transfer or device-local staging only when semantics are
  unchanged.

Guardrails:

- views must preserve aliasing expectations;
- removing a copy must not expose later in-place mutation differences;
- allocator improvements must be measured separately from compute improvements.

### 6. Launch Overhead Reduction

Reduce the number of small kernels without changing the model-level semantic
boundary.

Possible targets:

- batch identical small kernels over heads, modalities, or blocks;
- combine scalar arithmetic and pointwise post-processing;
- persistent buffers for repeated step-local temporaries;
- precomputed static metadata such as shape descriptors, offsets, or launch
  parameters;
- move Python-side loops into a vectorized or fused callable when the loop body
  is semantically identical.

Evidence to collect:

- launch count before/after;
- CPU-side enqueue time and GPU timeline gaps;
- whether the speedup persists under paired OFF/ON warm repeated runs in the
  DiT/module context rather than a one-off full-video run against a historical
  baseline.

### 7. Overlap, Streams, and Pipeline Scheduling

Overlap independent work only when dependency proofs are explicit.

Possible targets:

- independent modality branches;
- asynchronous H2D/D2H transfers that are not on the critical path;
- VAE/postprocess overlap with next independent stage when outputs are not read
  early;
- separate CUDA streams with events for exact dependency ordering.

Guardrails:

- no data race, aliasing conflict, RNG reordering, or hidden sync;
- deterministic outputs must remain within the same numeric tolerance as the
  single-stream baseline;
- timeline evidence must show actual overlap rather than shifted idle time.

### 8. Decode, VAE, and Postprocess Fusion

Apply fusions outside the denoiser when profiling shows they matter.

Possible targets:

- tiled VAE compile/capture;
- decoder norm/activation chains;
- exact pixel-space postprocessing, scaling, clamp, cast, and layout conversion;
- chunk/tile loop overhead reduction with identical tile boundaries.

Guardrails:

- frame count, tile overlap, blending weights, color transform, and output dtype
  must match baseline semantics;
- postprocess fusions cannot hide missing or reordered frames.

## Search Axes

- hot operator pattern: GEMM epilogue, norm/modulate, attention-adjacent,
  layout/copy, compile/graph, launch batching, stream overlap, decode/postprocess
- scope: one module, one block family, attention-only, FFN-only, VAE-only,
  postprocess-only, whole repeated denoising region
- implementation path: eager PyTorch reference, custom Triton, cuBLASLt/CUTLASS
  epilogue, project-local CUDA/C++ op, TorchInductor/CUDA graph only after
  module-local transfeat are exhausted
- guard: env flag, module flag, shape/dtype guard, warm-cache guard, fallback
  policy
- numerical tolerance: bit-exact, dtype-rounding-only, reduction-order drift,
  FMA/epilogue drift, fast-math drift, approximate-kernel drift
- timing state: cold compile, warm compile, autotuned, CUDA graph replay,
  cache-reused
- validation surface: warm paired DiT/module latency, module tensor diff, OFF
  identity, launch/profile evidence, expected full contribution, full render
  visual gate only after microbench success

## Profiling Setup

```bash
# nsys for timeline:
nsys profile --trace cuda,nvtx -o /tmp/profile \
    python -m sglang.multimodal_gen ...

# ncu for kernel-level throughput:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --target-processes all python ...
```

Key metrics:

- latency and peak memory versus model baseline;
- time per kernel type: GEMM, elementwise, softmax/attention, norm, layout/copy,
  VAE/postprocess;
- kernel launch count per block and per denoising step;
- host enqueue gaps, host-device syncs, graph breaks, and dynamic-shape guards;
- memory bandwidth versus compute utilization;
- cold compile/autotune time versus warm steady-state time;
- paired OFF/ON warm medians for DiT/block/module-level tests. Do not claim a
  small KWL speedup from a single full run compared only against a historical
  canonical baseline.

## Structured Negative Standard

A structured negative is acceptable only after the subagent records:

- at least six KWL method families considered, including exact-preferred and
  quality-gated approximate variants where relevant, and why each is unsafe,
  unavailable, already fused, or not hot enough;
- profile or code evidence for the top remaining hot spots;
- kernel implementation availability and fallback evidence for custom ops;
- OFF identity results for any touched guard path;
- expected speed ceiling explaining why more KWL work is unlikely to produce a
  useful retained frontier transfeat.

## Primary References

- PyTorch `torch.compile` and CUDA graph behavior:
  <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- NVIDIA CUDA Graphs programming guide:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- NVIDIA cuBLASLt epilogue enum reference:
  <https://docs.nvidia.com/cuda/nvmath-python/0.1.0/bindings/generated/nvmath.bindings.cublasLt.Epilogue.html>
- NVIDIA CUTLASS GEMM API and epilogue/mainloop fusion surface:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html>
