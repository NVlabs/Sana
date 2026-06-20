# Search Space: Kernel Fusion and Quality-Gated Operator Optimization

**Scope**: Find kernel-level and graph-level implementation optimizations that
preserve the same model algorithm. KWL candidates may change floating-point
operation order, fused-multiply-add behavior, launch grouping, memory layout, or
backend implementation, but must not change the scheduler, step count, token
set, prompt/guidance state, LoRA state, resolution, frame count, attention
semantics, cache semantics, pruning semantics, or quantization policy.

The `KWLFusions` transform is a build-time helper that emits
`SGLANG_HQ_KWL_*` flags. It is useful for ablations, but it is not the whole
search space. Subagents should inspect the target-model hot path directly and
may implement exact fused operators, backend swaps, compile wrappers, graph
capture, or layout fixes in the execution repo.

## Quality-Gated Frontier Contract

KWL is an implementation optimization dimension with a strict semantic boundary,
not a bit-exact-only dimension. Bit-exact or dtype-rounding-only candidates are
preferred when they exist, but non-bit-exact kernel/backend paths are valid
frontier candidates when their drift is declared and the authoritative visual
evidence is recorded.

- OFF path must be identity to baseline for guarded code paths.
- ON path may change floating-point order, FMA/epilogue behavior, compiler
  lowering, backend implementation, or use a declared approximate kernel path.
- Every candidate must record its expected tolerance class: bit-exact,
  dtype-rounding-only, reduction-order drift, FMA/epilogue drift, fast-math
  drift, or approximate-kernel drift.
- Use the standard fixed-budget frontier rule: retain a candidate when quality
  improves, latency improves, peak memory improves, or both quality and
  efficiency improve. Do not discard a speed/memory win only because it is not
  bit-exact; keep the aligned quality evidence for final tier selection.
- Final low/medium/high winners are selected after the 40-iteration budget by
  speed target and aligned quality ranking, the same as other dimensions.
- Any candidate that changes sampling, denoising steps, token count, attention
  density, cache reuse, quantization policy, prompt handling, or output shape is
  not KWL. Route it to the appropriate dimension instead.

## Required Preflight

Before proposing the first runnable candidate, record:

- hot-path profile or code-inspection evidence: dominant kernel families,
  launch count, memory traffic, tensor shapes, dtype, and repeated operator
  chains;
- environment and backend availability: PyTorch/CUDA versions, available
  attention kernels, Triton/Inductor availability, CUTLASS/cuBLASLt/FlashInfer
  or project-local fused kernels when relevant;
- compile/graph state: cold compile cost, warm steady-state timing, graph
  breaks, dynamic-shape guards, CUDA graph compatibility, and whether timing is
  cold, warm, or cache-reused;
- identity proof: OFF flag leaves the baseline path byte-identical or otherwise
  proves no guarded code executes;
- risk list: shape polymorphism, dtype casts, aliasing/in-place writes,
  stream/event ordering, RNG use, host-device syncs, and fallback path behavior.

## Method Families

These are method families, not a fixed grid. Each candidate should select one
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
  retained candidates.

### 3. Attention-Adjacent Fusion

Optimize work around attention without changing which tokens attend to which
tokens.

Possible targets:

- Q/K RMSNorm plus RoPE fusion;
- QKV projection packing or batching when weights and bias layout permit;
- bias or mask placement inside an attention backend that already supports the
  same dense attention semantics;
- packed QKV layout conversion removal;
- dense FlashAttention/FlashInfer/backend selection for the same mask, causal
  mode, scale, dropout-off state, and dtype.

Hard boundary:

- changing attention sparsity, windowing, token dropping, or approximate masks
  belongs to sparse attention or token pruning, not KWL.

### 4. Compile and Graph Capture

Use compiler or capture mechanisms to reduce launch overhead while preserving
the eager graph.

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
- whether the speedup persists under warm repeated runs.

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

### 9. Backend Selection and Fallback Policy

Try an equivalent or quality-gated approximate backend only when the semantic
boundary and fallback policy are proven.

Possible targets:

- dense attention backend selection among project-local kernels,
  FlashAttention/FlashInfer, Triton, or framework SDPA;
- GEMM backend selection among cuBLASLt, CUTLASS, Triton, or TorchInductor;
- exact fallback for unsupported shape/dtype/device combinations;
- warm cache and autotune-state management.

Guardrails:

- fallback must be visible in logs/manifests;
- a candidate cannot report speedup from silently skipping work or falling back
  to a different algorithm;
- autotune and compile state must be labelled in every timing result.

## Search Axes

- hot operator pattern: GEMM epilogue, norm/modulate, attention-adjacent,
  layout/copy, compile/graph, launch batching, stream overlap, decode/postprocess
- scope: one module, one block family, attention-only, FFN-only, VAE-only,
  postprocess-only, whole repeated denoising region
- backend path: eager PyTorch, TorchInductor, Triton, cuBLASLt, CUTLASS,
  project-local CUDA/C++ op, FlashAttention/FlashInfer, CUDA graph
- guard: env flag, module flag, shape/dtype guard, warm-cache guard, fallback
  policy
- numerical tolerance: bit-exact, dtype-rounding-only, reduction-order drift,
  FMA/epilogue drift, fast-math drift, approximate-kernel drift
- timing state: cold compile, warm compile, autotuned, CUDA graph replay,
  cache-reused
- validation surface: module tensor diff, OFF identity, full render gate,
  launch/profile evidence

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
- cold compile/autotune time versus warm steady-state time.

## Structured Negative Standard

A structured negative is acceptable only after the subagent records:

- at least six KWL method families considered, including exact-preferred and
  quality-gated approximate variants where relevant, and why each is unsafe,
  unavailable, already fused, or not hot enough;
- profile or code evidence for the top remaining hot spots;
- backend availability and fallback evidence;
- OFF identity results for any touched guard path;
- expected speed ceiling explaining why more KWL work is unlikely to produce a
  useful retained frontier candidate.

## Primary References

- PyTorch `torch.compile` and CUDA graph behavior:
  <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- NVIDIA CUDA Graphs programming guide:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- NVIDIA cuBLASLt epilogue enum reference:
  <https://docs.nvidia.com/cuda/nvmath-python/0.1.0/bindings/generated/nvmath.bindings.cublasLt.Epilogue.html>
- NVIDIA CUTLASS GEMM API and epilogue/mainloop fusion surface:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html>
