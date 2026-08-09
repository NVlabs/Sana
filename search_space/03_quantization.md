# Search Space: NVFP4 Linear Quantization

**Scope**: Explore low-precision linear-layer paths for the target model using
NVIDIA Transformer Engine NVFP4 or a model-specific FP4 equivalent. Start from
profiling and code inspection, not a fixed grid. The goal is to find retained
frontier config that improve quality or speed/memory, then let the main
agent select final low/medium/high winners after the 40-iteration budget.

This dimension is hardware-sensitive. A config that cannot exercise real FP4
hardware or a validated fallback should be recorded as a blocker or diagnostic,
not as a speed result.

---

## Background

Transformer Engine exposes `NVFP4BlockScaling` for Blackwell-class GPUs. The
recipe uses two-level scaling: local block scaling over groups of 16 values plus
a global tensor scale. Current TE docs describe 2D weight quantization, random
Hadamard transforms (RHT), stochastic rounding, and row-scaled activation as
recipe-level choices. The current target runtime already consumes the three
disable flags for RHT, stochastic rounding, and 2D quantization; other axes may
require config-side loader wiring.

Primary references:

- NVIDIA Transformer Engine NVFP4 documentation.
- Transformer Engine common API for `NVFP4BlockScaling`.
- NVIDIA technical note on NVFP4 micro-block scaling.

---

## Required Hardware And Runtime Preflight

Run this before config iterations consume GPU budget:

- GPU architecture: confirm Blackwell/SM100 or later for native NVFP4.
- CUDA, cuDNN, FlashInfer, and TransformerEngine versions.
- `transformer_engine.common.recipe.NVFP4BlockScaling` import.
- Minimal `te.Linear` smoke, or a target-loader smoke if direct TE import is not
  enough.
- FP4 GEMM backend availability, for example auto, cutlass, cudnn, trtllm, or
  target-supported FlashInfer backends.
- OFF path identity: disabled NVFP4 must recover the baseline path.
- Env consumption: prove that every env var used by the config is read by the
  target loader, or mark it metadata-only.

If this preflight fails because hardware or libraries are unavailable, report a
real blocker with exact versions and commands.

---

## Method Families

### 1. Conservative FFN-Only NVFP4

Quantize the FFN linear layers with the smallest plausible scope. For DiT-like
models, this usually starts with `proj_in` and `proj_out` or equivalent
gate/up/down projections. Exclude attention, output heads, embeddings, and tiny
linears until profiling shows they matter.

Useful axes:

- FFN submodule subset.
- Stage/component scope.
- Layer windows to keep BF16.
- Step windows to keep BF16.
- BF16 fallback on unsupported shapes.

### 2. Selective Hot-Linear NVFP4

Profile end-to-end runtime and apply FP4 only to linears that materially affect
latency or peak memory. This can include attention projections or output
projections, but only after profiling shows real benefit.

Useful axes:

- Top-K linear layers by CUDA time.
- Exclude small GEMMs where quantization overhead dominates.
- Exclude quality-sensitive early/late blocks.
- Separate stage-1, stage-2, text, and video modules when the model has them.

### 3. TE Recipe Variants

Explore recipe variants instead of assuming the historical "disable everything"
recipe is best.

Useful axes:

- `disable_rht`: true/false.
- `disable_stochastic_rounding`: true/false.
- `disable_2d_quantization`: true/false.
- `row_scaled_activation`: true/false when the installed TE version and loader
  support it.
- Weight-only vs weight+activation FP4 when the runtime exposes that distinction.

Record which axes are already consumed by the runtime and which were newly wired
by the config.

### 4. Dense Guard Policies

Use BF16 for sensitive windows while keeping FP4 where speed/memory improves.

Useful axes:

- Guard first N blocks.
- Guard last N blocks.
- Guard both ends.
- Guard early denoising steps.
- Guard late denoising steps.
- Content or prompt class fallback when a specific artifact pattern appears.

### 5. Backend And Padding Policy

FP4 speed can depend more on kernel/backend and shape alignment than on nominal
precision. Treat backend and padding as first-class search axes.

Useful axes:

- FP4 GEMM backend: auto, cutlass, cudnn, trtllm, or target-supported FlashInfer
  variants.
- Row padding, for example `pad_m_to=16` or backend-preferred multiples.
- Warm vs cold compile state.
- Per-shape fallback to BF16 when padding or quantization overhead dominates.

### 6. Fused Epilogue Paths

When the target runtime has TE fused paths, evaluate them separately from plain
NVFP4 linear replacement.

Useful axes:

- `proj_in + GELU`.
- `proj_out + bias/gate`.
- Fused path only for profiled modules.
- Fused path disabled on shape/runtime failure with clear fallback.

---

## Config Loop Template

Each iteration should write a hypothesis like:

```text
Config <id>:
- module scope:
- TE recipe:
- backend/padding:
- dense guards:
- expected win:
- prior failure avoided:
- env vars consumed by loader:
- OFF identity check:
- rejection evidence:
```

Then launch exactly one config, run the authoritative gate, and record one of:

- `quality_improved`
- `speed_improved`
- `quality_and_speed_improved`
- `discarded_regression`
- `rejected`
- `blocked`
- `structured_negative`

Do not treat "failed tier budget" as loop completion. Retain a config when
quality improves or speed/memory improves; discard it only when neither quality
nor speed/memory improves.

---

## Quality And Artifact Risks

Common failure signatures:

- Temporal hallucination or new objects from aggressive FP4 scope.
- Texture shimmer, static, snow, or flicker.
- Detail loss in text, faces, hands, or high-frequency structure.
- OFF identity break because disabled env still changes loader state.
- No-op config where env is set but target loader never consumes it.
- Speed regression due to quantization overhead, padding, fallback, or compile
  state.
- Inconclusive result because the run used non-Blackwell hardware or missing TE
  kernels.

Use aligned LPIPS and aligned pairwise Gemini as the quality source of truth for
final tier selection. Collector `quality.json` is telemetry when it conflicts
with aligned artifacts.

---

## Retained Frontier Record

For every retained config, record:

- config id and manifest;
- changed files and exact env vars;
- quantized module list;
- TE recipe flags and backend;
- dense layer/step guards;
- hardware, CUDA, cuDNN, FlashInfer, TransformerEngine versions;
- warm/cold compile state;
- run dir, benchmark, peak memory, side-by-side video, LPIPS, pairwise Gemini;
- whether the config improved quality, speed/memory, or both.

---

## Structured Negative Requirements

Do not declare structured negative until the summary covers:

- hardware/runtime preflight;
- at least one conservative FFN-only attempt;
- at least one recipe variant;
- at least one dense-guard attempt;
- at least one backend or padding check when backend control exists;
- best speed/memory point and its quality failure;
- best quality point and its speed/memory failure;
- why remaining module scopes or recipe variants are redundant or lower value.
