# Skill: Operator Fusion (the "KWL" idea) — model-specific, methodology not code

> **When to use:** you are an agent optimizing a NEW model (after its SGLang
> baseline is correct) and want the "KWL"-style speedup. There is **no generic
> runnable KWL** — the fusions target a specific model's ops. This skill is the
> **recipe** to find and implement them for your model, plus how to register the
> result as a build-time `ModelTransform` in the efficiency framework.

## What operator fusion is

Collapse adjacent **small ops** (norms, RoPE, adaLN scale/shift, activations,
gating, residual adds) into a **single kernel or a `torch.compile` region**, to
cut (a) per-op kernel-launch overhead and (b) HBM round-trips of intermediates.
Lossless or near-lossless — it changes *how* the math runs, not the math. This is
a **③ BUILD-phase** optimization (chosen at module construction), not a runtime
data-flow technique.

## Why it's model-specific (and what transfers)

The exact op sequences differ per model, so the kernels differ — that's why it
can't be a generic function. **What transfers is the catalog of recurring DiT
fusion patterns + the procedure.** Look for these in your model:

| pattern | what to fuse | typical win |
|---|---|---|
| **QK-norm + RoPE** | RMSNorm on Q/K together with the rotary embedding | medium |
| **adaLN modulation** | the timestep-embedding scale/shift/gate elementwise with the surrounding norm | medium |
| **FFN proj_in + activation** | the up-projection bias + GELU/SiLU as a GEMM epilogue | high (FFN is hot) |
| **gate → out + residual** | the gated output projection + residual add | medium |
| **QKV projection** | 3 separate Q/K/V projections → 1 batched GEMM | medium |
| **dual / cross-attn modulation** | conditioning applied to both streams | model-dependent |

## Procedure (the skill)

1. **Profile first.** Run the event profile, rank steady-state hotspots, find
   **chains of small ops sitting between GEMMs** (those are the launch/HBM-bound
   victims). Don't fuse blindly — fuse what the profile says is hot.
2. **Match a pattern** from the catalog to your model's actual ops.
3. **Implement, cheapest first:**
   - **`torch.compile` region** around the op chain (Inductor fuses elementwise +
     epilogues automatically). Usually the best effort/reward.
   - **Fused Triton/CUDA kernel** only when compile leaves it bound (e.g. a custom
     epilogue compile won't capture).
4. **Flag-gate it.** Every fusion behind its own env/config flag; **OFF must be
   byte-identical to baseline** (so you can LOO-ablate and ship safely).
5. **Verify numerics.** Same-noise compare vs baseline. Watch the known traps:
   - **GELU tanh vs erf**: a fused GELU may pick the other approximation → real
     drift. Pin the same one.
   - **fp32 accumulation**: a fused kernel may accumulate differently; check.
   - quantized (FP4) paths interacting with the fused epilogue.
6. **Measure per fusion (leave-one-out).** Keep only fusions with a real win;
   some compile regions cost more than they save.
7. **Register as a build-time `ModelTransform`** for your model:
   ```python
   # the framework provides the generic env-trigger mechanism; YOUR model's
   # build code reads these flags and swaps in the fused kernels.
   class MyKWL(ModelTransform):
       name = "mymodel_fusions"; phase = TransformPhase.BUILD
       writes = frozenset({Seam.KERNEL_FUSION})        # shared seam, composes freely
       def applies_to(self, spec): return spec.name == "MyModel"   # NOT a no-op elsewhere
       def set_env(self, ctx): ctx.env.update({"MYMODEL_FUSE_QK_ROPE": "1", ...})
   ```
   Then add it to that model's `*_full_opt()` preset. `applies_to` keeps it from
   silently no-op'ing on models that don't honor the flags.

## Hard-won pitfalls (from the LTX-2.3 instance)

- **`torch.compile` graph breaks**: a custom `autograd.Function` (e.g. an in-place
  fused norm) can't be traced → graph break + no fusion. Prefer a pure-torch
  `CustomOp.forward` that Inductor can fuse, and skip the inplace path under
  `torch.compiler.is_compiling()`.
- **In-place ops break CUDA-graph / inference tensors**: an in-place mutation
  bumps a version counter that fails on inference-mode / NVFP4-TE tensors
  ("Inference tensors do not track version counter"). Use **out-of-place** ops in
  fused paths (this exact trap broke the LTX2 token-prune until the rope-cache
  `_version` read was guarded).
- **VAE / decoder compile** is a separate fusion target (tiled VAE decode).
- Fusions interact: a fused FFN proj_in+GELU and an NVFP4 FFN want a *single*
  combined path, not two competing ones.

## Worked example

LTX-2.3's KWL bundle lives in `runtime/models/dits/ltx_2.py` (the `SGLANG_LTX2_
FUSED_*` / `SGLANG_HQ_KWL_*` flags: fused QK-RoPE, RMS-AdaLN, dual/CA-dual
modulation, FFN proj_in+GELU, compiled gate-to-out, audio QKVG, tiled-VAE
compile). Measured LOO win ≈ **1.23×**. Read it as a concrete template — then
build YOUR model's equivalent following the procedure above.

---

*This is a methodology skill, not runnable code. The same shape applies to other
inherently model-specific optimizations (e.g. NVFP4 FFN quantization): the
framework provides the generic mechanism + applicability check; you implement the
model-specific kernels/quant following a recipe like this and register a
`ModelTransform`.*
