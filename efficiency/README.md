# `efficiency/` — the LTX-2.3 acceleration engine (ported into autovideo)

This package is the **generic inference-acceleration framework** proven on
**LTX-2.3** in `Sol-LTX-Infer`, ported here as first-class repo code (not
vendored). It is the shared substrate every acceleration loop in `loops/` builds
on: each loop reuses these techniques/transforms and the `compose()` type-check,
and only adds the model-specific seam wiring + recipe for its direction.

## Provenance
- Source: `Sol-LTX-Infer` @ branch `codex/cosmos3-run-env` (`29d0d9e`),
  `python/sglang/multimodal_gen/runtime/efficiency/`.
- Ported verbatim with the import prefix rewritten
  `sglang.multimodal_gen.runtime.efficiency` → `efficiency`. No logic changes.
- The original ran on a GPU-less login node via a parent-package stub hack; in
  this repo it is a plain top-level package, so `selftest.py` imports it directly.

## What's here
- `spec.py` — `ModelSpec`: a model's declaration of the seams it exposes
  (capabilities + accessors). compose() type-checks a technique's
  `required_capabilities` against it.
- `technique.py` — `Technique`, `Capability`, `Seam`, `Phase`, `TechniqueContext`.
- `transform.py` — `ModelTransform` (build/load-time env/graph changes).
- `compose.py` — `compose(items, spec) -> Plan`: capability check + seam-conflict
  detection (exclusive writers, schedule-aware) + ordered execution plan.
- `schedule.py` — the schedule DSL (`const`, `before`, `at_steps`, `by_stage`).
- `registry.py` — `register_model_spec`, `get_model_spec`, `build_technique`.
- `presets.py` — `ltx_full_opt()`: the proven LTX-2.3 5-component config.
- `techniques/` — runtime per-step (OFF == byte-identical baseline):
  `token_prune.py` (fully-worked reference), `step_cache.py`, `teacache.py`.
- `transforms/` — build/load-time: `kwl_fusions.py`, `nvfp4_ffn.py`,
  `sparse_attention.py` (these delegate to the existing `SGLANG_HQ_*` env, they
  do not reimplement the kernels).
- `models/ltx2_spec.py` — the **worked LTX-2.3 spec** (the reference template).
- `models/cosmos3_spec.py` — the **Cosmos3 target spec** (get_blocks wired;
  technique seams declared conservatively, each loop wires the rest).

## Verify (independent test)
Needs `torch` (use the sana env, which has it):
```bash
~/lustre/miniconda3/envs/sana/bin/python efficiency/selftest.py
```
Exercises: schedule DSL, capability rejection, seam-conflict detection,
token-prune OFF==identity + real gather/scatter round-trip, the registry, and
the LTX-2.3 `ltx_full_opt` 5-component compose (KWL/NVFP4/PISA env + per-stage
gating). Expect `23 passed, 0 failed`.

## How loops use it
A `loops/<direction>/` reuses one technique/transform here, wires the matching
seam onto `models/cosmos3_spec.py`, and supplies the LTX-2.3 recipe + report
that direction was proven with. See each loop's `README.md`.
