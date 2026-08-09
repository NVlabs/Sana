## Layer-Step-Density PISA Search

The search product is a measured policy over attention layer, denoising step,
and exact-block density. Do not stop at one global sparsity number.

Represent each policy explicitly as:

```text
mode[layer_index, step_index, attention_type] = dense | pisa
density[layer_index, step_index, attention_type] = value in (0, 1]
```

`density=1.0` is dense-equivalent. A `pisa` cell must execute the real PISA
exact and approximate phases. Cross-attention and text/prompt attention remain
dense by default; approximate them only as a separately identified transfeat
with independent quality evidence.

### Required Preflight

Before tuning, inspect and record:

- actual transformer layer count and stable layer identifiers;
- all self-, cross-, linear-, dense-softmax-, and model-specific attention
  paths, including which ones dominate runtime;
- step indexing and solver call count for the fixed official-step workload;
- Q/K/V shapes, head count, head dimension, GQA or Q/K-length mismatch, token
  frame/tile layout, masks, and positional encoding;
- existing dense backend, available PISA kernel/backend, block-size constraints,
  and whether the transfeat really dispatches sparse kernels;
- per-layer and per-step dense attention latency plus mask-selection overhead.

Write this to `runs/pisa_preflight/attention_map.json` and reference it from
`PISA-SEARCH-STATE.json`.

### Adaptive Search Order

Use evidence-driven bracketing rather than a blind Cartesian grid:

1. Validate one faithful global PISA path at high exact density and prove OFF
   identity, dispatch, fallback counters, and complete output materialization.
2. Profile layer sensitivity by keeping all steps fixed and changing one layer
   group at a time. Identify mandatory-dense and PISA-tolerant layers.
3. Profile step sensitivity by keeping the layer policy fixed and changing one
   contiguous step window at a time. Test early, middle, and tail regions rather
   than assuming all steps have equal sensitivity.
4. Bracket exact density inside tolerant regions. A density near `0.10` is a
   reasonable initial prior, so useful probes include `0.05`, `0.10`, `0.15`,
   and `0.25`. This prior does not hold for every model, layer, step, shape, or
   backend: skip, widen, or interpolate points when the target model's measured
   quality and speed identify a different boundary.
5. Compose the measured layer and step policies, then rerun the complete
   full-evaluation contract. Interaction results, not multiplied isolated
   speedups, determine the recipe.
6. Refine around each quality boundary until another density or schedule change
   is unlikely to alter its recipe classification.

Quality movement beyond the target tier should move the next child toward more
dense layers/steps, higher density, smaller sparse windows, or denser fallback,
while preserving the measured point as frontier evidence. Quality pass with
weak speed should reduce density or expand PISA only in measured tolerant
regions. If dispatch occurs but wall time does not improve, optimize mask
selection, block layout, and kernel integration before increasing sparsity.

### Durable Search State

Maintain `PISA-SEARCH-STATE.json` with at least:

```json
{
  "schema_version": 1,
  "active_transfeat_id": "<id>",
  "attention_map": "runs/pisa_preflight/attention_map.json",
  "trials": [
    {
      "transfeat_id": "<id>",
      "parent_transfeat_id": "<id or empty>",
      "backend": "<actual implementation>",
      "block_size": [64, 64],
      "route_mode": "score",
      "route_bias": false,
      "only_video_self_attention": true,
      "density": 0.5,
      "sparsity": 0.5,
      "layer_policy": {},
      "step_policy": {},
      "attention_types": {},
      "adaptation_reason": "<prior evidence selecting this child>",
      "run_dir": "runs/<run>",
      "dispatch": {},
      "speedup": 0.0,
      "lpips_max": 0.0,
      "codex_visual_overall": "pass | fail",
      "max_artifact_severity": "none | low | medium | high",
      "outcome": "retain | refine | discard | retry"
    }
  ],
  "next_trial": {}
}
```

Every child must name its parent and explain how prior full-run speed, LPIPS,
blind Codex visual artifacts, dispatch, and overhead selected the changed axis.
