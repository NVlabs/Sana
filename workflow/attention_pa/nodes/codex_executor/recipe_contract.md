## Required PISA Recipe Set

The final deliverable is `PISA-RECIPES.json` with three measured operating
points. A recipe is an executable implementation contract, not a qualitative
label or an untested suggestion.

### Recipe Tiers

- `conservative`: highest measured quality at the least aggressive useful
  compute reduction; disclose any none/low-severity shift.
- `balanced`: highest measured quality at a middle compute budget; disclose any
  none/low/medium-severity shift.
- `aggressive`: highest measured quality at the fastest useful compute budget;
  high-severity loss may be disclosed, but critical degradation is not a
  deliverable recipe.

### Required JSON Shape

```json
{
  "schema_version": 1,
  "model_id": "<experiment model_id>",
  "workflow_uid": "attention_pa",
  "recipes": {
    "conservative": {},
    "balanced": {},
    "aggressive": {}
  }
}
```

Each recipe object must include:

- `status="measured"`, transfeat id, parent id, source commit/hash, and run dir;
- authoritative local PISA source path, commit and SHA-256, copied/adapted
  experiment-local source hashes, actual backend/kernel, block size, route
  mode, exact-remainder approximation configuration, dense fallback, and guard
  name;
- scalar default `density` and `sparsity`, plus complete per-layer and per-step
  overrides and attention-type policy;
- dense layer groups, dense step windows, PISA layer groups, PISA step windows,
  and any head-specific policy;
- Q/K mismatch, GQA, masks, RoPE/layout, and only-video-self-attention handling;
- PISA dispatches, dense fallbacks, mask-search overhead, kernel timing, peak
  memory, full end-to-end wall time, and speedup versus the canonical dense run;
- LPIPS, `codex_visual_overall`, maximum artifact severity, artifact list,
  independent reviewer verdict path, and aligned assessment path;
- exact launch environment/config and a concise explanation of why this point
  belongs in the tier.

All three recipe entries must use distinct transfeat ids, strictly increasing
measured speedups, and distinct existing `assess_verdict.json` files from
fixed-contract full runs. A visual fail remains valid measured evidence and may
define a balanced or aggressive point when its severity fits that tier. Do not mark
`AGENT-STATUS.json.status=complete`
until this file is valid and consistent with `PISA-SEARCH-STATE.json`,
`SEARCH_JOURNAL.md`, and `SUMMARY.md`.
