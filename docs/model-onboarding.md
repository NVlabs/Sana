# Onboarding a model & expanding its interface (any model)

This is the **repeatable procedure** for making the acceleration search work on
**any** served model, and for **expanding which search dimensions it can run** by
wiring more seams. It generalizes the Cosmos3 work into a model-agnostic playbook.

The search engine is generic; a model plugs in through exactly two artifacts and a
set of **seams**. "Expanding the interface" = declaring + wiring more seams so more
`loops/*` dimensions become eligible (and functional) for that model.

---

## 0. The contract — what a model provides
1. **`efficiency/models/<id>_spec.py`** — a `ModelSpec` (register via
   `@register_model_spec("<Key>", "<TransformerClassName>")`): declares the
   `capabilities` the model exposes + a few seam **accessors**.
2. **`models/<id>.toml`** — a profile: official config, `[baseline]`, run entry,
   base env, and `[seam_status]` (the human tracker of which seams are wired).

`compose()` type-checks every technique's `required_capabilities` against the
spec's `capabilities`. A dimension whose capability the model does NOT declare is
auto-skipped — that is the eligibility gate (`search.py --model <id>` shows it).

---

## 1. The three levels of "wired" (the mental model)
For each seam, wiring proceeds through three levels — do not conflate them:

| Level | Means | How to check |
| --- | --- | --- |
| **(A) compose-eligible** | the capability is declared in the spec | `search.py --model <id>` lists the dimension (not `[skip]`) |
| **(B) runtime-functional** | the model actually honors the hook (accessor returns the real thing / pipeline honors the env / the forward stashes the signal) | a real run: the technique ON actually changes compute |
| **(C) quality-valid** | technique OFF == baseline (byte/numeric identical) AND ON passes the tier quality gate (incl. the Gemini judge) | GPU run + the eval pipeline (`docs/search-architecture.md`) |

Declaring a capability you have NOT wired (B) makes the search *try* the dimension
and it will be wrong at runtime — so declare only what you have wired, and track
the rest in `[seam_status]`.

---

## 2. Seam catalog — per seam: what it is, what it unlocks, how to wire
`Capability` / `Seam` enums live in `efficiency/technique.py`; accessors in
`efficiency/spec.py`. `efficiency/models/ltx2_spec.py` is the **fully-wired
reference** for every row below.

### Capability seams (gate ELIGIBILITY)
| Capability | Unlocks dimension(s) | Declare in spec | Model-code requirement (level B) |
| --- | --- | --- | --- |
| `BLOCKS` | foundation for block-level techniques | `Capability.BLOCKS` + `get_blocks=lambda tf: tf.<block_list>` | a transformer-block `ModuleList` (LTX: `transformer_blocks`; Cosmos3: `gen_layers`) |
| `PRUNABLE_TOKENS` | `token_prune` | `Capability.PRUNABLE_TOKENS` + `prunable_segment(hidden, ctx)->(start,end)`; add `prune_gather`/`prune_scatter` if the pruned forward needs per-token coords/timestep/masks (not plain `[B,S,C]`) | a separable prunable token span along `seq_dim` |
| `SWAPPABLE_ATTENTION` | `sparse_attention` (PISA) | `Capability.SWAPPABLE_ATTENTION` | attention routes through a swappable backend layer that honors `SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS` (LTX & Cosmos3 both use the `supported_attention_backends` / USPAttention mechanism — so usually declare + map the component name) |
| `RESIDUAL_TUPLE` | residual-cache-style techniques | `Capability.RESIDUAL_TUPLE` | block forward returns a residual-compatible tuple |

### Env-transform seams (compose on ANY model; need pipeline support for level B)
These transforms set `SGLANG_HQ_*` env and require **no** capability, so they are
eligible everywhere — but only DO something if the model's SGLang pipeline honors
that env at build/run. Track that under `[seam_status]`.
| Seam (written) | Dimension | Level-B requirement |
| --- | --- | --- |
| `ATTENTION_BACKEND` | `sparse_attention` | same as `SWAPPABLE_ATTENTION` above (it is the capability-gated form) |
| `FFN_PRECISION` | `nvfp4_ffn` | model's FFN honors the NVFP4 TE env (GB200/TE build) |
| `KERNEL_FUSION` | `kwl_fusion` | model honors the KWL fusion env |

### Runtime signal seams (no capability; gate FUNCTIONALITY, not eligibility)
| Signal | Dimension | Level-B requirement |
| --- | --- | --- |
| `scratch[("teacache_signal", cache_key)]` | `teacache` | the forward stashes the timestep-modulated-input each step (else TeaCache composes but no-ops to full compute) |
| (whole-step wrap) | `step_cache` | none — `Plan.on_step` wraps the step; eligible+functional on any model |

---

## 3. The procedure (any model `<id>`)
```
1. efficiency/models/<id>_spec.py: register a ModelSpec; declare Capability.BLOCKS
   + get_blocks. Add other capabilities ONLY as you wire them.
2. models/<id>.toml: official_config + [baseline] + run_script + [seam_status]
   (mark each seam wired / declared / todo / transform-env).
3. For each seam to expand:
     a. (level B) wire the model hook / confirm pipeline honors the env / stash the
        runtime signal — in the model code (the submodule), referencing ltx2_spec.py
        + ltx_2.py as the worked example.
     b. (level A) declare the Capability (+ accessor) in <id>_spec.py.
     c. update [seam_status].
4. python search/search.py --model <id>   # eligible dimensions grew
5. (level C, GPU) per newly-wired seam: a real run with the technique OFF must be
   byte/numeric-identical to baseline; ON must engage and pass the tier gate.
```

## 4. Verification
- **CPU (no GPU)**: `python search/search.py --model <id>` shows the new dimension
  as composable (not `[skip]`); mirror `efficiency/selftest.py` to assert
  compose() accepts the technique against the spec and OFF==identity on a fixture.
- **GPU (level C)**: launch the candidate (`scripts/launch_candidate.py` from the
  profile), collect `benchmark.json`/`quality.json`, confirm OFF==baseline and run
  the 3-stage eval (off_identity → LPIPS → Gemini visual judge) → tier.

## 5. Worked examples
- **LTX-2** (`efficiency/models/ltx2_spec.py`): the full reference — declares
  `BLOCKS, PRUNABLE_TOKENS, SWAPPABLE_ATTENTION, RESIDUAL_TUPLE`; all dimensions
  eligible; `ltx_full_opt` composes all five.
- **Cosmos3** (`efficiency/models/cosmos3_spec.py`, `models/cosmos3.toml`): current
  state — `BLOCKS` (`get_blocks→gen_layers`) + `PRUNABLE_TOKENS` declared →
  eligible: `step_cache`, `teacache`, `token_prune` (+ env transforms `nvfp4_ffn`,
  `kwl_fusion`). **TODO** (tracked in `[seam_status]`): declare
  `SWAPPABLE_ATTENTION` (Cosmos3 already routes attention through
  `supported_attention_backends`/USPAttention — declare + map the component name to
  unblock `sparse_attention`); stash `teacache_signal` in the forward (make TeaCache
  functional); refine `prunable_segment` to the video-token span.

## See also
`models/README.md` (the two artifacts), `docs/search-architecture.md` (the search
+ eval pipeline), `efficiency/README.md` (the engine), `efficiency/selftest.py`
(the seam/compose test pattern).
