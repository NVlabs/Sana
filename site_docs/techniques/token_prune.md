# Token pruning

Token pruning removes low-salience video tokens in selected steps. It targets model-level redundancy in long spatiotemporal sequences.

The paper treats token pruning as a sequence-reduction family: the token-pruning agent tunes the pruning criterion, pruning ratio, layer schedule, timestep schedule, and reconstruction rule, then validates temporal coherence and late-step fidelity.

## In Sol-Engine

LTX-2.3 uses token pruning in the full optimization stack together with kernel fusion, cache, PISA sparse attention, and NVFP4.

Implemented entries:

- `python/sglang/multimodal_gen/runtime/efficiency/techniques/token_prune.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`
- `python/sglang/multimodal_gen/runtime/efficiency/models/ltx2_spec.py`
- `python/sglang/multimodal_gen/runtime/efficiency/spec.py`

## Methods

| Method | Open-source status | Integration | Role in the design space |
|---|---|---|---|
| [Feature-norm pruning](token_prune/feature_norm.md) | open-source local | local LTX runtime | Sol-Engine LTX-2.3 midpoint token pruning implementation |
| [ToMe-SD](token_prune/tome_sd.md) | open-source | external adapter | diffusion token merging baseline family |

## Practical notes

- Prune only in steps where visual sensitivity is acceptable.
- Keep prompt, seed, scheduler, and resolution fixed when comparing variants.
- Inspect videos side by side; scalar metrics are not enough for final acceptance.
