# Goal: sparse_attention

## Objective

Wire PISA sparse video self-attention onto Cosmos3 through the existing
`efficiency.transforms.sparse_attention.SparseAttention` transform and the
SGLang HQ `SGLANG_HQ_*` attention-backend configuration path.

## Context

- Execution repo: `Sol-LTX-Infer`
- Orchestration repo: `autovideo`
- Eval profile: `evals/profiles/official_video_t2v.toml`
- LTX-2.3 source proof: `Sol-LTX-Infer` @ `29d0d9e`

## Constraints

- Preserve the official Cosmos3-Super config in
  `evals/profiles/official_video_t2v.toml`.
- Do not reimplement the sparse-attention kernel; delegate to the existing
  `piecewise_attn` backend selected by `SGLANG_HQ_*`.
- Keep the disabled path equivalent to baseline by leaving the sparse env unset
  or routing all components to the dense fallback.
- Confirm Cosmos3 component names and layer guards before reusing the LTX
  `transformer` / `transformer_2` names.
- Keep generated artifacts under the canonical names from
  `docs/artifact-contract.md`.

## Done When

- `candidates/sparse_attention.toml` launches with
  `scripts/launch_candidate.py`.
- Cosmos3 declares `Capability.SWAPPABLE_ATTENTION` only after the attention seam
  is actually wired.
- OFF equals the same-seed baseline under the official config.
- Sparse attention demonstrates at least exploratory speedup and passes
  quantitative plus visual quality gates.
- `scripts/collect_run.py` produces `benchmark.json`, `quality.json`,
  `risk_notes.md`, `patch_summary.md`, and `collection.json`.
