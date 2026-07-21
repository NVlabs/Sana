# Runtime Onboarding Notes

This repo no longer ships built-in per-model `efficiency/models/*_spec.py`
files. New model work should start from the actual inference code and a model
profile, then express each candidate's required capabilities in its manifest.

## Current Contract

1. `models/<id>.toml` records official config, runtime env, baseline metadata,
   run script, and human seam/status notes. Fully onboarded models ALSO have a
   canonical `models/<id>/model.toml` (schema_version 1) with a `[baseline]`
   table (manifest / runtime_root / eval_profile) and a `[baseline.copy]`
   worktree contract.
2. Candidate TOML files declare `[requirements].capabilities`.
3. `techniques.candidate_manifest.dry_run_manifest()` synthesizes a minimal
   `ModelSpec` from those manifest capabilities and runs `compose()`.
4. The vendored runtime under `models/<id>/{baseline,optimized}/` is the only
   place model-specific hooks, env consumers, and adapter glue should live
   (the former `Sol-LTX-Infer/` engine submodule is removed; see
   `snippets/sol-ltx-infer-reference.md`).

## Onboarded Models

| Model | Runtime | Canonical `model.toml` | Notes |
| --- | --- | --- | --- |
| **HunyuanVideo-13B** | `models/hunyuan_video/` | yes | joint `[video, text]` self-attention; **SOL Attention v2/v3** (`techniques/sparse_backends/sol_attn_hunyuan_v{2,3}.py`) + TeaCache + compile; audited hot delivery 5.03x (`candidates/hunyuan_video_full_v3.toml`) |
| **Wan2.1-T2V-14B** | `models/wan21_t2v_14b/` | yes | pure-video self-attention; **SOL Attention v1** (`techniques/sparse_backends/sol_attn_backend.py`, colmask + Morton + dense guard) + EasyCache + compile; 3.48x (`candidates/wan21_14b_fullstack.toml`) |
| Wan2.2 TI2V-5B / T2V-A14B | `models/wan22_*/` | yes | kernel + EasyCache (+ PISA on A14B CP4) |
| SANA-Video 5B | `models/sana_video/` | yes | kernel/cache + EasyCache + PISA |
| LingBot-Video | `models/lingbot_video/` | yes | kernel + phase-specific PISA |
| Cosmos3 / Bernini | flat profile only | no | see `optimization_reports/` |

## Adding Or Porting A Model

1. Add or update `models/<id>.toml`, then the canonical `models/<id>/model.toml`
   once the baseline is reproducible.
2. Vendor the runnable baseline into `models/<id>/baseline/` and confirm it can
   run the official config; mirror it into `models/<id>/optimized/` with
   env-gated technique seams (OFF identity: no env flags = byte-identical
   baseline path).
3. For each candidate, implement the pure algorithm in `techniques/` when it is
   model-agnostic, and keep model-specific consumption in the model runtime.
4. Declare only the capabilities that candidate needs in its TOML.
5. Run dry-run validation before GPU work; use GPU only to validate that the
   runtime actually consumes the path and preserves quality.
6. Measurement convention: `benchmark.json` schema v2 (`total_s`, `timing_scope`,
   `warm_steady_state`), hot after a warmup pass with all technique clocks
   (SOL step clock, cache controllers) cold-started for the timed pass; quality
   via `scripts/video_quality_metrics.py` against baseline frames.

`ModelSpec` remains as the compose-time type object, but concrete built-in
Cosmos/LTX/Wan/Hunyuan spec files are intentionally absent until a runtime
adapter is proven and worth preserving.
