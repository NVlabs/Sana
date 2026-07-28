# Runtime Onboarding Notes

This repo no longer ships built-in per-model `efficiency/models/*_spec.py`
files. New model work should start from the actual inference code and a model
profile, then express each candidate's required capabilities in its manifest.

## Current Contract

1. `models/<id>.toml` records official config, runtime env, baseline metadata,
   run script, and human seam/status notes.
2. Candidate TOML files declare `[requirements].capabilities`.
3. `efficiency.candidate_manifest.dry_run_manifest()` synthesizes a minimal
   `ModelSpec` from those manifest capabilities and runs `compose()`.
4. The runtime code under `Sol-LTX-Infer/` is the only place model-specific
   hooks, env consumers, and adapter glue should live.

## Adding Or Porting A Model

1. Add or update `models/<id>.toml`.
2. Confirm the runtime entrypoint can run the official baseline.
3. For each candidate, implement the pure algorithm in `efficiency/` when it is
   model-agnostic, and keep model-specific consumption in `Sol-LTX-Infer/`.
4. Declare only the capabilities that candidate needs in its TOML.
5. Run dry-run validation before GPU work; use GPU only to validate that the
   runtime actually consumes the path and preserves quality.

`ModelSpec` remains as the compose-time type object, but concrete built-in
Cosmos/LTX/Wan/Hunyuan spec files are intentionally absent until a runtime
adapter is proven and worth preserving.

## SOL Attention Models

Two models are onboarded with **Sol-Attn** (`techniques/sparse_backends/`)
in their full-optimization stacks:

- [**HunyuanVideo-13B**](../models/hunyuan_video/) — release benchmark pending
  (`candidates/hunyuan_video_full.toml`).
- [**Wan2.1-T2V-14B**](../models/wan21_t2v_14b/) — release benchmark pending
  (`candidates/wan21_14b_fullstack.toml`).
