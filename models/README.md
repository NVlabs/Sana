# `models/` — runtime model profiles

This directory stores model profiles: official benchmark config, baseline
numbers, run entrypoint, base env, and seam/status notes. The actual runtime
adapter code lives under `Sol-LTX-Infer/`; the reusable algorithm layer lives in
`efficiency/`.

## A model profile contains

- official config and runtime env
- baseline timing/quality metadata
- run script and submodule commit
- human seam/status notes for what the current runtime consumes

Candidate manifests declare their required capabilities. During dry-run,
`efficiency.candidate_manifest` builds a minimal manifest-derived `ModelSpec`
and `compose()` checks the selected technique/transform against that contract.

Current profile: `cosmos3` for `nvidia/Cosmos3-Super`.
