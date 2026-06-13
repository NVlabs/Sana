# Goal: nvfp4_ffn

## Objective

Wire Cosmos3-Super to support an opt-in TE NVFP4 FFN load-time transform through
the shared `efficiency/` engine, using the LTX-2.3 NVFP4 FFN recipe as the
reference methodology.

## Context

- Execution repo: `Sol-LTX-Infer`
- Orchestration repo: `autovideo`
- Eval profile: `evals/profiles/official_video_t2v.toml`
- Existing transform: `efficiency.transforms.nvfp4_ffn.NVFP4FFN`

## Implementation Bounds

- Preserve the official Cosmos3-Super config exactly.
- Do not run GPU or Slurm jobs during implementation-only validation.
- Keep the NVFP4 path guarded by `SGLANG_HQ_ENABLE_TE_NVFP4_FFN`.
- Leave the disabled path on the baseline BF16 loader.
- Restrict quantization to FFN projection linears unless a separate candidate
  explicitly expands scope.
- Treat TE, CUDA, and Blackwell requirements as runtime prerequisites.

## Done When

- `candidates/nvfp4_ffn.toml` prepares through `scripts/launch_candidate.py`.
- `NVFP4FFN` composes through `efficiency/` for Cosmos3.
- Cosmos3 loader wiring consumes the env flag and can be disabled cleanly.
- `scripts/collect_run.py` produces canonical artifacts after a real run.
- `outputs/side_by_side.mp4` and visual judge output are present for enabled
  NVFP4 runs.
- Any blocker, such as missing TransformerEngine NVFP4 support, is recorded in
  `outputs/risk_notes.md`.
