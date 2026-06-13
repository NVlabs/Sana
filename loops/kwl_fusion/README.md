# Loop: kwl_fusion

## Purpose

Bring the LTX-2.3 kernel-wise lossless (KWL) operator-fusion experience into a
Cosmos3 acceleration loop. This loop does not copy the shared efficiency
framework; it preserves the LTX recipe, helper, and report evidence, then tests
that the in-repo `efficiency/transforms/kwl_fusions.py` transform emits the
expected `SGLANG_HQ_KWL_*` build-time flags.

## LTX-2.3 Provenance

Source checkout: `/home/haozhel/lustre/auto-video/Sol-LTX-Infer` at `29d0d9e`.

- Recipe wrapper: `scripts/run_ltx23_sglang_hq_kwl_1080p10s.sh`
- Env mapping: `scripts/run_ltx23_sglang_hq_1080p10s.sh`
- Helper code: `scripts/ltx23_official_kwl_ops.py`
- Report: `docs/ltx23_official_hq_kwl_report.md`
- Detailed fusion report: `docs/diffusion/ltx2_dit_fusion_report.md`

The migrated reference files live under `reference/`. They are not runtime
dependencies for Cosmos3.

## Success Story

The LTX KWL path kept the official HQ two-stage pipeline and applied only
operator-level execution changes inside the transformer build path. The report
states that the KWL branch did not include sparse attention, FP4, step-count
changes, scheduler changes, prompt changes, CFG changes, or LoRA-strength
changes. The measured official HQ KWL run improved end-to-end wall time from
`321.71s` to `256.27s` (`1.26x`) while preserving the algorithmic graph.

The reusable lesson is the fusion catalog and flag discipline:

- Q/K RMSNorm plus RoPE
- RMS/AdaLN and dual modulation chains
- FFN `proj_in + GELU`
- gate-to-output compiled subgraphs
- audio Q/K/V/gate projection fusion
- tiled VAE decoder compile

Each path is individually flag-gated. OFF must recover the baseline path, and
ON is accepted only after same-config benchmark and quality review.

## Efficiency Mapping

The generic entrypoint is `efficiency.transforms.kwl_fusions.KWLFusions`.
It is a build-time `ModelTransform` that writes `Seam.KERNEL_FUSION` and sets
the `SGLANG_HQ_KWL_*` env keys consumed by model-specific build code.

This loop's independent test composes `KWLFusions` through `efficiency.compose`
and checks that `plan.apply_transforms(None, stage, env)` sets the exact KWL
bundle, including `SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE=1`.

## Cosmos3 Wiring

To run this on Cosmos3, a future implementation branch must wire the flags into
the Cosmos3 implementation in `Sol-LTX-Infer`:

- keep `efficiency/models/cosmos3_spec.py` as the target `ModelSpec` for
  `get_model_spec("Cosmos3")`;
- make Cosmos3 module construction read the `SGLANG_HQ_KWL_*` flags or a
  Cosmos3-specific alias set produced from them;
- add model-specific fused operator paths around the hot Cosmos3 DiT op chains;
- keep every fused path independently flag-gated with OFF equal to baseline;
- add same-noise/off-identity and official-profile quality artifacts before
  promotion.

Cosmos3 currently declares `BLOCKS` and `PRUNABLE_TOKENS`; KWL does not require
a new capability in the current framework because it is an env-only build
transform. The missing work is the model-specific fused kernels and build-time
flag consumption.

## Candidate

`candidate.toml` is mirrored at `../../candidates/kwl_fusion.toml` for
`scripts/launch_candidate.py`.

## Eval

`eval.toml` points at `evals/profiles/official_video_t2v.toml`.

## Independent Test

Run:

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/kwl_fusion/test_kwl_fusion.py
```

The test is CPU-only and does not execute the reference LTX recipe or helper.

## Status

ready-for-codex
