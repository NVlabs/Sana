# Sol-LTX-Infer Reference Snippets

This page captures the minimum reusable patterns from `Sol-LTX-Infer`.

## Candidate Families

| Family | Existing source | What to reuse |
| --- | --- | --- |
| KWL fusion | `ltx2-dit-fusion-report`, `docs/ltx23_official_hq_kwl_report.md` | Lossless/operator-only acceptance language, leave-one-out reporting, side-by-side artifacts. |
| Sparse attention | `ltx-sparse-attn-bringup`, `ltx-stage1-sparse-schedule`, `docs/ltx23_sglang_hq_variants.md` | Piecewise attention env shape, dense warmup schedule, layer dense guards. |
| Cache | `sglang-ltx-cache`, `scripts/make_ltx23_cache_report.py` | Cache stats parsed from logs, speedup table, skipped-step reporting. |
| NVFP4 | `ltx2-nvfp4-two-stage-cleanup` | Quantized path must include side-by-side video and visual judge; treat PSNR as diagnostic. |
| Cosmos3 sparse | `codex/pisa0-cosmos3-sparse` | Candidate target for first Cosmos3 env/patch goal. |

## Sparse Attention Env Shape

From the LTX reports, a reusable candidate snippet is:

```bash
SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS="transformer=piecewise_attn,transformer_2=piecewise_attn"
SGLANG_HQ_ATTENTION_BACKEND_CONFIG="piecewise_sparsity=0.9,piecewise_block_size=64,piecewise_only_video_self_attention=true,piecewise_stage1_dense_steps=3,piecewise_stage2_dense_layers=0,piecewise_dense_fallback=fa"
```

Cosmos3 may need different component names and layer guards, so this is a
starting point, not a copy-paste guarantee.

## Cache Report Shape

Cache reports should include:

- baseline total/denoise/stage time
- candidate total/denoise/stage time
- speedup ratios
- skipped steps or hit/compute counts parsed from logs
- notes on which stages/layers are eligible

## Visual Report Shape

Every lossy or numerically sensitive candidate should write:

```text
outputs/side_by_side.mp4
outputs/frame_metrics.json
outputs/visual_judge.json
```

The existing `make_side_by_side_video.py` and `make_multiway_video.py` scripts
are good enough reference implementations for the first version.
