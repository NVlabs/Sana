# References

## Upstream Checkout

- `Sol-LTX-Infer` commit: `29d0d9e464000a2472345dcad51054b15aacca8d`
- Branch containing the commit: `origin/codex/cosmos3-run-env`

## Migrated Files

- `Sol-LTX-Infer/scripts/slurm_ltx23_best_nvfp4_video_attn_ffn_sglquant_1080p10s.sh`
  -> `loops/nvfp4_ffn/reference/recipe.sh`
- `Sol-LTX-Infer/scripts/bench_te_nvfp4_gelu_epilogue.py`
  -> `loops/nvfp4_ffn/reference/bench_te_nvfp4_gelu_epilogue.py`
- `Sol-LTX-Infer/docs/diffusion/quantization.md`
  -> `loops/nvfp4_ffn/reference/report.md`

## Studied But Not Copied

- `Sol-LTX-Infer/scripts/slurm_ltx23_best_nvfp4_video_qkv_ffnout_1080p10s.sh`
  confirmed the related QKV/FFN-out recipe shape.
- `Sol-LTX-Infer/scripts/bench_ltx2_nvfp4_vs_bf16_gemm.py`
  confirmed the GEMM microbench shape; the TE FFN epilogue bench is the migrated
  representative helper.
- `Sol-LTX-Infer/scripts/run_ltx23_sglang_hq_1080p10s.sh`
  confirmed how `SGLANG_HQ_ENABLE_TE_NVFP4_FFN` expands into the lower-level
  LTX TE NVFP4 env.

## In-Repo References

- `efficiency/transforms/nvfp4_ffn.py`: `NVFP4FFN` transform under test.
- `efficiency/selftest.py`: section `[7]` full-opt NVFP4 env and no-FP4 variant.
- `efficiency/models/cosmos3_spec.py`: target model spec for future wiring.
