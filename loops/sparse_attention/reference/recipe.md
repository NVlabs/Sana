<!-- ported from Sol-LTX-Infer scripts/run_ltx23_sglang_hq_1080p10s.sh @ 29d0d9e -->

# Sparse Attention Recipe Excerpt

The LTX-2.3 runner selected PISA sparse attention through SGLang HQ environment
keys rather than by reimplementing the kernel in the acceleration framework.
The stage-2 focused recipe routes `transformer_2` to `piecewise_attn` and keeps
`transformer` dense:

```bash
SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS="transformer=fa,transformer_2=piecewise_attn"
SGLANG_HQ_ATTENTION_BACKEND_CONFIG="piecewise_sparsity=0.9,piecewise_block_size=64,piecewise_only_video_self_attention=true,piecewise_stage1_schedule=false,piecewise_stage1_dense_steps=3,piecewise_stage2_dense_layers=0,piecewise_approx_remainder=true,piecewise_route_mode=score,piecewise_dense_fallback=fa"
```

Cosmos3 component names and layer guards may differ. Before promotion, confirm
which Cosmos3 attention modules correspond to LTX's `transformer` and
`transformer_2`, then update the manifest env only after the runtime seam is
wired.
