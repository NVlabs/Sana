#!/usr/bin/env bash
# ported from Sol-LTX-Infer scripts/run_ltx23_sglang_hq_1080p10s.sh @ 29d0d9e
#
# Reference only. Do not execute this loop reference as a Cosmos3 launcher.
# Cosmos3 component names and layer guards may differ from the LTX-2.3
# `transformer` / `transformer_2` names used below.

enable_stage2_sparse_env() {
  export SGLANG_PIECEWISE_ATTN_SPARSITY="${SGLANG_PIECEWISE_ATTN_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_BLOCK_SIZE="${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE:-64}"
  export SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF="${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF:-true}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE=false
  export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS=0
  export SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY="${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY:-0.9}"
  export SGLANG_PIECEWISE_ATTN_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS:-none}"
  export SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS:-none}"
  export SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS="${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS:-none}"
  export SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER="${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER:-true}"
  export SGLANG_PIECEWISE_ATTN_ROUTE_MODE="${SGLANG_PIECEWISE_ATTN_ROUTE_MODE:-score}"
  export SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK="${SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK:-fa}"
  COMPONENT_ATTENTION_BACKENDS="${SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS:-transformer=fa,transformer_2=piecewise_attn}"
  ATTENTION_BACKEND_CONFIG="piecewise_sparsity=${SGLANG_PIECEWISE_ATTN_SPARSITY},piecewise_block_size=${SGLANG_PIECEWISE_ATTN_BLOCK_SIZE},piecewise_only_video_self_attention=${SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF},piecewise_stage1_schedule=false,piecewise_stage1_dense_steps=0,piecewise_stage1_start_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY},piecewise_stage1_end_sparsity=${SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY},piecewise_dense_layers=${SGLANG_PIECEWISE_ATTN_DENSE_LAYERS},piecewise_stage1_dense_layers=${SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS},piecewise_stage2_dense_layers=${SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS},piecewise_approx_remainder=${SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER},piecewise_route_mode=${SGLANG_PIECEWISE_ATTN_ROUTE_MODE},piecewise_dense_fallback=${SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK}"
}

# Equivalent concise env shape used by the in-repo SparseAttention transform for
# this loop's test and candidate manifest.
sparse_attention_sglang_hq_env() {
  export SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS="${SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS:-transformer=fa,transformer_2=piecewise_attn}"
  export SGLANG_HQ_ATTENTION_BACKEND_CONFIG="${SGLANG_HQ_ATTENTION_BACKEND_CONFIG:-piecewise_sparsity=0.9,piecewise_block_size=64,piecewise_only_video_self_attention=true,piecewise_stage1_schedule=false,piecewise_stage1_dense_steps=3,piecewise_stage2_dense_layers=0,piecewise_approx_remainder=true,piecewise_route_mode=score,piecewise_dense_fallback=fa}"
}
