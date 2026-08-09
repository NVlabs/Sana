#!/usr/bin/env bash
# LTX-2.3 BASELINE technique env — the official two-stage pipeline with every
# acceleration seam explicitly OFF.
#
# Every knob is written out with an explicit value rather than left unset, so a
# stale exported variable in the caller's shell cannot silently turn part of the
# optimized stack on inside a "baseline" run. This file is the control arm: if a
# knob is not listed here, it is not a seam this pipeline has.
#
# Provenance: expanded from clear_lossy_env() + disable_kwl_env() in
# Sol-LTX-Infer scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh (see
# models/ltx23/model.toml [[baseline.reference_only]] for the pinned commit).

# --- Attention math path -----------------------------------------------------
# LTX-2.3 official selects FlashAttention4 for unmasked DiT attention on B200.
# The dense baseline must stay on the same attention math, otherwise
# video-to-audio cross attention falls back to SDPA and drifts from official.
export SGLANG_LTX2_OFFICIAL_FA4_ATTENTION="${SGLANG_LTX2_OFFICIAL_FA4_ATTENTION:-1}"

# --- KWL operator fusion: all off -------------------------------------------
export SGLANG_LTX2_SHARE_BLOCK0_SELF_ATTN=0
export SGLANG_LTX2_SHARE_GUIDANCE_PREFIX=0
export SGLANG_LTX2_COMPILE_MARK_STEP_BEGIN=0
export SGLANG_LTX2_COMPILE_PREWARM_PERTURBATION_MASKS=0
export SGLANG_LTX2_FUSED_QK_ROPE=0
export SGLANG_LTX2_FUSED_RMS_ADALN=0
export SGLANG_LTX2_FUSED_ADALN=0
export SGLANG_LTX2_FUSED_MODULATE=0
export SGLANG_LTX2_FUSED_RESIDUAL_GATE=0
export SGLANG_LTX2_FUSED_QKNORM=0
export SGLANG_LTX2_FUSED_QKNORM_ROPE=0
export SGLANG_LTX2_FUSED_DUAL_MODULATE=0
export SGLANG_LTX2_FUSED_CA_DUAL_MODULATE=0
export SGLANG_LTX2_FUSED_ADA_VALUES=0
export SGLANG_LTX2_FUSED_ADA_VALUES_ALL=0
export SGLANG_LTX2_FUSED_ADA_DIRECT=0
export SGLANG_LTX2_FUSED_Q_GATE=0
export SGLANG_LTX2_FUSED_QKV=0
export SGLANG_LTX2_FUSED_AUDIO_QKVG=0
export SGLANG_LTX2_FUSED_KV=0
export SGLANG_LTX2_FUSED_FFN_PROJ_IN_GELU=0
export SGLANG_LTX2_FUSED_GELU_INPLACE=0
export SGLANG_LTX2_COMPILE_GATE_TO_OUT=0
export SGLANG_LTX2_COMPILE_GATE_TO_OUT_RESIDUAL=0
export SGLANG_LTX2_COMPILE_A2V_GATE_TO_OUT=0
export SGLANG_LTX2_COMPILE_VAE_DECODER=0
export SGLANG_LTX2_COMPILE_TILED_VAE_DECODER=0
export SGLANG_ENABLE_FUSED_QKNORM_ROPE=0

# --- Stage-1 step-skip cache: off -------------------------------------------
export SGLANG_LTX2_STAGE1_CACHE_CORE_ENABLED=0

# --- NVFP4 video FFN: off ----------------------------------------------------
export SGLANG_LTX2_TE_NVFP4_VIDEO_FFN=0
export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU=0
export SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE=0

# --- Stage-2 midpoint token prune: off --------------------------------------
unset SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_RATIO
unset SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_METHOD
unset SGLANG_LTX2_STAGE2_MIDPOINT_PRUNE_STEPS

# --- Stage-2 PISA sparse attention: off -------------------------------------
unset SGLANG_PIECEWISE_ATTN_SPARSITY
unset SGLANG_PIECEWISE_ATTN_DENSITY
unset SGLANG_PIECEWISE_ATTN_BLOCK_SIZE
unset SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF
unset SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE
unset SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS
unset SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY
unset SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY
unset SGLANG_PIECEWISE_ATTN_DENSE_LAYERS
unset SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS
unset SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS
unset SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER
unset SGLANG_PIECEWISE_ATTN_ROUTE_MODE
unset SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK

# The generate CLI takes no attention-backend overrides in the baseline arm.
LTX23_COMPONENT_ATTENTION_BACKENDS=""
LTX23_ATTENTION_BACKEND_CONFIG=""
LTX23_CACHE_ALGO="none"
LTX23_VARIANT="baseline"
