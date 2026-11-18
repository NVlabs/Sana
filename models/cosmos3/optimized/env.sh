#!/usr/bin/env bash
# Cosmos3-Super OPTIMIZED technique env — the shipped acceleration stack.
#
# Two techniques compose here:
#   1. TeaCache  — residual-similarity step reuse, threshold 1.15, start step 10,
#                  max 3 consecutive skips (encoded in the variant string that
#                  the upstream matrix driver parses)
#   2. NVFP4     — step-selective FP4 linear on gate_up/down/qkv/out, with the
#                  first 3 and last 3 denoising steps kept dense
#
# Provenance: the `fullopt` branch of the SGLang runtime
# scripts/cosmos/slurm_cosmos3_super.sh (pinned commit in
# models/cosmos3/model.toml). Published speedup: 2.26x
# (site_docs/pipelines/cosmos3.md), measured on 4x GB200 at 1280x720 / 189
# frames / 35 steps with warmup excluded.

# TeaCache thr 1.15 / start 10 / max 3. The knobs live in this variant string
# rather than in env vars — changing the numbers means changing the string, and
# the upstream driver must recognise it.
COSMOS3_VARIANT="teacache_c115_s10_m3"
COSMOS3_ARM="fullopt"

# --- step-selective NVFP4 linear ---------------------------------------------
export SGLANG_COSMOS3_FP4_LINEAR=1
export SGLANG_COSMOS3_FP4_TARGETS="${SGLANG_COSMOS3_FP4_TARGETS:-gate_up,down,qkv,out}"
# First/last steps stay dense: FP4 on the earliest steps moves the trajectory
# far more than on the middle ones, and on the last steps it shows up directly
# as visible artifacts.
export SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS="${SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS:-3}"
export SGLANG_COSMOS3_FP4_SKIP_LAST_STEPS="${SGLANG_COSMOS3_FP4_SKIP_LAST_STEPS:-3}"
