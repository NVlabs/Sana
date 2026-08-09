#!/usr/bin/env bash
# Cosmos3-Super BASELINE technique env — official 64B 4-GPU sequence-parallel
# inference with every acceleration seam explicitly OFF.
#
# Every knob carries an explicit value rather than being left unset, so a stale
# export in the caller's shell cannot switch part of the optimized stack on
# inside a "baseline" run and quietly corrupt the control arm.
#
# Provenance: the `baseline` branch of Sol-LTX-Infer
# scripts/cosmos/slurm_cosmos3_super.sh (pinned commit in
# models/cosmos3/model.toml).

# Cache variant string consumed by the upstream matrix driver. "baseline" means
# no step reuse at all.
COSMOS3_VARIANT="baseline"
COSMOS3_ARM="baseline"

# --- step-selective NVFP4 linear: off ----------------------------------------
export SGLANG_COSMOS3_FP4_LINEAR=0
unset SGLANG_COSMOS3_FP4_TARGETS
unset SGLANG_COSMOS3_FP4_SKIP_FIRST_STEPS
unset SGLANG_COSMOS3_FP4_SKIP_LAST_STEPS
