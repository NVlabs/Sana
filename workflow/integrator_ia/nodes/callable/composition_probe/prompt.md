Run all eight kernel/PISA/cache toggle conditions from the integration
experiment's source tree. Load and warm the registry-resolved target model, text
encoder, VAE, and one-time compile/autotune paths before measurement. Time each
sample from immediately before text-encoder compute through synchronized VAE
decode completion. Record exact guard values, dispatches, fallbacks, per-stage
timings, spanning `sample_total_s`, timing distribution, and source hash.

Pairwise conditions are mandatory. Exclude process startup, model/text
encoder/VAE loading, compile, warmup, frame/video output, upload, and teardown.
Do not require the full stack to win: identify the measured frontier and use it
to choose conservative, balanced, and aggressive recipe candidates.
