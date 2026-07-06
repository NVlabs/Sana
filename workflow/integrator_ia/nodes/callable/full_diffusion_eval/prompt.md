Launch through the experiment-local runtime using the fixed first-five-prompt,
720p, 8-second, 193-frame, 24-fps, 50-step contract. Keep cfg 8, flow shift 12,
motion score 20, checkpoint, scheduler, VAE, text encoder, and seed policy fixed.

Load and warm the model, text encoder, VAE, compile/autotune paths, and each
selected recipe before measurement. Time each sample from immediately before
text-encoder compute through synchronized VAE-decode completion. Exclude
startup, all weight loading, compilation, warmup, frame extraction, video
encoding, video writing, upload, and teardown. Record the exact scope id
`warm_single_sample_text_encoder_through_vae_decode`, per-stage timings, and the
spanning `sample_total_s`.

Collect benchmark, run config, five videos or grouped frames, Slurm accounting,
and `integration_stats.json` for conservative, balanced, and aggressive
recipes. Generate visual artifacts outside the timed region. Stats must prove
enabled components really executed with zero fallback and disabled components
had zero activity. Do not report success from an incomplete run.
