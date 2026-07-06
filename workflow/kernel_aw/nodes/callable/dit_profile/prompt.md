Use this contract for the initial kernel preflight and for cumulative composition
checkpoints. Load the actual checkpoint and registry-resolved target model
transformer, construct the target model's official latent and conditioning
shapes, and time a complete DiT forward corresponding to one diffusion step. Do
not run the full multi-step sampler, VAE, or video writer.

Required evidence:

- warm repeated full-DiT OFF and ON timings in the same process and allocation;
- explicit warmup, alternating order, median/p25/p75/min/max, and repeat count;
- block-family and dominant-kernel profile with call counts and tensor layouts;
- active implementation classes and dispatch/fallback counters;
- output tensor max/mean error, cosine similarity, shape, and dtype;
- peak allocated/reserved memory when meaningful;
- cold initialization/compile time separated from warm DiT time;
- hashes for the candidate manifest and touched implementation files.

Write `dit_profile.json` and `gate_assess.json`. Set `measurement_scope` to
`registry_resolved_full_dit`, include `candidate_id`, and update
`AGENT-STATUS.json.active_gate` to this gate when it is the current iteration's
authoritative evidence. A module or synthetic trace may supplement this result
but may not replace it.
