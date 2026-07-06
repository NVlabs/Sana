# Composition And Evaluation Policy

## Integration Order

1. Establish an all-guards-off identity path in the integrated source tree and
   verify it against the existing immutable baseline; do not run another
   baseline.
2. Port and validate the kernel canonical-ON stack.
3. Port PISA into the resulting attention implementation without overwriting
   retained kernel changes.
4. Port cache around the composed denoiser without bypassing the guarded kernel
   or PISA paths.
5. Expose independent kernel, PISA, and cache guards and validate enabled and
   disabled activity counters.

## Composition Matrix

Write `COMPOSITION-MATRIX.json` with eight conditions. Reuse the one locked
baseline measurement for `baseline`; measure the other seven conditions:

- `baseline` (`000`): all three components off;
- `kernel_only` (`100`);
- `pisa_only` (`010`);
- `cache_only` (`001`);
- `kernel_pisa` (`110`);
- `kernel_cache` (`101`);
- `pisa_cache` (`011`);
- `full_stack` (`111`).

Every non-baseline condition must run from the same integrated source tree and use the same
fixed model, prompt/seed policy, node, allocation, precision, and timing scope.
Pairwise conditions are mandatory because they distinguish useful composition
from negative interaction. Donor timings are reference context, never
substitutes.

The matrix is an attribution experiment, not a rule that `111` must be shipped.
Record incremental and cumulative effects without multiplying independent
speedups. Select delivered recipes from measured evidence, and omit a runtime
component when its measured marginal effect is negative for that tier.

## Warm Sample Timing

Use `timing.scope = "warm_single_sample_text_encoder_through_vae_decode"`.
Before measurement, load the model, text encoder, and VAE, complete compilation,
autotuning, graph capture, and at least one representative warmup. For each
measured sample:

1. synchronize CUDA and start the spanning wall timer immediately before the
   text-encoder forward;
2. include text-encoder compute, conditioning/pipeline work, all DiT denoising
   steps, and VAE decode;
3. synchronize CUDA after VAE decode and stop the timer;
4. encode/write frames or video only after the timer has stopped.

Each benchmark must record `text_encoder_s`, `dit_denoise_s`, `vae_decode_s`,
and the spanning `sample_total_s`. It must also record that warmup completed,
CUDA synchronization was used, and these stages were excluded:
`process_startup`, `model_load`, `text_encoder_load`, `vae_load`, `compile`,
`warmup`, `video_encode`, and `video_write`. Recipe speedup is the graph-locked
baseline `sample_total_s` divided by the recipe `sample_total_s`. Never
substitute a new all-off denominator.

## Three Recipe Frontier

Write `COMPOSITION-MATRIX.json.recipes` and
`INTEGRATION-STATUS.json.recipes` with exactly `conservative`, `balanced`, and
`aggressive`. Each entry has a distinct candidate id and run directory, its
actual component enablement/settings, and a measured warm-sample speedup.
Speedups must increase strictly in that order. A recipe can use a tuned parameter
point not identical to an untuned toggle-matrix condition, but it must use the
same integrated source and timing contract.

## Official Final Workload

Use exactly the first five prompts of the target model's validation set at the
model's official eval profile (resolution, duration, frame count, fps, steps,
guidance, flow shift, motion score), and unchanged model, scheduler, VAE, text
encoder, and seed policy. Preserve:

- `outputs/benchmark.json`;
- `outputs/run_config.json`;
- `outputs/integration_stats.json`;
- five prompt videos, or all 5 x 193 generated frames in grouped form;
- Slurm accounting and logs.

Run this workload for all three delivered recipes. `integration_stats.json`
must prove every enabled component executed with zero fallback, and every
disabled component had zero activity. PISA-enabled recipes must show exact and
approximate phases; cache-enabled recipes must record the expected number of
denoiser calls with positive hits. A component's environment variable alone is
not evidence.

Visual policy is tiered. Conservative accepts candidate-side severity through
`low`. Balanced accepts `medium`. Aggressive may accept fully disclosed `high`
differences. No tier accepts `critical` or inconclusive evidence. Visual
difference positions a recipe on the frontier rather than automatically
rejecting it. LPIPS is always recorded but is not a standalone hard threshold.
